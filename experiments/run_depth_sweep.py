#!/usr/bin/env python3
"""
Depth sweep: test whether depth and degree are two routes to the same collapse.

The over-smoothing literature treats collapse as a consequence of DEPTH. This
paper observes it at two layers, driven by DEGREE. If both act through the same
quantity -- how much of the graph a node's receptive field covers -- then the
governing variable is the saturation

    s(L) = min(k^L, N) / N          k = mean degree, L = layers, N = nodes

and MAD should fall as s -> 1 regardless of which of k or L produced it.

Interpretation note: report mad_random alongside mad. Low mad with HIGH
mad_random is homophily, not collapse.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from run_grid import (GNN, HPARAMS, _train, bootstrap_eval, f1_at_percentiles,
                      load_care_gnn, load_elliptic, mad_metric, stratified_split)

def mad_random(emb, sample=20000, seed=0):
    """Mean cosine distance between RANDOM node pairs.

    Defined here rather than imported so these sweeps run against ANY version of
    run_grid.py. MAD over connected pairs alone cannot separate two situations:
    a graph that links genuinely similar nodes (R-U-R links reviews by one user)
    gives low connected-pair distance because the data says so, which is
    informative; representational collapse gives low connected-pair distance
    because every pair is close. Comparing against this random-pair baseline
    distinguishes them:

        mad_random near 0         -> collapse (everything is one point)
        mad_random high, mad low  -> homophily (structure preserved)
    """
    g = torch.Generator().manual_seed(seed)
    n = emb.size(0)
    i = torch.randint(0, n, (sample,), generator=g)
    j = torch.randint(0, n, (sample,), generator=g)
    k = i != j
    a = F.normalize(emb[i[k]], dim=1)
    b = F.normalize(emb[j[k]], dim=1)
    return float((1 - (a * b).sum(1)).mean())


# depths worth testing per dataset: sparse graphs need many layers to saturate,
# dense graphs saturate at 2 so the interesting question is what happens at 1
DEPTHS = {"elliptic": [1, 2, 3, 4, 6, 8], "yelp": [1, 2, 3], "amazon": [1, 2, 3]}
NORMS = ["none", "graph"]        # the contrast that matters; skip the middle


def saturation(k, L, N):
    return min(k ** L, N) / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--datasets", nargs="+", default=["elliptic", "yelp", "amazon"])
    ap.add_argument("--models", nargs="+", default=["GCN", "GAT", "SAGE"])
    ap.add_argument("--norms", nargs="+", default=NORMS)
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="override the per-dataset depth list")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    a = ap.parse_args()
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for ds in a.datasets:
        base = (load_elliptic(a.data_root) if ds == "elliptic"
                else load_care_gnn(a.data_root, ds))
        N = base.x.size(0)
        k = base.edge_index.size(1) / N
        depths = a.layers or DEPTHS[ds]
        print(f"\n{ds}: N={N}, mean degree {k:.1f}")
        for L in depths:
            print(f"   L={L}: receptive-field saturation {saturation(k, L, N):.3f}")

        for L in depths:
            for m in a.models:
                for nrm in a.norms:
                    for s in a.seeds:
                        f = O / f"{ds}_L{L}_{m}_{nrm}_seed{s}.json"
                        if f.exists():
                            print(f"  skip {f.name}"); continue
                        data = (base if ds == "elliptic"
                                else stratified_split(base.clone(), seed=s))
                        hp = HPARAMS[ds][m]
                        torch.manual_seed(s); np.random.seed(s)
                        X, EI = data.x.to(device), data.edge_index.to(device)
                        Y = data.y.clamp(min=0).to(device)
                        trm, vam, tem = (t.to(device) for t in
                                         (data.train_mask, data.val_mask, data.test_mask))
                        try:
                            model = GNN(m, X.size(1), hp["hidden"], hp["emb"], L,
                                        hp["dropout"], nrm, init="default",
                                        aggregator=hp.get("aggr", "mean")).to(device)
                            t0 = time.time()
                            _train(model, X, EI, Y, trm | vam, hp["epochs"], hp["lr"])
                            model.eval()
                            with torch.no_grad():
                                logits, emb = model(X, EI, return_emb=True)
                                prob = F.softmax(logits, dim=1)[:, 1]
                        except torch.cuda.OutOfMemoryError:
                            print(f"  OOM at {ds} L={L} {m}; skipping")
                            torch.cuda.empty_cache()
                            continue
                        yt = Y[tem].cpu().numpy(); ys = prob[tem].cpu().numpy()
                        res = dict(dataset=ds, model=m, norm=nrm, init="default",
                                   seed=s, layers=L, mean_degree=round(k, 2),
                                   n_nodes=int(N),
                                   saturation=round(saturation(k, L, N), 6),
                                   auc=float(roc_auc_score(yt, ys)),
                                   auprc=float(average_precision_score(yt, ys)),
                                   mad=mad_metric(emb.cpu(), data.edge_index),
                                   mad_random=mad_random(emb.cpu()),
                                   train_time_s=round(time.time() - t0, 1))
                        res.update({f"boot_{kk}": v for kk, v in
                                    bootstrap_eval(yt, ys, seed=s).items()})
                        res.update(f1_at_percentiles(yt, ys))
                        f.write_text(json.dumps(res, indent=1))
                        print(f"  {f.name}: s={res['saturation']:.3f} "
                              f"AUPRC={res['auprc']:.4f} MAD={res['mad']:.2e} "
                              f"MADrand={res['mad_random']:.3f} "
                              f"({res['train_time_s']:.0f}s)")
            torch.cuda.empty_cache()

    rows = [json.loads(p.read_text()) for p in O.glob("*.json")]
    pd.DataFrame(rows).to_csv(O.parent / "depth_sweep.csv", index=False)
    print(f"\nMerged {len(rows)} runs -> {O.parent / 'depth_sweep.csv'}")


if __name__ == "__main__":
    main()

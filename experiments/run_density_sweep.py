#!/usr/bin/env python3
"""
Density sweep: fill the gap between mean degree 5 and 300.

The density claim rests on three graphs (mean degree 4.6, 335, 737), i.e. 
one sparse point and two dense ones. This script varies density
on data we already have, by using the *relation-specific* adjacencies that the
YelpChi and Amazon benchmarks ship with, instead of only their union.

  YelpChi   R-U-R  same reviewer                    ~2 mean degree
            R-T-R  same rating, same month          ~25
            R-S-R  same rating on the same product  ~148
            union  (used in the main paper)          335
  Amazon    U-P-U  reviewed the same product        ~29
            U-V-U  top-5% mutual TF-IDF similarity  ~174
            U-S-U  same rating within a week        ~597
            union  (used in the main paper)          737

Nodes, features, labels and splits are identical across relations, so density is
manipulated with everything else held constant. That converts a two-regime
categorical claim into a continuous one.

"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from run_grid import (CARE_GNN_MAT, GNN, HPARAMS, WEIGHT_DECAY, _train,
                      bootstrap_eval, f1_at_percentiles, load_care_gnn,
                      mad_metric, stratified_split)

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


MODELS = ["GCN", "GAT", "SAGE"]
NORMS = ["none", "layer", "graph"]   # within-node vs across-node vs nothing


def relation_graphs(root: str, ds: str):
    """Every relation matrix in the .mat, plus the union used in the main paper."""
    from scipy.io import loadmat
    load_care_gnn(root, ds)                      # ensures the file is downloaded
    m = loadmat(Path(root) / ds / CARE_GNN_MAT[ds])
    feats = m["features"]
    x = torch.tensor(np.asarray(feats.todense() if hasattr(feats, "todense") else feats),
                     dtype=torch.float)
    y = torch.tensor(m["label"].flatten(), dtype=torch.long)
    n = x.size(0)
    out = {}
    for k in m:
        if k.startswith("__") or k in ("features", "label"):
            continue
        try:
            a = csr_matrix(m[k]).tocoo()
        except Exception:
            continue
        if a.shape != (n, n):
            continue
        ei = to_undirected(torch.tensor(np.vstack([a.row, a.col]), dtype=torch.long),
                           num_nodes=n)
        out[k] = (ei, 2 * ei.size(1) / 2 / n)     # edge_index, mean degree
    return x, y, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--datasets", nargs="+", default=["yelp", "amazon"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--norms", nargs="+", default=NORMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    a = ap.parse_args()
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for ds in a.datasets:
        x, y, rels = relation_graphs(a.data_root, ds)
        print(f"\n{ds}: {x.size(0)} nodes, {x.size(1)} feats, relations:")
        for r, (ei, deg) in sorted(rels.items(), key=lambda kv: kv[1][1]):
            print(f"    {r:10s} {ei.size(1):9d} directed edges  mean degree {deg:7.1f}")

        for rel, (ei, deg) in sorted(rels.items(), key=lambda kv: kv[1][1]):
            base = Data(x=x, y=y, edge_index=ei)
            for m in a.models:
                for nrm in a.norms:
                    for s in a.seeds:
                        f = O / f"{ds}_{rel}_{m}_{nrm}_seed{s}.json"
                        if f.exists():
                            print(f"  skip {f.name}"); continue
                        data = stratified_split(base.clone(), seed=s)
                        hp = HPARAMS[ds][m]
                        torch.manual_seed(s); np.random.seed(s)
                        X, EI = data.x.to(device), data.edge_index.to(device)
                        Y = data.y.clamp(min=0).to(device)
                        trm, vam, tem = (t.to(device) for t in
                                         (data.train_mask, data.val_mask, data.test_mask))
                        model = GNN(m, X.size(1), hp["hidden"], hp["emb"], hp["layers"],
                                    hp["dropout"], nrm, init="default",
                                    aggregator=hp.get("aggr", "mean")).to(device)
                        t0 = time.time()
                        _train(model, X, EI, Y, trm | vam, hp["epochs"], hp["lr"])
                        model.eval()
                        with torch.no_grad():
                            logits, emb = model(X, EI, return_emb=True)
                            prob = F.softmax(logits, dim=1)[:, 1]
                        yt = Y[tem].cpu().numpy(); ys = prob[tem].cpu().numpy()
                        res = dict(dataset=ds, relation=rel, mean_degree=round(deg, 2),
                                   n_nodes=int(x.size(0)), n_edges=int(ei.size(1)),
                                   model=m, norm=nrm, init="default", seed=s,
                                   layers=hp["layers"],
                                   auc=float(roc_auc_score(yt, ys)),
                                   auprc=float(average_precision_score(yt, ys)),
                                   mad=mad_metric(emb.cpu(), data.edge_index),
                                   mad_random=mad_random(emb.cpu()),
                                   train_time_s=round(time.time() - t0, 1))
                        res.update({f"boot_{k}": v for k, v in
                                    bootstrap_eval(yt, ys, seed=s).items()})
                        res.update(f1_at_percentiles(yt, ys))
                        f.write_text(json.dumps(res, indent=1))
                        print(f"  {f.name}: k={deg:.0f} AUPRC={res['auprc']:.4f} "
                              f"MAD={res['mad']:.4f} ({res['train_time_s']:.0f}s)")

    rows = [json.loads(p.read_text()) for p in O.glob("*.json")]
    pd.DataFrame(rows).to_csv(O.parent / "density_sweep.csv", index=False)
    print(f"\nMerged {len(rows)} runs -> {O.parent / 'density_sweep.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Re-run the best configuration per architecture per dataset, saving per-node test
scores so that precision-recall curves can be drawn (Figure 6).

run_grid.py stored only summary metrics, so full PR curves cannot be recovered
from the existing results. This script retrains 9 models (3 datasets x 3
architectures) at the configuration that maximised mean test AUPRC, using the
same seed-0 protocol, and writes the raw scores.

Cost: roughly 25-40 min on an L4 (Amazon and YelpChi GAT dominate).

Usage (Colab, after Cells 1-3):
  !cd /content && python save_predictions.py \
      --results /content/drive/MyDrive/journal-ext/results \
      --out     /content/drive/MyDrive/journal-ext/predictions
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from run_grid import (HPARAMS, GNN, load_care_gnn, load_elliptic,
                      stratified_split, _train)

DSETS = ["elliptic", "yelp", "amazon"]
MODELS = ["GCN", "GAT", "SAGE"]


def best_configs(results: Path, ds: str):
    """Configuration with the highest mean test AUPRC for each architecture."""
    df = pd.read_csv(results / f"{ds}_all.csv")
    g = df.groupby(["model", "init", "norm"]).auprc.mean().reset_index()
    return {m: (r["init"], r["norm"])
            for m in MODELS
            for _, r in [g[g.model == m].sort_values("auprc").iloc[[-1]].iterrows().__next__()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--seed", type=int, default=1,
                    help="Use a seed OTHER than run_grid's --convergence-seed "
                         "(default 0) so the RNG path matches exactly.")
    ap.add_argument("--convergence-seed", type=int, default=0,
                    help="Must match the value used by run_grid.py.")
    ap.add_argument("--emb-sample", type=int, default=3000,
                    help="Negative test nodes kept for the embedding figure; "
                         "all positives are kept regardless.")
    a = ap.parse_args()
    R, O = Path(a.results), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Configurations to capture. Beyond the best per architecture, we add
    # unnormalised counterparts on the dense datasets: those are the collapse
    # cases, and the embedding figure needs both sides of the contrast.
    def targets(ds, cfg):
        out = [(m, cfg[m][0], cfg[m][1]) for m in MODELS]
        if ds != "elliptic":
            for m in MODELS:
                t = (m, cfg[m][0], "none")
                if t not in out:
                    out.append(t)
        return out

    for ds in DSETS:
        if not (R / f"{ds}_all.csv").exists():
            print(f"skip {ds} (no results CSV)")
            continue
        cfg = best_configs(R, ds)
        base = (load_elliptic(a.data_root) if ds == "elliptic"
                else load_care_gnn(a.data_root, ds))
        for m, ini, nrm in targets(ds, cfg):
            f = O / f"{ds}_{m}_{ini}_{nrm}_seed{a.seed}.npz"
            if f.exists():
                print(f"skip {f.name}")
                continue
            data = (base if ds == "elliptic"
                    else stratified_split(base.clone(), seed=a.seed))
            hp = HPARAMS[ds][m]

            # RNG protocol must match run_grid.run_one exactly, or the reported
            # model differs. run_grid runs an extra train-only diagnostic pass
            # when seed == its --convergence-seed (default 0), which advances the
            # generator before the reported model is built. Seeds != 0 avoid this
            # entirely; for seed 0 we reproduce the same consumption.
            torch.manual_seed(a.seed)
            np.random.seed(a.seed)
            x, ei = data.x.to(device), data.edge_index.to(device)
            y = data.y.clamp(min=0).to(device)
            trm, vam, tem = (t.to(device) for t in
                             (data.train_mask, data.val_mask, data.test_mask))

            def build():
                return GNN(m, data.x.size(1), hp["hidden"], hp["emb"],
                           hp["layers"], hp["dropout"], nrm, init=ini,
                           aggregator=hp.get("aggr", "mean")).to(device)

            if a.seed == a.convergence_seed:
                _train(build(), x, ei, y, trm, hp["epochs"], hp["lr"],
                       val_mask=vam)

            model = build()
            _train(model, x, ei, y, trm | vam, hp["epochs"], hp["lr"])

            model.eval()
            with torch.no_grad():
                logits, emb = model(x, ei, return_emb=True)
                prob = F.softmax(logits, dim=1)[:, 1]

            # subsample test-node embeddings for the projection figure, keeping
            # every positive so the minority class is not lost to sampling
            te = torch.nonzero(tem).flatten()
            yt = y[te]
            pos = te[yt == 1]
            neg = te[yt == 0]
            g = torch.Generator(device="cpu").manual_seed(a.seed)
            keep_neg = neg[torch.randperm(neg.numel(), generator=g)[
                :min(a.emb_sample, neg.numel())].to(neg.device)]
            sel = torch.cat([pos, keep_neg])

            np.savez_compressed(
                f,
                y_true=y[tem].cpu().numpy().astype(np.int8),
                y_score=prob[tem].cpu().numpy().astype(np.float32),
                emb=emb[sel].cpu().numpy().astype(np.float32),
                emb_y=y[sel].cpu().numpy().astype(np.int8),
                meta=json.dumps(dict(dataset=ds, model=m, init=ini, norm=nrm,
                                     seed=a.seed, emb_dim=int(emb.size(1)))))
            print(f"wrote {f.name}  ({int(tem.sum())} test nodes, "
                  f"{sel.numel()} embeddings)")
    print(f"\nDone -> {O}")


if __name__ == "__main__":
    main()

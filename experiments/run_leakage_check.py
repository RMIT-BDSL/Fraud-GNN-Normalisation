#!/usr/bin/env python3
"""
C3: do normalisation statistics leak test-node information, and does removing
that leak change our finding?

The paper calls the Elliptic protocol leakage-free because the temporal split is
edge-disjoint. That argument covers message passing but not normalisation.
BatchNorm, GraphNorm and PairNorm all pool statistics over nodes, and full-graph
training passes every node through each forward pass. Test-node features
therefore influence training-time activations even though no edge crosses the
split.

This matters because the operators that can leak are exactly the ones the paper
reports as winning:

    can leak    BatchNorm, GraphNorm, PairNorm  -- pool statistics across nodes
    cannot      LayerNorm                       -- per node, over features
    cannot      no normalisation                -- no statistics at all

"Across-node normalisation wins" and "across-node normalisation sees the test
set" are currently supported by the same runs. This script separates them.

METHOD. Every configuration is run twice through the SAME operator
implementation, differing only in which rows the statistics are pooled over:

    stats = all     pool over every node                  (current protocol)
    stats = notest  pool over every node except test      (no test leakage)

Using one implementation for both arms is essential. An earlier draft compared
nn.BatchNorm1d against a masked reimplementation, which differed in two ways at
once: the rows pooled over, AND running-versus-current statistics at eval time.
That confounds the very thing being measured. Here both arms use MaskedNorm and
the only difference is the mask.

MASK CHOICE. The mask is ~test_mask, not train|val. On Elliptic only ~46k of
203,769 nodes carry labels, so masking to train|val would drop ~170k UNLABELLED
nodes from the statistics -- nodes that are not test data and whose removal is
not the intervention we intend. Excluding exactly the evaluated nodes isolates
the leak.

SCOPE. This removes test-LABEL-node leakage. On Elliptic, unlabelled nodes from
future time steps remain visible, so this is a lower bound on strict temporal
isolation, not the strictest possible protocol. State that in the paper.

READING THE OUTPUT.
  1. Sanity block first. 'none' and 'layer' cannot pool across nodes, so they
     must be identical between modes. A difference means the harness is broken
     and nothing else in the run is interpretable.
  2. Comparison block. If batch/graph hold up under notest, the finding survives
     and we report this as a robustness check. If they drop, part of the headline
     was leakage and the claim must change.

NOTE. The 'all' arm here will not be bitwise identical to the main grid, because
the main grid used nn.BatchNorm1d with running statistics at eval. The internal
comparison is what this experiment is for; treat the main grid separately.

Usage:
  python run_leakage_check.py --out /content/drive/MyDrive/journal-ext/leakage
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from run_grid import (GNN, HPARAMS, _train, bootstrap_eval, f1_at_percentiles,
                      load_care_gnn, load_elliptic, mad_metric, stratified_split)


def mad_random(emb, sample=20000, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = emb.size(0)
    i = torch.randint(0, n, (sample,), generator=g)
    j = torch.randint(0, n, (sample,), generator=g)
    k = i != j
    a = F.normalize(emb[i[k]], dim=1)
    b = F.normalize(emb[j[k]], dim=1)
    return float((1 - (a * b).sum(1)).mean())


class MaskedNorm(nn.Module):
    """Across-node normalisation pooling statistics over `mask` rows only.

    mask=None reproduces the usual all-node behaviour, so the same class serves
    both arms of the comparison. Arithmetic mirrors the operator replaced:

      batch : per-feature standardisation across nodes   (cf. nn.BatchNorm1d)
      graph : per-feature, learnable mean shift          (cf. PyG GraphNorm)
      pair  : per-feature centring, ONE global scale     (cf. PyG PairNorm)

    Statistics are computed from the current forward pass in both train and eval
    mode, so the two arms differ only in the mask.
    """

    def __init__(self, kind, dim, eps=1e-5):
        super().__init__()
        self.kind, self.eps, self.mask = kind, eps, None
        if kind in ("batch", "graph"):
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        if kind == "graph":
            self.mean_scale = nn.Parameter(torch.ones(dim))
        if kind == "pair":
            self.scale = 1.0

    def forward(self, x):
        ref = x if self.mask is None else x[self.mask]
        if self.kind == "batch":
            mu = ref.mean(0, keepdim=True)
            var = ref.var(0, unbiased=False, keepdim=True)
            return self.weight * (x - mu) / (var + self.eps).sqrt() + self.bias
        if self.kind == "graph":
            mu = ref.mean(0, keepdim=True)
            shift = self.mean_scale * mu
            var = (ref - shift).pow(2).mean(0, keepdim=True)
            return self.weight * (x - shift) / (var + self.eps).sqrt() + self.bias
        if self.kind == "pair":
            mu = ref.mean(0, keepdim=True)
            denom = (self.eps + (ref - mu).pow(2).sum(-1).mean()).sqrt()
            return self.scale * (x - mu) / denom
        raise ValueError(self.kind)


def norm_width(model, i):
    """Width of the i-th normalisation slot, read from the module it replaces."""
    mod = model.norms[i]
    if hasattr(mod, "num_features"):
        return mod.num_features
    if getattr(mod, "weight", None) is not None:
        return mod.weight.numel()
    out = model.convs[i].out_channels          # PairNorm carries no parameters
    return out * 4 if model.arch == "GAT" else out


def swap_in_masked(model, kind, mask, device):
    """Replace across-node operators with MaskedNorm. `mask=None` -> all nodes."""
    if kind in ("none", "layer"):
        return model                           # cannot pool across nodes
    for i in range(len(model.norms)):
        new = MaskedNorm(kind, norm_width(model, i)).to(device)
        new.mask = mask
        model.norms[i] = new
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--datasets", nargs="+", default=["elliptic", "yelp"])
    ap.add_argument("--models", nargs="+", default=["GCN", "SAGE"])
    ap.add_argument("--norms", nargs="+", default=["none", "layer", "batch", "graph"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    a = ap.parse_args()
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for ds in a.datasets:
        base = (load_elliptic(a.data_root) if ds == "elliptic"
                else load_care_gnn(a.data_root, ds))
        print(f"\n{ds}: {base.x.size(0)} nodes")
        for stats in ["all", "notest"]:
            for m in a.models:
                for nrm in a.norms:
                    for s in a.seeds:
                        f = O / f"{ds}_{m}_{nrm}_{stats}_seed{s}.json"
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
                        model = GNN(m, X.size(1), hp["hidden"], hp["emb"], hp["layers"],
                                    hp["dropout"], nrm, init="default",
                                    aggregator=hp.get("aggr", "mean")).to(device)
                        # both arms use MaskedNorm; only the mask differs
                        model = swap_in_masked(model, nrm,
                                               None if stats == "all" else ~tem, device)
                        t0 = time.time()
                        _train(model, X, EI, Y, trm | vam, hp["epochs"], hp["lr"])
                        model.eval()
                        with torch.no_grad():
                            logits, emb = model(X, EI, return_emb=True)
                            prob = F.softmax(logits, dim=1)[:, 1]
                        yt = Y[tem].cpu().numpy(); ys = prob[tem].cpu().numpy()
                        res = dict(dataset=ds, model=m, norm=nrm, stats=stats, seed=s,
                                   init="default", layers=hp["layers"],
                                   n_stat_nodes=int(X.size(0) if stats == "all"
                                                    else (~tem).sum()),
                                   auc=float(roc_auc_score(yt, ys)),
                                   auprc=float(average_precision_score(yt, ys)),
                                   mad=mad_metric(emb.cpu(), data.edge_index),
                                   mad_random=mad_random(emb.cpu()),
                                   train_time_s=round(time.time() - t0, 1))
                        res.update({f"boot_{k}": v for k, v in
                                    bootstrap_eval(yt, ys, seed=s).items()})
                        res.update(f1_at_percentiles(yt, ys))
                        f.write_text(json.dumps(res, indent=1))
                        print(f"  {f.name}: AUPRC={res['auprc']:.4f} "
                              f"MADrand={res['mad_random']:.4f} ({res['train_time_s']:.0f}s)")

    rows = [json.loads(p.read_text()) for p in O.glob("*.json")]
    df = pd.DataFrame(rows)
    df.to_csv(O.parent / "leakage_check.csv", index=False)
    print(f"\nMerged {len(rows)} runs -> {O.parent / 'leakage_check.csv'}")
    if not len(df):
        return

    print("\n=== SANITY: 'none' and 'layer' must be identical between modes ===")
    ok = True
    for nrm in ["none", "layer"]:
        g = df[df.norm == nrm]
        if not len(g):
            continue
        p = g.pivot_table(index=["dataset", "model", "seed"], columns="stats", values="auprc")
        if {"all", "notest"} <= set(p.columns):
            dmax = (p["all"] - p["notest"]).abs().max()
            good = dmax < 1e-6
            ok &= good
            print(f"  {nrm:6s} max |all - notest| = {dmax:.2e}   "
                  f"{'OK' if good else '*** HARNESS BUG — stop and investigate ***'}")
    print("\n=== EFFECT OF REMOVING TEST-NODE STATISTICS ===")
    g = df[df.norm.isin(["batch", "graph"])]
    if len(g):
        p = g.pivot_table(index=["dataset", "model", "norm"], columns="stats", values="auprc")
        if {"all", "notest"} <= set(p.columns):
            p["delta"] = p["notest"] - p["all"]
            print(p.round(4).to_string())
            print(f"\n  largest drop from removing leakage: {p['delta'].min():+.4f}")
            print("  (a large negative delta means the reported result depended on it)")
    if not ok:
        print("\nSANITY FAILED — do not interpret the comparison above.")


if __name__ == "__main__":
    main()

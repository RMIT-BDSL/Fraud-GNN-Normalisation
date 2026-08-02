#!/usr/bin/env python3
"""
Edge homophily per relation graph. No training, no GPU -- seconds on CPU.

The density sweep varies mean degree across relation-specific adjacencies while
holding nodes, features, labels and splits fixed. But it does NOT hold edge
SEMANTICS fixed: R-U-R links reviews by the same author, R-T-R links reviews
sharing a rating and month. Those differ in how strongly an edge predicts a
shared label, and AUPRC tracks that far more than it tracks density:

    YelpChi   AUPRC  0.59 (k=2) -> 0.21 (k=25) -> 0.26 (k=148) -> 0.26 (k=167)
    Amazon    AUPRC  0.62 (k=29) -> 0.67 (k=174) -> 0.53 (k=597) -> 0.56 (k=737)

Non-monotonic in both. Without this measurement the paper has to concede the
confound; with it, the paper can show the MECHANISM result (mad_random vs
degree) is robust while the ACCURACY result is homophily-driven, which is the
honest and much stronger position.

Reports three quantities per relation:

  h_edge   P(y_u == y_v) over edges. Inflated by class imbalance: a graph that
           is 93% legitimate scores h=0.87 from random wiring alone.
  h_adj    (h_edge - sum_k p_k^2) / (1 - sum_k p_k^2). Zero for random wiring,
           1 for perfect homophily, negative for heterophily. This is the
           comparable number and the one to put in the table.
  h_fraud  P(both endpoints fraudulent | edge). Fraud-ring density; the
           camouflage literature (dou2020caregnn) predicts this is low.

Usage:
  python compute_homophily.py --data-root data
  python compute_homophily.py --data-root data --out ../experiments-output/homophily.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_density_sweep import relation_graphs


def homophily(edge_index: np.ndarray, y: np.ndarray) -> dict:
    """Edge homophily, imbalance-adjusted homophily, and fraud-fraud edge rate."""
    u, v = edge_index[0], edge_index[1]
    same = (y[u] == y[v])
    h_edge = float(same.mean())

    # expected agreement under random wiring with the same class marginals
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    h_rand = float((p ** 2).sum())
    h_adj = (h_edge - h_rand) / (1.0 - h_rand)

    both_fraud = float(((y[u] == 1) & (y[v] == 1)).mean())
    return dict(h_edge=round(h_edge, 4), h_random=round(h_rand, 4),
                h_adj=round(h_adj, 4), h_fraud=round(both_fraud, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--datasets", nargs="+", default=["yelp", "amazon"])
    ap.add_argument("--out", default="../experiments-output/homophily.csv")
    a = ap.parse_args()

    rows = []
    for ds in a.datasets:
        x, y, rels = relation_graphs(a.data_root, ds)
        yn = y.numpy()
        print(f"\n{ds}: {x.size(0)} nodes, fraud rate {(yn == 1).mean():.4f}")
        for rel, (ei, deg) in sorted(rels.items(), key=lambda kv: kv[1][1]):
            h = homophily(ei.numpy(), yn)
            rows.append(dict(dataset=ds, relation=rel, mean_degree=round(deg, 2),
                             n_edges=int(ei.size(1)), **h))
            print(f"  {rel:10s} k={deg:7.2f}  h_edge={h['h_edge']:.4f}  "
                  f"h_adj={h['h_adj']:+.4f}  h_fraud={h['h_fraud']:.4f}")

    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows -> {out}")
    print("\nTo interpret: if h_adj explains the AUPRC ordering better than "
          "mean_degree does, say so in the paper. That is the honest reading, "
          "and it does not touch the mad_random-vs-degree result.")


if __name__ == "__main__":
    main()

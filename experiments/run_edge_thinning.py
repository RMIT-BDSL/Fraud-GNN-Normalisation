#!/usr/bin/env python3
"""
A5: a genuinely controlled density intervention (internal review, P0).

The relation-graph sweep varied mean degree, but relations also differ in edge
semantics, homophily, degree distribution and component structure. Our own sweep
showed that confound: at matched degree, YelpChi GCN = 0.218 against Amazon
GCN = 0.489, and dataset explained more variance than degree (eta^2 0.183 vs
0.090). That experiment supports an association, not a causal claim.

This script fixes it. Take ONE relation graph and randomly delete edges from it.
Edge semantics, node set, features and labels are then identical by construction
across every density level; only the edge count changes. `stratified_split`
draws from `data.y` and the seed alone and never touches `edge_index`, so for a
fixed model seed the train/val/test partition is also bit-identical across
levels. The single manipulated variable is density.

SEEDS ARE DECOUPLED. The internal review (A6/E11) criticised the main grid for
coupling split and parameter seeds, so it would be self-defeating to ship a new
experiment where one seed drives the edge sample, the split AND the
initialisation at once -- variance across seeds could not then be attributed to
edge sampling. `--edge-seeds` and `--seeds` are crossed:

    edge_seed  -> which edges survive
    seed       -> split and parameter initialisation

3 x 3 gives edge-sampling variance at fixed model seed, and model variance at
fixed graph, separately.

BUILT-IN CONTROLS. Two, in the spirit of the leakage check:
  1. keep=1.00 is a no-op, so all three edge seeds must give an identical graph
     and identical diagnostics. Any spread there is a harness bug.
  2. The run asserts that the keep=1.00 edge count matches the untouched graph.
     Earlier this script sampled over `ei[0] < ei[1]`, which silently dropped
     every self-loop at all levels including 100% -- so the anchor would not
     have been the graph the density sweep measured. Sampling is over `<=`.

DIAGNOSTICS. Degree quantiles, isolated-node rate, component count,
largest-component share and edge homophily at every level. Thinning does not
only sparsify, it fragments, and the paper must not attribute fragmentation
effects to density.

Usage:
  python run_edge_thinning.py --out /content/drive/MyDrive/journal-ext/thinning
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
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from run_grid import (CARE_GNN_MAT, GNN, HPARAMS, _train, bootstrap_eval,
                      f1_at_percentiles, load_care_gnn, mad_metric, stratified_split)

# Mid-density single relations. Thinning these spans ~20x in degree. A single
# relation is preferable to the union graph (`homo`), whose edges mix three
# different semantics -- thinning it would vary composition as well as density.
BASE_RELATION = {"yelp": "net_rsr", "amazon": "net_uvu"}
LEVELS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]


def mad_random(emb, sample=20000, seed=0):
    """Mean cosine distance between RANDOM node pairs. Defined here rather than
    imported so the sweep runs against any version of run_grid.py; arithmetic is
    identical to run_grid.mad_random."""
    g = torch.Generator().manual_seed(seed)
    n = emb.size(0)
    i = torch.randint(0, n, (sample,), generator=g)
    j = torch.randint(0, n, (sample,), generator=g)
    k = i != j
    a = F.normalize(emb[i[k]], dim=1)
    b = F.normalize(emb[j[k]], dim=1)
    return float((1 - (a * b).sum(1)).mean())


def load_relation(root, ds, rel):
    from scipy.io import loadmat
    load_care_gnn(root, ds)                       # ensures the file is present
    m = loadmat(Path(root) / ds / CARE_GNN_MAT[ds])
    feats = m["features"]
    x = torch.tensor(np.asarray(feats.todense() if hasattr(feats, "todense") else feats),
                     dtype=torch.float)
    y = torch.tensor(m["label"].flatten(), dtype=torch.long)
    a = csr_matrix(m[rel]).tocoo()
    ei = to_undirected(torch.tensor(np.vstack([a.row, a.col]), dtype=torch.long),
                       num_nodes=x.size(0))
    return x, y, ei


def thin(ei, keep, edge_seed, n):
    """Retain `keep` of the undirected edges, then re-symmetrise.

    Sampling is over unique i <= j pairs, so an edge is dropped in both
    directions -- thinning the directed list would leave half-edges and silently
    change each operator's neighbourhood. `<=` rather than `<` because `<` drops
    self-loops entirely, which would make keep=1.00 a different graph from the
    one the density sweep measured.
    """
    m = ei[0] <= ei[1]
    up = ei[:, m]
    n_keep = int(round(keep * up.size(1)))
    g = torch.Generator().manual_seed(edge_seed)
    idx = torch.randperm(up.size(1), generator=g)[:n_keep]
    return to_undirected(up[:, idx], num_nodes=n)


def diagnostics(ei, y, n):
    """Structure at this retention level, so density and fragmentation can be
    told apart when the results are read."""
    deg = torch.bincount(ei[0], minlength=n).float()
    out = dict(mean_degree=float(deg.mean()),
               deg_p50=float(deg.median()), deg_p90=float(deg.quantile(0.9)),
               deg_max=float(deg.max()),
               isolated_frac=float((deg == 0).float().mean()))
    lab = y >= 0
    e = ei[:, lab[ei[0]] & lab[ei[1]]] if ei.size(1) else ei
    out["edge_homophily"] = (float((y[e[0]] == y[e[1]]).float().mean())
                             if e.size(1) else float("nan"))
    A = csr_matrix((np.ones(ei.size(1)), (ei[0].numpy(), ei[1].numpy())), shape=(n, n))
    ncomp, lbl = connected_components(A, directed=False)
    out["n_components"] = int(ncomp)
    out["largest_cc_frac"] = float(np.bincount(lbl).max() / n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--datasets", nargs="+", default=["yelp", "amazon"])
    ap.add_argument("--models", nargs="+", default=["GCN", "SAGE"])
    ap.add_argument("--norms", nargs="+", default=["none", "graph"])
    ap.add_argument("--levels", nargs="+", type=float, default=LEVELS)
    ap.add_argument("--edge-seeds", nargs="+", type=int, default=[1, 2, 3],
                    help="controls which edges survive")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3],
                    help="controls split and parameter initialisation")
    a = ap.parse_args()
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for ds in a.datasets:
        rel = BASE_RELATION[ds]
        x, y, ei_full = load_relation(a.data_root, ds, rel)
        n = x.size(0)
        print(f"\n{ds} / {rel}: {n} nodes, {ei_full.size(1)} directed edges, "
              f"mean degree {ei_full.size(1)/n:.1f}")

        # CONTROL 2: keep=1.00 must reproduce the untouched graph exactly.
        chk = thin(ei_full, 1.0, a.edge_seeds[0], n)
        assert chk.size(1) == ei_full.size(1), (
            f"keep=1.0 gave {chk.size(1)} edges, expected {ei_full.size(1)} — "
            "thinning is not identity at full retention; do not trust the sweep")
        print(f"  control: keep=1.00 reproduces the graph exactly "
              f"({chk.size(1)} edges) OK")

        for keep in a.levels:
            for es in a.edge_seeds:
                ei = thin(ei_full, keep, edge_seed=es, n=n)
                dg = diagnostics(ei, y, n)
                print(f"  keep={keep:5.0%} es={es}  k={dg['mean_degree']:7.1f}  "
                      f"isolated={dg['isolated_frac']:.3f}  "
                      f"cc={dg['n_components']:6d}  "
                      f"largest={dg['largest_cc_frac']:.3f}  "
                      f"homoph={dg['edge_homophily']:.3f}")
                base = Data(x=x, y=y, edge_index=ei)
                for m in a.models:
                    for nrm in a.norms:
                        for s in a.seeds:
                            f = O / (f"{ds}_{rel}_keep{int(keep*100):03d}"
                                     f"_e{es}_{m}_{nrm}_seed{s}.json")
                            if f.exists():
                                continue
                            data = stratified_split(base.clone(), seed=s)
                            hp = HPARAMS[ds][m]
                            torch.manual_seed(s); np.random.seed(s)
                            X, EI = data.x.to(device), data.edge_index.to(device)
                            Y = data.y.clamp(min=0).to(device)
                            trm, vam, tem = (t.to(device) for t in
                                             (data.train_mask, data.val_mask,
                                              data.test_mask))
                            model = GNN(m, X.size(1), hp["hidden"], hp["emb"],
                                        hp["layers"], hp["dropout"], nrm,
                                        init="default",
                                        aggregator=hp.get("aggr", "mean")).to(device)
                            t0 = time.time()
                            _train(model, X, EI, Y, trm | vam, hp["epochs"], hp["lr"])
                            model.eval()
                            with torch.no_grad():
                                logits, emb = model(X, EI, return_emb=True)
                                prob = F.softmax(logits, dim=1)[:, 1]
                            yt = Y[tem].cpu().numpy(); ys = prob[tem].cpu().numpy()
                            res = dict(dataset=ds, relation=rel, keep=keep,
                                       edge_seed=es, seed=s, model=m, norm=nrm,
                                       init="default", layers=hp["layers"],
                                       n_nodes=int(n), n_edges=int(ei.size(1)),
                                       auc=float(roc_auc_score(yt, ys)),
                                       auprc=float(average_precision_score(yt, ys)),
                                       mad=mad_metric(emb.cpu(), data.edge_index),
                                       mad_random=mad_random(emb.cpu()),
                                       train_time_s=round(time.time() - t0, 1), **dg)
                            res.update({f"boot_{k}": v for k, v in
                                        bootstrap_eval(yt, ys, seed=s).items()})
                            res.update(f1_at_percentiles(yt, ys))
                            f.write_text(json.dumps(res, indent=1))
                            print(f"    {m:5s} {nrm:6s} e{es}s{s} "
                                  f"AUPRC={res['auprc']:.4f} "
                                  f"MADrand={res['mad_random']:.4f} "
                                  f"({res['train_time_s']:.0f}s)")

    rows = [json.loads(p.read_text()) for p in O.glob("*.json")]
    df = pd.DataFrame(rows)
    df.to_csv(O.parent / "edge_thinning.csv", index=False)
    print(f"\nMerged {len(rows)} runs -> {O.parent / 'edge_thinning.csv'}")
    if not len(df):
        return

    # CONTROL 1: at 100% retention the edge seed is a no-op, so spread across
    # edge seeds there is pure run-to-run noise -- the floor everything else
    # must clear, exactly as the control arms served in the leakage check.
    print("\n=== CONTROL: spread across edge seeds at keep=1.00 (must be ~0) ===")
    full = df[df.keep == 1.0]
    if len(full):
        sp = (full.groupby(["dataset", "model", "norm", "seed"]).auprc
                  .agg(lambda v: v.max() - v.min()))
        print(f"  max spread over edge seeds at full retention: {sp.max():.2e}")
        print("  (a nonzero value means the graph changed when it should not have)")

    print("\n=== VARIANCE SOURCES (std of AUPRC) ===")
    for (ds, m, nrm), g in df.groupby(["dataset", "model", "norm"]):
        e_var = g.groupby(["keep", "seed"]).auprc.std().mean()      # edge sampling
        m_var = g.groupby(["keep", "edge_seed"]).auprc.std().mean() # split + init
        print(f"  {ds:7s} {m:5s} {nrm:6s}  edge-sample {e_var:.4f}   model {m_var:.4f}")

    print("\n=== degree -> collapse, WITHIN one relation graph (semantics fixed) ===")
    from scipy import stats
    for (ds, m), g in df[df.norm == "none"].groupby(["dataset", "model"]):
        r, p = stats.spearmanr(g.mean_degree, g.mad_random)
        r2, p2 = stats.spearmanr(g.mean_degree, g.auprc)
        print(f"  {ds:7s} {m:5s}  rho(k, MADrand) = {r:+.2f}{'*' if p < .05 else ' '}"
              f"   rho(k, AUPRC) = {r2:+.2f}{'*' if p2 < .05 else ' '}")
    print("\n  A negative rho(k, MADrand) is causal evidence here: nothing but the")
    print("  edge count differs between levels. Before attributing it to density,")
    print("  check isolated_frac and largest_cc_frac — at low retention the graph")
    print("  fragments, and that is a different mechanism.")

    # ------------------------------------------------------------------
    # C1 taxonomy readout — only meaningful once layer/pair (and ideally
    # batch) have been run. The corrected C1 axis predicts that operators
    # standardising per-feature variance ACROSS nodes (batch, graph) hold
    # dispersion roughly constant as density rises, while within-node (layer)
    # and single-global-scalar (pair) operators collapse like `none`. This
    # block prints exactly that contrast under the controlled manipulation.
    # ------------------------------------------------------------------
    CLASS = {"none": "baseline", "graph": "across-node std",
             "batch": "across-node std", "layer": "within-node",
             "pair": "global-scalar"}
    present = [n for n in ["none", "graph", "batch", "layer", "pair"]
               if n in set(df.norm.unique())]
    if {"layer", "pair"} & set(present):
        print("\n=== C1 TAXONOMY UNDER CONTROLLED DENSITY ===")
        print("  MADrand at sparsest kept level (keep=0.20) vs densest (keep=1.00),")
        print("  averaged over edge/model seeds. Across-node operators should stay")
        print("  flat (fold ~1); within-node / global-scalar should collapse like none.\n")
        lo, hi = 0.20, 1.00
        for (ds, m), g in df.groupby(["dataset", "model"]):
            print(f"  {ds} {m}")
            for nrm in present:
                gg = g[g.norm == nrm]
                a = gg[gg.keep == lo].mad_random.mean()
                b = gg[gg.keep == hi].mad_random.mean()
                ap = gg[gg.keep == hi].auprc.mean()
                fold = (a / b) if b > 0 else float("inf")
                print(f"    {nrm:6s} [{CLASS[nrm]:16s}]  MADrand {a:.3f}->{b:.3f}  "
                      f"fold {fold:5.1f}   AUPRC@full {ap:.3f}")
        print("\n  Read: a within-node (layer) or global-scalar (pair) fold close to")
        print("  none's, against an across-node (graph/batch) fold near 1, is the")
        print("  controlled evidence for the C1 axis. If layer/pair instead hold")
        print("  dispersion like graph, C1 is wrong and must be revisited BEFORE")
        print("  submission — that is the check this arm exists to make.")
    else:
        print("\n  [C1 taxonomy readout skipped: run with --norms layer pair "
              "(and ideally batch) to populate it.]")


if __name__ == "__main__":
    main()

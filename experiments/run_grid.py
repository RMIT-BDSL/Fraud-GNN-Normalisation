#!/usr/bin/env python3
"""
Grid: {GCN, GAT, SAGE} x {default, xavier, kaiming} x {none, batch, layer, graph, pair}
      x 5 seeds x {elliptic, yelp, amazon}

Design goals:
  - Resume-safe: each (dataset, model, init, norm, seed) run writes one JSON.
    Existing JSONs are skipped, so Colab disconnects cost nothing.
  - No Kaggle auth needed: Elliptic comes via PyG's mirror; Yelp/Amazon via CARE-GNN repo.
  - Reuses conference-paper hyperparameters (no new Optuna) for comparability.

Usage:
  python run_grid.py --dataset elliptic --out /content/drive/MyDrive/journal-ext/results
  python run_grid.py --dataset yelp     --out ...
  python run_grid.py --dataset amazon   --out ...
  Optional filters: --models GCN SAGE --inits xavier --norms graph pair --seeds 0 1 2
"""
import argparse
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.nn import GATConv, GCNConv, GraphNorm, PairNorm, SAGEConv
from torch_geometric.utils import to_undirected

# ----------------------------------------------------------------------------
# Conference-paper tuned hyperparameters (Table 1 of EANN-26)
# ----------------------------------------------------------------------------
HPARAMS = {
    "elliptic": {
        "GAT":  dict(lr=6.999e-4, hidden=148, emb=89,  layers=2, dropout=0.2522, epochs=508),
        "GCN":  dict(lr=8.475e-4, hidden=211, emb=90,  layers=2, dropout=0.2361, epochs=497),
        "SAGE": dict(lr=5.302e-4, hidden=140, emb=103, layers=2, dropout=0.1135, epochs=397),
    },
    # YelpChi and Amazon use the union of relation-specific graphs and are therefore
    # very dense (mean degree ~335 and ~737). GAT materialises one message per edge of
    # width hidden*heads, so hidden=128 with 4 heads needs ~16 GB on YelpChi and OOMs a
    # 16 GB GPU. We set GAT hidden=32 with 4 heads, giving a concatenated width of 128 —
    # equal to the GCN/SAGE hidden width, so effective layer capacity is matched.
    "yelp": {
        "GAT":  dict(lr=5e-4, hidden=32,  emb=64, layers=2, dropout=0.2, epochs=300),
        "GCN":  dict(lr=5e-4, hidden=128, emb=64, layers=2, dropout=0.2, epochs=300),
        "SAGE": dict(lr=5e-4, hidden=128, emb=64, layers=2, dropout=0.2, epochs=300),
    },
    "amazon": {
        "GAT":  dict(lr=5e-4, hidden=32,  emb=64, layers=2, dropout=0.2, epochs=300),
        "GCN":  dict(lr=5e-4, hidden=128, emb=64, layers=2, dropout=0.2, epochs=300),
        "SAGE": dict(lr=5e-4, hidden=128, emb=64, layers=2, dropout=0.2, epochs=300),
    },
}
WEIGHT_DECAY = 5e-4

CARE_GNN_URLS = {
    "yelp":   "https://github.com/YingtongDou/CARE-GNN/raw/master/data/YelpChi.zip",
    "amazon": "https://github.com/YingtongDou/CARE-GNN/raw/master/data/Amazon.zip",
}
CARE_GNN_MAT = {"yelp": "YelpChi.mat", "amazon": "Amazon.mat"}


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_elliptic(root: str) -> Data:
    """Download via PyG mirror, then rebuild from raw CSVs so we keep the time
    step column and can apply the paper's 29/10/10 temporal split."""
    ds = EllipticBitcoinDataset(root=str(Path(root) / "elliptic"))  # triggers download
    raw = Path(ds.raw_dir)
    feats = pd.read_csv(raw / "elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(raw / "elliptic_txs_classes.csv")
    edges = pd.read_csv(raw / "elliptic_txs_edgelist.csv")

    txid = feats[0].values
    tstep = feats[1].values.astype(int)          # 1..49
    x = torch.tensor(feats.iloc[:, 2:].values, dtype=torch.float)  # 165 aggregated+local feats
    id2idx = {t: i for i, t in enumerate(txid)}

    lbl_map = {"1": 1, "2": 0, "unknown": -1}    # 1=illicit -> positive class 1
    y = torch.full((len(txid),), -1, dtype=torch.long)
    for t, c in zip(classes["txId"].values, classes["class"].astype(str).values):
        y[id2idx[t]] = lbl_map[c]

    ei = torch.tensor(
        [[id2idx[a] for a in edges["txId1"].values],
         [id2idx[b] for b in edges["txId2"].values]], dtype=torch.long)
    ei = to_undirected(ei, num_nodes=len(txid))

    t = torch.tensor(tstep)
    labeled = y >= 0
    data = Data(x=x, edge_index=ei, y=y)
    data.train_mask = labeled & (t <= 29)
    data.val_mask = labeled & (t > 29) & (t <= 39)
    data.test_mask = labeled & (t > 39)
    return data


def load_care_gnn(root: str, name: str) -> Data:
    """YelpChi / Amazon fraud graphs (homogeneous adjacency) from the CARE-GNN repo.
    No auth needed. Stratified 60/20/20 split is applied per-seed at train time."""
    from scipy.io import loadmat
    from scipy.sparse import csr_matrix

    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    matfile = d / CARE_GNN_MAT[name]
    if not matfile.exists():
        zpath = d / "data.zip"
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(CARE_GNN_URLS[name], zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(d)
        # the zip may nest the .mat; locate it
        for p in d.rglob("*.mat"):
            p.rename(matfile)
            break
    m = loadmat(matfile)
    adj = csr_matrix(m["homo"]).tocoo()
    x = torch.tensor(np.asarray(m["features"].todense() if hasattr(m["features"], "todense")
                                else m["features"]), dtype=torch.float)
    y = torch.tensor(m["label"].flatten(), dtype=torch.long)
    ei = to_undirected(torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long),
                       num_nodes=x.size(0))
    return Data(x=x, edge_index=ei, y=y)


def stratified_split(data: Data, seed: int, frac=(0.6, 0.2, 0.2)):
    g = torch.Generator().manual_seed(seed)
    n = data.y.size(0)
    train = torch.zeros(n, dtype=torch.bool)
    val = torch.zeros(n, dtype=torch.bool)
    test = torch.zeros(n, dtype=torch.bool)
    for c in torch.unique(data.y):
        idx = torch.nonzero(data.y == c).flatten()
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        n1, n2 = int(frac[0] * idx.numel()), int((frac[0] + frac[1]) * idx.numel())
        train[idx[:n1]] = True
        val[idx[n1:n2]] = True
        test[idx[n2:]] = True
    data.train_mask, data.val_mask, data.test_mask = train, val, test
    return data


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def make_norm(kind: str, dim: int) -> nn.Module:
    return {
        "none":  nn.Identity(),
        "batch": nn.BatchNorm1d(dim),
        "layer": nn.LayerNorm(dim),
        "graph": GraphNorm(dim),
        "pair":  PairNorm(),
    }[kind]


class GNN(nn.Module):
    """Faithful replication of the conference repo's GCN/GAT/SAGE modules
    (RMIT-BDSL/Blockchain-Anomaly-Detection, model/*.py), generalised so any
    norm layer can occupy the GraphNorm slot:

      conv_1(in->hidden) -> [norm] -> ReLU -> dropout ->
      conv_L(hidden->emb)                                  (no norm/act after last conv)
      -> Linear(emb->2)

    GAT specifics from the repo: 4 heads with concat on hidden layers, final
    GAT layer heads=1 concat=False, attention dropout = dropout, slope 0.2.
    Init semantics from the repo: convs ALWAYS use default reset_parameters();
    the init scheme applies to the output Linear head only."""

    def __init__(self, arch, in_dim, hidden, emb, layers, dropout, norm,
                 init="default", aggregator="mean", n_heads=4):
        super().__init__()
        self.arch, self.dropout, self.init = arch, dropout, init
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        def block(i, d_in, d_out, last):
            if arch == "GCN":
                self.convs.append(GCNConv(d_in, d_out, cached=True))
            elif arch == "SAGE":
                self.convs.append(SAGEConv(d_in, d_out, aggr=aggregator))
            elif arch == "GAT":
                if last:
                    self.convs.append(GATConv(d_in, d_out, heads=1, concat=False,
                                              dropout=dropout, negative_slope=0.2))
                else:
                    self.convs.append(GATConv(d_in, d_out, heads=n_heads, concat=True,
                                              dropout=dropout, negative_slope=0.2))
            if not last:
                width = d_out * (n_heads if arch == "GAT" else 1)
                self.norms.append(make_norm(norm, width))

        if layers == 1:
            block(0, in_dim, emb, last=True)
        else:
            block(0, in_dim, hidden, last=False)
            w = hidden * (n_heads if arch == "GAT" else 1)
            for i in range(layers - 2):
                block(i + 1, w, hidden, last=False)
            block(layers - 1, w, emb, last=True)

        self.head = nn.Linear(emb, 2)
        self.reset_parameters()

    def reset_parameters(self):
        for c in self.convs:
            c.reset_parameters()
        for n in self.norms:
            if hasattr(n, "reset_parameters"):
                n.reset_parameters()
        if self.init == "xavier":
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        elif self.init == "kaiming":
            nn.init.kaiming_uniform_(self.head.weight, nonlinearity="relu")
            nn.init.zeros_(self.head.bias)
        else:  # default
            self.head.reset_parameters()

    def forward(self, x, ei, return_emb=False):
        last = len(self.convs) - 1
        for i, conv in enumerate(self.convs):
            x = conv(x, ei)
            if i < last:
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if return_emb:
            return self.head(x), x
        return self.head(x)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def mad_metric(emb: torch.Tensor, ei: torch.Tensor, sample=20000, seed=0) -> float:
    """Mean Average Distance (cosine) over connected node pairs — over-smoothing
    proxy (lower = more over-smoothed)."""
    g = torch.Generator().manual_seed(seed)
    m = ei.size(1)
    idx = torch.randperm(m, generator=g)[: min(sample, m)]
    a = F.normalize(emb[ei[0, idx]], dim=1)
    b = F.normalize(emb[ei[1, idx]], dim=1)
    return float((1 - (a * b).sum(1)).mean())


def mad_random(emb, sample=20000, seed=0):
    """Mean cosine distance between RANDOM node pairs.

    MAD over connected pairs cannot on its own distinguish two very different
    situations. If a graph links genuinely similar nodes (R-U-R links reviews by
    one user), connected pairs are close because the data says so, and that is
    informative. Under representational collapse, connected pairs are close
    because *every* pair is close. Comparing the connected-pair statistic with
    this random-pair baseline separates the two:

        mad_random near 0            -> collapse (everything is one point)
        mad_random high, mad low     -> homophily (the model preserved structure)
    """
    import torch as _t
    g = _t.Generator().manual_seed(seed)
    n = emb.size(0)
    i = _t.randint(0, n, (sample,), generator=g)
    j = _t.randint(0, n, (sample,), generator=g)
    k = i != j
    a = F.normalize(emb[i[k]], dim=1)
    b = F.normalize(emb[j[k]], dim=1)
    return float((1 - (a * b).sum(1)).mean())


def bootstrap_eval(y_true, y_score, n_boot=100, frac=0.5, seed=0):
    """Conference paper protocol: 100x subsample 50% of test nodes."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs, aps = [], []
    for _ in range(n_boot):
        idx = rng.choice(n, int(frac * n), replace=False)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        aps.append(average_precision_score(y_true[idx], y_score[idx]))
    return dict(auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
                auprc_mean=float(np.mean(aps)), auprc_std=float(np.std(aps)))


def f1_at_percentiles(y_true, y_score, pcts=(90, 99, 99.9)):
    out = {}
    for p in pcts:
        thr = np.percentile(y_score, p)
        pred = (y_score >= thr).astype(int)
        out[f"f1@{p}"] = float(f1_score(y_true, pred, zero_division=0))
        out[f"prec@{p}"] = float(precision_score(y_true, pred, zero_division=0))
        out[f"rec@{p}"] = float(recall_score(y_true, pred, zero_division=0))
    return out


# ----------------------------------------------------------------------------
# Single run
# ----------------------------------------------------------------------------
def _train(model, x, ei, y, train_mask, epochs, lr, val_mask=None):
    """Repo-faithful loop: full-graph, Adam, UNWEIGHTED CrossEntropy, no early
    stopping. If val_mask given, records per-epoch val AUPRC (diagnostics)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    val_curve = []
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, ei)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()
        if val_mask is not None:
            model.eval()
            with torch.no_grad():
                prob = F.softmax(model(x, ei), dim=1)[:, 1]
            vt = y[val_mask].cpu().numpy()
            vs = prob[val_mask].cpu().numpy()
            val_curve.append(average_precision_score(vt, vs)
                             if len(np.unique(vt)) > 1 else 0.0)
    return val_curve


def run_one(data, dataset, arch, init, norm, seed, device, convergence=False):
    """Two phases, replicating the conference protocol (scripts/train.py):
      1. (optional, diagnostics) train on train-only, track val AUPRC per epoch
         -> convergence metrics (epoch-to-best-val, best val AUPRC).
      2. (headline) train a FRESH model on train UNION val for the tuned epoch
         count, evaluate the final-epoch model once on the held-out test set.
    """
    hp = HPARAMS[dataset][arch]
    torch.manual_seed(seed)
    np.random.seed(seed)

    def build():
        return GNN(arch, data.x.size(1), hp["hidden"], hp["emb"], hp["layers"],
                   hp["dropout"], norm, init=init,
                   aggregator=hp.get("aggr", "mean")).to(device)

    y = data.y.clamp(min=0).to(device)
    x, ei = data.x.to(device), data.edge_index.to(device)
    trm, vam, tem = (m.to(device) for m in (data.train_mask, data.val_mask, data.test_mask))

    t0 = time.time()
    best_val, best_epoch = float("nan"), -1
    if convergence:
        val_curve = _train(build(), x, ei, y, trm, hp["epochs"], hp["lr"], val_mask=vam)
        best_val = float(np.max(val_curve))
        best_epoch = int(np.argmax(val_curve))

    # Phase 2: final model on train+val, last epoch = reported model
    model = build()
    _train(model, x, ei, y, trm | vam, hp["epochs"], hp["lr"])
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        logits, emb = model(x, ei, return_emb=True)
        prob = F.softmax(logits, dim=1)[:, 1]
    yt = y[tem].cpu().numpy()
    ys = prob[tem].cpu().numpy()

    res = dict(dataset=dataset, model=arch, init=init, norm=norm, seed=seed,
               auc=float(roc_auc_score(yt, ys)),
               auprc=float(average_precision_score(yt, ys)),
               val_auprc=best_val, best_epoch=best_epoch,
               train_time_s=round(train_time, 1),
               mad=mad_metric(emb.cpu(), data.edge_index))
    res.update({f"boot_{k}": v for k, v in bootstrap_eval(yt, ys, seed=seed).items()})
    res.update(f1_at_percentiles(yt, ys))
    return res


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["elliptic", "yelp", "amazon"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--models", nargs="+", default=["GCN", "GAT", "SAGE"])
    ap.add_argument("--inits", nargs="+", default=["default", "xavier", "kaiming"])
    ap.add_argument("--norms", nargs="+", default=["none", "batch", "layer", "graph", "pair"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--convergence-seed", type=int, default=0,
                    help="Seed for which the extra train-only convergence phase runs "
                         "(diagnostics roughly double that seed's runtime). -1 disables.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    out = Path(args.out) / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "elliptic":
        base = load_elliptic(args.data_root)
    else:
        base = load_care_gnn(args.data_root, args.dataset)
    print(f"{args.dataset}: {base.x.size(0)} nodes, {base.edge_index.size(1)} edges, "
          f"{base.x.size(1)} feats")

    todo = [(m, i, n, s) for m in args.models for i in args.inits
            for n in args.norms for s in args.seeds]
    for k, (m, i, n, s) in enumerate(todo, 1):
        f = out / f"{m}_{i}_{n}_seed{s}.json"
        if f.exists():
            print(f"[{k}/{len(todo)}] skip {f.name}")
            continue
        data = base if args.dataset == "elliptic" else stratified_split(base.clone(), seed=s)
        t0 = time.time()
        res = run_one(data, args.dataset, m, i, n, s, device,
                      convergence=(s == args.convergence_seed))
        f.write_text(json.dumps(res, indent=1))
        print(f"[{k}/{len(todo)}] {f.name}: AUPRC={res['auprc']:.4f} "
              f"AUC={res['auc']:.4f} MAD={res['mad']:.3f} ({time.time()-t0:.0f}s)")

    # merge everything present into one CSV
    rows = [json.loads(p.read_text()) for p in out.glob("*.json")]
    pd.DataFrame(rows).to_csv(out.parent / f"{args.dataset}_all.csv", index=False)
    print(f"Merged {len(rows)} runs -> {out.parent / f'{args.dataset}_all.csv'}")


if __name__ == "__main__":
    main()

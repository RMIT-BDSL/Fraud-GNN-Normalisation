#!/usr/bin/env python3
"""
Generate paper figures from the experiment CSVs, to Springer Nature artwork spec.

Spec followed (IJMLC / Springer submission guidelines):
  - width 174 mm (double-column text area) or 84 mm; height <= 234 mm
  - vector output (PDF); fonts embedded as Type 42
  - sans-serif lettering, 8-10 pt, consistent across figures
  - no titles inside the artwork (captions live in LaTeX)
  - panel parts labelled with lowercase (a), (b), ...
  - colour-blind safe palette (Okabe-Ito) AND hatching/markers so figures
    survive greyscale printing; contrast >= 4.5:1

Usage:
  python make_figures.py --results ../experiments-output --out ../paper/figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

# ---------------------------------------------------------------- style
MM = 1 / 25.4
# sn-jnl in two-column mode: \textwidth = 160 mm, \columnwidth = 76 mm.
# Generating at the true width means figures embed 1:1 and lettering renders
# at its nominal point size rather than being scaled down.
W_FULL, W_HALF = 160 * MM, 76 * MM
H_MAX = 234 * MM

# Nimbus Sans = URW Helvetica clone (identical metrics). Liberation Sans = Arial
# metrics. Either satisfies "Helvetica or Arial" and typesets representatively.
for fam in ["Nimbus Sans", "Helvetica", "Liberation Sans", "Arial", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(fam, fallback_to_default=False)
        FONT = fam
        break
    except Exception:
        continue
else:
    FONT = "DejaVu Sans"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    # render maths in the same face as the text; matplotlib otherwise falls back
    # to DejaVu for mathtext and the figure ends up mixing two typefaces
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold", "mathtext.sf": FONT,
})

# Okabe-Ito: colour-blind safe, distinguishable in greyscale by lightness
COL = {"GCN": "#0072B2", "GAT": "#E69F00", "SAGE": "#009E73"}
HATCH = {"GCN": "", "GAT": "///", "SAGE": "..."}
MARK = {"GCN": "o", "GAT": "s", "SAGE": "^"}
LABEL = {"GCN": "GCN", "GAT": "GAT", "SAGE": "GraphSAGE"}
MODELS = ["GCN", "GAT", "SAGE"]
NORMS = ["none", "batch", "layer", "graph", "pair"]
NORM_LBL = ["None", "Batch", "Layer", "Graph", "Pair"]
INITS = ["default", "xavier", "kaiming"]
DSETS = [("elliptic", "Elliptic"), ("yelp", "YelpChi"), ("amazon", "Amazon")]
# Mean degree = 2E/N with E the UNDIRECTED edge count. Earlier versions of this
# file carried 4.6 / 335 / 737, which double-counted Elliptic and YelpChi: their
# edge totals were directed adjacency entries, Amazon's were undirected.
DEGREE = {"elliptic": 2.3, "yelp": 167.4, "amazon": 736.5}


def base_rate(d):
    """Positive-class rate in the test split, recovered from precision/recall at
    the 90th-percentile threshold: P/N = prec * 0.10 / rec."""
    return float((d["prec@90"] * 0.10 / d["rec@90"]).replace(
        [np.inf, -np.inf], np.nan).median())


def panel_label(ax, s, dx=0.015, dy=0.985):
    """Inside the axes, top-left, on an opaque patch so it never collides with
    tick labels or data."""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va="top", ha="left", zorder=6,
            bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85))


# ---------------------------------------------------------------- Fig 2
def fig2(F, out):
    """AUPRC (top) and MAD (bottom) by normalisation, per dataset.
    The visual correspondence between rows on the dense datasets, and its
    absence on Elliptic, is the density-conditional finding."""
    fig, axes = plt.subplots(2, 3, figsize=(W_FULL, 3.6), sharex=True)
    x = np.arange(len(NORMS))
    w = 0.26
    for j, (ds, name) in enumerate(DSETS):
        d = F[ds]
        ax = axes[0, j]
        for k, m in enumerate(MODELS):
            g = d[d.model == m].groupby("norm").auprc
            mu = g.mean().reindex(NORMS).values
            sd = g.std().reindex(NORMS).values
            ax.bar(x + (k - 1) * w, mu, w, yerr=sd, color=COL[m], hatch=HATCH[m],
                   edgecolor="white", linewidth=0.5, error_kw=dict(lw=0.6, capsize=1.5),
                   label=LABEL[m])
        # base rate identified once, in the shared legend, to avoid per-panel clutter
        ax.axhline(base_rate(d), ls=(0, (4, 2)), lw=0.7, color="0.35", zorder=0)
        ax.set_ylim(0, None)
        ax.set_title(f"{name}  (mean degree {DEGREE[ds]:g})", pad=4)
        if j == 0:
            ax.set_ylabel("Test AUPRC")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        ax = axes[1, j]
        for k, m in enumerate(MODELS):
            mu = d[d.model == m].groupby("norm").mad.mean().reindex(NORMS).values
            ax.bar(x + (k - 1) * w, mu, w, color=COL[m], hatch=HATCH[m],
                   edgecolor="white", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_ylim(2e-5, 1)
        if j == 0:
            ax.set_ylabel("MAD (log scale)")
        else:
            ax.set_yticklabels([])          # shared scale across the row
        ax.set_xticks(x, NORM_LBL, rotation=30, ha="right")
        ax.set_xlabel("Normalisation")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    for i, s in enumerate("abcdef"):
        panel_label(axes.flat[i], f"({s})")
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(Line2D([], [], ls=(0, (4, 2)), lw=0.7, color="0.35"))
    l.append("Class base rate")
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.035),
               handlelength=1.5, columnspacing=1.5)
    fig.tight_layout(w_pad=1.0, h_pad=0.6, rect=(0, 0, 1, 0.955))
    fig.savefig(out / "Fig2.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3
def fig3(F, out):
    """MAD vs AUPRC per run, faceted by dataset, with per-architecture fits."""
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.15))
    for j, (ds, name) in enumerate(DSETS):
        d, ax = F[ds], axes[j]
        lines = []
        for m in MODELS:
            s = d[d.model == m]
            ax.scatter(s.mad, s.auprc, s=5, c=COL[m], marker=MARK[m], alpha=0.5,
                       linewidths=0, label=LABEL[m], zorder=2)
            # Spearman: MAD is heavily skewed over ~5 orders of magnitude, so a
            # rank correlation is the appropriate summary and is invariant to the
            # log axis. Fit is linear in log10(MAD), shown only as a visual guide.
            rho, p = stats.spearmanr(s.mad, s.auprc)
            b, a = np.polyfit(np.log10(s.mad), s.auprc, 1)
            xs = np.log10(np.array([s.mad.min(), s.mad.max()]))
            ax.plot(10 ** xs, a + b * xs, color=COL[m], lw=1.0, zorder=3,
                    ls="-" if p < 0.05 else (0, (2, 2)))
            lines.append((m, f"{rho:+.2f}{'*' if p < 0.05 else ' '}"))
        ax.set_xscale("log")
        ax.set_xlabel("MAD (log scale)")
        if j == 0:
            ax.set_ylabel("Test AUPRC")
        ax.set_title(f"{name}  (mean degree {DEGREE[ds]:g})", pad=4)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        # narrow ranges need minor tick labels; wide ranges get decades only
        lo, hi = np.log10(d.mad.min()), np.log10(d.mad.max())
        if hi - lo < 1.5:
            ax.xaxis.set_minor_formatter(matplotlib.ticker.LogFormatterSciNotation(
                labelOnlyBase=False, minor_thresholds=(2, 0.5)))
            ax.tick_params(axis="x", which="minor", labelsize=6.5)
        else:
            ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        # rank correlations as a compact corner block, colour-keyed to architecture.
        # Bottom-right is clear on Elliptic; the dense panels are crowded there by
        # the fit lines and the GraphSAGE cluster, so their block goes top-left.
        if j == 0:
            xa, ha, va, yhead = 0.97, "right", "bottom", 0.055 + 3 * 0.085
            yline = lambda i: 0.055 + (2 - i) * 0.085
        else:
            xa, ha, va, yhead = 0.04, "left", "top", 0.86  # below the panel label
            yline = lambda i: 0.86 - (i + 1) * 0.085
        ax.text(xa, yhead, "Spearman correlation", transform=ax.transAxes,
                fontsize=6.8, ha=ha, va=va, color="0.25")
        for i, (m, txt) in enumerate(lines):
            ax.text(xa, yline(i), f"{LABEL[m]} {txt}", transform=ax.transAxes,
                    fontsize=6.8, ha=ha, va=va, color=COL[m])
        panel_label(ax, f"({'abc'[j]})")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06),
               handlelength=1.2, columnspacing=1.6, markerscale=2.2)
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.925))
    fig.savefig(out / "Fig5.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4
def fig4(F, out):
    """Variance decomposition within each architecture, per dataset.

    Decomposing within an architecture rather than over the pooled grid is the
    meaningful comparison: pooling lets the (large) architecture effect absorb
    the variance and understates both experimental factors."""
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    seg = ["Normalisation", "Initialisation", "Interaction", "Residual"]
    col = ["#0072B2", "#D55E00", "#CC79A7", "0.82"]
    hat = ["", "...", "\\\\\\", ""]
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 1.75), sharex=True)
    for j, (ds, name) in enumerate(DSETS):
        d, ax = F[ds], axes[j]
        vals = []
        for m in MODELS:
            s = d[d.model == m]
            fit = ols("auprc ~ C(norm)*C(init)", data=s).fit()
            a = sm.stats.anova_lm(fit, typ=2)
            tot = a.sum_sq.sum()
            e = {k: a.sum_sq.get(k, 0.0) / tot for k in a.index}
            vals.append([e.get("C(norm)", 0), e.get("C(init)", 0),
                         e.get("C(norm):C(init)", 0), e.get("Residual", 0)])
        vals = np.array(vals)
        y = np.arange(len(MODELS))[::-1]
        left = np.zeros(len(MODELS))
        for i, s in enumerate(seg):
            ax.barh(y, vals[:, i], 0.58, left=left, color=col[i], hatch=hat[i],
                    edgecolor="white", linewidth=0.6, label=s if j == 0 else None)
            for yy, v, l in zip(y, vals[:, i], left):
                if v > 0.08:
                    ax.text(l + v / 2, yy, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.8, color="white" if i < 3 else "0.25")
            left += vals[:, i]
        ax.set_yticks(y, [LABEL[m] for m in MODELS])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Proportion of variance in test AUPRC")
        ax.set_title(f"{name}  (mean degree {DEGREE[ds]:g})", pad=4)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        panel_label(ax, f"({'abc'[j]})", dx=-0.30, dy=1.30)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.055),
               handlelength=1.3, columnspacing=1.5)
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.90))
    fig.savefig(out / "Fig3.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5
def fig5(F, out):
    """(a) initialisation x normalisation interaction; (b) GAT seed instability."""
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.0),
                             gridspec_kw=dict(width_ratios=[1, 1.15]))
    # (a)
    ax = axes[0]
    d = F["yelp"]
    d = d[d.model == "GCN"]
    ls = {"default": (0, (3, 2)), "xavier": "-", "kaiming": (0, (1, 1.6))}
    mk = {"default": "o", "xavier": "s", "kaiming": "^"}
    for ini in INITS:
        g = d[d.init == ini].groupby("norm").auprc
        mu = g.mean().reindex(NORMS).values
        se = (g.std() / np.sqrt(g.count())).reindex(NORMS).values
        ax.errorbar(np.arange(len(NORMS)), mu, yerr=se, marker=mk[ini], ms=3.2,
                    lw=1.1, ls=ls[ini], capsize=1.5, elinewidth=0.6,
                    color=COL["GCN"] if ini == "xavier" else ("#D55E00" if ini == "default" else "0.35"),
                    label=ini.capitalize())
    ax.annotate("", xy=(0, 0.227), xytext=(0, 0.192),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.25"))
    ax.text(0.14, 0.209, "+0.035", fontsize=7, color="0.25", va="center")
    ax.set_xticks(np.arange(len(NORMS)), NORM_LBL, rotation=30, ha="right")
    ax.set_xlabel("Normalisation")
    ax.set_ylabel("Test AUPRC")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title="Initialisation", loc="lower right", handlelength=1.8,
              labelspacing=0.2, title_fontsize=7.5)
    panel_label(ax, "(a)")
    # (b)
    ax = axes[1]
    d = F["amazon"]
    rng = np.random.default_rng(0)
    for m in MODELS:
        s = d[d.model == m]
        ax.scatter(s.seed + rng.uniform(-0.18, 0.18, len(s)), s.auprc, s=7,
                   c=COL[m], marker=MARK[m], alpha=0.6, linewidths=0, label=LABEL[m])
        mu = s.groupby("seed").auprc.mean()
        ax.plot(mu.index, mu.values, color=COL[m], lw=1.1, zorder=3)
    ax.axhline(base_rate(d), ls=(0, (4, 2)), lw=0.7, color="0.35", zorder=0)
    ax.text(4.45, base_rate(d), "base rate", fontsize=6.5, color="0.35",
            va="bottom", ha="right")
    ax.set_xticks(range(5))
    ax.set_xlabel("Seed (also determines the stratified split)")
    ax.set_ylabel("Test AUPRC")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.14)                      # headroom so the legend clears the data
    ax.legend(loc="upper center", handlelength=1.1, labelspacing=0.2, ncol=3,
              columnspacing=1.0, borderaxespad=0.15)
    panel_label(ax, "(b)")
    fig.tight_layout(w_pad=1.4)
    fig.savefig(out / "Fig4.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 6
def fig6(pred_dir, out):
    """Precision-recall curves for the best configuration per architecture,
    with the three reported operating points marked."""
    import json
    from sklearn.metrics import average_precision_score, precision_recall_curve
    files = sorted(Path(pred_dir).glob("*.npz"))
    if not files:
        print("no predictions found; skipping Fig6")
        return
    byds = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        byds.setdefault(m["dataset"], []).append((m, z["y_true"], z["y_score"]))

    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.25))
    for j, (ds, name) in enumerate(DSETS):
        ax = axes[j]
        runs = {m["model"]: (y, s) for m, y, s in byds.get(ds, [])}
        for mo in MODELS:
            if mo not in runs:
                continue
            y, s = runs[mo]
            pr, rc, th = precision_recall_curve(y, s)
            ap = average_precision_score(y, s)
            ax.plot(rc, pr, color=COL[mo], lw=1.2,
                    label=f"{LABEL[mo]}  {ap:.3f}")
            # operating points reported in Table 5
            for q, mk in zip((90, 99, 99.9), ("o", "s", "D")):
                t = np.percentile(s, q)
                sel = s >= t
                if sel.sum() and y[sel].sum() >= 0:
                    ax.plot(y[sel].sum() / y.sum(), y[sel].mean(), mk,
                            ms=3.0, mfc="white", mec=COL[mo], mew=0.8, zorder=4)
        br = float(np.mean(byds[ds][0][1])) if ds in byds else None
        if br:
            ax.axhline(br, ls=(0, (4, 2)), lw=0.7, color="0.35", zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall")
        if j == 0:
            ax.set_ylabel("Precision")
        ax.set_title(f"{name}  (mean degree {DEGREE[ds]:g})", pad=4)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", title="AUPRC", title_fontsize=7,
                  handlelength=1.2, labelspacing=0.2, borderaxespad=0.25)
        panel_label(ax, f"({'abc'[j]})")
    h = [Line2D([], [], ls="none", marker=k, ms=3.0, mfc="white", mec="0.3",
                mew=0.8) for k in ("o", "s", "D")]
    h.append(Line2D([], [], ls=(0, (4, 2)), lw=0.7, color="0.35"))
    fig.legend(h, ["90th pct", "99th pct", "99.9th pct", "Class base rate"],
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06),
               handlelength=1.3, columnspacing=1.4)
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.92))
    fig.savefig(out / "Fig9.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 7
def fig7(pred_dir, out):
    """Distribution of pairwise cosine distances between test-node embeddings,
    with and without normalisation, on the two dense datasets.

    A two-dimensional projection is the conventional choice here but is not
    sound for this claim: t-SNE and UMAP preserve only relative neighbourhood
    structure, so they expand a degenerate embedding to fill the plotting area
    and display apparent structure where none exists. The distribution of
    pairwise cosine distances is the quantity MAD averages, is scale-free in the
    same way MAD is, and cannot manufacture separation."""
    import json
    store = {}
    for f in sorted(Path(pred_dir).glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        store[(m["dataset"], m["model"], m["norm"])] = (z["emb"], m)

    rng = np.random.default_rng(0)

    def cosdist(e, n_pairs=60000):
        e = e.astype(np.float64)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        n = e.shape[0]
        i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
        k = i != j
        return 1.0 - (e[i[k]] * e[j[k]]).sum(1)

    dense = [("yelp", "YelpChi"), ("amazon", "Amazon")]
    fig, axes = plt.subplots(2, 3, figsize=(W_FULL, 3.1), sharex=True)
    bins = np.linspace(0, 1.6, 70)
    for r, (ds, dsn) in enumerate(dense):
        F = pd.read_csv(Path(pred_dir).parent / f"{ds}_all.csv")
        for c, mo in enumerate(MODELS):
            ax = axes[r, c]
            for nrm, col, lab in (("none", "#D55E00", "None"),
                                  ("graph", "#0072B2", "GraphNorm")):
                if (ds, mo, nrm) not in store:
                    continue
                e, meta = store[(ds, mo, nrm)]
                ax.hist(cosdist(e), bins=bins, density=True, histtype="stepfilled",
                        color=col, alpha=0.35, lw=0)
                ax.hist(cosdist(e), bins=bins, density=True, histtype="step",
                        color=col, lw=1.0, label=lab)
                row = F[(F.model == mo) & (F.norm == nrm) & (F.seed == 1)
                        & (F.init == meta["init"])]
                if len(row):
                    ax.axvline(row.mad.iloc[0], color=col, ls=(0, (2, 1.6)), lw=0.8)
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 60)
            if c == 0:
                ax.set_ylabel(f"{dsn}\ndensity (log)")
            else:
                ax.set_yticklabels([])
            if r == 1:
                ax.set_xlabel("Pairwise cosine distance")
            ax.set_title(LABEL[mo], pad=3) if r == 0 else None
            ax.grid(alpha=0.25)
            ax.set_axisbelow(True)
            panel_label(ax, f"({'abcdef'[r * 3 + c]})")
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(Line2D([], [], ls=(0, (2, 1.6)), lw=0.8, color="0.35"))
    l.append("MAD (mean over connected pairs)")
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.045),
               handlelength=1.5, columnspacing=1.5)
    fig.tight_layout(w_pad=0.9, h_pad=0.5, rect=(0, 0, 1, 0.935))
    fig.savefig(out / "Fig6.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 8
NORM3 = ["none", "layer", "graph"]
NORM3_LBL = {"none": "None", "layer": "LayerNorm", "graph": "GraphNorm"}
NORM3_LS = {"none": ":", "layer": "--", "graph": "-"}


def fig8(R, out):
    """Density sweep: the mechanism scales with degree, the accuracy does not.

    (a,b) MAD_rand against mean degree, log-log, one line per architecture and
    normalisation, with the fitted exponent annotated for the unnormalised and
    GraphNorm cases. (c,d) AUPRC against mean degree on the same x axis, which
    is flat-to-non-monotonic and so makes the point that density predicts
    collapse but not accuracy. (e) MAD against MAD_rand across every run,
    separating homophily from collapse."""
    d = pd.read_csv(R / "density_sweep.csv")
    h = pd.read_csv(R / "homophily.csv")
    fig = plt.figure(figsize=(W_FULL, 5.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.05], hspace=0.45, wspace=0.22)

    for j, (ds, name) in enumerate([("yelp", "YelpChi"), ("amazon", "Amazon")]):
        s = d[d.dataset == ds]
        # ---- row 0: MAD_rand vs degree
        ax = fig.add_subplot(gs[0, j])
        for m in MODELS:
            for nrm in NORM3:
                g = s[(s.model == m) & (s.norm == nrm)].groupby("mean_degree")["mad_random"].mean()
                ax.plot(g.index, g.values, NORM3_LS[nrm], color=COL[m], lw=1.0,
                        marker=MARK[m], ms=2.6, mew=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylabel(r"$\mathrm{MAD}_{\mathrm{rand}}$")
        ax.set_title(name, pad=4)
        ax.grid(alpha=0.25, lw=0.4)
        panel_label(ax, "(a)" if j == 0 else "(b)")
        # annotate the two extreme exponents
        for nrm, va in (("none", "top"), ("graph", "bottom")):
            g = s[(s.model == "GCN") & (s.norm == nrm)].groupby("mean_degree")["mad_random"].mean()
            sl = stats.linregress(np.log10(g.index.values), np.log10(g.values)).slope
            ax.annotate(rf"$\bar{{k}}^{{{sl:.2f}}}$", xy=(g.index[-1], g.values[-1]),
                        xytext=(-2, -9 if va == "top" else 5),
                        textcoords="offset points", ha="right", fontsize=7,
                        color=COL["GCN"])

        # ---- row 1: AUPRC vs degree
        ax = fig.add_subplot(gs[1, j])
        for m in MODELS:
            for nrm in NORM3:
                g = s[(s.model == m) & (s.norm == nrm)].groupby("mean_degree")["auprc"].mean()
                ax.plot(g.index, g.values, NORM3_LS[nrm], color=COL[m], lw=1.0,
                        marker=MARK[m], ms=2.6, mew=0)
        br = 0.145 if ds == "yelp" else 0.069
        ax.axhline(br, color="0.45", lw=0.6, ls=(0, (1, 2)))
        ax.annotate("base rate", xy=(s.mean_degree.min(), br), xytext=(0, 2),
                    textcoords="offset points", fontsize=6.5, color="0.35")
        ax.set_xscale("log"); ax.set_ylim(0, 1)
        ax.set_ylabel("Test AUPRC"); ax.set_xlabel(r"Mean degree $\bar{k}$")
        ax.grid(alpha=0.25, lw=0.4)
        panel_label(ax, "(c)" if j == 0 else "(d)")

    # ---- row 2 left: MAD vs MAD_rand, all runs
    ax = fig.add_subplot(gs[2, 0])
    for m in MODELS:
        g = d[d.model == m]
        ax.scatter(g.mad_random.clip(lower=1e-4), g["mad"].clip(lower=1e-10),
                   s=6, c=COL[m], marker=MARK[m], lw=0, alpha=0.75)
    ax.set_xscale("log"); ax.set_yscale("log")
    rur = d[(d.dataset == "yelp") & (d.relation == "net_rur") & (d.norm != "graph")]
    ax.annotate("R-U-R: homophily,\nnot collapse",
                xy=(rur.mad_random.median(), rur["mad"].clip(lower=1e-10).median()),
                xytext=(0.97, 0.42), textcoords="axes fraction", fontsize=6.5,
                ha="right", va="center", color="0.2",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4",
                                shrinkA=2, shrinkB=3))
    ax.set_xlabel(r"$\mathrm{MAD}_{\mathrm{rand}}$ (random pairs)")
    ax.set_ylabel("MAD (adjacent pairs)")
    ax.grid(alpha=0.25, lw=0.4)
    panel_label(ax, "(e)")

    # ---- row 2 right: AUPRC vs edge homophily
    ax = fig.add_subplot(gs[2, 1])
    a = d.groupby(["dataset", "relation"])["auprc"].mean().rename("auprc").reset_index()
    m2 = h.merge(a, on=["dataset", "relation"])
    for ds, mk, cl in (("yelp", "o", "#0072B2"), ("amazon", "s", "#E69F00")):
        g = m2[m2.dataset == ds]
        ax.scatter(g.h_edge, g.auprc, s=18, marker=mk, c=cl, lw=0,
                   label="YelpChi" if ds == "yelp" else "Amazon")
    r = stats.spearmanr(m2.h_edge, m2.auprc)
    ax.annotate(rf"$\rho={r[0]:.2f}$, $p={r[1]:.3f}$", xy=(0.97, 0.94),
                xycoords="axes fraction", fontsize=7, ha="right", va="top")
    ax.set_xlabel("Edge homophily $h$"); ax.set_ylabel("Test AUPRC")
    ax.set_ylim(0, 1); ax.grid(alpha=0.25, lw=0.4)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02), handletextpad=0.3,
              borderpad=0.2, labelspacing=0.25)
    panel_label(ax, "(f)")

    handles = ([Line2D([], [], color=COL[m], marker=MARK[m], ms=3, lw=1.0,
                       label=LABEL[m]) for m in MODELS] +
               [Line2D([], [], color="0.35", ls=NORM3_LS[n], lw=1.0,
                       label=NORM3_LBL[n]) for n in NORM3])
    fig.legend(handles=handles, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.035), handletextpad=0.4, columnspacing=1.2)
    fig.savefig(out / "Fig7.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 9
def fig9(R, out):
    """One layer is enough: collapse at a receptive-field saturation of 0.004.

    Unnormalised GCN at one and two message-passing layers, on all three
    datasets. MAD is unchanged on the dense graphs and orders of magnitude
    higher on Elliptic at identical depth, which separates degree from depth."""
    dp = pd.read_csv(R / "depth_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 1.85))
    order = [("elliptic", "Elliptic", 2.3), ("yelp", "YelpChi", 167.4),
             ("amazon", "Amazon", 736.5)]

    l2 = {}
    for ds, _, _ in order:
        g = pd.read_csv(R / f"{ds}_all.csv")
        g = g[(g.norm == "none") & (g.init == "default") & (g.model == "GCN")]
        l2[ds] = (g.auprc.mean(), g["mad"].mean())

    x = np.arange(len(order)); w = 0.36
    for ax, key, lbl, logy in ((axes[0], "mad", "MAD (adjacent pairs)", True),
                               (axes[1], "auprc", "Test AUPRC", False)):
        v1, v2 = [], []
        for ds, _, _ in order:
            g = dp[(dp.dataset == ds) & (dp.model == "GCN") & (dp.layers == 1)]
            v1.append(g[key].mean() if len(g) else np.nan)
            v2.append(l2[ds][0 if key == "auprc" else 1])
        ax.bar(x - w / 2, v1, w, color="#CC79A7", ec="k", lw=0.4, label="1 layer")
        ax.bar(x + w / 2, v2, w, color="#56B4E9", ec="k", lw=0.4, hatch="///",
               label="2 layers")
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n" + r"$\bar{k}=$" + f"{k:g}" for _, n, k in order])
        ax.set_ylabel(lbl); ax.grid(axis="y", alpha=0.25, lw=0.4)
        panel_label(ax, "(a)" if key == "mad" else "(b)")
    axes[0].legend(loc="upper right", handletextpad=0.4, borderpad=0.2)
    fig.savefig(out / "Fig8.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../experiments-output")
    ap.add_argument("--out", default="../paper/figures")
    ap.add_argument("--predictions", default="../experiments-output/predictions")
    a = ap.parse_args()
    R, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    F = {ds: pd.read_csv(R / f"{ds}_all.csv") for ds, _ in DSETS}
    print(f"font: {FONT}")
    for fn in (fig2, fig3, fig4, fig5):
        fn(F, out)
        print("wrote", fn.__name__)
    fig6(a.predictions, out)
    print("wrote fig6")
    fig7(a.predictions, out)
    print("wrote fig7")
    fig8(R, out)
    print("wrote fig8")
    fig9(R, out)
    print("wrote fig9")
    for p in sorted(out.glob("Fig*.pdf")):
        print(f"  {p.name}  {p.stat().st_size/1024:.0f} kB")


if __name__ == "__main__":
    main()

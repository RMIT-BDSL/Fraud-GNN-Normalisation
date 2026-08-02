#!/usr/bin/env python3
"""
Regenerate the Elliptic ANOVA (Table 4) and its inline statistics from the
canonical per-run results, so the manuscript numbers are never transcribed by
hand (internal review E9/A1).

Two-way factorial ANOVA of test AUPRC on normalisation x initialisation per
architecture, Type II sums of squares, eta^2 = SS_factor / SS_total. Residual df
is 60 (75 runs = 15 cells x 5 seeds, minus 15 fitted cell means). Also refits
with seed as a blocking factor and reports whether any verdict changes.

Usage:
  python make_anova_table.py --csv ../experiments-output/elliptic_all.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

TERMS = [("C(norm)", "Normalisation"),
         ("C(init)", "Initialisation"),
         ("C(norm):C(init)", "Interaction")]
ARCHES = ["GCN", "GAT", "SAGE"]


def fit(g, formula):
    m = smf.ols(formula, data=g).fit()
    a = anova_lm(m, typ=2)
    return a, a["sum_sq"].sum(), int(a.loc["Residual", "df"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../experiments-output/elliptic_all.csv")
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    for c in ("init", "norm", "seed"):
        d[c] = d[c].astype("category")

    rows = []
    print("=== two-way ANOVA (manuscript Table 4) ===")
    for arch in ARCHES:
        g = d[d.model == arch]
        aov, sstot, rdf = fit(g, "auprc ~ C(norm)*C(init)")
        for term, label in TERMS:
            F, p = aov.loc[term, "F"], aov.loc[term, "PR(>F)"]
            df1, eta = int(aov.loc[term, "df"]), aov.loc[term, "sum_sq"] / sstot
            print(f"  {arch:5s} {label:14s} F({df1},{rdf})={F:6.2f}  "
                  f"p={p:.3f}  eta2={eta:.3f}")
            rows.append((arch, label, F, p, eta))

    print("\n=== seed-blocked robustness (auprc ~ C(norm)*C(init)+C(seed)) ===")
    for arch in ARCHES:
        g = d[d.model == arch]
        aov, sstot, rdf = fit(g, "auprc ~ C(norm)*C(init) + C(seed)")
        out = []
        for term, label in TERMS + [("C(seed)", "Seed")]:
            F, p = aov.loc[term, "F"], aov.loc[term, "PR(>F)"]
            out.append(f"{label} F({int(aov.loc[term,'df'])},{rdf})={F:.2f} p={p:.3f}")
        print(f"  {arch:5s} " + " | ".join(out))

    # emit LaTeX table body matching tab:anova
    print("\n=== LaTeX (paste into tab:anova) ===")
    for arch in ARCHES:
        g = rows_for(rows, arch)
        print(f"\\multirow{{3}}{{*}}{{{arch if arch!='SAGE' else 'GraphSAGE'}}}")
        for label, F, p, eta in g:
            ps = "$<0.001$" if p < 0.001 else f"{p:.3f}"
            print(f" & {label} & {F:.2f} & {ps} & {eta:.3f} \\\\")
        print("\\midrule" if arch != "SAGE" else "\\bottomrule")


def rows_for(rows, arch):
    return [(lab, F, p, e) for ar, lab, F, p, e in rows if ar == arch]


if __name__ == "__main__":
    main()

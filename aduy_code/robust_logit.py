"""
Logistic regression: robustness ~ domain group + tile support (+ strength).

Why this is needed. The Pfam group table shows CXXC at 1.00 robust, KRAB at
0.95, WD40 at 0.00 against a 0.27 background -- but an earlier check found
that keeping only domains with >= 3 tiles lifts the background from 0.2706 to
0.4665. Tile support is therefore a strong predictor on its own, and a group
that happens to contain long, well-covered domains would look robust for that
reason alone. Nested likelihood-ratio tests separate the two.

Models (all on the domains that carry a Pfam group label):
  M0  intercept only
  M1  ~ n_tiles                      tile support alone
  M2  ~ group                        group alone
  M3  ~ group + n_tiles              does group survive tile support?
  M4  ~ group + n_tiles + max_pEF    and survive repression strength too?

The test that matters is M3 vs M1: it asks whether domain group adds anything
once tile support is known. M4 vs M3 adds strength, which has its own AUC of
0.73, so a group effect that dies in M4 was strength in disguise.

Groups with fewer than `--min-group` domains are pooled into "other": a level
with 3 members contributes a coefficient with no information and destabilises
the fit.

Usage:
  python robust_logit.py --annotated pfam_domains_annotated.csv --out logit
"""

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--min-group", type=int, default=10)
    ap.add_argument("--out", default="logit")
    args = ap.parse_args()

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy.stats import chi2

    d = pd.read_csv(args.annotated)
    d["robust"] = d.pgk_hit.astype(int)
    d = d[d.pfam_group.notna()].copy()
    print(f"{len(d):,} domains carry a Pfam group label "
          f"({d.robust.mean():.4f} robust)")

    small = d.pfam_group.value_counts()
    keep = set(small[small >= args.min_group].index)
    d["grp"] = np.where(d.pfam_group.isin(keep), d.pfam_group, "other")
    print(f"groups kept (>= {args.min_group}): {len(keep)}; "
          f"{(d.grp == 'other').sum()} domains pooled into 'other'")
    print("\ntile support by group (the thing being controlled for):")
    tbl = d.groupby("grp").agg(n=("robust", "size"), robust=("robust", "mean"),
                               mean_tiles=("n_tiles", "mean"),
                               median_tiles=("n_tiles", "median"),
                               median_pEF=("max_pEF", "median")).round(3)
    print(tbl.sort_values("robust", ascending=False).to_string())

    # reference level = the largest group, so coefficients read against it
    ref = d.grp.value_counts().index[0]
    d["grp"] = pd.Categorical(d.grp, categories=[ref] +
                              sorted(set(d.grp) - {ref}))
    print(f"\nreference level for group = '{ref}'")

    specs = {
        "M0 intercept": "robust ~ 1",
        "M1 n_tiles": "robust ~ n_tiles",
        "M2 group": "robust ~ grp",
        "M3 group + n_tiles": "robust ~ grp + n_tiles",
        "M4 group + n_tiles + max_pEF": "robust ~ grp + n_tiles + max_pEF",
    }
    fits = {}
    print(f"\n{'model':<30}{'df':>5}{'logLik':>11}{'AIC':>10}{'pseudoR2':>10}")
    for name, f in specs.items():
        m = smf.logit(f, data=d).fit(disp=0)
        fits[name] = m
        print(f"{name:<30}{int(m.df_model):>5}{m.llf:>11.2f}{m.aic:>10.1f}"
              f"{m.prsquared:>10.4f}")

    def lr(bigger, smaller):
        b, s = fits[bigger], fits[smaller]
        stat = 2 * (b.llf - s.llf)
        df = int(b.df_model - s.df_model)
        return stat, df, chi2.sf(stat, df)

    print("\nnested likelihood-ratio tests")
    for big, sml, question in [
        ("M1 n_tiles", "M0 intercept", "does tile support matter at all?"),
        ("M2 group", "M0 intercept", "does group matter at all?"),
        ("M3 group + n_tiles", "M1 n_tiles",
         "does GROUP add anything once tile support is known?"),
        ("M3 group + n_tiles", "M2 group",
         "does TILE SUPPORT add anything once group is known?"),
        ("M4 group + n_tiles + max_pEF", "M3 group + n_tiles",
         "does strength add anything on top?"),
    ]:
        stat, df, p = lr(big, sml)
        print(f"  {question}")
        print(f"    chi2 = {stat:.2f}, df = {df}, p = {p:.3ე}"
              .replace("ე", "e") if False else
              f"    chi2 = {stat:.2f}, df = {df}, p = {p:.3e}")

    print("\ncoefficients of the full model (log-odds)")
    m4 = fits["M4 group + n_tiles + max_pEF"]
    summ = pd.DataFrame({"coef": m4.params, "se": m4.bse,
                         "z": m4.tvalues, "p": m4.pvalues,
                         "OR": np.exp(m4.params)})
    summ["OR_lo"] = np.exp(m4.params - 1.96 * m4.bse)
    summ["OR_hi"] = np.exp(m4.params + 1.96 * m4.bse)
    print(summ.round(4).to_string())

    # Does the group effect hold at fixed tile support? Stratified rates.
    print("\nrobust rate by group WITHIN tile-support bands")
    d["band"] = pd.cut(d.n_tiles, [0, 1, 2, 4, 100],
                       labels=["1 tile", "2", "3-4", "5+"])
    piv = d.pivot_table(index="grp", columns="band", values="robust",
                        aggfunc="mean", observed=False).round(2)
    cnt = d.pivot_table(index="grp", columns="band", values="robust",
                        aggfunc="size", observed=False)
    print(piv.to_string())
    print("\n(n per cell)")
    print(cnt.to_string())
    print("\nA group effect that is real shows up inside the bands, not only")
    print("across them. Cells with n < 5 carry no information.")

    # ------------------------------------------------------------------
    # Separation. A group that is 100% or 0% robust has no finite maximum
    # likelihood estimate: its coefficient runs off to +/- infinity and the
    # reported se and p are numerical artefacts, not evidence of "no effect".
    # Two fixes, both reported.
    # ------------------------------------------------------------------
    rates = d.groupby("grp", observed=True).robust.agg(["mean", "size"])
    sep = rates[(rates["mean"] == 0) | (rates["mean"] == 1)]
    if len(sep):
        print("\n" + "=" * 68)
        print("SEPARATION: these groups have no finite MLE")
        print("=" * 68)
        for g, r in sep.iterrows():
            print(f"  {g:<16} {r['mean']*100:5.1f}% robust on n = {int(r['size'])}"
                  f"  -> coefficient and p in the table above are meaningless")
        print("  The separation is itself the finding: the group is perfectly")
        print("  predictive in this sample. What cannot be done is to put a")
        print("  number and a confidence interval on it from this fit.")

        # (a) refit without the separated groups, for a clean LR test
        sub = d[~d.grp.isin(sep.index)].copy()
        # drop the now-empty categories, or the design matrix keeps an
        # all-zero column per removed level and becomes singular
        sub["grp"] = pd.Categorical(sub.grp.astype(str))
        m1b = smf.logit("robust ~ n_tiles", data=sub).fit(disp=0)
        m3b = smf.logit("robust ~ grp + n_tiles", data=sub).fit(disp=0)
        stat = 2 * (m3b.llf - m1b.llf)
        df = int(m3b.df_model - m1b.df_model)
        print(f"\n  refit on the {len(sub):,} domains in non-separated groups:")
        print(f"    does group still add over tile support? "
              f"chi2 = {stat:.2f}, df = {df}, p = {chi2.sf(stat, df):.3e}")

        # (b) L2-penalised fit, which keeps every coefficient finite
        from sklearn.linear_model import LogisticRegression
        X = pd.get_dummies(d[["grp"]], drop_first=True).astype(float)
        X["n_tiles"] = d.n_tiles.values
        X["max_pEF"] = d.max_pEF.values
        lr2 = LogisticRegression(penalty="l2", C=1.0, max_iter=5000)
        lr2.fit(X.values, d.robust.values)
        pen = pd.Series(lr2.coef_[0], index=X.columns).sort_values(ascending=False)
        print("\n  L2-penalised coefficients (log-odds, finite by construction):")
        for k, v in pen.items():
            print(f"    {k:<28}{v:+.3f}   OR {np.exp(v):8.2f}")
        print("  Penalised estimates are biased toward zero on purpose; read")
        print("  them for sign and ordering, not as effect sizes.")

    summ.to_csv(f"{args.out}_coefficients.csv")
    tbl.to_csv(f"{args.out}_group_table.csv")
    print(f"\nwrote {args.out}_coefficients.csv, {args.out}_group_table.csv")


if __name__ == "__main__":
    main()

"""Does robustness come from the annotated domain, or from the flanking sequence?

Section 4.4 lists this as a competing explanation it could not rule out, and
section 3.7 leans on it as an unrun hypothesis. The test it names: compute how
much of each 80-aa window actually falls inside the Pfam-annotated interval,
then re-run the analysis on high-overlap domains only.

The audit split this into two questions, because they have very different power:

  DEAD   the PRD/ANTP contrast of section 3.7. Only 93 domains carry a
         Homeodomain annotation, 38 survive a 70% overlap cut and ZERO survive
         80%. Detecting 59.3% vs 34.0% at 80% power needs 60 per group; the cut
         leaves roughly 13 vs 20. Reported here as description only -- a null
         would carry no information, so it is not tested.

  ALIVE  the global question behind section 3.6: is robustness itself a
         property of the annotated domain or of what surrounds it? 617 domains
         survive a 70% cut and 452 an 80% cut, and the tertile means already
         point one way.

Two confounds the audit flagged and this script controls:
  * overlap is negatively correlated with domain length (-0.193), and length is
    already known to drive the hit rate, so length enters every model;
  * domains cluster in proteins (2.26 per protein), so every p-value is GEE
    clustered by protein, never a bare Fisher test.

Threshold is pre-declared at 70% (the highest cut that keeps the homeodomain
group non-empty); the full 50-90% curve is reported as sensitivity.

Usage:
  python flanking.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu

ROOT = "/home/dmin/ai&stem_cell"
THRESH = 0.70
GROUPS = ["KRAB", "bHLH", "homeodomain", "C2H2", "PHD", "WD40", "SCAN",
          "CXXC", "leucine_zipper", "chromo_bromo"]


def gee(df, formula):
    m = smf.gee(formula, groups="protein", data=df,
                family=sm.families.Binomial()).fit()
    return m


def main():
    d = pd.read_csv(f"{ROOT}/run/pfam_domains_annotated.csv")
    a = d[d.pfam_name.notna()].copy()
    a["ovl"] = (a.ali_to - a.ali_from + 1) / a.aa_len
    a["logt"] = np.log(a.n_tiles)
    a["robust"] = a.pgk_hit.astype(int)
    print(f"{len(a):,} Pfam-annotated domains on {a.protein.nunique():,} proteins")
    print(f"overlap: median {a.ovl.median():.3f}  IQR "
          f"[{a.ovl.quantile(.25):.3f}, {a.ovl.quantile(.75):.3f}]\n")

    # ---- 1. is robustness associated with overlap, controlling for length? ----
    print("=== robustness vs annotation overlap (GEE, clustered by protein) ===")
    base = "robust ~ logt + max_pEF"
    full = base + " + ovl"
    m0, m1 = gee(a, base), gee(a, full)
    b, se = m1.params["ovl"], m1.bse["ovl"]
    print(f"  n = {len(a):,}  clusters = {a.protein.nunique():,}")
    print(f"  beta(ovl) = {b:+.4f}  SE {se:.4f}  z {b/se:+.2f}  "
          f"p = {m1.pvalues['ovl']:.2e}")
    print(f"  OR per +0.1 overlap = {np.exp(b*0.1):.4f} "
          f"[{np.exp((b-1.96*se)*0.1):.4f} - {np.exp((b+1.96*se)*0.1):.4f}]")
    print(f"  OR low->high (0.44 -> 0.86) = {np.exp(b*(0.855-0.438)):.4f}")

    print("\n  tertiles of overlap:")
    a["tert"] = pd.qcut(a.ovl, 3, labels=["low", "mid", "high"])
    print(a.groupby("tert", observed=True)
          .agg(n=("robust", "size"), robust=("robust", "mean"),
               med_ovl=("ovl", "median"), med_len=("aa_len", "median"),
               med_tiles=("n_tiles", "median")).round(3).to_string())

    # ---- 2. sensitivity: does the section 3.6 group table survive the cut? ----
    print(f"\n=== group stratification among high-overlap domains "
          f"(>= {THRESH:.0%}) ===")
    hi = a[a.ovl >= THRESH]
    print(f"  n = {len(hi):,} on {hi.protein.nunique():,} proteins  "
          f"(base robust rate {hi.robust.mean():.3f} vs {a.robust.mean():.3f} overall)")
    rows = []
    for g in GROUPS:
        for name, sub in (("all", a), (f">={THRESH:.0%}", hi)):
            s = sub[sub.pfam_group.astype(str) == g]
            if len(s) == 0:
                rows.append(dict(group=g, set=name, k=0, n=0, frac=np.nan))
                continue
            rows.append(dict(group=g, set=name, k=int(s.robust.sum()),
                             n=len(s), frac=float(s.robust.mean()),
                             n_prot=int(s.protein.nunique())))
    t = pd.DataFrame(rows)
    piv = t.pivot_table(index="group", columns="set", values="frac")
    cnt = t.pivot_table(index="group", columns="set", values="n")
    out = pd.concat([piv.round(3).add_prefix("frac_"),
                     cnt.add_prefix("n_")], axis=1)
    print(out.to_string())

    # ---- 3. sensitivity curve over the threshold ----
    print("\n=== sensitivity: threshold curve ===")
    crows = []
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        s = a[a.ovl >= th]
        mm = gee(a.assign(hi=(a.ovl >= th).astype(int)),
                 "robust ~ logt + max_pEF + hi")
        crows.append(dict(threshold=th, n_above=len(s),
                          robust_above=float(s.robust.mean()),
                          robust_below=float(a[a.ovl < th].robust.mean()),
                          beta_hi=float(mm.params["hi"]),
                          p_hi=float(mm.pvalues["hi"])))
        r = crows[-1]
        print(f"  >= {th:.0%}: n {r['n_above']:4d}  robust above "
              f"{r['robust_above']:.3f} vs below {r['robust_below']:.3f}  "
              f"beta {r['beta_hi']:+.3f}  p {r['p_hi']:.2e}")
    pd.DataFrame(crows).to_csv(f"{ROOT}/run/rev2/flanking_curve.csv", index=False)

    # ---- 4. PRD/ANTP: description only, explicitly underpowered ----
    print("\n=== PRD/ANTP: DESCRIPTION ONLY (underpowered, no test) ===")
    cls = pd.read_csv(f"{ROOT}/run/homeodomain_classes.csv")
    col = [c for c in cls.columns if "class" in c.lower()]
    print(f"  homeodomain_classes.csv cols: {list(cls.columns)}")
    h = a[a.pfam_name.astype(str).str.contains("Homeo", case=False)]
    print(f"  homeodomain-annotated domains: {len(h)}")
    for th in (0.5, 0.6, 0.7, 0.8):
        print(f"    >= {th:.0%}: {int((h.ovl >= th).sum())}")
    if col:
        key = col[0]
        pc = cls[[c for c in cls.columns if c.lower() in ("protein", "gene")][0]]
        cls2 = cls.assign(_p=pc)
        hm = h.merge(cls2[["_p", key]], left_on="protein", right_on="_p",
                     how="left")
        for th in (0.0, 0.5, 0.7):
            s = hm[hm.ovl >= th]
            g = (s[s[key].isin(["PRD", "ANTP"])]
                 .groupby(key).agg(n=("robust", "size"),
                                   robust=("robust", "sum"),
                                   frac=("robust", "mean")))
            print(f"\n    overlap >= {th:.0%}:")
            print(g.round(3).to_string() if len(g) else "      (empty)")
        print("\n  Power: detecting 59.3% vs 34.0% at 80% power needs 60/group.")
        print("  No test is reported for the restricted sets.")

    a[["protein", "domain_id", "pfam_name", "pfam_group", "ovl", "aa_len",
       "n_tiles", "max_pEF", "robust"]].to_csv(
        f"{ROOT}/run/rev2/flanking_overlap.csv", index=False)
    print(f"\nwrote {ROOT}/run/rev2/flanking_overlap.csv and flanking_curve.csv")
    print("DONE_FLANKING")


if __name__ == "__main__":
    main()

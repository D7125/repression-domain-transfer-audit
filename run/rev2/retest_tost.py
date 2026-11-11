"""Recompute the primary ceiling on each seed's scored subset (retest_seeds.py
dropped it to NaN: a few tiles carry only one pEF replicate), then run the TOST.

Takes the input/output stems as arguments so it can be pointed at the
deterministic rerun instead of only the original pre-patch file:

  python retest_tost.py retest_det10 retest_det10_final
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import pearsonr, t as tdist
sys.path.insert(0, "/home/dmin/ai&stem_cell/cnn_AI&stemcell")
from train_cnn import grouped_split

ROOT = "/home/dmin/ai&stem_cell"
MARGIN = 5.0
sb = lambda r: 2 * r / (1 + r)

df = pd.read_csv(f"{ROOT}/run/rev2/master.csv", low_memory=False)
has_rt = df.Avg_pEF_retest.notna().values
cl = pd.read_csv(f"{ROOT}/run/protein_clusters.tsv", sep="\t", header=None,
                 names=["rep", "member"])
cmap = dict(zip(cl["member"], cl["rep"]))
groups = np.array([cmap.get(g, g) for g in df.protein.astype(str)])

IN_STEM = sys.argv[1] if len(sys.argv) > 1 else "retest_seeds"
OUT_STEM = sys.argv[2] if len(sys.argv) > 2 else "retest_seeds_final"

res = pd.read_json(f"{ROOT}/run/rev2/{IN_STEM}.json")
fix = []
for seed in res.seed:
    _, _, te = grouped_split(groups, seed)
    te = te[has_rt[te]]
    sub = df.iloc[te]
    m = sub.R1_pEF.notna() & sub.R2_pEF.notna()
    r = float(pearsonr(sub.R1_pEF[m], sub.R2_pEF[m])[0])
    fix.append(dict(seed=int(seed), r_prim_reps_fixed=r,
                    ceil_primary_fixed=float(np.sqrt(sb(r)))))
res = res.merge(pd.DataFrame(fix), on="seed")
res["pct_ceil_primary"] = 100 * res.r_pred_primary / res.ceil_primary_fixed
res["pct_ceil_retest"] = 100 * res.r_pred_retest / res.ceil_retest_measured
res["diff_pct"] = res.pct_ceil_retest - res.pct_ceil_primary
res.to_json(f"{ROOT}/run/rev2/{OUT_STEM}.json", orient="records", indent=2)

print(res[["seed", "n_test_retest", "ceil_primary_fixed", "ceil_retest_measured",
           "r_pred_primary", "r_pred_retest", "pct_ceil_primary",
           "pct_ceil_retest", "diff_pct"]].round(4).to_string(index=False))

d = res.diff_pct.values
n = len(d); mean, sd = d.mean(), d.std(ddof=1); se = sd / np.sqrt(n)
t1, t2 = (mean + MARGIN) / se, (MARGIN - mean) / se
p_tost = max(tdist.sf(t1, n - 1), tdist.sf(t2, n - 1))
half = tdist.ppf(0.95, n - 1) * se
p_two = 2 * tdist.sf(abs(mean) / se, n - 1)
print("\n" + "=" * 66)
print(f"seeds                        : {n}")
print(f"% ceiling, training screen   : {res.pct_ceil_primary.mean():.2f} "
      f"(sd {res.pct_ceil_primary.std(ddof=1):.2f})")
print(f"% ceiling, independent screen: {res.pct_ceil_retest.mean():.2f} "
      f"(sd {res.pct_ceil_retest.std(ddof=1):.2f})")
print(f"paired difference            : {mean:+.2f} points "
      f"(sd {sd:.2f}, {abs(mean)/sd:.2f}x sd, SE {se:.2f})")
print(f"two-sided test of zero       : p = {p_two:.4f}")
print(f"90% CI                       : [{mean-half:+.2f}, {mean+half:+.2f}]")
print(f"TOST vs +-{MARGIN:.0f} points          : p = {p_tost:.4g} -> "
      f"{'EQUIVALENT within the margin' if p_tost < 0.05 else 'NOT equivalent'}")
print(f"measured retest ceiling      : {res.ceil_retest_measured.mean():.4f} "
      f"(sd {res.ceil_retest_measured.std(ddof=1):.4f})")
print(f"derived  retest ceiling      : {res.ceil_retest_derived.mean():.4f} "
      f"(sd {res.ceil_retest_derived.std(ddof=1):.4f})")
print(f"\nwrote {ROOT}/run/rev2/{OUT_STEM}.json")
print("DONE_RETEST_TOST")

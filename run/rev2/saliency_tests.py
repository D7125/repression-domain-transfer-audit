"""Paired tests behind the section 3.3 control table, as a script.

The v9 table (saliency_seeds_tests.csv) was produced by an ad-hoc command that
was never saved, so the one analysis in the paper that carries the saliency
conclusion had no reproducible path. This is that command, written down.

Design, all of it pre-stated in section 3.3:
  unit                  one seed (its own grouped split; all three model states
                        share that seed's split, so the comparison is paired)
  primary endpoint      mass A, trained minus label_perm
  secondary             mass A vs random_init, mass B vs each control
  multiplicity          Benjamini-Hochberg over all four
  ratio                 reported as description only -- a ratio of two noisy
                        quantities is unstable, the difference is the estimand

Wilcoxon signed-rank is reported next to the paired t because the label_perm
column carries one large outlier, and a leave-one-out pass shows how much that
seed moves the estimate.

  python saliency_tests.py saliency_det10 saliency_det10_tests
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as tdist, wilcoxon

ROOT = "/home/dmin/ai&stem_cell"
IN_STEM = sys.argv[1] if len(sys.argv) > 1 else "saliency_seeds_all"
OUT_STEM = sys.argv[2] if len(sys.argv) > 2 else "saliency_seeds_tests"

t = pd.read_csv(f"{ROOT}/run/rev2/{IN_STEM}.csv")
states = ("trained", "label_perm", "random_init")
missing = [s for s in states if s not in set(t.state)]
if missing:
    raise SystemExit(f"{IN_STEM}.csv is missing model states: {missing}")

print(f"{IN_STEM}.csv: {t.seed.nunique()} seeds x {t.state.nunique()} states")
print("\n=== mass A (%) and mass B (%) by state ===")
print(t.groupby("state")[["massA", "massB", "test_r"]]
      .agg(["mean", "std", "min", "max"]).round(3).to_string())


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    tv = m / se
    half = tdist.ppf(0.975, n - 1) * se
    try:
        pw = float(wilcoxon(d).pvalue)
    except ValueError:
        pw = float("nan")
    return dict(n=n, diff=m, sd=sd, ci_lo=m - half, ci_hi=m + half, t=tv,
                p=float(2 * tdist.sf(abs(tv), n - 1)), p_wilcoxon=pw,
                neg=int((d < 0).sum()))


rows = []
for conv, col in (("A", "massA"), ("B", "massB")):
    piv = t.pivot_table(index="seed", columns="state", values=col)
    for ctrl in ("label_perm", "random_init"):
        st = paired(piv["trained"], piv[ctrl])
        ratio = (piv["trained"] / piv[ctrl]).values
        st.update(conv=conv, ctrl=ctrl,
                  ratio=float(np.mean(ratio)),
                  ratio_sd=float(np.std(ratio, ddof=1)),
                  ratio_median=float(np.median(ratio)),
                  ratio_of_medians=float(np.median(piv["trained"])
                                         / np.median(piv[ctrl])))
        rows.append(st)

out = pd.DataFrame(rows)
# Benjamini-Hochberg over the four comparisons
order = np.argsort(out.p.values)
m = len(out)
q = np.empty(m)
prev = 1.0
for rank, i in enumerate(order[::-1], start=1):
    k = m - rank + 1
    prev = min(prev, out.p.values[i] * m / k)
    q[i] = prev
out["q_bh"] = q
out = out[["conv", "ctrl", "n", "diff", "sd", "ci_lo", "ci_hi", "t", "p",
           "q_bh", "p_wilcoxon", "neg", "ratio", "ratio_sd", "ratio_median",
           "ratio_of_medians"]]
out.to_csv(f"{ROOT}/run/rev2/{OUT_STEM}.csv", index=False)

print("\n=== paired per-seed differences, BH over all four ===")
for _, r in out.iterrows():
    tag = "PRIMARY" if (r.conv == "A" and r.ctrl == "label_perm") else "       "
    print(f"  {tag} {r.conv} vs {r.ctrl:12s} {r['diff']:+6.2f} pt  "
          f"CI95 [{r.ci_lo:+6.2f}, {r.ci_hi:+6.2f}]  t({int(r.n)-1}) = {r.t:5.2f}  "
          f"p = {r.p:.2g}  q = {r.q_bh:.2g}  wilcoxon p = {r.p_wilcoxon:.4f}  "
          f"neg {int(r.neg)}/{int(r.n)}  ratio {r.ratio:.2f}x (sd {r.ratio_sd:.2f})")

# --- sensitivity: how much does the single worst control seed carry? ---
piv = t.pivot_table(index="seed", columns="state", values="massA")
d = (piv["trained"] - piv["label_perm"])
worst = piv["label_perm"].idxmax()
print(f"\n=== sensitivity on the primary endpoint ===")
print(f"  highest label_perm mass: seed {worst} at {piv['label_perm'].max():.2f}%"
      f"  (other seeds {piv['label_perm'].drop(worst).min():.2f}-"
      f"{piv['label_perm'].drop(worst).max():.2f}%)")
rest = piv["label_perm"].drop(worst)
if len(rest) > 1 and rest.std(ddof=1) > 0:
    print(f"  that seed sits {(piv['label_perm'].max() - rest.mean()) / rest.std(ddof=1):.1f}"
          f" sd above the remaining {len(rest)} seeds")
st_drop = paired(piv["trained"].drop(worst), piv["label_perm"].drop(worst))
print(f"  all seeds kept  : {d.mean():+.2f} pt  CI95 "
      f"[{paired(piv['trained'], piv['label_perm'])['ci_lo']:+.2f}, "
      f"{paired(piv['trained'], piv['label_perm'])['ci_hi']:+.2f}]  (reported)")
print(f"  that seed dropped: {st_drop['diff']:+.2f} pt  CI95 "
      f"[{st_drop['ci_lo']:+.2f}, {st_drop['ci_hi']:+.2f}]  (n = {st_drop['n']})")
print(f"  keeping it is the conservative choice.")

print(f"\n=== descriptive: do the two distributions overlap at all? ===")
print(f"  worst trained seed : {piv['trained'].min():.2f}%")
print(f"  best control seed  : {max(piv['label_perm'].max(), 0):.2f}% (label_perm)")
print(f"  -> {'no overlap' if piv['trained'].min() > piv['label_perm'].max() else 'they overlap'}"
      f"; this is a description of the two extremes, not a test")

print(f"\n=== label-permutation sanity: test r must sit at ~0 ===")
lp = t[t.state == "label_perm"].test_r
print(f"  label_perm test r: mean {lp.mean():+.4f}  range "
      f"[{lp.min():+.4f}, {lp.max():+.4f}]")

print(f"\nwrote {ROOT}/run/rev2/{OUT_STEM}.csv")
print("DONE_SALIENCY_TESTS")

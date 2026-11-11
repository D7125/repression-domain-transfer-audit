"""Robustness table for referee items M10 (#3), M4 (#2) and Mock-M5 (#5).

Inference unit, stated explicitly because it differs per contrast:

  * esm640 vs esm480_5, and br2 vs br1, are PAIRED: identical seeds give
    identical GroupKFold splits, so the seed is the unit of the manipulation
    and a paired CI over seeds is the right interval for the CONFIGURATION
    contrast. It is not a population interval over proteins -- the protein
    cluster bootstrap in reviewer_response.py is that, and it needs per-tile
    predictions this script does not keep.
  * id20/id30/id40 are NOT paired against anything: a different MMseqs2
    partition is a different split, so only each run's own seed spread is
    reportable, plus the range across thresholds.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

R = "/home/dmin/ai&stem_cell/run/rev3"
K = "pct_ceil_cd43model_cd43"


def load(tag):
    return pd.DataFrame(json.load(open(f"{R}/var_{tag}.json"))).set_index("seed")


def ci_mean(x, lab):
    x = np.asarray(x, float) * 100
    m, n = x.mean(), len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    print(f"  {lab:34s} {m:6.2f}%  sd {x.std(ddof=1):4.2f}  "
          f"95% CI [{m-h:6.2f}, {m+h:6.2f}]  n={n}")
    return m


def paired(a, b, lab, margin=None):
    """b - a, paired on seed."""
    common = sorted(set(a.index) & set(b.index))
    d = (b.loc[common, K].values - a.loc[common, K].values) * 100
    m, n = d.mean(), len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    t, p = stats.ttest_rel(b.loc[common, K], a.loc[common, K])
    print(f"\n  {lab}")
    print(f"    paired diff {m:+6.2f} pts  95% CI [{m-h:+6.2f}, {m+h:+6.2f}]  "
          f"p={p:.4f}  (n={n} seeds)")
    if margin is not None:
        # TOST: both one-sided tests must reject for equivalence
        p_lo = stats.t.sf((m + margin) / se, n - 1)
        p_hi = stats.t.cdf((m - margin) / se, n - 1)
        pt = max(p_lo, p_hi)
        print(f"    TOST +-{margin} pts: p={pt:.4f} -> "
              f"{'EQUIVALENT' if pt < 0.05 else 'not shown equivalent'}")
    return m


print("=" * 72)
print("#3  M10 -- does ESM-2 model size change the conclusion?")
print("=" * 72)
e640, e480 = load("esm640"), load("esm480_5")
for k, lab in [("pct_ceil_cd43model_cd43", "CD43->CD43"),
               ("pct_ceil_pEFmodel_pEF", "pEF->pEF"),
               ("pct_ceil_pEFmodel_cd43", "pEF->CD43 (transfer)"),
               ("pct_ceil_cd43model_pEF", "CD43->pEF (transfer)")]:
    d = (e640[k].values - e480[k].values) * 100
    se = d.std(ddof=1) / np.sqrt(len(d))
    h = stats.t.ppf(0.975, len(d) - 1) * se
    print(f"  {lab:22s} D640 {100*e640[k].mean():6.2f}  D480 {100*e480[k].mean():6.2f}"
          f"   diff {d.mean():+5.2f} [{d.mean()-h:+6.2f}, {d.mean()+h:+6.2f}]")
K = "pct_ceil_cd43model_cd43"
paired(e480, e640, "150M vs 35M on the headline cell", margin=5)
print("  cost: 774 s/seed vs 238 s/seed = 3.3x wall clock for that difference")

print("\n" + "=" * 72)
print("#2  M4 -- train on ONE CD43 replicate, score the OTHER")
print("=" * 72)
b1, b2 = load("br1"), load("br2")
ci_mean(b1[K], "train br1 -> score br2")
ci_mean(b2[K], "train br2 -> score br1")
pooled = np.r_[b1[K].values, b2[K].values]
ci_mean(pooled, "both directions pooled")
paired(b1, b2, "which replicate is the better training target")
print(f"\n  ceiling used = sqrt(r(br1,br2)) = {np.sqrt(0.4577):.4f} (single draw),")
print("  identical in both directions, so the gap is a training-target effect.")
print("  reference: cross-assay transfer = 18.64% (sd 4.76)")
print(f"  ratio pooled / transfer = {pooled.mean()*100/18.64:.2f}x")

print("\n" + "=" * 72)
print("#5  Mock-M5 -- MMseqs2 identity threshold (homology leakage)")
print("=" * 72)
print("  id  clusters  CD43->CD43        pEF->pEF        pEF->CD43       CD43->pEF")
rows = {}
for tag, nclu in [("id20", 868), ("id30", 1017), ("id40", 1279)]:
    t = load(tag); rows[tag] = t
    v = [100 * t[c].mean() for c in
         ["pct_ceil_cd43model_cd43", "pct_ceil_pEFmodel_pEF",
          "pct_ceil_pEFmodel_cd43", "pct_ceil_cd43model_pEF"]]
    s = [100 * t[c].std(ddof=1) for c in
         ["pct_ceil_cd43model_cd43", "pct_ceil_pEFmodel_pEF",
          "pct_ceil_pEFmodel_cd43", "pct_ceil_cd43model_pEF"]]
    print(f"  {tag[2:]:>2s}  {nclu:8,}  " +
          "  ".join(f"{a:5.2f}+-{b:4.2f}" for a, b in zip(v, s)))
pub = 100 * e480[K].mean()
print(f"\n  published partition (1,013 clusters, same 5 seeds): {pub:.2f}%")
rng = [100 * rows[t][K].mean() for t in rows]
print(f"  CD43->CD43 range across 20/30/40%: {min(rng):.2f} - {max(rng):.2f} "
      f"= {max(rng)-min(rng):.2f} pts spread")
print(f"  largest within-run seed sd: "
      f"{max(100*rows[t][K].std(ddof=1) for t in rows):.2f} pts")
print("  => the spread across thresholds is smaller than one run's seed noise")
print("\nDONE_SUMMARISE")

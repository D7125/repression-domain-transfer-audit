"""Consolidate every v9 arm that has a deterministic 10-seed file on disk.

Three arms of report v9 (2026-10-30 22:38) quote fewer seeds than the runs that
actually exist in run/rev2, because the deterministic reruns either landed after
the report was written or were never picked up:

  arm                     v9 quotes                  file that supersedes it
  CD43 context transfer   3 seeds, pre-patch         cd43_fixed_det10_final.json (10 det)
  precision@k             2 seeds, pre-patch         cd43_topk3.csv              (10 det)
  de-noising test         2 seeds, pre-patch         recomputed from cd43_topk3_preds.csv

This script recomputes the three tables at n = 10 and prints them next to the
numbers v9 printed, so each change can be checked line by line. It trains
nothing; everything here is arithmetic on saved results.

  python seed10_consolidate.py
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t as tdist, wilcoxon

ROOT = "/home/dmin/ai&stem_cell"
REV = f"{ROOT}/run/rev2"


def paired(a, b, label):
    """Per-seed paired difference a - b, reported the way the paper does."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    tv = m / se
    p = 2 * tdist.sf(abs(tv), n - 1)
    half = tdist.ppf(0.975, n - 1) * se
    try:
        pw = float(wilcoxon(d).pvalue)
    except ValueError:
        pw = float("nan")
    return dict(label=label, n=n, mean=m, sd=sd, se=se,
                ci_lo=m - half, ci_hi=m + half, t=tv, p=p, p_wilcoxon=pw,
                sd_multiple=abs(m) / sd if sd else float("nan"),
                n_neg=int((d < 0).sum()), n_pos=int((d > 0).sum()))


def show(st):
    print(f"  {st['label']:<46s} {st['mean']:+.4f}  sd {st['sd']:.4f}  "
          f"CI95 [{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}]  "
          f"t({st['n']-1}) = {st['t']:+.2f}  p = {st['p']:.2g}  "
          f"{st['sd_multiple']:.1f}x sd  neg {st['n_neg']}/{st['n']}")


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.4f} +- {v.std(ddof=1):.4f}"


# ---------------------------------------------------------------- arm 1
print("=" * 78)
print("ARM 1 -- CD43 context transfer (v9 section 3.2 subsection, 3 seeds)")
print("=" * 78)

old = pd.read_json(f"{REV}/cd43_fixed_final.json")
new = pd.read_json(f"{REV}/cd43_fixed_det10_final.json")
print(f"old file: {len(old)} seeds {list(old.seed)}   "
      f"new file: {len(new)} seeds {list(new.seed)}")
print(f"n_test per seed (new): {list(new.n_test)}")
assert len(new) == 10, "cd43_fixed_det10_final.json is not 10 seeds"

rows = [
    ("ceiling, Avg_pEF target (rTetR)", new.ceil_pEF, None),
    ("  MotifCNN trained on Avg_pEF", new.r_pEFmodel_pEF, new.pct_ceil_pEFmodel_pEF),
    ("ceiling, CD43 target (dCas9)", new.ceil_cd43, None),
    ("  Avg_pEF measurement used directly", new.r_meas_pEF_cd43,
     new.r_meas_pEF_cd43 / new.ceil_cd43),
    ("  MotifCNN trained on Avg_pEF", new.r_pEFmodel_cd43, new.pct_ceil_pEFmodel_cd43),
    ("  MotifCNN trained on CD43", new.r_cd43model_cd43, new.pct_ceil_cd43model_cd43),
]
print(f"\n{'row':<40s} {'r (mean +- sd)':>20s} {'% ceiling':>18s}")
for name, r, pct in rows:
    pc = "" if pct is None else f"{100*np.mean(pct):.1f}% +- {100*np.std(pct, ddof=1):.1f}"
    print(f"{name:<40s} {ms(r):>20s} {pc:>18s}")

print("\npaired per-seed differences, n = 10:")
st_meas = paired(new.r_pEFmodel_cd43, new.r_meas_pEF_cd43,
                 "pEF-model r minus raw Avg_pEF r (on CD43)")
st_coll = paired(new.pct_ceil_cd43model_cd43, new.pct_ceil_pEFmodel_cd43,
                 "%ceiling: CD43-trained minus pEF-trained")
st_sym = paired(new.pct_ceil_cd43model_cd43, new.pct_ceil_pEFmodel_pEF,
                "%ceiling: CD43 on CD43 minus pEF on pEF")
for s in (st_meas, st_coll, st_sym):
    show(s)

print("\nv9 printed (3 seeds, pre-patch) -> this run (10 seeds, deterministic):")
cmp_rows = [
    ("ceiling pEF", 0.7641, new.ceil_pEF),
    ("MotifCNN/pEF r", 0.5850, new.r_pEFmodel_pEF),
    ("MotifCNN/pEF %ceil", 76.6, 100 * new.pct_ceil_pEFmodel_pEF),
    ("ceiling CD43", 0.7931, new.ceil_cd43),
    ("raw Avg_pEF r on CD43", 0.2183, new.r_meas_pEF_cd43),
    ("pEF-model r on CD43", 0.1303, new.r_pEFmodel_cd43),
    ("pEF-model %ceil on CD43", 16.4, 100 * new.pct_ceil_pEFmodel_cd43),
    ("CD43-model r on CD43", 0.6267, new.r_cd43model_cd43),
    ("CD43-model %ceil on CD43", 79.0, 100 * new.pct_ceil_cd43model_cd43),
]
for name, v9, series in cmp_rows:
    mu = float(np.mean(series))
    print(f"  {name:<28s} v9 {v9:>8.4f}   n=10 {mu:>8.4f}   "
          f"delta {mu - v9:+8.4f}")

json.dump(dict(
    n_seeds=int(len(new)),
    table={name: dict(r_mean=float(np.mean(r)), r_sd=float(np.std(r, ddof=1)),
                      pct_mean=None if pct is None else float(np.mean(pct)),
                      pct_sd=None if pct is None else float(np.std(pct, ddof=1)))
           for name, r, pct in rows},
    paired=[st_meas, st_coll, st_sym],
), open(f"{REV}/cd43_transfer_10seed.json", "w"), indent=1)
print(f"\nwrote {REV}/cd43_transfer_10seed.json")

# ---------------------------------------------------------------- arm 2
print()
print("=" * 78)
print("ARM 2 -- precision@k (v9 section 4.3, 2 seeds)")
print("=" * 78)

o = pd.read_csv(f"{REV}/cd43_topk.csv")
n3 = pd.read_csv(f"{REV}/cd43_topk3.csv")
print(f"old cd43_topk.csv : {o.seed.nunique()} seeds, rankers {sorted(o.ranker.unique())}")
print(f"new cd43_topk3.csv: {n3.seed.nunique()} seeds, rankers {sorted(n3.ranker.unique())}")
assert n3.seed.nunique() == 10

base = n3.base_rate.mean()
print(f"\nbase rate (hit in both CD43 bioreps): {100*base:.2f}% "
      f"(range {100*n3.base_rate.min():.2f}-{100*n3.base_rate.max():.2f})")

KEEP = [r for r in ("pred_cd43", "meas_Avg_pEF", "pred_pEF") if r in set(n3.ranker)]
ks = sorted(n3.k.unique())
print(f"\nprecision@k, mean +- sd over 10 deterministic seeds "
      f"(enrichment vs {100*base:.2f}% in brackets)")
hdr = f"{'k':>6s}" + "".join(f"{r:>28s}" for r in KEEP)
print(hdr)
tbl = {}
for k in ks:
    line = f"{k:>6d}"
    for r in KEEP:
        g = n3[(n3.k == k) & (n3.ranker == r)].sort_values("seed")
        mu, sd = g.precision.mean(), g.precision.std(ddof=1)
        tbl[(k, r)] = (mu, sd, g.precision.values)
        line += f"{100*mu:>14.1f}+-{100*sd:<4.1f}({mu/base:>4.1f}x)"
    print(line)

print("\nis the CD43-trained model ahead of the raw rTetR measurement, per seed?")
for k in ks:
    a = tbl[(k, "pred_cd43")][2]
    b = tbl[(k, "meas_Avg_pEF")][2]
    st = paired(a, b, f"k = {k:<6d} pred_cd43 - meas_Avg_pEF")
    show(st)

print("\nand ahead of the pEF-trained model?")
for k in ks:
    st = paired(tbl[(k, "pred_cd43")][2], tbl[(k, "pred_pEF")][2],
                f"k = {k:<6d} pred_cd43 - pred_pEF")
    show(st)

print("\nv9 printed (2 seeds) -> this run (10 deterministic seeds), precision:")
v9_tbl = {25: (0.800, 0.600, 0.620), 50: (0.720, 0.550, 0.610),
          100: (0.610, 0.575, 0.520), 200: (0.508, 0.528, 0.428),
          500: (0.356, 0.376, 0.291), 1000: (0.251, 0.247, 0.182)}
for k, trio in v9_tbl.items():
    if k not in ks:
        continue
    line = f"  k = {k:<6d}"
    for r, v9 in zip(KEEP, trio):
        mu = tbl[(k, r)][0]
        line += f"  {r}: {100*v9:.1f} -> {100*mu:.1f} ({100*(mu-v9):+.1f})"
    print(line)

n3.to_csv(f"{REV}/precision_at_k_10seed.csv", index=False)
print(f"\nwrote {REV}/precision_at_k_10seed.csv")
print("DONE_SEED10_CONSOLIDATE")

"""Is the model a de-noiser, or only an interpolator?

The CD43 ceiling is 0.79, so the two-biorep measurement itself correlates only
~0.79 with the true value, and the model reaches 79% of that ceiling. Classical
test theory then says model and measurement are equally good estimates of the
truth, and that averaging them beats either. That is an inference. This script
turns it into a measurement, with no test-theory assumption at all:

  held-out replicate = br2
  competitor 1       = br1 alone                 (what the assay gives you)
  competitor 2       = mean(z(br1), z(pred))     (assay + model)
  competitor 3       = pred alone

If competitor 2 beats competitor 1 at predicting br2, the model carries
information the assay's own replicate does not, and ranking by the blend is
strictly better than ranking by the measurement. br1/br2 are symmetric, so both
directions are reported.

Takes the prediction dump and output stems as arguments so it can be pointed at
the deterministic 10-seed rerun instead of only the original two-seed file:

  python denoise_test.py cd43_topk3_preds denoise_test_10seed
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t as tdist

ROOT = "/home/dmin/ai&stem_cell"
z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)

IN_STEM = sys.argv[1] if len(sys.argv) > 1 else "cd43_topk_preds"
OUT_STEM = sys.argv[2] if len(sys.argv) > 2 else "denoise_test"

df = pd.read_csv(f"{ROOT}/run/rev2/{IN_STEM}.csv")
rows = []
for seed, g in df.groupby("seed"):
    m = g.cd43_br1.notna() & g.cd43_br2.notna() & g.pred_cd43.notna()
    g = g[m]
    for held, other in (("cd43_br2", "cd43_br1"), ("cd43_br1", "cd43_br2")):
        y = g[held].values
        a, p = z(g[other].values), z(g.pred_cd43.values)
        for name, s in (("replicate alone", a),
                        ("model alone", p),
                        ("replicate + model", (a + p) / 2)):
            rows.append(dict(seed=seed, held=held, n=len(g), predictor=name,
                             r=float(pearsonr(s, y)[0]),
                             rho=float(spearmanr(s, y)[0])))
t = pd.DataFrame(rows)
t.to_csv(f"{ROOT}/run/rev2/{OUT_STEM}.csv", index=False)

print("=== predicting one held-out CD43 biological replicate ===")
print(t.pivot_table(index="predictor", columns="held", values="r")
      .round(4).to_string())
print("\n=== averaged over both directions and seeds ===")
mu = t.pivot_table(index="predictor", values=["r", "rho"]).round(4)
print(mu.to_string())
base = t[t.predictor == "replicate alone"].r.mean()
blend = t[t.predictor == "replicate + model"].r.mean()
print(f"\nblend minus replicate alone: {blend - base:+.4f} "
      f"({(blend/base - 1)*100:+.1f}%)")

# The two held-out directions reuse the same replicate pair, so they are not
# independent; average them within a seed and let the seed be the unit, which is
# what every other paired statistic in the paper does.
per_seed = (t.pivot_table(index="seed", columns="predictor", values=["r", "rho"])
            .sort_index())
n = len(per_seed)
print(f"\n=== paired per-seed differences, n = {n} seeds "
      f"(both directions averaged within a seed) ===")
for metric in ("r", "rho"):
    for a, b in (("model alone", "replicate alone"),
                 ("replicate + model", "replicate alone"),
                 ("replicate + model", "model alone")):
        d = (per_seed[(metric, a)] - per_seed[(metric, b)]).values
        m, sd = float(d.mean()), float(d.std(ddof=1))
        if n > 1 and sd > 0:
            se = sd / np.sqrt(n)
            tv, p = m / se, 2 * tdist.sf(abs(m / se), n - 1)
            half = tdist.ppf(0.975, n - 1) * se
            ci = f"[{m-half:+.4f}, {m+half:+.4f}]"
        else:
            tv, p, ci = float("nan"), float("nan"), "[n/a]"
        print(f"  {metric:>4s}  {a:<18s} - {b:<18s} {m:+.4f}  sd {sd:.4f}  "
              f"CI95 {ci}  t = {tv:+.2f}  p = {p:.2g}  "
              f"neg {int((d < 0).sum())}/{n}")
print(f"\n=== mean over seeds, both directions averaged ===")
print(per_seed.mean().round(4).to_string())
print(f"wrote {ROOT}/run/rev2/{OUT_STEM}.csv")
print("DONE_DENOISE")

"""The analytic ceiling on cross-context transfer, and what it forecloses.

A model trained on assay A cannot, at its own optimum, output anything other
than the latent rTetR quantity tau_pEF. So its correlation with the CD43 truth
is bounded by the correlation between the two LATENT quantities, which is the
observed cross-assay correlation disattenuated by both ceilings:

    r(tau_pEF, tau_cd43) = r(Avg_pEF, cd43) / (ceil_pEF * ceil_cd43)

Expressed on the paper's own scale -- percent of the CD43 noise ceiling, which
is r(pred, cd43)/ceil_cd43 -- the bound for ANY rTetR-trained model is

    bound = r(Avg_pEF, cd43) / (ceil_pEF * ceil_cd43)

computed per seed on that seed's own test tiles and ceilings, then averaged.
Writes bound.json so the prose and figure 3 read one number, not two
transcriptions of it.
"""
import json
import numpy as np

SRC = "/home/dmin/ai&stem_cell/run/rev2/cd43_fixed_det10_final.json"
OUT = "/home/dmin/ai&stem_cell/run/rev3/bound.json"
rows = json.load(open(SRC))

per = []
for r in rows:
    per.append(dict(
        seed=r["seed"],
        # raw measurement used directly as the ranking score, on CD43's ceiling
        raw=r["r_meas_pEF_cd43"] / r["ceil_cd43"],
        # our rTetR-trained CNN scored on CD43
        cnn=r["r_pEFmodel_cd43"] / r["ceil_cd43"],
        # the bound: disattenuate the raw cross-assay r by the pEF ceiling too
        bound=r["r_meas_pEF_cd43"] / (r["ceil_pEF"] * r["ceil_cd43"]),
        # what training on the right assay delivers
        right=r["pct_ceil_cd43model_cd43"],
        ceil_pEF=r["ceil_pEF"], ceil_cd43=r["ceil_cd43"],
    ))

def ms(k):
    v = np.array([p[k] for p in per], float) * 100
    return v.mean(), v.std(ddof=1)

out = {k: dict(zip(("mean", "sd"), ms(k))) for k in
       ("raw", "cnn", "bound", "right")}
out["n_seeds"] = len(per)
out["per_seed"] = per
out["headroom_pts"] = out["bound"]["mean"] - out["cnn"]["mean"]
out["gap_to_right_assay_pts"] = out["right"]["mean"] - out["bound"]["mean"]
json.dump(out, open(OUT, "w"), indent=1)

print(f"{'raw Avg_pEF as the score':38s} {out['raw']['mean']:6.2f}%  sd {out['raw']['sd']:4.2f}")
print(f"{'our rTetR-trained CNN':38s} {out['cnn']['mean']:6.2f}%  sd {out['cnn']['sd']:4.2f}")
print(f"{'BOUND for ANY rTetR-trained model':38s} {out['bound']['mean']:6.2f}%  sd {out['bound']['sd']:4.2f}")
print(f"{'CD43-trained CNN (right assay)':38s} {out['right']['mean']:6.2f}%  sd {out['right']['sd']:4.2f}")
print(f"\nheadroom left for a better rTetR model : {out['headroom_pts']:+5.1f} pts")
print(f"gap the bound still leaves to the right assay : {out['gap_to_right_assay_pts']:+5.1f} pts")
# sanity: the bound must dominate both observed rTetR-trained quantities
assert out["bound"]["mean"] > out["cnn"]["mean"], "bound below the model it bounds"
assert out["bound"]["mean"] > out["raw"]["mean"], "bound below the raw score"
assert out["bound"]["mean"] < out["right"]["mean"], "bound above right-assay training"
nviol = sum(p["bound"] < max(p["cnn"], p["raw"]) for p in per)
print(f"seeds where the bound is violated by an observed value: {nviol}/10")
print("wrote", OUT)
print("DONE_BOUND")

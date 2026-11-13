"""Decompose the cross-context collapse into an irreducible part and a model part.

Section 3.2 reports three quantities on one scale (% of the CD43 noise ceiling):

    right  = r(pred_cd43, cd43) / ceil_cd43      model trained on the scored assay
    cross  = r(pred_pEF , cd43) / ceil_cd43      model trained on the other assay
    cap    = r(Avg_pEF, cd43) / (ceil_pEF * ceil_cd43)
             the analytic ceiling for ANY model that attains its rTetR objective

and notes that the collapse `right - cross` is large while `cap - cross` leaves
head-room. What it never does is divide the collapse into its two parts, which is
the quantity that answers "is the signal itself context-dependent, or is the model
failing to carry it across?":

    total = right - cross          the whole collapse
    A     = right - cap            irreducible under the rTetR training objective
    B     = cap   - cross          model/representation shortfall, recoverable

    total = A + B    identically, so A/total is the irreducible SHARE.

Every term is formed WITHIN a seed before averaging, because `cap` and `right`
share the per-seed ceilings and the per-seed test split; averaging the three
levels first and subtracting afterwards would discard that pairing and give a
confidence interval that is too wide.

Scope, which must travel with the number: A is irreducible *given a model that
attains the rTetR objective*, not in principle. A function can beat `cap` on CD43
while being worse on pEF (section 3.2, scope note). A therefore bounds the family
of models defined by the training objective, not every function from sequence to
CD43 score.

The strong reading of "the representation cannot preserve the needed features" is
separately contradicted by `right` itself: the identical frozen ESM-2
representation reaches ~78% of the CD43 ceiling when the labels are CD43.

Usage:
    python transfer_decomp.py --in ../run/rev2/cd43_fixed_det10_final.json \
                              --out ../run/rev2/transfer_decomp.json
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats


def ci95(x):
    """Paired-sample t interval on the mean. n=10, so t not z."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return (float(x.mean() - t * se), float(x.mean() + t * se))


def describe(name, x, unit=""):
    lo, hi = ci95(x)
    x = np.asarray(x, dtype=float)
    return {"quantity": name, "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "ci95_lo": lo, "ci95_hi": hi, "min": float(x.min()),
            "max": float(x.max()), "n": int(len(x)), "unit": unit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="per-seed JSON from cd43_model.py --mode fixed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_json(args.inp)
    need = ["seed", "ceil_pEF", "ceil_cd43", "r_meas_pEF_cd43",
            "pct_ceil_cd43model_cd43", "pct_ceil_pEFmodel_cd43"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"input lacks {missing}; columns present: {list(df.columns)}")

    # pct_ceil_* are stored as FRACTIONS (0.777), not percentages. Put every
    # level on the same 0-100 scale before any subtraction.
    right = df["pct_ceil_cd43model_cd43"].values * 100.0
    cross = df["pct_ceil_pEFmodel_cd43"].values * 100.0
    cap = (df["r_meas_pEF_cd43"].values
           / (df["ceil_pEF"].values * df["ceil_cd43"].values)) * 100.0

    total = right - cross
    A = right - cap          # irreducible under the rTetR objective
    B = cap - cross          # model shortfall
    share_A = A / total      # formed per seed, then averaged
    share_B = B / total

    # Sanity: the identity must hold to floating point, and no seed may sit
    # above its own cap -- that was the pre-declared check in section 3.2.
    resid = np.abs((A + B) - total).max()
    if resid > 1e-9:
        raise SystemExit(f"identity A+B=total broken, max residual {resid:.3e}")
    above_cap = int((cross > cap).sum())
    cap_below_right = int((cap < right).sum())

    rows = [
        describe("right_assay (CD43-trained -> CD43)", right, "% of CD43 ceiling"),
        describe("cross_context (pEF-trained -> CD43)", cross, "% of CD43 ceiling"),
        describe("analytic_cap (any pEF-trained model)", cap, "% of CD43 ceiling"),
        describe("total_collapse (right - cross)", total, "ceiling points"),
        describe("A_irreducible (right - cap)", A, "ceiling points"),
        describe("B_model_shortfall (cap - cross)", B, "ceiling points"),
        describe("share_A (A / total)", share_A * 100.0, "% of the collapse"),
        describe("share_B (B / total)", share_B * 100.0, "% of the collapse"),
    ]
    out = pd.DataFrame(rows)

    meta = {
        "n_seeds": int(len(df)),
        "seeds": [int(s) for s in df["seed"].values],
        "seeds_above_own_cap": above_cap,
        "seeds_with_cap_below_right": cap_below_right,
        "identity_max_residual": float(resid),
        "source": args.inp,
    }

    with open(args.out, "w") as fh:
        json.dump({"meta": meta, "per_seed": {
            "seed": [int(s) for s in df["seed"].values],
            "right": right.tolist(), "cross": cross.tolist(), "cap": cap.tolist(),
            "total": total.tolist(), "A": A.tolist(), "B": B.tolist(),
            "share_A_pct": (share_A * 100).tolist(),
        }, "summary": rows}, fh, indent=2)

    pd.set_option("display.width", 140)
    print(out.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print()
    print(f"n = {meta['n_seeds']} seeds")
    print(f"seeds whose observed cross value exceeds their own cap: {above_cap} "
          f"(pre-declared check: must be 0)")
    print(f"seeds with cap < right: {cap_below_right} of {len(df)} "
          f"(if this were not all of them, the decomposition would have a "
          f"negative part)")
    print(f"identity |A+B-total| max residual: {resid:.3e}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

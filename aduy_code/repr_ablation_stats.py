"""Formal analysis of the 3 x 2 representation x pooling grid, as pre-registered.

Reads the per-seed JSON written by repr_ablation.py and applies exactly the plan
in thiet_ke_onehot_2x2.md section 6, with nothing added after seeing the numbers:

  PRIMARY endpoint, one only:
      D = (R1_esm480, MotifCNN) - (R2_embed480, MotifCNN)
      paired per seed, on the % noise-ceiling scale, n = 10
      two-sided paired t  +  TOST against the +/-5 ceiling-point margin

  SECONDARY, 5 comparisons, Benjamini-Hochberg:
      R1,P2-R3,P2   R2,P2-R3,P2   R1,P2-R1,P1   R2,P2-R2,P1   R3,P2-R3,P1

The last two separate the pooling contribution WITHIN a representation, which is
the second, independent handle on "convolution or pretraining".

Scale. The pre-registered margin is +/-5 percentage points of the noise ceiling,
the same margin section 3.12 used for ESM-2 35M vs 150M. Raw test r is converted
with the ceiling of the seed's OWN test split -- sqrt of the Spearman-Brown
reliability of the two pEF replicates on exactly those tiles -- recomputed here
from master.csv and the cluster file, because grouped_split is a pure function of
(groups, seed) and so the split is reproducible without re-training. Dividing both
arms of a paired difference by the same per-seed ceiling keeps the pairing intact.

Tiles with only one pEF replicate are masked out of the reliability estimate; an
unmasked pearsonr returns NaN there and would silently void every percentage.

QC gates re-checked here, both declared before the run:
  - mean best_epoch < 90 of the 100-epoch cap in every cell, else that cell did
    not converge and the whole grid must be re-run
  - sd of the primary difference <= 5 points, else the pre-registered stopping
    rule calls for 20 seeds (and that decision may not be taken after reading)

Usage:
    python repr_ablation_stats.py --in ../run/rev2/repr_ablation.json \
        --csv ../run/rev2/master.csv.gz --clusters ../run/protein_clusters.tsv \
        --out ../run/rev2/repr_ablation_stats.json
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

MARGIN = 5.0            # ceiling points, pre-registered
PRIMARY = (("R1_esm480", "MotifCNN"), ("R2_embed480", "MotifCNN"))
SECONDARY = [
    (("R1_esm480", "MotifCNN"), ("R3_onehot20", "MotifCNN")),
    (("R2_embed480", "MotifCNN"), ("R3_onehot20", "MotifCNN")),
    (("R1_esm480", "MotifCNN"), ("R1_esm480", "MeanPool")),
    (("R2_embed480", "MotifCNN"), ("R2_embed480", "MeanPool")),
    (("R3_onehot20", "MotifCNN"), ("R3_onehot20", "MeanPool")),
]


def seed_ceilings(csv, clusters, target, reps, group_col, seeds):
    """Per-seed noise ceiling on that seed's own test split.

    grouped_split is a pure function of (groups, seed), so the split is
    reproducible here without re-training anything.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_cnn import grouped_split

    df = pd.read_csv(csv, low_memory=False)
    keep = df[target].notna().values
    groups = df[group_col].astype(str).values
    if clusters:
        cl = pd.read_csv(clusters, sep="\t", header=None, names=["rep", "member"])
        m = dict(zip(cl["member"], cl["rep"]))
        groups = np.array([m.get(g, g) for g in groups])

    out = {}
    for seed in seeds:
        te = [i[keep[i]] for i in grouped_split(groups, seed)][2]
        sub = df.iloc[te]
        ok = sub[reps[0]].notna().values & sub[reps[1]].notna().values
        r = float(stats.pearsonr(sub[reps[0]][ok], sub[reps[1]][ok])[0])
        out[int(seed)] = {"rel_halves": r,
                          "reliability_sb": 2 * r / (1 + r),
                          "ceiling": float(np.sqrt(2 * r / (1 + r))),
                          "n_test": int(len(te)), "n_both_reps": int(ok.sum())}
    return out


def tost(d, margin):
    """Two one-sided paired t tests; the TOST p-value is the larger of the two."""
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t_lo = (d.mean() + margin) / se          # H0: mean <= -margin
    t_hi = (d.mean() - margin) / se          # H0: mean >= +margin
    p_lo = 1 - stats.t.cdf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    return float(max(p_lo, p_hi)), float(p_lo), float(p_hi)


def paired(d, margin=MARGIN):
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    t, p = stats.ttest_rel(d, np.zeros(n))
    p_tost, p_lo, p_hi = tost(d, margin)
    return {
        "n": int(n), "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
        "ci95_lo": float(d.mean() - tcrit * se),
        "ci95_hi": float(d.mean() + tcrit * se),
        "t": float(t), "p_two_sided": float(p),
        "p_tost": p_tost, "p_tost_lower": p_lo, "p_tost_upper": p_hi,
        "equivalent_at_margin": bool(p_tost < 0.05),
        "ci_excludes_zero": bool((d.mean() - tcrit * se) * (d.mean() + tcrit * se) > 0),
        "same_sign": int(max((d > 0).sum(), (d < 0).sum())),
        "cohens_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan"),
    }


def flat(frame):
    """MultiIndex -> 'repr/arch' string keys, because json rejects tuple keys."""
    f = frame.copy()
    f.index = [f"{a}/{b}" for a, b in f.index]
    return f.round(6).to_dict()


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = n - rank + 1
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return adj.tolist()


def verdict(res):
    """The pre-registered decision table, applied without re-interpretation."""
    if res["equivalent_at_margin"]:
        return ("EQUIVALENT within +/-5 ceiling points: pretraining does NOT carry "
                "the gap. Section 3.2 must be rewritten to attribute it to "
                "windowed convolution rather than to pretraining.")
    if abs(res["mean"]) > MARGIN and res["ci_excludes_zero"]:
        return ("PRETRAINING CARRIES THE GAP: |delta| > 5 points and the interval "
                "excludes 0. Section 3.2's sentence is attributed correctly for "
                "the first time and section 2.1 gains a real justification.")
    return ("UNDERPOWERED / INCONCLUSIVE: the interval spans both 0 and the "
            "margin. Report the interval and draw no conclusion. Do NOT add "
            "seeds after looking -- that decision was pre-registered.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--target", default="Avg_pEF")
    ap.add_argument("--reps", nargs=2, default=["R1_pEF", "R2_pEF"])
    ap.add_argument("--group-col", default="protein")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_json(args.inp)
    seeds = sorted(df["seed"].unique().tolist())
    cells = sorted({(r, a) for r, a in zip(df["representation"], df["arch"])})
    print(f"{len(df)} runs | {len(cells)} cells | {len(seeds)} seeds")

    # ---------------- QC gate 1: convergence inside the epoch cap
    ep = df.groupby(["representation", "arch"])["best_epoch"].agg(["mean", "max"])
    print("\n=== QC 1: best_epoch per cell (gate: mean < 90 of 100) ===")
    print(ep.round(2).to_string())
    qc1 = bool((ep["mean"] < 90).all())
    print("QC 1:", "PASS" if qc1 else "FAIL -- raise --epochs and re-run ALL cells")

    # ---------------- ceilings
    ceil = seed_ceilings(args.csv, args.clusters, args.target, args.reps,
                         args.group_col, seeds)
    print(f"\n=== per-seed noise ceiling on '{args.target}' "
          f"(own test split, both replicates present) ===")
    cs = pd.DataFrame(ceil).T
    print(cs.round(4).to_string())
    print(f"mean ceiling {cs['ceiling'].mean():.4f} "
          f"(sd {cs['ceiling'].std(ddof=1):.4f}); "
          f"{MARGIN:.0f} ceiling points = "
          f"{MARGIN/100*cs['ceiling'].mean():.4f} raw r on average")

    df["pct_ceil"] = [row.test_r / ceil[int(row.seed)]["ceiling"] * 100
                      for row in df.itertuples()]
    piv = df.pivot_table(index="seed", columns=["representation", "arch"],
                         values="pct_ceil")

    print("\n=== cell means, % of the per-seed noise ceiling ===")
    lvl = df.groupby(["representation", "arch"])["pct_ceil"].agg(["mean", "std"])
    lvl["params"] = df.groupby(["representation", "arch"])["params"].first()
    print(lvl.round(2).to_string())

    # ---------------- primary
    a, b = PRIMARY
    if a not in piv.columns or b not in piv.columns:
        raise SystemExit(f"primary cells missing: {a} or {b}")
    d = (piv[a] - piv[b]).values
    res = paired(d)
    res["contrast"] = f"{a[0]}/{a[1]} - {b[0]}/{b[1]}"
    res["per_seed"] = d.tolist()

    print("\n" + "=" * 72)
    print(f"PRIMARY  {res['contrast']}  (ceiling points, paired by seed)")
    print("=" * 72)
    print(f"  mean  {res['mean']:+.2f}   sd {res['sd']:.2f}   "
          f"95% CI [{res['ci95_lo']:+.2f}, {res['ci95_hi']:+.2f}]   "
          f"dz {res['cohens_dz']:+.2f}")
    print(f"  two-sided paired t: t = {res['t']:+.3f}, p = {res['p_two_sided']:.2e}")
    print(f"  TOST +/-{MARGIN:.0f}: p = {res['p_tost']:.4f} "
          f"(lower {res['p_tost_lower']:.4f}, upper {res['p_tost_upper']:.4f})")
    print(f"  same sign in {res['same_sign']}/{res['n']} seeds")

    qc2 = res["sd"] <= 5.0
    print(f"\n  QC 2 (pre-registered stopping rule): sd = {res['sd']:.2f} points "
          f"-> {'PASS' if qc2 else 'FAIL, the plan calls for 20 seeds'}")
    print(f"\n  VERDICT: {verdict(res)}")

    # ---------------- secondary, BH over the 5 declared comparisons
    sec = []
    for x, y in SECONDARY:
        if x not in piv.columns or y not in piv.columns:
            continue
        dd = (piv[x] - piv[y]).values
        r = paired(dd)
        r["contrast"] = f"{x[0]}/{x[1]} - {y[0]}/{y[1]}"
        sec.append(r)
    if sec:
        for r, q in zip(sec, bh([s["p_two_sided"] for s in sec])):
            r["p_bh"] = float(q)
        print("\n=== SECONDARY (5 declared, Benjamini-Hochberg) ===")
        print(f"{'contrast':54s} {'mean':>8} {'95% CI':>18} {'p':>10} "
              f"{'p_BH':>9} {'TOST':>8} {'sign':>6}")
        for r in sec:
            print(f"{r['contrast']:54s} {r['mean']:>+8.2f} "
                  f"[{r['ci95_lo']:+6.2f},{r['ci95_hi']:+6.2f}] "
                  f"{r['p_two_sided']:>10.2e} {r['p_bh']:>9.2e} "
                  f"{r['p_tost']:>8.4f} {r['same_sign']:>4d}/{r['n']}")

    # ---------------- cross-context arm, if the run carried one
    cross = None
    if "pct_ceil_cross" in df.columns and df["pct_ceil_cross"].notna().any():
        pc = df.pivot_table(index="seed", columns=["representation", "arch"],
                            values="pct_ceil_cross") * 100
        capv = df.groupby("seed")["cap"].first().values * 100
        tgt = df["cross_target"].dropna().iloc[0]
        print(f"\n=== CROSS-CONTEXT arm: scored on '{tgt}', never trained on it "
              f"(% of the {tgt} ceiling) ===")
        cl_ = (df.groupby(["representation", "arch"])["pct_ceil_cross"]
                 .agg(["mean", "std"]) * 100)
        print(cl_.round(2).to_string())
        print(f"\n  analytic cap for ANY {args.target}-trained model: "
              f"{capv.mean():.2f} (sd {capv.std(ddof=1):.2f})")
        spread = cl_["mean"].max() - cl_["mean"].min()
        train_spread = lvl["mean"].max() - lvl["mean"].min()
        print(f"  spread across the 6 cells: {spread:.2f} points on the cross "
              f"target vs {train_spread:.2f} on the training target")
        xs = []
        for x, y in [PRIMARY] + SECONDARY:
            if x not in pc.columns or y not in pc.columns:
                continue
            dd = (pc[x] - pc[y]).values
            r = paired(dd)
            r["contrast"] = f"{x[0]}/{x[1]} - {y[0]}/{y[1]}"
            xs.append(r)
        for r, q in zip(xs, bh([s["p_two_sided"] for s in xs])):
            r["p_bh"] = float(q)
        print(f"\n{'contrast':54s} {'mean':>8} {'95% CI':>18} {'p_BH':>9} "
              f"{'TOST':>8} {'sign':>6}")
        for r in xs:
            print(f"{r['contrast']:54s} {r['mean']:>+8.2f} "
                  f"[{r['ci95_lo']:+6.2f},{r['ci95_hi']:+6.2f}] "
                  f"{r['p_bh']:>9.2e} {r['p_tost']:>8.4f} "
                  f"{r['same_sign']:>4d}/{r['n']}")
        n_over = int((pc.max(axis=1).values > capv).sum())
        print(f"\n  seeds where the BEST cell beats its own analytic cap: "
              f"{n_over}/{len(capv)}")
        print("  Reading rule from the design: if the representations sit within "
              "a few points of each other here while differing on the training "
              "target, the transfer shortfall is not the representation's.")
        cross = {"cells": flat(cl_), "cap_mean": float(capv.mean()),
                 "cap_sd": float(capv.std(ddof=1)), "cross_target": tgt,
                 "contrasts": xs, "spread_cross": float(spread),
                 "spread_train": float(train_spread), "seeds_over_cap": n_over}

    with open(args.out, "w") as fh:
        json.dump({"margin_ceiling_points": MARGIN, "n_seeds": len(seeds),
                   "qc_convergence_pass": qc1, "qc_sd_pass": bool(qc2),
                   "ceilings": ceil,
                   "cell_levels_pct_ceiling": flat(lvl),
                   "primary": res, "secondary": sec, "cross_context": cross,
                   "verdict": verdict(res), "source": args.inp}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

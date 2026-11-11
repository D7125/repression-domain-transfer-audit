"""The strong version of "we already have the measurements, why do we need a model?"

cd43_topk compared the sequence model against ONE measured column, Avg_pEF.
That is the weak form of the objection and it is not the one a reviewer will
make. The strong form is: take EVERY column the screen already produced, throw
the best non-sequence learner you have at it, and see whether reading the amino
acid sequence still buys anything.

So this adds two rankers on exactly the tiles and splits cd43_topk3 used:

  tab_meas    gradient boosting on the measured columns only -> predicted cd43
  meas_FLAG   Avg_FLAG on its own (the "just screen on expression" strawman,
              included so it is refuted with a number instead of an opinion)

No GPU and no retraining: pred_cd43 is read back from cd43_topk3_preds.csv, so
the sequence model being compared against is byte-identical to the published one.

FEATURES ARE MEASUREMENTS ONLY. Every cd43-derived column is excluded -- it is
the label. protein and seq are excluded too: one leaks cluster identity, the
other is the sequence, which is the thing this baseline is supposed to not have.

PRE-COMMITTED (fixed before the script was first run):
  Primary     blend_z - tab_meas, precision@50, paired by seed, 10 seeds.
              MBRE +10 pp. Observed SD of paired diffs ~9 pp at n=10 gives an
              MDE near 8.5 pp, so the design can see the MBRE.
  Secondary   pred_cd43 - tab_meas @50; tab_meas - meas_Avg_pEF @50;
              the same two contrasts at k in {20,100,200}; BH within the family.
  Reading     If blend_z - tab_meas straddles 0, the honest claim shrinks to
              "the sequence model matches what the measurements already give"
              and the poster's selection story has to say so.

Usage:
  python cd43_tabular_baseline.py --seeds 0 1 2 3 4 5 6 7 8 9 \
      --preds cd43_topk3_preds.csv --out cd43_tabular
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/home/dmin/ai&stem_cell/cnn_AI&stemcell")
from train_cnn import grouped_split

ROOT = "/home/dmin/ai&stem_cell"
KS = [20, 25, 50, 100, 200, 500, 1000]

# Measured columns a lab holds before any sequence model exists, split in two.
#
# CLEAN: real measurements, all under 25% missing. This is the honest baseline.
# SELECTION: the extra columns whose MISSINGNESS is itself informative --
#   Hit_pEF is 91.6% absent, Hit_minCMV 99.0%, the retest columns 91.6%, and
#   absent means "never entered into the confirmation screen". Entry was decided
#   by Avg_pEF, so a tree can read those NaNs as "somebody already thought this
#   tile was worth a second look". That is the same defect that made
#   run/cls_homology.json unusable (98.2% of its negative class never tested),
#   so it is kept in a SEPARATE arm. If tab_sel beats tab_meas, the gap is the
#   selection artifact, not the measurements.
FEATS_CLEAN = ["Avg_minCMV", "Avg_FLAG", "Avg_pEF", "SE_pEF", "Avg_PGK",
               "SE_PGK", "R1_pEF", "R2_pEF", "Hit_PGK", "Hit_FLAG",
               "n_elem", "tile_num", "tile_start", "tile_end"]
FEATS_SEL = FEATS_CLEAN + ["Hit_pEF", "Hit_minCMV", "Avg_pEF_retest",
                           "R1_pEF_retest", "R2_pEF_retest"]
HITCOLS = ["Hit_PGK", "Hit_FLAG", "Hit_pEF", "Hit_minCMV"]
BANNED = {"cd43", "cd43_br1", "cd43_br2", "cd43_hit_both", "protein", "seq",
          "sequence"}


def z(v):
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def prec_at(score, hit, k):
    return hit[np.argsort(-np.asarray(score, dtype=float))[:k]].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(range(10)))
    ap.add_argument("--preds", default=f"{ROOT}/run/rev2/cd43_topk3_preds.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert not (set(FEATS_SEL) & BANNED), "a label column leaked into FEATS"

    df = pd.read_csv(f"{ROOT}/run/rev2/master.csv", low_memory=False)
    # 'Hit' / 'Not Hit' -> 1 / 0, NaN kept as NaN (the tree splits on it).
    for c in HITCOLS:
        df[c] = df[c].map({"Hit": 1.0, "Not Hit": 0.0}).astype(float)
    saved = pd.read_csv(args.preds, low_memory=False)
    has = (df.cd43.notna().values & df.cd43_br1.notna().values
           & df.cd43_br2.notna().values)
    cl = pd.read_csv(f"{ROOT}/run/protein_clusters.tsv", sep="\t", header=None,
                     names=["rep", "member"])
    cmap = dict(zip(cl["member"], cl["rep"]))
    groups = np.array([cmap.get(g, g) for g in df.protein.astype(str)])
    print(f"tab_meas, {len(FEATS_CLEAN)} measured feats: "
          f"{', '.join(FEATS_CLEAN)}", flush=True)
    print(f"tab_sel adds {len(FEATS_SEL)-len(FEATS_CLEAN)} selection-flavoured: "
          f"{', '.join(FEATS_SEL[len(FEATS_CLEAN):])}", flush=True)

    rows = []
    for seed in args.seeds:
        tr_i, va_i, te_i = grouped_split(groups, seed)
        tr_c, te_c = tr_i[has[tr_i]], te_i[has[te_i]]
        sub = df.iloc[te_c].reset_index(drop=True)
        prev = saved[saved.seed == seed].reset_index(drop=True)

        # The saved rows must be the same tiles in the same order, or the
        # sequence model's scores would be pasted onto the wrong tiles.
        if len(prev) != len(sub) or not np.allclose(
                prev.Avg_pEF.values, sub.Avg_pEF.values, equal_nan=True):
            raise SystemExit(f"seed {seed}: saved preds do not line up with "
                             f"the recomputed split ({len(prev)} vs {len(sub)})")

        fits = {}
        for nm, fs in (("tab_meas", FEATS_CLEAN), ("tab_sel", FEATS_SEL)):
            gb = HistGradientBoostingRegressor(max_iter=300, random_state=seed)
            gb.fit(df.iloc[tr_c][fs], df.iloc[tr_c].cd43.values)
            fits[nm] = gb.predict(sub[fs])

        hit = sub.cd43_hit_both.values.astype(bool)
        base = hit.mean()
        rankers = {
            "tab_meas": fits["tab_meas"],
            "tab_sel": fits["tab_sel"],
            "meas_FLAG": sub.Avg_FLAG.values,
            "meas_Avg_pEF": sub.Avg_pEF.values,
            "pred_cd43": prev.pred_cd43.values,
            "blend_z": prev.blend_z.values,
            "blend_tab": (z(fits["tab_meas"])
                          + z(prev.pred_cd43.values)) / 2.0,
        }
        print(f"\n=== seed {seed}: test {len(sub):,}  base {base*100:.2f}% ===",
              flush=True)
        for name, sc in rankers.items():
            for k in KS:
                if k > len(sub):
                    continue
                p = float(prec_at(sc, hit, k))
                rows.append(dict(ranker=name, k=k, precision=p, seed=seed,
                                 n_hit=int(p * k), base_rate=base))
            p50 = prec_at(sc, hit, 50) * 100
            print(f"  {name:>13s} @50 = {p50:5.1f}%", flush=True)
        pd.DataFrame(rows).to_csv(f"{args.out}.csv", index=False)

    t = pd.DataFrame(rows)
    print("\n=== precision@k, mean over seeds (%) ===")
    print((t.pivot_table(index="k", columns="ranker",
                         values="precision") * 100).round(1).to_string())
    print(f"\nwrote {args.out}.csv")
    print("DONE_CD43_TABULAR")


if __name__ == "__main__":
    main()

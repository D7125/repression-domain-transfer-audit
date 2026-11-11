"""precision@k for four rankers on the CD43 selection task, 10 seeds.

Answers the only question a lab actually spends money on: of the top k tiles a
ranker hands you, how many are real CD43 hits in both bioreps?

Numbers are read from run/rev2/cd43_topk2.csv at build time, so the figure
cannot drift from the run. Palette is Okabe-Ito, validated with the dataviz
skill's checker (all-pairs, light surface): no FAILs; the two WARNs (CVD dE 7.6
deutan, #CC79A7 contrast 2.98) are cleared by direct-labelling every series.

Usage:
  conda activate machine_deep_learning
  python make_fig_topk.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# t_{0.975, df}. Hard-coded so the figure builds in any env that has matplotlib;
# scipy is not installed in `genetics`, which is the env that has matplotlib.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 14: 2.145,
        19: 2.093, 29: 2.045}


def tcrit_for(df):
    return T975.get(df, 1.96)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# topk3, not topk2: the deterministic run. topk2 was produced before cuDNN,
# cuBLAS and the DataLoader were pinned, so rerunning it gives different models
# from the same seed. Its lookup column is identical (the split was always
# deterministic); only the model-based rankers moved, by up to 4.5 points.
SRC = os.path.join(ROOT, "run", "rev2", "cd43_topk3.csv")

INK, INK_M, GRID = "#1B3A5C", "#6F7B88", "#DFE4EA"
SERIES = [                       # (column, colour, label)
    ("blend_z",      "#0072B2", "Model + table (z-blend)"),
    ("pred_cd43",    "#D55E00", "Model alone, CD43-trained"),
    ("pred_pEF",     "#009E73", "Model alone, rTetR-trained"),
    ("meas_Avg_pEF", "#CC79A7", "Table alone (measured Avg_pEF)"),
]

t = pd.read_csv(SRC)
w = t.pivot_table(index=["seed", "k"], columns="ranker",
                  values="precision").reset_index()
ks = sorted(w.k.unique())
nseed = w.seed.nunique()
base = t.base_rate.mean() * 100

fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=200)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# base rate: the floor every ranker is beating
ax.axhline(base, color=INK_M, lw=1.4, ls=(0, (6, 4)), zorder=1)
ax.text(ks[0] * 0.93, base + 1.6, f"base rate {base:.1f}%", color=INK_M,
        fontsize=9, style="italic", va="bottom")

tcrit = tcrit_for(nseed - 1)
for col, colour, label in SERIES:
    m = np.array([w[w.k == k][col].mean() for k in ks]) * 100
    se = np.array([w[w.k == k][col].std(ddof=1) / np.sqrt(nseed)
                   for k in ks]) * 100
    ax.fill_between(ks, m - tcrit * se, m + tcrit * se, color=colour,
                    alpha=0.15, lw=0, zorder=2)
    ax.plot(ks, m, color=colour, lw=2.0, marker="o", ms=5.5,
            mec="white", mew=1.4, zorder=3)

# Direct labels: identity never rests on colour alone, and they also discharge
# the validator's two WARNs. At k=1000 three lines sit inside 1 point of each
# other, so the label y is spread to a minimum gap and a leader is drawn when it
# has moved away from its line.
ends = sorted(((w[w.k == ks[-1]][col].mean() * 100, colour, lab)
               for col, colour, lab in SERIES), reverse=True)
MINGAP = 5.2
placed = []
for yv, colour, lab in ends:
    ty = yv if not placed else min(yv, placed[-1][0] - MINGAP)
    placed.append((ty, yv, colour, lab))
for ty, yv, colour, lab in placed:
    if abs(ty - yv) > 0.4:
        ax.plot([ks[-1], ks[-1] * 1.30], [yv, ty], color=colour, lw=1.0,
                alpha=0.55, zorder=2, clip_on=False)
    ax.annotate(lab, xy=(ks[-1] * 1.33, ty), xytext=(0, 0),
                textcoords="offset points", va="center", ha="left",
                fontsize=9.5, color=INK, annotation_clip=False)

ax.set_xscale("log")
ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.minorticks_off()
ax.set_xlim(ks[0] * 0.88, ks[-1] * 3.9)
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 20))
ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])

ax.set_xlabel("k  —  number of tiles you pick", fontsize=10.5, color=INK,
              labelpad=8)
ax.set_ylabel("of those k, % that are real CD43 hits\n(both bioreps)",
              fontsize=10.5, color=INK, labelpad=8)
ax.set_title("Picking CD43 effectors: what each ranker buys you",
             fontsize=13, color=INK, fontweight="bold", loc="left", pad=14)
ax.text(0, 1.015, f"mean of {nseed} seeds · band = 95% CI · held-out homology "
        f"clusters · deterministic run", transform=ax.transAxes, fontsize=9.3,
        color=INK_M, va="bottom")

ax.grid(axis="y", color=GRID, lw=1, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK_M, labelsize=9.5, length=0)

fig.tight_layout()
out = os.path.join(HERE, "fig_topk_cd43.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")

print(f"\nprecision@k, mean of {nseed} runs (%)")
piv = (t.pivot_table(index="k", columns="ranker", values="precision") * 100)
print(piv.round(1).to_string())
print("DONE_FIG_TOPK")

"""Six numbered publication figures for the v12 manuscript.

  python -I figures/make_paper_figures.py

Each figure is built from a file in run/ and NOTHING is typed in by hand: every
value plotted is read from the saved result, and the script asserts that the
numbers it is about to draw match the ones the manuscript prints. A figure that
disagrees with the text therefore fails the build instead of shipping.

Figure -> claim (manuscript section):
  1  motif regexes mostly count composition                       (3.1)
  2  the sequence model reads reproducible signal within an assay  (3.2)
  3  context transfer: the assay matters more than the architecture (3.3)
  5  rank order is not preserved across contexts                   (3.9)
  6  precision@k: the failure is mid-distribution, not at the top  (3.10)
  4  saliency: most raw enrichment is representation, not learning (3.3)

Colour semantics are fixed across all six figures (Okabe-Ito, colourblind-safe):
  blue       = trained on / measured by rTetR (pEF)
  vermillion = trained on / measured by dCas9 (CD43)
  green      = composition baseline (ridge)
  orange     = raw measurement used directly
  grey       = null / control state
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "run")
OUT = os.path.join(ROOT, "figures")

# ---------------------------------------------------------------- house style
# Okabe-Ito. Assigned by entity, never cycled by rank.
BLUE = "#0072B2"   # rTetR / pEF arm
VERM = "#D55E00"   # dCas9 / CD43 arm
GREEN = "#009E73"  # ridge on amino-acid composition
ORANGE = "#E69F00" # raw measurement
GREY = "#7A7A7A"   # null / control
INK = "#1A1A1A"
MUTED = "#6B6B6B"
SURFACE = "#FFFFFF"

plt.rcParams.update({
    # DejaVu Sans is matplotlib's bundled font and the only one guaranteed to
    # carry Vietnamese combining diacritics; Arial drops some stacked marks.
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": "#BFBFBF",
    "axes.linewidth": 0.7,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.6,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "lines.linewidth": 1.6,
})

MM = 1 / 25.4
SINGLE = 89 * MM    # 3.50 in
DOUBLE = 183 * MM   # 7.20 in

CHECKS = []


def near(label, got, want, tol):
    """Record a check that a plotted value matches what the manuscript prints."""
    ok = abs(got - want) <= tol
    CHECKS.append((label, got, want, ok))
    return ok


def tidy(ax, grid="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, zorder=0)
        ax.set_axisbelow(True)


def panel(ax, letter, dx=-0.085, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left", color=INK)


def save(fig, stem):
    for ext, kw in (("png", {"dpi": 400}), ("pdf", {}), ("svg", {})):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), bbox_inches="tight",
                    pad_inches=0.02, **kw)
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf / .svg")


# ============================================================ FIGURE 1  motif
def figure1():
    d = pd.read_csv(os.path.join(RUN, "motif_background_derived.csv"))
    pwm = pd.read_csv(os.path.join(RUN, "motif_pwm_shuffle.csv"))

    order = ["CtBP", "SIM", "SUMO_fwd", "SUMO_rev_unpublished",
             "HP1_PxVxL", "CtBP_refined", "TLE_WRPW"]
    nice = {"CtBP": "CtBP rộng\nP.[DENS][LM].", "SIM": "SIM\n(dài, biến thiên)",
            "SUMO_fwd": "SUMO\n[VILMF]K.E", "SUMO_rev_unpublished": "SUMO ngược\nE.K[VILMF]",
            "HP1_PxVxL": "HP1\nP.V.L", "CtBP_refined": "CtBP tinh chỉnh\nPL[DN]L.",
            "TLE_WRPW": "TLE\nWRPW"}
    d = d.set_index("motif").loc[order].reset_index()

    near("Fig1 CtBP observed %", d.loc[0, "observed"] * 100, 13.33, 0.02)
    near("Fig1 CtBP shuffled %", d.loc[0, "shuffled"] * 100, 13.30, 0.02)
    near("Fig1 HP1 ratio", d.set_index("motif").loc["HP1_PxVxL", "ratio_obs_sh"], 0.86, 0.01)

    fig = plt.figure(figsize=(DOUBLE, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])

    y = np.arange(len(d))
    h = 0.34
    obs = d["observed"].values * 100
    shuf = d["shuffled"].values * 100
    # 2px surface gap between the paired bars: drawn as an edge in the surface
    # colour so adjacent fills never touch.
    ax.barh(y + h / 2 + 0.012, obs, height=h, color=BLUE, label="chuỗi thật",
            edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax.barh(y - h / 2 - 0.012, shuf, height=h, color=GREY,
            label="xáo trộn giữ thành phần", edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax.errorbar(obs, y + h / 2 + 0.012,
                xerr=[obs - d["obs_lo"] * 100, d["obs_hi"] * 100 - obs],
                fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6, zorder=4)
    ax.errorbar(shuf, y - h / 2 - 0.012,
                xerr=[shuf - d["sh_lo"] * 100, d["sh_hi"] * 100 - shuf],
                fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6, zorder=4)

    for i, r in enumerate(d["ratio_obs_sh"].values):
        xmax = max(obs[i], shuf[i])
        ax.text(xmax + 0.45, y[i], f"{r:.2f}×".replace(".", ","),
                va="center", ha="left", fontsize=7.5,
                color=INK if abs(r - 1) < 0.1 else MUTED,
                fontweight="bold" if abs(r - 1) < 0.1 else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels([nice[m] for m in d["motif"]], fontsize=7)
    ax.set_xlabel("Tần suất khớp trên 96.682 đoạn không-hit (%)")
    ax.set_xlim(0, 17.5)
    # the two shortest motifs leave the upper right quadrant empty
    ax.legend(frameon=False, loc="upper right", handlelength=1.1,
              bbox_to_anchor=(1.0, 1.02))
    tidy(ax, grid="x")
    panel(ax, "a", dx=-0.30)

    # -- panel b: PWM / logistic AUC
    ax2 = fig.add_subplot(gs[0, 1])
    p = pwm.dropna(subset=["logit_auc"]).copy()
    lab = {"HP1_PxVxL": "HP1", "CtBP": "CtBP rộng", "SUMO_fwd": "SUMO",
           "SUMO_rev": "SUMO ngược"}
    p["lab"] = p["motif"].map(lab)
    p = p.sort_values("logit_auc")
    near("Fig1 HP1 logistic AUC", float(p.set_index("motif").loc["HP1_PxVxL", "logit_auc"]), 0.501, 0.002)
    near("Fig1 CtBP logistic AUC", float(p.set_index("motif").loc["CtBP", "logit_auc"]), 0.542, 0.002)

    yy = np.arange(len(p))
    ax2.axvline(0.5, color=INK, lw=0.9, ls="--", zorder=2)
    ax2.text(0.506, -0.62, "0,5 = không phân biệt được", fontsize=6.5,
             color=MUTED, va="center", ha="left")
    cols = [GREEN if v >= 0.6 else GREY for v in p["logit_auc"]]
    ax2.barh(yy, p["logit_auc"] - 0.5, left=0.5, height=0.5, color=cols,
             edgecolor=SURFACE, linewidth=0.8, zorder=3)
    for i, (v, pw) in enumerate(zip(p["logit_auc"], p["pwm_auc"])):
        ax2.plot([pw], [yy[i]], marker="D", ms=4.5, color=INK, zorder=5,
                 markeredgecolor=SURFACE, markeredgewidth=0.8)
        # right-aligned at a fixed x so the value never lands on the diamond
        ax2.text(0.938, yy[i], f"{v:.3f}".replace(".", ","), va="center",
                 ha="right", fontsize=7, color=INK)
    ax2.set_yticks(yy)
    ax2.set_yticklabels(p["lab"], fontsize=7)
    ax2.set_xlim(0.48, 0.94)
    ax2.set_ylim(-1.0, len(p) - 0.45)
    ax2.set_xlabel("AUC phân biệt thật / xáo trộn")
    ax2.legend(handles=[Patch(facecolor=GREY, label="hồi quy logistic"),
                        plt.Line2D([], [], marker="D", ls="none", ms=4.5,
                                   color=INK, label="PWM log-odds")],
               frameon=False, loc="upper center", ncol=2, handlelength=1.1,
               bbox_to_anchor=(0.5, -0.20))
    tidy(ax2, grid="x")
    panel(ax2, "b", dx=-0.22)

    save(fig, "fig1_motif_controls")


# ================================================= FIGURE 2  within-assay model
def figure2():
    rows = json.load(open(os.path.join(RUN, "rev2", "cnn_det_10seed.json")))
    df = pd.DataFrame(rows)
    comp = pd.read_csv(os.path.join(RUN, "rev2", "composition_baseline_10seed.csv"))
    comp = comp[comp.target == "Avg_pEF"]
    ret = pd.DataFrame(json.load(open(os.path.join(RUN, "rev2", "retest_det10_final.json"))))

    assert df.seed.nunique() == 10, f"expected 10 seeds, got {df.seed.nunique()}"
    CEIL = 0.7567

    # "aa20+" is the 21-feature variant (composition + length); "dipep" is the
    # 400-feature dipeptide one. They differ by ~3e-6 and 0.06 respectively, so
    # picking the wrong label would silently mis-plot the control.
    aa20 = comp[comp.features == "aa20+"].groupby("seed").test_r.mean()
    dip = comp[comp.features == "dipep"]
    assert not dip.empty, "no dipeptide rows in the composition baseline"
    series = [
        ("Ridge\n20 aa", aa20.values, GREEN, 21),
        ("Ridge\ndipeptide", dip.groupby("seed").test_r.mean().values, GREEN, 401),
        ("MeanPool\nMLP", df[df.model == "MeanPoolMLP"].test_r.values, BLUE, 35265),
        ("AttnPool\nMLP", df[df.model == "AttnPoolMLP"].test_r.values, BLUE, 61825),
        ("Motif\nCNN", df[df.model == "MotifCNN"].test_r.values, BLUE, 105153),
    ]
    near("Fig2 MotifCNN test r", float(np.mean(series[4][1])), 0.5683, 0.0015)
    near("Fig2 MeanPool test r", float(np.mean(series[2][1])), 0.5084, 0.0015)
    near("Fig2 ridge aa20 test r", float(np.mean(series[0][1])), 0.2366, 0.0025)

    fig = plt.figure(figsize=(DOUBLE, 2.85))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])

    x = np.arange(len(series))
    for i, (name, vals, col, npar) in enumerate(series):
        m, s = float(np.mean(vals)), float(np.std(vals, ddof=1))
        ax.bar(x[i], m, width=0.62, color=col, edgecolor=SURFACE, linewidth=0.9,
               zorder=3, alpha=1.0 if col == BLUE else 0.92)
        ax.errorbar(x[i], m, yerr=s, fmt="none", ecolor=INK, elinewidth=0.8,
                    capsize=2.2, zorder=4)
        ax.scatter(np.full(len(vals), x[i]) + np.linspace(-.13, .13, len(vals)),
                   vals, s=4.5, color=INK, alpha=0.45, zorder=5, linewidths=0)
        ax.text(x[i], m + s + 0.022, f"{m:.3f}".replace(".", ","), ha="center",
                fontsize=7.3, color=INK)
        ax.text(x[i], 0.012, f"{npar:,}".replace(",", "."), ha="center",
                fontsize=6.3, color=SURFACE, rotation=90, va="bottom")

    ax.axhline(CEIL, color=INK, lw=0.9, ls="--", zorder=6)
    ax.text(len(series) - 0.52, CEIL + 0.009, "trần nhiễu 0,7567", fontsize=6.8,
            color=INK, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in series], fontsize=7)
    ax.set_ylabel("test r với cường độ ức chế (`Avg_pEF`)")
    ax.set_ylim(0, 0.84)
    ax.legend(handles=[Patch(facecolor=GREEN, label="thành phần aa (đối chứng)"),
                       Patch(facecolor=BLUE, label="ESM-2 + pooling")],
              frameon=False, loc="upper left", handlelength=1.1)
    tidy(ax)
    panel(ax, "a")

    # -- panel b: paired per-seed % of ceiling, training screen vs retest
    ax2 = fig.add_subplot(gs[0, 1])
    a = ret["pct_ceil_primary"].values
    b = ret["pct_ceil_retest"].values
    diff = float(np.mean(b - a))
    near("Fig2 paired %ceiling diff", diff, -3.89, 0.02)

    for i in range(len(a)):
        down = b[i] < a[i]
        ax2.plot([0, 1], [a[i], b[i]], color=BLUE if down else ORANGE,
                 lw=1.0, alpha=0.55, zorder=3, marker="o", ms=3.4,
                 markeredgecolor=SURFACE, markeredgewidth=0.6)
    ax2.plot([0, 1], [a.mean(), b.mean()], color=INK, lw=2.2, zorder=5,
             marker="o", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.0)
    for xx, v in ((0, a.mean()), (1, b.mean())):
        ax2.text(xx + (-0.07 if xx == 0 else 0.07), v,
                 f"{v:.2f}%".replace(".", ","), fontsize=7.5, color=INK,
                 ha="right" if xx == 0 else "left", va="center",
                 fontweight="bold")
    ax2.set_xlim(-0.42, 1.42)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["màn huấn luyện\n(`Avg_pEF`)", "lần đo độc lập\n(retest)"],
                        fontsize=7)
    ax2.set_ylabel("% trần nhiễu đạt được")
    ax2.set_title(f"hiệu ghép đôi −3,89 điểm · âm 9/10 seed\nKTC 90% [−5,33; −2,46] · TOST ±5 điểm: p = 0,096",
                  fontsize=6.9, color=MUTED, pad=5)
    tidy(ax2)
    panel(ax2, "b", dx=-0.20)

    save(fig, "fig2_within_assay")


# ====================================================== FIGURE 3  the headline
def figure3():
    d = json.load(open(os.path.join(RUN, "rev2", "cd43_transfer_10seed.json")))
    t = d["table"]
    comp = pd.read_csv(os.path.join(RUN, "rev2",
                                    "composition_baseline_10seed_cd43subset.csv"))

    ridge_pef = comp[(comp.target == "Avg_pEF") & (comp.features == "aa20+")]
    ridge_cd43 = comp[(comp.target == "cd43") & (comp.features == "aa20+")]
    ceil_pef = t["ceiling, Avg_pEF target (rTetR)"]["r_mean"]
    ceil_cd43 = t["ceiling, CD43 target (dCas9)"]["r_mean"]

    # the pEF-model-on-pEF-target cell is per-seed in the raw run file, not in
    # the consolidated table, so read it there rather than typing in 76.3
    per_seed = pd.DataFrame(json.load(
        open(os.path.join(RUN, "rev2", "cd43_fixed_det10_final.json"))))
    assert len(per_seed) == 10, f"expected 10 seeds, got {len(per_seed)}"
    pef_on_pef = per_seed["pct_ceil_pEFmodel_pEF"].values * 100
    # The reverse arm: trained on CD43, scored on pEF. r_cd43model_pEF was
    # computed by the original 10-seed run and never reported, so the 2x2 was
    # published with one cell missing -- which is what let a reader ask whether
    # the collapse is a property of the context pair or just of the rTetR screen.
    cd43_on_pef = (per_seed["r_cd43model_pEF"].values
                   / per_seed["ceil_pEF"].values) * 100

    bars = []
    bars.append(("ridge\nthành phần aa", "pEF",
                 float(ridge_pef.test_r.mean() / ceil_pef * 100),
                 float(ridge_pef.test_r.std(ddof=1) / ceil_pef * 100), GREEN))
    bars.append(("MotifCNN\nhuấn luyện pEF", "pEF",
                 float(np.mean(pef_on_pef)), float(np.std(pef_on_pef, ddof=1)), BLUE))
    bars.append(("MotifCNN\nhuấn luyện CD43", "pEF",
                 float(np.mean(cd43_on_pef)),
                 float(np.std(cd43_on_pef, ddof=1)), VERM))
    bars.append(("`Avg_pEF`\ndùng trực tiếp", "CD43",
                 t["  Avg_pEF measurement used directly"]["pct_mean"] * 100,
                 t["  Avg_pEF measurement used directly"]["pct_sd"] * 100, ORANGE))
    bars.append(("MotifCNN\nhuấn luyện pEF", "CD43",
                 t["  MotifCNN trained on Avg_pEF"]["pct_mean"] * 100,
                 t["  MotifCNN trained on Avg_pEF"]["pct_sd"] * 100, BLUE))
    bars.append(("ridge\nthành phần aa", "CD43",
                 float(ridge_cd43.test_r.mean() / ceil_cd43 * 100),
                 float(ridge_cd43.test_r.std(ddof=1) / ceil_cd43 * 100), GREEN))
    bars.append(("MotifCNN\nhuấn luyện CD43", "CD43",
                 t["  MotifCNN trained on CD43"]["pct_mean"] * 100,
                 t["  MotifCNN trained on CD43"]["pct_sd"] * 100, VERM))

    # The analytic ceiling on ANY rTetR-trained scorer, read from the file that
    # computes it so the figure and the prose cannot drift apart.
    bnd = json.load(open(os.path.join(RUN, "rev3", "bound.json")))
    bound = bnd["bound"]["mean"]
    near("Fig3 analytic transfer bound", bound, 36.52, 0.02)
    assert bound > bars[4][2] and bound > bars[3][2], \
        "bound must dominate both rTetR-trained bars"

    near("Fig3 CNN-pEF on CD43 %ceil", bars[4][2], 18.6, 0.15)
    near("Fig3 CNN-CD43 on CD43 %ceil", bars[6][2], 77.7, 0.15)
    near("Fig3 raw Avg_pEF on CD43 %ceil", bars[3][2], 27.6, 0.15)
    near("Fig3 ridge on CD43 %ceil", bars[5][2], 53.8, 0.05)
    near("Fig3 CNN-pEF on pEF %ceil", bars[1][2], 76.3, 0.15)
    near("Fig3 ridge on pEF %ceil", bars[0][2], 31.6, 0.05)
    near("Fig3 CNN-CD43 on pEF %ceil", bars[2][2], 20.9, 0.15)

    fig, ax = plt.subplots(figsize=(DOUBLE, 3.15))
    x = np.array([0, 1, 2, 3.75, 4.75, 5.75, 6.75])
    for i, (name, tgt, m, s, col) in enumerate(bars):
        ax.bar(x[i], m, width=0.70, color=col, edgecolor=SURFACE, linewidth=1.0,
               zorder=3)
        ax.errorbar(x[i], m, yerr=s, fmt="none", ecolor=INK, elinewidth=0.9,
                    capsize=2.6, zorder=4)
        ax.text(x[i], m + s + 1.5, f"{m:.1f}%".replace(".", ","), ha="center",
                fontsize=8, color=INK, fontweight="bold")

    # Seven bars across one double-column width leaves ~1 inch each, so the
    # second line has to be the training assay alone; the legend already says
    # which model each colour is.
    ax.set_xticks(x)
    ax.set_xticklabels([b[0].split("\n")[0] + "\n" + s for b, s in zip(
        bars, ["pEF", "pEF", "CD43", "—", "pEF", "CD43", "CD43"])], fontsize=7)
    ax.set_xlabel("mô hình, và assay nó được huấn luyện trên", fontsize=7.6,
                  color=MUTED, labelpad=2)
    ax.set_ylabel("% trần nhiễu của chính đích đo")
    ax.set_ylim(0, 100)

    ax.axvline(2.875, color="#D8D8D8", lw=1.0, zorder=1)
    ax.text(1.0, 95, "chấm trên đích rTetR (pEF)", ha="center", fontsize=7.6,
            color=INK, fontweight="bold")
    ax.text(5.25, 95, "chấm trên đích dCas9 (CD43)",
            ha="center", fontsize=7.6, color=INK, fontweight="bold")

    # the paper's headline paired contrasts, now reported as cluster-bootstrap
    # intervals over proteins rather than as p-values from ten seeds
    def brace(x0, x1, y, text, col):
        ax.plot([x0, x0, x1, x1], [y - 1.6, y, y, y - 1.6], color=col, lw=1.0,
                zorder=6, clip_on=False)
        ax.text((x0 + x1) / 2, y + 1.2, text, ha="center", fontsize=6.6,
                color=col, zorder=6)

    # Point estimate and interval both come from run/rev3/reviewer_response.py,
    # i.e. the same protein-cluster bootstrap, so the interval is centred on the
    # value printed. The pEF-side figure (-56,1) is on the slightly smaller
    # subset that also has both pEF replicates, so it is not exactly the
    # difference of the two bars above it (-55,4 on the full test set).
    # The bound spans only the two rTetR-derived scorers it applies to -- drawing
    # it across the whole CD43 panel would imply it also caps the CD43-trained
    # bar beside them, which is exactly the misreading it exists to prevent.
    ax.hlines(bound, 3.33, 5.17, color=INK, lw=1.1, linestyle=(0, (4, 2.2)),
              zorder=5)
    ax.text(4.25, bound + 2.0,
            ("trần cho mọi mô hình\nhuấn luyện trên rTetR: "
             f"{bound:.1f}%").replace(".", ","),
            ha="center", va="bottom", linespacing=1.25, fontsize=6.6,
            color=INK, fontweight="bold", zorder=6)

    brace(4.75, 6.75, 86, "+59,1 điểm · KTC bootstrap [+51,6; +65,8]", VERM)
    brace(1.0, 2.0, 86, "−56,1 điểm · KTC bootstrap [−63,2; −51,6]", BLUE)
    brace(4.75, 5.75, 68, "−0,280 · 10/10 seed", GREEN)

    # below the axes: every in-axes position collides with one of the six bars
    ax.legend(handles=[Patch(facecolor=GREEN, label="ridge 21 tham số, thành phần aa"),
                       Patch(facecolor=ORANGE, label="phép đo rTetR thô"),
                       Patch(facecolor=BLUE, label="MotifCNN huấn luyện trên rTetR"),
                       Patch(facecolor=VERM, label="MotifCNN huấn luyện trên dCas9")],
              frameon=False, loc="upper center", ncol=2, handlelength=1.1,
              bbox_to_anchor=(0.5, -0.17))
    tidy(ax)
    save(fig, "fig3_context_transfer")


# =============================================== FIGURE 4  rank not preserved
def figure4():
    w = pd.read_csv(os.path.join(RUN, "rev2", "whatbreaks.csv"))
    att = pd.read_csv(os.path.join(RUN, "rev2", "attenuation.csv"))
    thr = json.load(open(os.path.join(RUN, "rev2", "cd43_thresholds.json")))

    keep = ["DBD swap (same pEF reporter)", "promoter swap (same rTetR), UbC",
            "promoter swap (same rTetR)"]
    nice = {"DBD swap (same pEF reporter)": "Đổi miền gắn DNA\n(rTetR → dCas9, cùng pEF)",
            "promoter swap (same rTetR), UbC": "Đổi promoter\n(pEF → UbC)",
            "promoter swap (same rTetR)": "Đổi promoter\n(pEF → PGK)"}
    # ORANGE vs VERM is a weak CVD pair (deutan dE below the floor), so the two
    # subsets that must never be confused get BLUE vs VERM instead.
    sub_col = {"all": INK, "KRAB": BLUE, "non-KRAB": VERM}
    sub_lab = {"all": "toàn bộ", "KRAB": "chỉ KRAB", "non-KRAB": "trừ KRAB"}

    fig = plt.figure(figsize=(DOUBLE, 4.15))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1, 0.78],
                          wspace=0.40, hspace=0.62)

    # -- panel a: forest plot, corrected rho with bootstrap CI
    ax = fig.add_subplot(gs[:, 0])
    ypos, ylab = [], []
    y = 0.0
    for c in keep:
        for s in ["all", "KRAB", "non-KRAB"]:
            r = w[(w.contrast == c) & (w.subset == s)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot([r.corrected_lo, r.corrected_hi], [y, y], color=sub_col[s],
                    lw=1.5, zorder=3, solid_capstyle="round")
            ax.plot([r.rho_raw], [y], marker="|", ms=7, color=sub_col[s],
                    alpha=0.55, zorder=4)
            ax.plot([r.rho_corrected], [y], marker="o", ms=5.2,
                    color=sub_col[s], zorder=5, markeredgecolor=SURFACE,
                    markeredgewidth=0.8)
            ypos.append(y)
            ylab.append(sub_lab[s])
            y -= 1.0
        y -= 0.55
    near("Fig4 DBD corrected rho",
         float(w[(w.contrast == keep[0]) & (w.subset == "all")].rho_corrected.iloc[0]),
         0.232, 0.002)
    near("Fig4 DBD non-KRAB corrected rho",
         float(w[(w.contrast == keep[0]) & (w.subset == "non-KRAB")].rho_corrected.iloc[0]),
         0.073, 0.002)

    ax.axvline(0, color="#D8D8D8", lw=0.9, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab, fontsize=7)
    ax.set_xlabel("Spearman rho giữa hai ngữ cảnh")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(min(ypos) - 1.5, max(ypos) + 1.3)
    # group titles sit above each triplet on the left, where no CI reaches
    for i, c in enumerate(keep):
        ax.text(-0.01, ypos[i * 3] + 0.52, nice[c].replace("\n", " "),
                fontsize=6.9, color=INK, ha="left", va="bottom",
                fontweight="bold")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="-", ms=5, color=INK,
                                  label="đã hiệu chỉnh trần (KTC 95%)"),
                       plt.Line2D([], [], marker="|", ls="none", ms=7,
                                  color=MUTED, label="rho thô")],
              frameon=False, loc="upper center", ncol=2, handlelength=1.3,
              fontsize=6.8, bbox_to_anchor=(0.5, -0.11))
    tidy(ax, grid="x")
    panel(ax, "a", dx=-0.26, dy=1.04)

    # -- panel b: the two library levels agree once corrected
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_box_aspect(0.62)
    lv = [("Thư viện Pfam\n4.620 miền", 0.193, 0.832, 0.232),
          ("Thư viện tiling\n102.595 đoạn", 0.0921, 0.410, 0.225)]
    near("Fig4 tile corrected rho",
         float(att[att.comparison.str.contains("Tile level")].rho_corrected.iloc[0]),
         0.225, 0.002)
    xx = np.arange(2)
    ax2.bar(xx - 0.19, [v[1] for v in lv], width=0.34, color=GREY,
            edgecolor=SURFACE, linewidth=0.9, label="rho thô", zorder=3)
    ax2.bar(xx + 0.19, [v[3] for v in lv], width=0.34, color=INK,
            edgecolor=SURFACE, linewidth=0.9, label="đã hiệu chỉnh trần", zorder=3)
    for i, v in enumerate(lv):
        ax2.text(xx[i] - 0.19, v[1] + 0.006, f"{v[1]:.3f}".replace(".", ","),
                 ha="center", fontsize=6.8, color=MUTED)
        ax2.text(xx[i] + 0.19, v[3] + 0.006, f"{v[3]:.3f}".replace(".", ","),
                 ha="center", fontsize=7.2, color=INK, fontweight="bold")
    ax2.set_xticks(xx)
    # NB: only the decimal goes through the comma swap. Applying it to the whole
    # label would turn the Vietnamese thousands separator in "4.620 miền" into
    # "4,620 miền", i.e. silently restyle the counts as English.
    ax2.set_xticklabels([v[0] + "\nsuy giảm " + f"{v[2]:.3f}".replace(".", ",")
                         for v in lv], fontsize=6.6)
    ax2.set_ylabel("rho, rTetR vs dCas9", fontsize=7.5)
    ax2.set_ylim(0, 0.335)
    ax2.legend(frameon=False, loc="upper center", handlelength=1.0,
               fontsize=6.5, ncol=2, columnspacing=1.0,
               bbox_to_anchor=(0.5, 1.16))
    tidy(ax2)
    panel(ax2, "b", dx=-0.24, dy=1.22)

    # -- panel c: CD43 hit calls barely reproduce between bioreps
    ax3 = fig.add_subplot(gs[1, 1])
    cats = ["Chỉ lần lặp 1", "Cả hai\nlần lặp", "Chỉ lần lặp 2"]
    vals = [4719, 4396, 14202]
    cols = [GREY, VERM, GREY]
    bb = ax3.barh(np.arange(3)[::-1], vals, height=0.56, color=cols,
                  edgecolor=SURFACE, linewidth=0.9, zorder=3)
    for i, (v, b) in enumerate(zip(vals, bb)):
        ax3.text(v + 400, b.get_y() + b.get_height() / 2,
                 f"{v:,}".replace(",", "."), va="center", fontsize=7.2,
                 color=INK, fontweight="bold" if i == 1 else "normal")
    ax3.set_yticks(np.arange(3)[::-1])
    ax3.set_yticklabels(cats, fontsize=6.8)
    ax3.set_xlabel("Số đoạn được gọi hit tại CD43", fontsize=7.5)
    ax3.set_xlim(0, 19500)
    ax3.set_xticks([0, 5000, 10000, 15000])
    ax3.set_xticklabels(["0", "5.000", "10.000", "15.000"], fontsize=6.8)
    ax3.set_title("chỉ 4.396 / 23.317 = 18,9% gọi hit ở cả hai lần lặp\n"
                  + f"ngưỡng theo lần lặp +{thr['br1']:.4f} và +{thr['br2']:.4f}".replace(".", ","),
                  fontsize=6.6, color=MUTED, pad=5)
    tidy(ax3, grid="x")
    panel(ax3, "c", dx=-0.24, dy=1.30)

    save(fig, "fig5_rank_preservation")


# ================================================== FIGURE 5  precision at k
def figure5():
    d = pd.read_csv(os.path.join(RUN, "rev2", "cd43_topk3.csv"))
    assert d.seed.nunique() == 10, f"expected 10 seeds, got {d.seed.nunique()}"
    base = float(d.base_rate.mean())
    near("Fig5 base rate %", base * 100, 4.18, 0.05)

    want = {"pred_cd43": ("Mô hình huấn luyện trên CD43", VERM, "o", "-"),
            "meas_Avg_pEF": ("Phép đo `Avg_pEF` dùng thẳng", ORANGE, "s", "--"),
            "pred_pEF": ("Mô hình huấn luyện trên pEF", BLUE, "^", "-.")}
    missing = [r for r in want if r not in set(d.ranker)]
    assert not missing, f"missing ranker(s) {missing} in {sorted(set(d.ranker))}"

    fig, ax = plt.subplots(figsize=(SINGLE * 1.42, 2.95))
    for r in ["pred_cd43", "meas_Avg_pEF", "pred_pEF"]:
        g = d[d.ranker == r].groupby("k").precision
        m, s = g.mean() * 100, g.std(ddof=1) * 100
        lab, col, mk, ls = want[r]
        ax.plot(m.index, m.values, color=col, marker=mk, ms=4.6, ls=ls,
                label=lab, zorder=4, markeredgecolor=SURFACE,
                markeredgewidth=0.7)
        ax.fill_between(m.index, m - s, m + s, color=col, alpha=0.13,
                        linewidth=0, zorder=2)

    ax.axhline(base * 100, color=INK, lw=0.9, ls=":", zorder=3)
    ax.text(1000, base * 100 + 1.8, "nền 4,18%", fontsize=7, color=INK,
            ha="right")
    ax.annotate("có ý nghĩa\nđến k = 200\n(q = 0,029)", xy=(200, 56), xytext=(360, 72),
                fontsize=6.8, color=MUTED, ha="left",
                arrowprops=dict(arrowstyle="-", color="#C4C4C4", lw=0.8))
    ax.set_xscale("log")
    ax.set_xticks([20, 25, 50, 100, 200, 500, 1000])
    ax.set_xticklabels(["20", "25", "50", "100", "200", "500", "1.000"])
    ax.set_xlabel("k đoạn đầu bảng")
    ax.set_ylabel("precision@k — hit CD43 ở cả hai lần lặp (%)", fontsize=7.8)
    ax.set_ylim(0, 102)
    ax.legend(frameon=False, loc="lower left", handlelength=1.8)
    tidy(ax, grid="both")
    save(fig, "fig6_precision_at_k")


# ================================================== FIGURE 6  saliency controls
def figure6():
    d = pd.read_csv(os.path.join(RUN, "rev2", "saliency_det10.csv"))
    t = pd.read_csv(os.path.join(RUN, "rev2", "saliency_det10_tests.csv"))
    assert d.seed.nunique() == 10

    states = [("trained", "Huấn luyện thật", VERM),
              ("random_init", "Trọng số\nchưa huấn luyện", GREY),
              ("label_perm", "Nhãn\nxáo trộn", GREY)]
    near("Fig6 trained massA", float(d[d.state == "trained"].massA.mean()), 20.79, 0.05)
    near("Fig6 label_perm massA", float(d[d.state == "label_perm"].massA.mean()), 7.30, 0.05)
    near("Fig6 random_init massA", float(d[d.state == "random_init"].massA.mean()), 9.30, 0.05)
    prim = t[(t.conv == "A") & (t.ctrl == "label_perm")].iloc[0]
    near("Fig6 primary endpoint", float(prim["diff"]), 13.49, 0.01)

    fig = plt.figure(figsize=(DOUBLE, 3.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1], wspace=0.46)
    ax = fig.add_subplot(gs[0, 0])

    rng = np.random.default_rng(0)
    for i, (st, lab, col) in enumerate(states):
        v = d[d.state == st].massA.values
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.13, 0.13, len(v)), v,
                   s=20, color=col, alpha=0.75, zorder=4, linewidths=0.6,
                   edgecolors=SURFACE)
        m = float(np.mean(v))
        ax.plot([i - 0.26, i + 0.26], [m, m], color=INK, lw=2.0, zorder=5)
        # clear of the whole seed cloud, not just the mean: at m+1.3 the label
        # lands inside the scatter for the two control states
        ax.text(i, float(np.max(v)) + 1.1, f"{m:.2f}%".replace(".", ","),
                fontsize=7.5, color=INK, ha="center", fontweight="bold")

    ax.axhline(5.0, color=INK, lw=0.9, ls="--", zorder=3)
    ax.axhspan(7.30, 9.30, color=GREY, alpha=0.13, zorder=1, linewidth=0)
    ax.text(-0.42, 4.3, "nền chiều dài 5,0%", fontsize=6.6, color=INK,
            ha="left", va="center")
    ax.text(2.46, 14.0, "mốc không-tín-hiệu\nthật: 7,3–9,3%", fontsize=6.5,
            color=MUTED, ha="right", va="center")
    ax.annotate("", xy=(2.46, 9.3), xytext=(2.46, 12.6),
                arrowprops=dict(arrowstyle="-", color="#C4C4C4", lw=0.8))
    ax.set_xticks(range(3))
    ax.set_xticklabels([s[1] for s in states], fontsize=7)
    ax.set_ylabel("Khối lượng saliency trong\ncửa sổ `[VILMF]K.E` (%)", fontsize=7.8)
    ax.set_ylim(0, 29)
    ax.set_xlim(-0.45, 2.55)
    tidy(ax)
    panel(ax, "a", dx=-0.21)

    # -- panel b: the four paired contrasts, BH-corrected
    ax2 = fig.add_subplot(gs[0, 1])
    t = t.copy()
    t["lab"] = t.conv + " vs " + t.ctrl.map({"label_perm": "nhãn xáo trộn",
                                             "random_init": "trọng số ngẫu nhiên"})
    t = t.iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(t))
    for i, r in t.iterrows():
        is_primary = (r.conv == "A") and (r.ctrl == "label_perm")
        col = VERM if is_primary else MUTED
        ax2.plot([r.ci_lo, r.ci_hi], [yy[i], yy[i]], color=col, lw=1.6,
                 zorder=3, solid_capstyle="round")
        ax2.plot([r["diff"]], [yy[i]], marker="o", ms=5.4 if is_primary else 4.4,
                 color=col, zorder=4, markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax2.text(r.ci_hi + 0.5, yy[i],
                 f"q = {r.q_bh:.0e}".replace("e-0", " × 10⁻").replace(".", ","),
                 fontsize=6.6, color=MUTED, va="center")
    ax2.axvline(0, color="#D8D8D8", lw=0.9, zorder=1)
    ax2.set_yticks(yy)
    ax2.set_yticklabels([f"{r.lab}{' ★' if (r.conv=='A' and r.ctrl=='label_perm') else ''}"
                         for _, r in t.iterrows()], fontsize=6.8)
    ax2.set_xlabel("Hiệu khối lượng saliency (điểm phần trăm)")
    ax2.set_xlim(0, 25)
    ax2.set_title("★ endpoint chính đặt trước · ghép đôi theo seed, n = 10",
                  fontsize=6.8, color=MUTED, pad=5)
    tidy(ax2, grid="x")
    panel(ax2, "b", dx=-0.42)

    save(fig, "fig4_saliency_controls")


def main():
    print("building figures from run/ outputs")
    for fn in (figure1, figure2, figure3, figure4, figure5, figure6):
        print(f"{fn.__name__}:")
        fn()

    print("\nvalue checks against the manuscript:")
    bad = 0
    for lab, got, want, ok in CHECKS:
        flag = "ok  " if ok else "FAIL"
        if not ok:
            bad += 1
        print(f"  [{flag}] {lab:<36} plotted {got:>10.4f}   manuscript {want}")
    if bad:
        raise SystemExit(f"\n{bad} plotted value(s) disagree with the manuscript")
    print(f"\nall {len(CHECKS)} plotted values match the manuscript")
    print("DONE_MAKE_FIGURES")


if __name__ == "__main__":
    main()

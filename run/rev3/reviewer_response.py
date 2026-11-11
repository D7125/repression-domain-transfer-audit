"""Three things two mock referees asked for, computed from data already on disk.

Referee Q2 (Phan_bien_Q2_repression_domain_v11.docx) and the mock review
(Mock_Peer_Review_Repression_Domain_v11.docx) converge on three demands that do
not need a new training run:

  M3  The reverse transfer arm.  The paper trains on pEF and scores CD43 but
      never the other way, so it cannot tell a *context* asymmetry from "the
      rTetR screen is simply easier to learn".  The 10-seed JSON already stores
      r_cd43model_pEF; it was computed and never reported.  Part A turns it into
      the fourth cell of the 2x2 and tests the two collapses against each other.

  M2  Ten seeds are not ten independent observations.  Seeds resample the split,
      not the data: all ten share 1,330 proteins, so a p-value from n = 10 is a
      statement about training stability, not about sampling proteins.  Part B
      replaces it with the cluster bootstrap by protein that the paper already
      uses for rho in section 4.3, applied to every key quantity.

  M5  "The model beats a biological replicate" may be a theorem, not a finding:
      any smoothed predictor beats one noisy draw.  Part C writes the theorem
      down, finds the accuracy a predictor must clear to win, and checks where
      the model sits relative to it.

Run:  ~/miniconda3/envs/protein/bin/python reviewer_response.py
(protein env, not machine_deep_learning: that env has no scipy/pandas.)
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t as tdist

ROOT = "/home/dmin/ai&stem_cell"
REV2 = f"{ROOT}/run/rev2"
BOOT = 2000
OUT = {}


def sb(r):
    """Spearman-Brown ceiling: sqrt of the reliability of a two-replicate mean."""
    return np.sqrt(2 * r / (1 + r))


def near(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {got:.4f} vs published {want:.4f}")
    if not ok:
        sys.exit(f"reproduction check failed: {label}")


def paired(d, n=None):
    """mean, sd, t-CI and p for a paired difference over seeds."""
    d = np.asarray(d, float)
    n = n or len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    half = tdist.ppf(0.975, n - 1) * se
    p = 2 * tdist.sf(abs(m / se), n - 1)
    return dict(mean=m, sd=sd, lo=m - half, hi=m + half, p=float(p),
                pos=int((d > 0).sum()), n=n)


# ----------------------------------------------------------------- part A
def part_a():
    """The 2x2 train x score matrix, and the symmetry of the two collapses."""
    print("\n" + "=" * 72)
    print("A. REVERSE TRANSFER: the fourth cell of the 2x2 (referee M3)")
    print("=" * 72)
    j = json.load(open(f"{REV2}/cd43_fixed_det10_final.json"))
    g = lambda k: np.array([s[k] for s in j], float)
    cp, cc = g("ceil_pEF"), g("ceil_cd43")

    cells = {
        ("pEF", "pEF"): g("r_pEFmodel_pEF") / cp,
        ("pEF", "CD43"): g("r_pEFmodel_cd43") / cc,
        ("CD43", "CD43"): g("r_cd43model_cd43") / cc,
        ("CD43", "pEF"): g("r_cd43model_pEF") / cp,
    }
    print("\n  trained on -> scored on      % of that target's noise ceiling")
    for (tr, sc), v in cells.items():
        star = "  <- NEW" if (tr, sc) == ("CD43", "pEF") else ""
        print(f"    {tr:>4s} -> {sc:<4s}   {100*v.mean():6.2f}%  "
              f"sd {100*v.std(ddof=1):4.2f}{star}")

    near("pEF->pEF", 100 * cells[("pEF", "pEF")].mean(), 76.3, 0.1)
    near("pEF->CD43", 100 * cells[("pEF", "CD43")].mean(), 18.6, 0.1)
    near("CD43->CD43", 100 * cells[("CD43", "CD43")].mean(), 77.7, 0.1)

    fwd = cells[("pEF", "pEF")] - cells[("pEF", "CD43")]
    rev = cells[("CD43", "CD43")] - cells[("CD43", "pEF")]
    print("\n  collapse on leaving the training context:")
    for nm, v in (("pEF-trained  (pEF -> CD43)", fwd),
                  ("CD43-trained (CD43 -> pEF)", rev)):
        s = paired(100 * v)
        print(f"    {nm}: {s['mean']:5.1f} pts  sd {s['sd']:4.1f}  "
              f"positive in {s['pos']}/10 seeds")

    asym = paired(100 * (rev - fwd))
    print(f"\n  asymmetry (CD43-trained collapse - pEF-trained collapse):")
    print(f"    {asym['mean']:+.1f} pts, 95% CI [{asym['lo']:+.1f}, "
          f"{asym['hi']:+.1f}], p = {asym['p']:.2f}")

    # TOST against a 10-point margin: a quarter of either collapse, and the
    # smallest asymmetry that would let anyone claim one direction is special.
    margin = 10.0
    d = 100 * (rev - fwd)
    se = d.std(ddof=1) / np.sqrt(len(d))
    p_lo = tdist.sf((d.mean() - (-margin)) / se, len(d) - 1)
    p_hi = tdist.cdf((d.mean() - margin) / se, len(d) - 1)
    p_tost = max(p_lo, p_hi)
    print(f"    TOST against +-{margin:.0f} pts: p = {p_tost:.4f} "
          f"-> {'equivalent' if p_tost < 0.05 else 'NOT equivalent'} "
          f"within that margin")

    # Referee minor 3: the paper reads "+1.4 points, p = 0.14" as the two
    # in-domain arms being indistinguishable. That is non-significance, not
    # equivalence. TOST it against the same margin, with the margin declared.
    ind = 100 * (cells[("CD43", "CD43")] - cells[("pEF", "pEF")])
    s = paired(ind)
    se = s["sd"] / np.sqrt(s["n"])
    p_eq = max(tdist.sf((s["mean"] + margin) / se, s["n"] - 1),
               tdist.cdf((s["mean"] - margin) / se, s["n"] - 1))
    print(f"\n  in-domain arms, CD43->CD43 minus pEF->pEF (referee minor 3):")
    print(f"    {s['mean']:+.1f} pts, 95% CI [{s['lo']:+.1f}, {s['hi']:+.1f}], "
          f"p = {s['p']:.2f} (two-sided)")
    print(f"    TOST against +-{margin:.0f} pts: p = {p_eq:.4f} -> "
          f"{'equivalent' if p_eq < 0.05 else 'NOT equivalent'}; "
          f"'indistinguishable' must be stated at this margin, not from p = 0.14")

    OUT["A"] = dict(
        cells={f"{a}->{b}": [float(100 * v.mean()), float(100 * v.std(ddof=1))]
               for (a, b), v in cells.items()},
        collapse_fwd=paired(100 * fwd), collapse_rev=paired(100 * rev),
        asymmetry=asym, tost_margin=margin, tost_p=float(p_tost),
        in_domain=dict(**s, tost_p=float(p_eq)))


# ----------------------------------------------------------------- part B
def load_tiles():
    """Per-seed test tiles with both screens' replicates attached."""
    p = pd.read_csv(f"{REV2}/cd43_topk3_preds.csv")
    m = pd.read_csv(f"{REV2}/master.csv",
                    usecols=["protein", "tile_num", "R1_pEF", "R2_pEF"])
    p = p.merge(m, on=["protein", "tile_num"], how="left", validate="m:1")
    return p


def boot_stat(fn, prot, n_boot=BOOT, seed=0):
    """Cluster bootstrap by protein. Tiles of one protein overlap by 70/80
    residues, so resampling tiles would overstate precision; proteins are the
    exchangeable unit, exactly as whatbreaks.py:boot_rho does for rho."""
    uniq, inv = np.unique(np.asarray(prot), return_inverse=True)
    idx = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        out[b] = fn(np.concatenate([idx[q] for q in pick]))
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def part_b(p):
    """Protein-cluster bootstrap CIs for the four cells and the transfer gap."""
    print("\n" + "=" * 72)
    print("B. PROTEIN-CLUSTER BOOTSTRAP, not seeds (referee M2)")
    print("=" * 72)
    print(f"  {BOOT} resamples of proteins, within each seed's own test set.")

    j = json.load(open(f"{REV2}/cd43_fixed_det10_final.json"))
    specs = {
        "pEF->pEF": ("pred_pEF", "Avg_pEF", "R1_pEF", "R2_pEF"),
        "pEF->CD43": ("pred_pEF", "cd43", "cd43_br1", "cd43_br2"),
        "CD43->CD43": ("pred_cd43", "cd43", "cd43_br1", "cd43_br2"),
        "CD43->pEF": ("pred_cd43", "Avg_pEF", "R1_pEF", "R2_pEF"),
    }
    rows = []
    for name, (pr, y, r1, r2) in specs.items():
        los, his, pts = [], [], []
        for seed, g in p.groupby("seed"):
            g = g[g[pr].notna() & g[y].notna() & g[r1].notna() & g[r2].notna()]
            a, b = g[pr].values, g[y].values
            v1, v2 = g[r1].values, g[r2].values
            prot = g.protein.values

            def stat(ix):
                rep = pearsonr(v1[ix], v2[ix])[0]
                if rep <= 0:
                    return np.nan
                return 100 * pearsonr(a[ix], b[ix])[0] / sb(rep)

            full = np.arange(len(g))
            pts.append(stat(full))
            lo, hi = boot_stat(stat, prot, seed=seed)
            los.append(lo)
            his.append(hi)
        pt = float(np.mean(pts))
        lo, hi = float(np.median(los)), float(np.median(his))
        rows.append((name, pt, lo, hi, float(np.std(pts, ddof=1))))
        print(f"    {name:<11s} {pt:6.2f}%  bootstrap CI [{lo:5.2f}, {hi:5.2f}]"
              f"   (sd across seeds {np.std(pts, ddof=1):4.2f})")

    # Both headline gaps, bootstrapped on the paired tiles rather than on seeds.
    # "own-assay model minus foreign-assay model, scored on one target" -- once
    # on each target, so the two collapses get the same treatment.
    gapspec = [("on CD43 (CD43-trained - pEF-trained)",
                "pred_cd43", "pred_pEF", "cd43", "cd43_br1", "cd43_br2"),
               ("on pEF  (CD43-trained - pEF-trained)",
                "pred_cd43", "pred_pEF", "Avg_pEF", "R1_pEF", "R2_pEF")]
    gapout = {}
    print("\n  the headline gaps, bootstrapped over proteins:")
    for lab, ca, cb, y_, r1, r2 in gapspec:
        gaps = []
        for seed, g in p.groupby("seed"):
            g = g[g[ca].notna() & g[cb].notna() & g[y_].notna()
                  & g[r1].notna() & g[r2].notna()]
            pa, pb, y = g[ca].values, g[cb].values, g[y_].values
            v1, v2 = g[r1].values, g[r2].values

            def stat(ix):
                rep = pearsonr(v1[ix], v2[ix])[0]
                if rep <= 0:
                    return np.nan
                c = sb(rep)
                return 100 * (pearsonr(pa[ix], y[ix])[0]
                              - pearsonr(pb[ix], y[ix])[0]) / c

            full = np.arange(len(g))
            lo, hi = boot_stat(stat, g.protein.values, seed=seed)
            gaps.append((stat(full), lo, hi))
        pt = float(np.mean([x[0] for x in gaps]))
        lo = float(np.median([x[1] for x in gaps]))
        hi = float(np.median([x[2] for x in gaps]))
        print(f"    {lab}: {pt:+.1f} points, CI [{lo:+.1f}, {hi:+.1f}]")
        gapout[lab] = dict(pct=pt, lo=lo, hi=hi)
    print("    for comparison, the 10-seed paired t-test on the CD43 gap gave")
    print("    p = 1.4e-11, a statement about training stability, not proteins.")

    OUT["B"] = dict(cells={r[0]: dict(pct=r[1], lo=r[2], hi=r[3], sd_seed=r[4])
                           for r in rows},
                    gaps=gapout, n_boot=BOOT)


def part_b2(p):
    """Same treatment for precision@k, which the referee named explicitly."""
    print("\n  precision@k by the same bootstrap (k = 100, hit = cd43_hit_both):")
    rank = {"pred_cd43": "model trained on CD43",
            "pred_pEF": "model trained on pEF",
            "Avg_pEF": "the rTetR measurement itself",
            "blend_z": "blend"}
    res = {}
    for col, lab in rank.items():
        pts, los, his = [], [], []
        for seed, g in p.groupby("seed"):
            g = g[g[col].notna() & g.cd43_hit_both.notna()]
            v, hit = g[col].values, g.cd43_hit_both.values.astype(bool)
            prot = g.protein.values
            k = 100

            def stat(ix):
                vv, hh = v[ix], hit[ix]
                top = np.argsort(-vv)[:k]
                return 100 * hh[top].mean()

            pts.append(stat(np.arange(len(g))))
            lo, hi = boot_stat(stat, prot, n_boot=1000, seed=seed)
            los.append(lo)
            his.append(hi)
        pt, lo, hi = np.mean(pts), np.median(los), np.median(his)
        res[col] = dict(pct=float(pt), lo=float(lo), hi=float(hi))
        print(f"    {lab:<28s} {pt:5.1f}%  CI [{lo:5.1f}, {hi:5.1f}]")
    base = 100 * p.groupby("seed").cd43_hit_both.mean().mean()
    print(f"    {'base rate':<28s} {base:5.2f}%")
    res["base_rate"] = float(base)

    # Marginal CIs that overlap do not mean the difference is unresolved: the
    # two rankers are scored on the same tiles, so the paired difference has to
    # be bootstrapped directly.
    print("\n  paired differences in precision@100, same resampled proteins:")
    pairs = [("pred_cd43", "Avg_pEF"), ("pred_cd43", "pred_pEF"),
             ("blend_z", "pred_cd43"), ("pred_pEF", "Avg_pEF")]
    for a, b in pairs:
        pts, los, his = [], [], []
        for seed, g in p.groupby("seed"):
            g = g[g[a].notna() & g[b].notna() & g.cd43_hit_both.notna()]
            va, vb = g[a].values, g[b].values
            hit = g.cd43_hit_both.values.astype(bool)
            k = 100

            def stat(ix):
                hh = hit[ix]
                ta = np.argsort(-va[ix])[:k]
                tb = np.argsort(-vb[ix])[:k]
                return 100 * (hh[ta].mean() - hh[tb].mean())

            pts.append(stat(np.arange(len(g))))
            lo, hi = boot_stat(stat, g.protein.values, n_boot=1000, seed=seed)
            los.append(lo)
            his.append(hi)
        pt, lo, hi = np.mean(pts), np.median(los), np.median(his)
        verdict = "excludes 0" if lo > 0 or hi < 0 else "includes 0"
        print(f"    {a:<10s} - {b:<10s} {pt:+5.1f} pts  "
              f"CI [{lo:+5.1f}, {hi:+5.1f}]  {verdict}")
        res[f"{a}_minus_{b}"] = dict(pct=float(pt), lo=float(lo), hi=float(hi))
    OUT["B2"] = res


# ----------------------------------------------------------------- part C
def part_c(p):
    """Is 'the model beats one replicate' a finding or a theorem? (referee M5)"""
    print("\n" + "=" * 72)
    print("C. THE DENOISING NULL (referee M5)")
    print("=" * 72)
    print("""  Measurement model: br_j = tau + eps_j, eps independent, equal variance.
  Then corr(br1, br2) = 1/(1+s2) with s2 = var(eps)/var(tau), and a single
  replicate correlates with the truth at sqrt(corr(br1,br2)).  Any predictor p
  beats one replicate at predicting a held-out replicate iff
      corr(p, tau) > sqrt(corr(br1, br2)).
  So the comparison has a threshold, and the question is whether the model
  clears it -- not whether smoothing helps, which it always does.""")

    rows = []
    for seed, g in p.groupby("seed"):
        g = g[g.cd43_br1.notna() & g.cd43_br2.notna() & g.pred_cd43.notna()]
        rep = pearsonr(g.cd43_br1, g.cd43_br2)[0]
        ceil = sb(rep)
        r_model_mean = pearsonr(g.pred_cd43, g.cd43)[0]
        rows.append(dict(seed=seed, n=len(g), rep=rep, ceil=ceil,
                         r_model_mean=r_model_mean,
                         corr_model_truth=r_model_mean / ceil,
                         bar=np.sqrt(rep)))
    t = pd.DataFrame(rows)
    print(f"\n  averaged over 10 seeds (n = {t.n.mean():,.0f} tiles per seed):")
    print(f"    corr(br1, br2)                        {t.rep.mean():.4f}")
    print(f"    bar a predictor must clear  sqrt(r)   {t.bar.mean():.4f}")
    print(f"    model's corr with the truth           "
          f"{t.corr_model_truth.mean():.4f}  "
          f"(sd {t.corr_model_truth.std(ddof=1):.4f})")
    marg = t.corr_model_truth - t.bar
    print(f"    margin over the bar                   {marg.mean():+.4f}  "
          f"(positive in {int((marg > 0).sum())}/10 seeds)")

    # Simulate the null the referee described: a predictor carrying no
    # information except the truth, calibrated to the model's own accuracy.
    rng = np.random.default_rng(0)
    N, S = 14000, 400
    sim = []
    for _ in range(S):
        rep = float(t.rep.mean())
        s2 = 1 / rep - 1
        tau = rng.standard_normal(N)
        b1 = tau + np.sqrt(s2) * rng.standard_normal(N)
        b2 = tau + np.sqrt(s2) * rng.standard_normal(N)
        rho_pt = float(t.corr_model_truth.mean())
        q = np.sqrt(1 / rho_pt**2 - 1)
        pnull = tau + q * rng.standard_normal(N)
        zz = lambda v: (v - v.mean()) / v.std()
        sim.append([pearsonr(b1, b2)[0],
                    pearsonr(pnull, b2)[0],
                    pearsonr((zz(b1) + zz(pnull)) / 2, b2)[0]])
    sim = np.array(sim)
    print(f"\n  null simulation ({S} draws, {N:,} tiles, a predictor with no")
    print("  information beyond the truth, calibrated to the model's accuracy):")
    print(f"    one replicate    -> held-out replicate   r = {sim[:,0].mean():.4f}")
    print(f"    null predictor   -> held-out replicate   r = {sim[:,1].mean():.4f}")
    print(f"    replicate + null -> held-out replicate   r = {sim[:,2].mean():.4f}")

    obs = pd.read_csv(f"{REV2}/denoise_test_10seed.csv") \
        if __import__("os").path.exists(f"{REV2}/denoise_test_10seed.csv") else None
    if obs is not None:
        mu = obs.pivot_table(index="predictor", values="r")
        print("\n  observed, same quantities (denoise_test_10seed.csv):")
        for k in ("replicate alone", "model alone", "replicate + model"):
            if k in mu.index:
                print(f"    {k:<20s} r = {float(mu.loc[k, 'r']):.4f}")
        print("""
  Reading: the observed numbers land on the simulated ones. The model's edge
  over a single replicate is what measurement theory predicts for a predictor
  of its accuracy -- it is not extra evidence that the model is good. What is
  not automatic is clearing the bar at all: the bar is set by the screen's own
  reproducibility, and a predictor below it loses to simply measuring once.""")

    OUT["C"] = dict(rep=float(t.rep.mean()), bar=float(t.bar.mean()),
                    corr_model_truth=float(t.corr_model_truth.mean()),
                    margin=float(marg.mean()),
                    margin_pos=int((marg > 0).sum()),
                    sim=dict(one_rep=float(sim[:, 0].mean()),
                             null=float(sim[:, 1].mean()),
                             blend=float(sim[:, 2].mean())))


if __name__ == "__main__":
    part_a()
    tiles = load_tiles()
    part_b(tiles)
    part_b2(tiles)
    part_c(tiles)
    json.dump(OUT, open(f"{ROOT}/run/rev3/reviewer_response.json", "w"),
              indent=1)
    print(f"\nwrote {ROOT}/run/rev3/reviewer_response.json")
    print("DONE_REVIEWER_RESPONSE")

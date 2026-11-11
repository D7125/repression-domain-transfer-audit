"""What actually breaks rank order: the DBD, the sgRNA, the target, or the promoter?

Section 4.3 measures one contrast -- rTetR vs dCas9 on the same pEF reporter,
rho = 0.193 -- and reads it as "changing the DNA-binding domain does not
preserve rank". That reading is only licensed if the DBD is the operative
variable. Table S1 happens to contain the controls that test it, because the
same 5,092 Pfam domains were run through several K562 repression screens that
differ in exactly one factor at a time:

  DBD swap       K562 rTetR pEF D5   vs  K562 dCas9 pEF          (same reporter)
  sgRNA swap     K562 dCas9 CD43 sg10 vs K562 dCas9 CD43 sg15    (same DBD, same gene)
  target swap    K562 dCas9 pEF      vs  K562 dCas9 CD43 sg15    (same DBD)
  promoter swap  K562 rTetR pEF D5   vs  K562 rTetR PGK          (same DBD)

Each screen's own two-biological-replicate reliability gives its ceiling, so
every cross-screen rho is also reported attenuation-corrected, exactly as in
section 4.3. If the sgRNA swap is as bad as the DBD swap, the paper's causal
attribution to the DBD is wrong and has to be rewritten as a statement about
rank reliability in these screens generally.

Everything is Spearman, because section 2.2 showed Pearson on these screens is
carried by a minority of large-amplitude elements.

Usage:
  python whatbreaks.py
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = "/home/dmin/ai&stem_cell"
sb = lambda r: 2 * r / (1 + r)

SCREENS = ["K562 rTetR pEF D5", "K562 rTetR PGK", "K562 rTetR UbC",
           "K562 dCas9 pEF", "K562 dCas9 CD43 sg10", "K562 dCas9 CD43 sg15"]

CONTRASTS = [
    ("DBD swap (same pEF reporter)", "K562 rTetR pEF D5", "K562 dCas9 pEF"),
    ("target swap (same dCas9)", "K562 dCas9 pEF", "K562 dCas9 CD43 sg15"),
    ("promoter swap (same rTetR)", "K562 rTetR pEF D5", "K562 rTetR PGK"),
    ("promoter swap (same rTetR), UbC", "K562 rTetR pEF D5", "K562 rTetR UbC"),
]

# Deliberately NOT in CONTRASTS. The two CD43 sgRNA screens look like the one
# clean one-factor contrast in the table (same DBD, same gene, same library),
# and on the non-KRAB bulk they give rho = -0.249, which corrects to about
# -0.55 -- a spectacular result. It does not survive its own consistency check:
# the four samples are two library preps x two guides, and the two "both
# factors differ" diagonals give +0.212 and -0.335 where a factor structure
# requires them to be comparable. check_coherence() prints the 2x2 and the
# contrast is reported as uninterpretable, not as a measurement.
SGRNA = ("K562 dCas9 CD43 sg10", "K562 dCas9 CD43 sg15")
CD43_SAMPLES = ["216Nuc_CD43_sg10", "216Nuc_CD43_sg15",
                "216Pfam_CD43sg10_BR2", "216Pfam_CD43sg15_BR2"]
BOOT = 2000


def rho_on(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 30:
        return np.nan, int(m.sum())
    return float(spearmanr(x[m], y[m])[0]), int(m.sum())


def boot_rho(x, y, prot, n_boot=BOOT, seed=0):
    """Cluster bootstrap by protein: domains are 2+ per protein, so a naive CI
    on 4,800 domains overstates the precision by roughly sqrt(domains/protein)."""
    m = x.notna() & y.notna()
    x, y, prot = x[m].values, y[m].values, np.asarray(prot)[m.values]
    if len(x) < 30:
        return np.nan, np.nan
    uniq, inv = np.unique(prot, return_inverse=True)
    idx_by_prot = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by_prot[p] for p in pick])
        out[b] = spearmanr(x[sel], y[sel])[0]
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def check_coherence(w, krab):
    """Does the CD43 2x2 (prep x guide) behave like a factor design at all?"""
    have = [c for c in CD43_SAMPLES if c in w.columns]
    if len(have) < 4:
        print("  cannot run: missing samples")
        return None
    A, B, C, D = CD43_SAMPLES          # Nuc_sg10, Nuc_sg15, Pfam_sg10, Pfam_sg15
    nk = ~krab
    pairs = [("same prep Nuc,  guide differs", A, B),
             ("same prep Pfam, guide differs", C, D),
             ("same guide sg10, prep differs", A, C),
             ("same guide sg15, prep differs", B, D),
             ("diagonal 1 (both differ)", A, D),
             ("diagonal 2 (both differ)", B, C)]
    res = {}
    for lab, a, b in pairs:
        r_nk, n_nk = rho_on(w[a][nk], w[b][nk])
        r_all, _ = rho_on(w[a], w[b])
        res[lab] = (r_nk, r_all, n_nk)
        print(f"  {lab:32s} non-KRAB {r_nk:+.4f}  all {r_all:+.4f}  n {n_nk:,}")
    d1 = res["diagonal 1 (both differ)"][0]
    d2 = res["diagonal 2 (both differ)"][0]
    print(f"\n  diagonals differ by {abs(d1 - d2):.3f} in rho "
          f"({d1:+.3f} vs {d2:+.3f}).")
    print("  A factor design requires these to be comparable. They are not,")
    print("  so the guide contrast is NOT interpretable and is excluded.")
    return res


def main():
    dom = pd.read_csv(f"{ROOT}/run/tycko/s1_domains.csv")
    dat = pd.read_csv(f"{ROOT}/run/tycko/s1_data.csv")
    smp = pd.read_csv(f"{ROOT}/run/tycko/s1_samples.csv")
    elem = pd.read_csv(f"{ROOT}/run/tycko/s1_elem.csv")

    # drop the library's own random controls from every correlation
    ctrl = elem.label.astype(str).str.contains("random|Random", regex=True)
    is_ctrl = dict(zip(elem.label, ctrl))
    dom = dom[~dom.label.map(is_ctrl).fillna(False)].copy()
    krab = dom.Domain.astype(str).str.contains("KRAB", case=False).values
    print(f"{len(dom):,} non-control domains | {int(krab.sum())} KRAB\n")

    # per-screen biological-replicate reliability, from the per-sample table
    smap = smp.set_index("unique sample")
    wide = dat.pivot_table(index="label", columns="unique sample", values="rho")
    rel = {}
    print("=== each screen's own two-biorep reliability (Spearman) ===")
    for s in SCREENS:
        cols = [c for c in wide.columns
                if c in smap.index and smap.loc[c, "screen"] == s]
        br = {}
        for c in cols:
            br.setdefault(int(smap.loc[c, "biorep"]), []).append(c)
        if 1 not in br or 2 not in br:
            print(f"  {s:26s} only bioreps {sorted(br)} -- no reliability")
            continue
        x = wide[br[1]].mean(axis=1)
        y = wide[br[2]].mean(axis=1)
        x, y = x[x.index.isin(dom.label)], y[y.index.isin(dom.label)]
        j = x.notna() & y.notna()
        r = float(spearmanr(x[j], y[j])[0])
        rel[s] = sb(r)
        print(f"  {s:26s} rho(BR1,BR2) {r:+.4f}  n {int(j.sum()):,}  "
              f"-> rel {sb(r):.4f}  ceiling {np.sqrt(sb(r)):.4f}")

    print("\n=== EXCLUDED CONTRAST: the CD43 sgRNA swap, and why ===")
    check_coherence(wide.reindex(dom.label), krab)
    a_, b_ = SGRNA
    for subset, mask in (("all", np.ones(len(dom), bool)),
                         ("KRAB", krab), ("non-KRAB", ~krab)):
        d = dom[mask]
        r, n = rho_on(d[a_], d[b_])
        print(f"    [not used] sgRNA swap {subset:9s} n={n:5d} rho {r:+.4f}")

    rows = []
    print("\n=== one factor at a time (bootstrap CI clustered by protein) ===")
    for name, a, b in CONTRASTS:
        for subset, mask in (("all", np.ones(len(dom), bool)),
                             ("KRAB", krab), ("non-KRAB", ~krab)):
            d = dom[mask]
            r, n = rho_on(d[a], d[b])
            lo, hi = boot_rho(d[a], d[b], d["Gene entry name"].values)
            att = (np.sqrt(rel[a] * rel[b])
                   if a in rel and b in rel else np.nan)
            rows.append(dict(contrast=name, subset=subset, n=n, rho_raw=r,
                             ci_lo=lo, ci_hi=hi, attenuation=att,
                             rho_corrected=r / att if att == att else np.nan,
                             corrected_lo=lo / att if att == att else np.nan,
                             corrected_hi=hi / att if att == att else np.nan))
        for r_ in rows[-3:]:
            s = (f"  {r_['contrast']:34s} {r_['subset']:9s} n={r_['n']:5d} "
                 f"rho {r_['rho_raw']:+.4f} [{r_['ci_lo']:+.3f}, {r_['ci_hi']:+.3f}]")
            if r_["rho_corrected"] == r_["rho_corrected"]:
                s += (f"  corrected {r_['rho_corrected']:+.4f} "
                      f"[{r_['corrected_lo']:+.3f}, {r_['corrected_hi']:+.3f}]")
            print(s)
        print()

    t = pd.DataFrame(rows)
    t.to_csv(f"{ROOT}/run/rev2/whatbreaks.csv", index=False)

    # the hit-call view: does the hit survive the swap, even if the rank does not
    thr = pd.read_csv(f"{ROOT}/run/tycko/s1_thresholds.csv", index_col=0)
    print("=== hit-call concordance, same contrasts ===")
    hrows = []
    for name, a, b in CONTRASTS:
        if a not in thr.index or b not in thr.index:
            print(f"  {name:34s} no threshold for one side")
            continue
        ta, tb = float(thr.loc[a].iloc[0]), float(thr.loc[b].iloc[0])
        m = dom[a].notna() & dom[b].notna()
        ha, hb = dom[a][m] > ta, dom[b][m] > tb
        both = int((ha & hb).sum())
        print(f"  {name:34s} hits {int(ha.sum()):4d} / {int(hb.sum()):4d}  "
              f"both {both:4d}  "
              f"of-2nd {both/max(int(hb.sum()),1):.3f}  "
              f"Jaccard {both/max(int((ha|hb).sum()),1):.3f}")
        hrows.append(dict(contrast=name, n=int(m.sum()), hits_a=int(ha.sum()),
                          hits_b=int(hb.sum()), both=both,
                          frac_of_b=both / max(int(hb.sum()), 1),
                          jaccard=both / max(int((ha | hb).sum()), 1)))
    pd.DataFrame(hrows).to_csv(f"{ROOT}/run/rev2/whatbreaks_hits.csv",
                               index=False)

    # where the sgRNA disagreement lives
    a, b = "K562 dCas9 CD43 sg10", "K562 dCas9 CD43 sg15"
    d = dom[dom[a].notna() & dom[b].notna()].copy()
    d["dec"] = pd.qcut(d[b], 10, labels=False)
    print(f"\n=== sgRNA swap, by decile of {b} (n={len(d):,}) ===")
    for q, g in d.groupby("dec"):
        r = float(spearmanr(g[a], g[b])[0])
        print(f"  decile {q}: n={len(g):4d}  rho {r:+.3f}  "
              f"mean sg10 {g[a].mean():+.3f}  KRAB {int(g.Domain.astype(str).str.contains('KRAB',case=False).sum()):3d}")
    print(f"\nwrote {ROOT}/run/rev2/whatbreaks.csv and whatbreaks_hits.csv")
    print("DONE_WHATBREAKS")


if __name__ == "__main__":
    main()

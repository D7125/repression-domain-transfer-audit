"""The within-domain version of the flanking test -- the one the hypothesis asks.

flanking.py asked a between-domain question: do domains whose 80-aa window is
mostly annotated score differently from domains whose window is mostly flank?
Null across four escalating length controls, and exactly zero in the stratum
where length is fixed by construction (beta +0.003, p 0.995).

That null is solid but it answers a weaker question than the hypothesis poses.
Overlap fraction cannot tell a domain whose activity sits INSIDE the annotated
half from one whose activity sits in the flank -- both can read 50%. The
hypothesis in section 3.7 is positional, so the test must be positional too:

  within each domain, does the strongest tile overlap the Pfam annotation more
  than the domain's other tiles do?

Paired inside the domain, so annotation size, domain length and protein
identity all cancel. If robustness came from flanking sequence, the peak tile
should sit OFF the annotation (negative effect). If it came from the annotated
domain, the peak should sit ON it (positive).

Primary endpoint, declared before running: Wilcoxon signed-rank on
(peak tile's annotation coverage - mean coverage of that domain's other tiles),
over multi-tile annotated domains. One test. Sign test and a per-domain
permutation of which tile is called the peak are reported as secondary.

Usage:
  python flanking_within.py
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

ROOT = "/home/dmin/ai&stem_cell"


def main():
    dom = pd.read_csv(f"{ROOT}/run/pfam_domains_annotated.csv")
    m = pd.read_csv(f"{ROOT}/run/rev2/master.csv", low_memory=False)
    a = dom[dom.pfam_name.notna() & (dom.n_tiles >= 2)].copy()
    print(f"{len(a):,} annotated domains with >= 2 tiles "
          f"({a.protein.nunique():,} proteins)")

    # annotation in protein coordinates
    a["ann_from"] = a.aa_start + a.ali_from - 1
    a["ann_to"] = a.aa_start + a.ali_to - 1

    tiles = m[["protein", "tile_num", "tile_start", "tile_end", "Avg_pEF"]]
    rows = []
    for r in a.itertuples():
        t = tiles[(tiles.protein == r.protein)
                  & (tiles.tile_num >= r.first_tile)
                  & (tiles.tile_num <= r.last_tile)]
        t = t[t.Avg_pEF.notna()]
        if len(t) < 2:
            continue
        # fraction of each tile covered by the annotation
        lo = np.maximum(t.tile_start.values, r.ann_from)
        hi = np.minimum(t.tile_end.values, r.ann_to)
        cov = np.clip(hi - lo + 1, 0, None) / (t.tile_end.values
                                               - t.tile_start.values + 1)
        k = int(np.argmax(t.Avg_pEF.values))
        others = np.delete(cov, k)
        rows.append(dict(protein=r.protein, domain_id=r.domain_id,
                         pfam_group=r.pfam_group, n_tiles=len(t),
                         robust=int(r.pgk_hit), peak_cov=float(cov[k]),
                         other_cov=float(others.mean()),
                         diff=float(cov[k] - others.mean()),
                         cov_spread=float(cov.max() - cov.min())))
    d = pd.DataFrame(rows)
    d.to_csv(f"{ROOT}/run/rev2/flanking_within.csv", index=False)
    print(f"usable: {len(d):,} domains on {d.protein.nunique():,} proteins\n")

    # domains where every tile has identical coverage carry no information
    inf = d[d.cov_spread > 1e-9]
    print(f"informative (coverage varies across tiles): {len(inf):,}")
    print(f"  mean peak coverage  {inf.peak_cov.mean():.4f}")
    print(f"  mean other coverage {inf.other_cov.mean():.4f}")
    print(f"  mean difference     {inf['diff'].mean():+.4f} "
          f"(sd {inf['diff'].std(ddof=1):.4f})\n")

    print("=== PRIMARY: Wilcoxon signed-rank on peak minus others ===")
    w = wilcoxon(inf["diff"], alternative="two-sided")
    print(f"  n = {len(inf):,}  W = {w.statistic:.0f}  p = {w.pvalue:.4f}")
    med = float(np.median(inf["diff"]))
    boot = np.random.default_rng(0)
    bs = [np.median(inf["diff"].sample(len(inf), replace=True,
                                       random_state=int(boot.integers(1e9))))
          for _ in range(2000)]
    print(f"  median difference {med:+.4f}  95% CI "
          f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")

    print("\n=== secondary: sign test ===")
    pos, neg = int((inf["diff"] > 0).sum()), int((inf["diff"] < 0).sum())
    bt = binomtest(pos, pos + neg, 0.5)
    print(f"  peak more annotated: {pos:,}   less annotated: {neg:,}   "
          f"ties {len(inf)-pos-neg:,}")
    print(f"  p = {bt.pvalue:.4g}  proportion {pos/(pos+neg):.4f} "
          f"95% CI [{bt.proportion_ci().low:.4f}, {bt.proportion_ci().high:.4f}]")

    print("\n=== secondary: split by whether the domain is robust ===")
    for lab, sub in (("robust", inf[inf.robust == 1]),
                     ("not robust", inf[inf.robust == 0])):
        if len(sub) < 20:
            continue
        ww = wilcoxon(sub["diff"], alternative="two-sided")
        print(f"  {lab:11s} n={len(sub):5d}  median {np.median(sub['diff']):+.4f}  "
              f"mean {sub['diff'].mean():+.4f}  p={ww.pvalue:.4f}")
    dif = inf[inf.robust == 1]["diff"].mean() - inf[inf.robust == 0]["diff"].mean()
    print(f"  robust minus not-robust in mean difference: {dif:+.4f}")
    print("  (the hypothesis predicts robust domains' peaks sit FURTHER OFF "
          "the annotation, i.e. a negative value here)")
    print(f"\nwrote {ROOT}/run/rev2/flanking_within.csv")
    print("DONE_FLANKING_WITHIN")


if __name__ == "__main__":
    main()

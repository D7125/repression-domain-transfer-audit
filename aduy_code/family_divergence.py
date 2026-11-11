"""
Within-family divergence table (the deliverable the patch doc moves to centre
stage, replacing the ten-reprogramming-factor framing).

Families come from MMseqs2 protein clustering, NOT from gene-name prefixes:
"ZNF" is not a family, and SOX2 vs SRY share a prefix with very different
proteins. Each family gets both axes -- repression strength and promoter
robustness -- plus the tile-count normalisation, because a long protein gets
more tiles and therefore more chances to contain a hit.

The point of the table: "SOX2 ranks last of 19 SOX members" is a much stronger
statement than "SOX2 has 0 hits", because the family is its own control,
matched for length, structure and ancestry.

Usage:
  python family_divergence.py --csv data.csv --domains best_domains.csv \
      --clusters protein_clusters.tsv --min-members 5 --out families
"""

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--domains", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--min-members", type=int, default=5)
    ap.add_argument("--out", default="families")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    dom = pd.read_csv(args.domains)
    cl = pd.read_csv(args.clusters, sep="\t", header=None, names=["rep", "member"])
    fam = dict(zip(cl.member, cl.rep))

    # per-protein summary first: tiles, hits, strongest tile, robust domains
    per = df.groupby("protein").agg(
        n_tiles=("tile_num", "size"),
        n_hits=("Hit_pEF", lambda s: int((s == "Hit").sum())),
        max_pEF=("Avg_pEF", "max"),
    ).reset_index()
    dsum = dom.groupby("protein").agg(
        n_domains=("domain_id", "size"),
        n_robust=("pgk_hit", "sum"),
        dom_max_pEF=("max_pEF", "max"),
    ).reset_index()
    per = per.merge(dsum, on="protein", how="left").fillna(
        {"n_domains": 0, "n_robust": 0})
    per["family"] = per.protein.map(fam).fillna(per.protein)
    # hit RATE per tile: the normalisation the patch doc insists on
    per["hit_rate"] = per.n_hits / per.n_tiles
    per["has_RD"] = per.n_domains > 0

    rows = []
    for f, s in per.groupby("family"):
        if len(s) < args.min_members:
            continue
        nd = s.n_domains.sum()
        rows.append({
            "family": f,
            "n_members": len(s),
            "frac_with_RD": float(s.has_RD.mean()),
            "n_domains": int(nd),
            "frac_robust": float(s.n_robust.sum() / nd) if nd else np.nan,
            # divergence = spread of strength across members of one family
            "sd_max_pEF": float(s.max_pEF.std()),
            "iqr_max_pEF": float(s.max_pEF.quantile(.75) - s.max_pEF.quantile(.25)),
            "range_max_pEF": float(s.max_pEF.max() - s.max_pEF.min()),
            "sd_hit_rate": float(s.hit_rate.std()),
            "median_tiles": float(s.n_tiles.median()),
            "strongest": s.loc[s.max_pEF.idxmax(), "protein"],
            "weakest": s.loc[s.max_pEF.idxmin(), "protein"],
            "members": ", ".join(sorted(s.protein)[:8]),
        })
    fam_df = pd.DataFrame(rows).sort_values("sd_max_pEF", ascending=False)

    print(f"{len(fam_df)} families with >= {args.min_members} members "
          f"(covering {int(fam_df.n_members.sum()):,} proteins)")
    print("\nMOST DIVERGENT FAMILIES, by spread of repression strength")
    print(f"  {'family':<12}{'n':>4}{'%RD':>6}{'doms':>6}{'%rob':>7}"
          f"{'sd':>7}{'range':>7}  strongest / weakest")
    for _, r in fam_df.head(15).iterrows():
        rob = "   -  " if r.frac_robust != r.frac_robust else f"{r.frac_robust*100:5.0f}"
        print(f"  {r.family:<12}{int(r.n_members):>4}{r.frac_with_RD*100:>6.0f}"
              f"{int(r.n_domains):>6}{rob:>7}{r.sd_max_pEF:>7.2f}"
              f"{r.range_max_pEF:>7.2f}  {r.strongest} / {r.weakest}")

    # where do the reprogramming factors sit inside their own families?
    print("\n" + "=" * 70)
    print("REPROGRAMMING FACTORS, RANKED INSIDE THEIR OWN FAMILY")
    print("=" * 70)
    for g in ["POU5F1", "SOX2", "NANOG", "KLF4", "MYC"]:
        if g not in set(per.protein):
            print(f"  {g}: not in the library")
            continue
        f = per.loc[per.protein == g, "family"].iloc[0]
        sib = per[per.family == f].sort_values("max_pEF", ascending=False)
        rank = int((sib.protein == g).values.argmax()) + 1
        me = sib[sib.protein == g].iloc[0]
        print(f"  {g:<8} family {f:<10} rank {rank}/{len(sib)} by strength | "
              f"max_pEF {me.max_pEF:.2f} | {int(me.n_hits)} hit tiles | "
              f"{int(me.n_domains)} domains, {int(me.n_robust)} robust")
        if len(sib) > 1:
            print(f"           siblings: "
                  f"{', '.join(f'{r.protein}({r.max_pEF:.1f})' for _, r in sib.head(6).iterrows())}")

    fam_df.to_csv(f"{args.out}_divergence.csv", index=False)
    per.to_csv(f"{args.out}_per_protein.csv", index=False)
    print(f"\nwrote {args.out}_divergence.csv, {args.out}_per_protein.csv")


if __name__ == "__main__":
    main()

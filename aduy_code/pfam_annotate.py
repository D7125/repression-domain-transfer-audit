"""
Pfam annotation of repression domains. Serves three separate requirements.

  Task 3  -- the "how is your clustering different from Pfam?" control.
             AMI between the embedding clusters and Pfam family labels.
  Fix 1.3 -- the third mechanism group the pipeline was missing entirely:
             structured protein-binding domains. The patch doc is explicit
             that these must come from Pfam/UniProt, NOT from regex, which is
             why there is no regex fallback here.
  Item 6.9 -- is a given cluster KRAB or a C2H2 array? The cluster must be
             named from the domain annotation overlapping its coordinates,
             not from the gene symbols of its members.

Input is an hmmscan --domtblout run against Pfam-A:

  hmmpress Pfam-A.hmm
  hmmscan --cut_ga --cpu 8 --domtblout domains.pfam Pfam-A.hmm domains.fasta

Usage:
  python pfam_annotate.py --domtbl domains.pfam --domains best_domains.csv \
      --out pfam
"""

import argparse
import re

import numpy as np
import pandas as pd

# Pfam families for the structured-domain group. Matched case-insensitively
# against the Pfam "name" field, as a whole word or a prefix with a digit
# suffix (zf-C2H2, zf-C2H2_6, ...).
DOMAIN_GROUPS = {
    "C2H2":          [r"zf-C2H2.*", r"zf-met", r"zf-H2C2.*"],
    "KRAB":          [r"KRAB.*"],
    "SCAN":          [r"SCAN.*"],
    "PHD":           [r"PHD.*", r"zf-HC5HC2H.*"],
    "CXXC":          [r"zf-CXXC.*", r"CXXC.*"],
    "MYM":           [r"zf-MYM.*", r"zf-MYND.*"],
    "homeodomain":   [r"Homeodomain.*", r"Homeobox.*", r"HOX.*", r"Pou.*"],
    "bHLH":          [r"HLH", r"bHLH.*"],
    "leucine_zipper": [r"bZIP.*", r"BZIP.*", r"Leucine_zipper.*"],
    "BTB_POZ":       [r"BTB.*"],
    "SAND":          [r"SAND.*"],
    "WD40":          [r"WD40.*"],
    "chromo_bromo":  [r"Chromo.*", r"Bromodomain.*"],
    "MBD":           [r"MBD.*"],
}


def parse_domtbl(path):
    """hmmscan --domtblout is whitespace-aligned with a variable description."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split(None, 22)
            if len(f) < 22:
                continue
            rows.append({
                "pfam_name": f[0], "pfam_acc": f[1], "pfam_len": int(f[2]),
                "domain_id": f[3], "qlen": int(f[5]),
                "evalue_full": float(f[6]), "score_full": float(f[7]),
                "evalue_dom": float(f[11]), "score_dom": float(f[13]),
                "hmm_from": int(f[15]), "hmm_to": int(f[16]),
                "ali_from": int(f[17]), "ali_to": int(f[18]),
            })
    return pd.DataFrame(rows)


def group_of(name):
    for g, pats in DOMAIN_GROUPS.items():
        for p in pats:
            if re.fullmatch(p, name, flags=re.I):
                return g
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domtbl", required=True)
    ap.add_argument("--domains", required=True)
    ap.add_argument("--out", default="pfam")
    args = ap.parse_args()

    from sklearn.metrics import adjusted_mutual_info_score as ami
    from scipy.stats import fisher_exact

    hits = parse_domtbl(args.domtbl)
    dom = pd.read_csv(args.domains)
    print(f"{len(hits):,} Pfam domain hits over "
          f"{hits.domain_id.nunique():,} of {len(dom):,} repression domains "
          f"({hits.domain_id.nunique()/len(dom)*100:.0f}% annotated)")

    # best hit per repression domain, by domain bit score
    best = hits.sort_values("score_dom", ascending=False).groupby(
        "domain_id", as_index=False).first()
    dom = dom.merge(best[["domain_id", "pfam_name", "pfam_acc", "score_dom",
                          "ali_from", "ali_to"]], on="domain_id", how="left")
    dom["pfam_group"] = dom.pfam_name.map(
        lambda x: group_of(x) if isinstance(x, str) else None)

    print("\nmost common Pfam families among repression domains:")
    for n, c in dom.pfam_name.value_counts().head(15).items():
        g = group_of(n)
        print(f"  {n:<28} {c:>5}   {('[' + g + ']') if g else ''}")

    # -------- the structured-domain feature group (fix 1.3) --------------
    print("\nSTRUCTURED-DOMAIN GROUPS (the missing third mechanism class)")
    print(f"  {'group':<17}{'n':>6}{'% of all':>10}{'robust':>9}"
          f"{'median pEF':>12}")
    base = dom.pgk_hit.mean()
    rows = []
    for g in DOMAIN_GROUPS:
        s = dom[dom.pfam_group == g]
        if not len(s):
            print(f"  {g:<17}{0:>6}")
            continue
        # is this group enriched for promoter robustness vs everything else?
        a = int(s.pgk_hit.sum()); b = len(s) - a
        c = int(dom.pgk_hit.sum()) - a; d = len(dom) - len(s) - c
        _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        print(f"  {g:<17}{len(s):>6}{len(s)/len(dom)*100:>9.1f}%"
              f"{s.pgk_hit.mean():>9.2f}{s.max_pEF.median():>12.2f}")
        rows.append({"group": g, "n": len(s), "frac_robust": s.pgk_hit.mean(),
                     "median_max_pEF": s.max_pEF.median(), "p_robust": p})
    print(f"  {'(background)':<17}{len(dom):>6}{100.0:>9.1f}%{base:>9.2f}"
          f"{dom.max_pEF.median():>12.2f}")
    grp_df = pd.DataFrame(rows)

    # -------- Task 3: how different is the clustering from Pfam? ---------
    asg = dom[(dom.cluster >= 0) & dom.pfam_name.notna()]
    if len(asg) > 20:
        print(f"\nTASK 3 -- clustering vs Pfam, on {len(asg):,} domains that "
              f"have both labels")
        print(f"  AMI(embedding cluster, Pfam family) = "
              f"{ami(asg.pfam_name.astype(str), asg.cluster):.4f}")
        if "pfam_group" in asg:
            g = asg[asg.pfam_group.notna()]
            if len(g) > 20:
                print(f"  AMI(embedding cluster, Pfam GROUP) = "
                      f"{ami(g.pfam_group.astype(str), g.cluster):.4f} "
                      f"(on {len(g):,} domains)")
        print("  high -> a family classifier, so a positive control")
        print("  low  -> cuts across families, which is the only way a")
        print("          cluster can be a convergent mechanism")

    # -------- Item 6.9: name each cluster from annotation ----------------
    print("\nITEM 6.9 -- naming clusters from Pfam, not from gene symbols")
    print(f"  {'cluster':<9}{'n':>5}{'annot%':>8}{'robust':>8}  dominant Pfam "
          f"families (share of annotated)")
    for cl in sorted(c for c in dom.cluster.unique() if c >= 0):
        s = dom[dom.cluster == cl]
        ann = s[s.pfam_name.notna()]
        if not len(ann):
            print(f"  c{cl:<8}{len(s):>5}{0:>7.0f}%{s.pgk_hit.mean():>8.2f}  "
                  f"-- no Pfam hit --")
            continue
        vc = ann.pfam_name.value_counts().head(3)
        lab = ", ".join(f"{k} {v/len(ann)*100:.0f}%" for k, v in vc.items())
        print(f"  c{cl:<8}{len(s):>5}{len(ann)/len(s)*100:>7.0f}%"
              f"{s.pgk_hit.mean():>8.2f}  {lab}")

    # KRAB vs C2H2 specifically
    k = dom[dom.pfam_group == "KRAB"]; c2 = dom[dom.pfam_group == "C2H2"]
    print(f"\n  KRAB-annotated repression domains: {len(k)}  "
          f"(the paper removed 344 KRAB-containing TFs from the library)")
    print(f"  C2H2-annotated repression domains: {len(c2)}")
    if len(c2):
        print(f"    C2H2 robustness {c2.pgk_hit.mean():.2f} vs background "
              f"{base:.2f}")

    dom.to_csv(f"{args.out}_domains_annotated.csv", index=False)
    grp_df.to_csv(f"{args.out}_groups.csv", index=False)
    hits.to_csv(f"{args.out}_all_hits.csv", index=False)
    print(f"\nwrote {args.out}_domains_annotated.csv, {args.out}_groups.csv, "
          f"{args.out}_all_hits.csv")


if __name__ == "__main__":
    main()

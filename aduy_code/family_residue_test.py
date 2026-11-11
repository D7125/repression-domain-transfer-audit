"""
Does residue identity, or residue CLASS, track promoter robustness inside one
protein family? Generalised from the 44-homeodomain experiment.

The homeodomain run found: specific residues mark the sub-family, not
robustness (p = 0.065 once sub-family is held constant), while
physicochemical CLASS survives (p = 0.012). That was one family at n = 44.
This runs the same test on any Pfam family, so the result can be checked
where n is larger: HLH has 58 domains, zf-C2H2 has 116.

Stratification: the confound variable here is the MMseqs2 homology family,
not the embedding cluster. MMseqs2 families are defined purely by sequence
identity, so they are the right thing to hold constant when asking whether
sequence tracks a phenotype. The homeodomain run used embedding clusters,
which happened to coincide with sub-families; this is the cleaner control
and applies to families that were never split into clusters at all.

Usage:
  python family_residue_test.py --annotated pfam_domains_annotated.csv \
      --pfam HLH --hmm-db ../pfam_db/Pfam-A.hmm \
      --mmseqs dom_clu_cluster.tsv --out hlh
"""

import argparse
import os
import subprocess
import tempfile
from collections import Counter

import numpy as np
import pandas as pd

CLASS = {}
for aa, cl in [("KRH", "basic"), ("DE", "acidic"), ("NQST", "polar"),
               ("AVLIMFWYC", "hydrophobic"), ("GP", "small"), ("-", "gap")]:
    for a in aa:
        CLASS[a] = cl

HMMER_BIN = "/home/dmin/miniconda3/envs/mmseqs/bin"


def read_stockholm(path):
    seqs = {}
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        parts = line.split()
        if len(parts) == 2:
            seqs.setdefault(parts[0], "")
            seqs[parts[0]] += parts[1]
    return seqs


def mutual_info(col, labels):
    n = len(labels)
    mi = 0.0
    pos = sum(labels) / n
    for res in set(col):
        p_res = col.count(res) / n
        for lab in (0, 1):
            joint = sum(1 for c, l in zip(col, labels) if c == res and l == lab)
            if not joint:
                continue
            p_joint = joint / n
            p_lab = pos if lab else 1 - pos
            mi += p_joint * np.log2(p_joint / (p_res * p_lab))
    return mi


def max_mi(cols, labels):
    return max(mutual_info(c, labels) for c in cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--pfam", required=True, help="Pfam family name, e.g. HLH")
    ap.add_argument("--hmm-db", required=True)
    ap.add_argument("--mmseqs", required=True)
    ap.add_argument("--perms", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = pd.read_csv(args.annotated)
    fam = d[d.pfam_name == args.pfam].copy()
    print(f"=== Pfam family {args.pfam}: {len(fam)} repression domains ===")
    if len(fam) < 20:
        print("fewer than 20 domains; not worth testing")
        return
    print(f"  robust {int(fam.pgk_hit.sum())} | not robust "
          f"{int((~fam.pgk_hit.astype(bool)).sum())} "
          f"(background 27.1%)")

    t = pd.read_csv(args.mmseqs, sep="\t", header=None, names=["rep", "member"])
    fam["mm"] = fam.domain_id.map(dict(zip(t.member, t.rep))).fillna(fam.domain_id)
    sizes = fam.mm.value_counts()
    print(f"  MMseqs2 sub-families: {fam.mm.nunique()} "
          f"(largest {sizes.iloc[0]}, singletons {(sizes == 1).sum()})")

    # ---------------- align to the family HMM -----------------------------
    tmp = tempfile.mkdtemp()
    fa, hmm, sto = (os.path.join(tmp, f) for f in ("f.fasta", "f.hmm", "f.sto"))
    with open(fa, "w") as fh:
        for _, r in fam.iterrows():
            fh.write(f">{r.domain_id}\n{r.sequence}\n")
    subprocess.run([f"{HMMER_BIN}/hmmfetch", args.hmm_db, args.pfam],
                   stdout=open(hmm, "w"), check=True)
    subprocess.run([f"{HMMER_BIN}/hmmalign", "--amino", "--trim",
                    "-o", sto, hmm, fa], check=True)
    seqs = read_stockholm(sto)
    fam = fam[fam.domain_id.isin(seqs)].reset_index(drop=True)
    aln = [seqs[d] for d in fam.domain_id]
    L = len(aln[0])
    labels = fam.pgk_hit.astype(int).tolist()
    print(f"  aligned: {len(aln)} sequences x {L} columns")

    results = []
    for mode in ("raw residue", "physicochemical class"):
        cols = []
        for i in range(L):
            col = [a[i] for a in aln]
            if mode != "raw residue":
                col = [CLASS.get(c, "other") for c in col]
            cols.append(col)
        obs = max_mi(cols, labels)
        lab = np.array(labels)
        mm_arr = fam.mm.values

        # null A: free permutation (ignores sub-family)
        nullA = np.array([max_mi(cols, rng.permutation(lab).tolist())
                          for _ in range(args.perms)])
        pA = (1 + (nullA >= obs).sum()) / (args.perms + 1)

        # null B: permute WITHIN each MMseqs2 sub-family. Sub-families of
        # size 1, or with no label variation, contribute nothing and stay put.
        groups = [np.where(mm_arr == g)[0] for g in pd.unique(mm_arr)]
        usable = sum(1 for g in groups if len(g) > 1 and len(set(lab[g])) > 1)
        nullB = np.empty(args.perms)
        for p in range(args.perms):
            sh = lab.copy()
            for g in groups:
                if len(g) > 1:
                    sh[g] = rng.permutation(sh[g])
            nullB[p] = max_mi(cols, sh.tolist())
        pB = (1 + (nullB >= obs).sum()) / (args.perms + 1)

        print(f"\n  --- {mode} ---")
        print(f"    observed max MI = {obs:.4f} bits "
              f"(column {int(np.argmax([mutual_info(c, labels) for c in cols])) + 1})")
        print(f"    null A, free permutation      : 95th pct "
              f"{np.percentile(nullA, 95):.4f}  p = {pA:.4f}")
        print(f"    null B, within sub-family     : 95th pct "
              f"{np.percentile(nullB, 95):.4f}  p = {pB:.4f}")
        print(f"      ({usable} sub-families actually carry label variation,")
        print(f"       so null B has limited shuffling room)")
        if pB < 0.05:
            print("    -> SURVIVES the sub-family control")
        elif pA < 0.05:
            print("    -> only passes the free permutation: the signal is")
            print("       sub-family identity, not robustness")
        else:
            print("    -> no signal at all")
        results.append({"family": args.pfam, "mode": mode, "n": len(aln),
                        "n_robust": int(sum(labels)), "cols": L,
                        "max_mi": obs, "p_free": pA, "p_within": pB,
                        "usable_subfamilies": usable})

        mis = [mutual_info(c, labels) for c in cols]
        for i in np.argsort(mis)[::-1][:3]:
            rob = Counter(c for c, l in zip(cols[i], labels) if l == 1)
            fra = Counter(c for c, l in zip(cols[i], labels) if l == 0)
            print(f"      col {i+1:>3} MI={mis[i]:.3f}  "
                  f"robust={dict(rob.most_common(2))}  "
                  f"fragile={dict(fra.most_common(2))}")

    pd.DataFrame(results).to_csv(f"{args.out}_residue_test.csv", index=False)
    print(f"\nwrote {args.out}_residue_test.csv")


if __name__ == "__main__":
    main()

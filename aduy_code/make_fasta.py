"""
Build the two FASTA files MMseqs2 needs.

  proteins.fasta  one record per HGNC symbol, tiles stitched back together.
                  Used to cluster PROTEINS, which is what train_cnn.py's
                  --clusters flag consumes: splitting by protein still leaks
                  through paralogy (KLF4 in train, KLF7 in test).
  domains.fasta   one record per repression domain. Used to cluster DOMAINS,
                  which is what separates a homology family from a convergent
                  cluster -- the 5-mer Jaccard proxy returns a background of
                  0.0000 on this data, so it cannot make that call at all.

Stitching: tiles step 10 aa but not always (4 near C-termini, 20 where an
unscored tile was dropped), so residues are added by real coordinate. Where a
gap is too wide to bridge, the sequence is cut and the next piece is emitted as
a separate record with a #2, #3 suffix -- better than silently fusing two
distant stretches into one fake protein.

Usage:
    python make_fasta.py --csv data.csv --domains comp_domains.csv --out-dir .
"""

import argparse
import os

import pandas as pd


def stitch(sub):
    """Yield contiguous sequence pieces for one protein, in coordinate order."""
    sub = sub.sort_values("tile_start")
    starts = sub.tile_start.values
    seqs = sub.sequence.tolist()
    pieces, cur = [], seqs[0]
    for i in range(1, len(seqs)):
        shift = int(starts[i] - starts[i - 1])
        if 0 < shift < len(seqs[i]):
            cur += seqs[i][-shift:]           # overlapping: add only new residues
        else:
            pieces.append(cur)                # gap too wide to bridge
            cur = seqs[i]
    pieces.append(cur)
    return pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--domains", required=True)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    pf = os.path.join(args.out_dir, "proteins.fasta")
    n_rec = n_split = 0
    with open(pf, "w") as fh:
        for gene, sub in df.groupby("protein"):
            for j, piece in enumerate(stitch(sub)):
                name = gene if j == 0 else f"{gene}#{j + 1}"
                fh.write(f">{name}\n{piece}\n")
                n_rec += 1
            n_split += (j > 0)
    print(f"wrote {pf}: {n_rec:,} records from {df.protein.nunique():,} proteins "
          f"({n_split:,} proteins had an unbridgeable gap and were split)")

    dom = pd.read_csv(args.domains)
    dfp = os.path.join(args.out_dir, "domains.fasta")
    with open(dfp, "w") as fh:
        for _, r in dom.iterrows():
            fh.write(f">{r.domain_id}\n{r.sequence}\n")
    print(f"wrote {dfp}: {len(dom):,} domain records")
    print(f"  domain length: min {dom.sequence.str.len().min()} "
          f"max {dom.sequence.str.len().max()} "
          f"median {int(dom.sequence.str.len().median())}")


if __name__ == "__main__":
    main()

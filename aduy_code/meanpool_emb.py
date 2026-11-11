"""
Pool the per-residue ESM memmap down to one vector per tile.

Pooling level A of two. Level A is 80 residues -> 1 tile (this script);
level B is several tiles -> 1 domain (cluster_domains.py --domain-pool).
Neither choice is obvious, so both are switchable and the clusterings they
produce are compared by AMI rather than one being assumed.

  --pool mean   every residue weighted 1/L. A 4-aa motif is diluted ~20x,
                which is the exact objection cnn_model.py is built around.
  --pool max    per-dimension maximum over residues. Keeps the strongest
                local signal, so it is the pooling that matches a motif
                model -- but it discards how much of the tile supports it.

extract_per_residue.py writes [N, L, D] for the CNN. cluster_domains.py wants
[N, D], one row per tile, row-aligned to data.csv. This bridges the two so the
ESM extraction only has to run once.

Reads in chunks: the full tensor is ~8 GB and this machine has 9 GB of RAM.

Usage:
    python meanpool_emb.py --emb esm2_480_res --out esm2_480.npy
"""

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="prefix from extract_per_residue.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--pool", default="mean", choices=["mean", "max"],
                    help="how to reduce residues to one vector per tile")
    args = ap.parse_args()

    meta = json.load(open(args.emb + ".json"))
    n, L, d = meta["n"], meta["max_len"], meta["dim"]
    lens = np.load(args.emb + ".len.npy")
    mm = np.memmap(args.emb + ".dat", dtype=np.float16, mode="r", shape=(n, L, d))

    out = np.zeros((n, d), dtype=np.float32)
    for s in range(0, n, args.chunk):
        e = min(s + args.chunk, n)
        block = np.asarray(mm[s:e], dtype=np.float32)      # [b, L, D]
        # Average over REAL residues only. Every tile here is exactly 80 aa, so
        # this reduces to a plain mean -- but padding-aware anyway, because a
        # rerun at a different max_len would otherwise average in the zeros.
        ln = lens[s:e].astype(np.int32)
        m = (np.arange(L)[None, :] < ln[:, None])                      # [b, L]
        if args.pool == "mean":
            w = m.astype(np.float32)
            out[s:e] = (block * w[:, :, None]).sum(1) / np.maximum(
                w.sum(1, keepdims=True), 1)
        else:
            # padding must not win the max; -inf on masked positions
            block[~m] = -np.inf
            out[s:e] = block.max(1)
        if (s // args.chunk) % 5 == 0:
            print(f"  {s:,}/{n:,}", flush=True)

    np.save(args.out, out)
    print(f"wrote {args.out}  shape {out.shape}  "
          f"({out.nbytes / 1e6:.0f} MB, float32)")
    print(f"sanity: row 0 L2 norm = {np.linalg.norm(out[0]):.4f}, "
          f"all-finite = {np.isfinite(out).all()}")


if __name__ == "__main__":
    main()

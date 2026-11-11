"""
Extract PER-RESIDUE ESM-2 embeddings (not mean-pooled) to a memmap.

The existing pipeline stored one vector per tile (esm2_640.npy, [N, 640]).
The CNN needs the full [N, L, D] tensor. Row order is kept identical to
data.csv so every downstream join still works.

Outputs
  <out_prefix>.dat   fp16 memmap, shape [N, max_len, D]
  <out_prefix>.len   int16 npy,   shape [N]      true length of each tile
  <out_prefix>.json  shape / dtype / model metadata

Disk cost (max_len=80):
  D=320  ->  5.3 GB      D=640  -> 10.6 GB
  D=480  ->  7.9 GB      D=1280 -> 21.2 GB
Start with D=480 to test the idea cheaply, then rerun at 640.

Requires:  pip install fair-esm torch pandas numpy
Usage:
  python extract_per_residue.py --csv data.csv --model esm2_t30_150M_UR50D \
         --out esm2_640_res --batch 32
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

MODELS = {
    # fair-esm name            dim   last layer index
    "esm2_t6_8M_UR50D":       (320,  6),
    "esm2_t12_35M_UR50D":     (480, 12),
    "esm2_t30_150M_UR50D":    (640, 30),
    "esm2_t33_650M_UR50D":    (1280, 33),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="data.csv, must contain a sequence column")
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--model", default="esm2_t30_150M_UR50D", choices=list(MODELS))
    ap.add_argument("--layer", type=int, default=None,
                    help="representation layer; default = last. "
                         "Try ~60-80%% of depth too: intermediate layers often "
                         "probe better than the last for function prediction.")
    ap.add_argument("--out", required=True, help="output prefix, no extension")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import esm  # fair-esm

    dim, last_layer = MODELS[args.model]
    layer = args.layer if args.layer is not None else last_layer

    df = pd.read_csv(args.csv)
    if args.seq_col not in df.columns:
        raise SystemExit(
            f"column '{args.seq_col}' not in {args.csv}. columns = {list(df.columns)}"
        )
    seqs = df[args.seq_col].astype(str).str.strip().str.upper().tolist()
    n = len(seqs)

    lens = np.array([min(len(s), args.max_len) for s in seqs], dtype=np.int16)
    if (np.array([len(s) for s in seqs]) > args.max_len).any():
        print(f"warning: some sequences longer than max_len={args.max_len}; truncating")

    print(f"{n:,} sequences | model {args.model} | layer {layer} | dim {dim}")
    gb = n * args.max_len * dim * 2 / 1e9
    print(f"output size ~{gb:.1f} GB")

    model, alphabet = esm.pretrained.__dict__[args.model]()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(args.device).eval()
    if args.device == "cuda":
        model = model.half()

    out = np.memmap(args.out + ".dat", dtype=np.float16, mode="w+",
                    shape=(n, args.max_len, dim))

    with torch.no_grad():
        for start in range(0, n, args.batch):
            chunk = seqs[start:start + args.batch]
            data = [(f"t{start + i}", s[:args.max_len]) for i, s in enumerate(chunk)]
            _, _, toks = batch_converter(data)
            toks = toks.to(args.device)

            rep = model(toks, repr_layers=[layer], return_contacts=False)
            rep = rep["representations"][layer]       # [B, 1 + L + 1, D]

            for i, (_, s) in enumerate(data):
                L = len(s)
                # strip BOS (index 0) and EOS; keep exactly the L real residues
                out[start + i, :L] = rep[i, 1:1 + L].float().cpu().numpy().astype(np.float16)
                if L < args.max_len:
                    out[start + i, L:] = 0

            if (start // args.batch) % 50 == 0:
                print(f"  {start:,}/{n:,}", flush=True)

    out.flush()
    np.save(args.out + ".len", lens)
    with open(args.out + ".json", "w") as f:
        json.dump({"n": n, "max_len": args.max_len, "dim": dim,
                   "model": args.model, "layer": layer, "dtype": "float16"}, f, indent=2)

    print("done ->", args.out + ".dat")
    print("sanity: row 0 L2 norm of residue 0 =",
          float(np.linalg.norm(np.asarray(out[0, 0], dtype=np.float32))))


if __name__ == "__main__":
    main()

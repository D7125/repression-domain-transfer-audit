"""
Representation x pooling ablation: attribution control for the section-3.2 claim
that the gap over the amino-acid-composition baseline comes from "the ESM-2
representation PLUS windowed pooling".

That sentence conjoins two causes with no cell that separates them. This script
runs the full 3 x 2 factorial in ONE invocation so every contrast is paired by
seed:

  representation                                   pooling
  R1_esm480    ESM-2 t12_35M layer 12, frozen      MeanPool  (no position)
  R2_embed480  Linear(20->480, no bias) on one-hot MotifCNN  (conv over position)
               == nn.Embedding(20,480), trained from scratch
  R3_onehot20  one-hot 20 amino acids, fixed

R2 is the cell that isolates PRETRAINING: same dimensionality, byte-identical
downstream stack, differing only in whether the per-residue vectors were learned
in advance. R3 instead asks whether 480 dimensions are needed at all; it confounds
"no pretraining" with "20 dimensions", so it does not carry the pretraining claim.

Why all six cells are re-run rather than reusing published numbers: adding a second
dataset path can shift RNG consumption order, so a paired comparison is only safe
inside one invocation with one seed list.

MotifCNN's only representation-dependent layer is proj = Linear(in_dim, 64); the
conv bank, pooling and head are identical across cells. Parameter counts therefore
differ by construction and are reported, not matched -- padding one-hot up to 480
to equalise them would be an artefact.

Usage (deterministic mode needs the cuBLAS var set BEFORE python starts):

  CUBLAS_WORKSPACE_CONFIG=:4096:8 python repr_ablation.py \
      --emb ../run/esm2_480_res --csv ../run/rev2/master.csv.gz \
      --clusters ../run/protein_clusters.tsv --target Avg_pEF \
      --seeds 0 1 2 3 4 5 6 7 8 9 --deterministic \
      --out ../run/rev2/repr_ablation.json

  # bitwise determinism check on the primary cell before spending 60 runs:
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python repr_ablation.py ... --verify-determinism

With --cross-target cd43 every trained model is ALSO scored on the dCas9 label it
never saw, which turns the same 60 runs into the representation arm of the context
question: section 3.2 shows the cross-context collapse splits ~70/30 into a part
that is irreducible under the rTetR objective and a model shortfall, and this run
asks whether the remaining ~30% is a property of the REPRESENTATION. If R1, R2 and
R3 all land at the same cross-context value while differing on the training target,
the shortfall is not the representation's. Cross-scoring costs no extra training:
the test loader runs shuffle=False, so the test predictions are already aligned
with the test index and the cross subset is a positional mask.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cnn_model import MotifCNN, MeanPoolMLP, count_params          # noqa: E402
from train_cnn import TileDataset, grouped_split, fit, shortlist_metrics  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"          # 20 canonical residues; the library uses no others
AA_IDX = {a: i for i, a in enumerate(AA)}
N_AA = len(AA)


# ---------------------------------------------------------------- data

class SeqOneHotDataset(torch.utils.data.Dataset):
    """One-hot per-residue input, shaped exactly like the ESM memmap batch.

    Masks are taken from the SAME .len.npy the embedding dataset uses, so the two
    representations see byte-identical masks and the only difference between cells
    is the per-residue vector itself.
    """

    def __init__(self, codes, lens, max_len, idx, y):
        self.codes = codes              # (N, max_len) int8, -1 = padding
        self.lens = lens
        self.max_len = max_len
        self.idx = np.asarray(idx)
        self.y = np.asarray(y, dtype=np.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        row = int(self.idx[i])
        c = torch.from_numpy(self.codes[row].astype(np.int64))
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        L = int(self.lens[row])
        mask[:L] = True
        x = F.one_hot(c.clamp(min=0), N_AA).float()
        x = x * mask.unsqueeze(-1)      # padding rows stay all-zero
        return x, mask, torch.tensor(self.y[i])


def encode_sequences(seqs, max_len):
    """(N, max_len) int8 of amino-acid indices, -1 for padding."""
    codes = np.full((len(seqs), max_len), -1, dtype=np.int8)
    bad = 0
    for r, s in enumerate(seqs):
        for p, ch in enumerate(s[:max_len]):
            j = AA_IDX.get(ch, -1)
            if j < 0:
                bad += 1
            codes[r, p] = j
    if bad:
        print(f"warning: {bad} residues outside the 20-letter alphabet, encoded as padding")
    return codes


# ---------------------------------------------------------------- model wrapper

class EmbedWrap(nn.Module):
    """Trainable per-residue embedding in front of an unmodified head.

    Linear(20 -> out_dim, bias=False) applied to a one-hot vector is exactly an
    nn.Embedding(20, out_dim) lookup, and keeps the inner model untouched so the
    comparison against the ESM-2 cell is architecture-for-architecture.
    """

    def __init__(self, inner, in_dim=N_AA, out_dim=480):
        super().__init__()
        self.emb = nn.Linear(in_dim, out_dim, bias=False)
        self.inner = inner

    def forward(self, x, mask=None):
        return self.inner(self.emb(x), mask)


# ---------------------------------------------------------------- cells

def build_cells(dim, args):
    """(repr_name, arch_name, which_loader, constructor) -- order is fixed and
    every cell is built after the same torch.manual_seed(seed)."""
    def motif(d):
        return MotifCNN(d, proj_dim=args.proj_dim, channels=args.channels,
                        kernels=tuple(args.kernels), pooling=args.pooling,
                        attn_heads=args.attn_heads)
    return [
        ("R1_esm480",   "MeanPool", "esm", lambda: MeanPoolMLP(dim)),
        ("R1_esm480",   "MotifCNN", "esm", lambda: motif(dim)),
        ("R2_embed480", "MeanPool", "seq", lambda: EmbedWrap(MeanPoolMLP(dim), out_dim=dim)),
        ("R2_embed480", "MotifCNN", "seq", lambda: EmbedWrap(motif(dim), out_dim=dim)),
        ("R3_onehot20", "MeanPool", "seq", lambda: MeanPoolMLP(N_AA)),
        ("R3_onehot20", "MotifCNN", "seq", lambda: motif(N_AA)),
    ]


def make_loaders(kind, args, emb_prefix, codes, lens, max_len, splits, y_all, device, seed):
    out = []
    for idx, shuf in zip(splits, (True, False, False)):
        if kind == "esm":
            ds = TileDataset(emb_prefix, idx, y_all[idx])
        else:
            ds = SeqOneHotDataset(codes, lens, max_len, idx, y_all[idx])
        kw = {}
        if args.deterministic:
            g = torch.Generator()
            g.manual_seed(seed)
            kw["generator"] = g
            kw["worker_init_fn"] = (lambda wid, s=seed: np.random.seed(s * 1000 + wid))
        out.append(torch.utils.data.DataLoader(
            ds, batch_size=args.batch, shuffle=shuf, num_workers=args.workers,
            pin_memory=(device == "cuda"), **kw))
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="Avg_pEF")
    ap.add_argument("--group-col", default="protein")
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--proj-dim", type=int, default=64)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--kernels", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--pooling", default="maxattn",
                    choices=["max", "mean", "maxmean", "attn", "maxattn"])
    ap.add_argument("--attn-heads", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--out", default="repr_ablation.json")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--cross-target", default=None,
                    help="score every trained model on a SECOND label column it was "
                         "never trained on, e.g. cd43. This is the representation arm "
                         "of the context question: if all three representations land "
                         "at the same cross-context value, representation is not what "
                         "limits transfer. No extra training -- the test loader runs "
                         "shuffle=False, so test predictions are aligned with the test "
                         "index and the cross subset is taken by position.")
    ap.add_argument("--cross-reps", nargs=2, default=["cd43_br1", "cd43_br2"],
                    help="the two replicate columns of --cross-target, for its ceiling")
    ap.add_argument("--train-reps", nargs=2, default=["R1_pEF", "R2_pEF"],
                    help="the two replicate columns of --target, for the analytic cap")
    ap.add_argument("--cells", nargs="+", default=None,
                    help="restrict to cells named REPR/ARCH, e.g. R2_embed480/MotifCNN. "
                         "Determinism is verified by running the SAME command twice as two "
                         "separate processes and diffing the output JSONs -- not by repeating "
                         "a cell inside one process, because a DataLoader generator advances "
                         "its state across epochs and the second pass would see a different "
                         "shuffle order.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(args.emb + ".json"))
    dim, max_len = meta["dim"], meta["max_len"]
    lens = np.load(args.emb + ".len.npy")

    df = pd.read_csv(args.csv)
    if args.seq_col not in df.columns:
        raise SystemExit(f"--seq-col '{args.seq_col}' not in {list(df.columns)}")
    if len(df) != meta["n"]:
        raise SystemExit(f"row-count mismatch: csv {len(df)} vs memmap {meta['n']}; "
                         "master.csv row order is asserted identical to the memmap -- "
                         "do not reorder")

    keep = df[args.target].notna().values
    y_all = df[args.target].values.astype(np.float32)
    groups = df[args.group_col].astype(str).values

    has_cross = None
    if args.cross_target:
        need = [args.cross_target, *args.cross_reps, *args.train_reps]
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise SystemExit(f"--cross-target needs columns {miss}, not in the csv")
        # A tile enters the cross comparison only with BOTH replicates present:
        # the ceiling it is divided by is estimated on exactly these tiles, and a
        # single-replicate tile would contribute to the numerator but not the
        # denominator. Same rule as cd43_model.py, so the two are comparable.
        has_cross = (df[args.cross_target].notna().values
                     & df[args.cross_reps[0]].notna().values
                     & df[args.cross_reps[1]].notna().values)
        print(f"cross-scoring on '{args.cross_target}': {has_cross.sum():,} tiles "
              f"have the target and both replicates", flush=True)

    if args.clusters:
        cl = pd.read_csv(args.clusters, sep="\t", header=None, names=["rep", "member"])
        m = dict(zip(cl["member"], cl["rep"]))
        groups = np.array([m.get(g, g) for g in groups])
        print(f"splitting by {len(np.unique(groups)):,} homology clusters")

    if args.deterministic:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise SystemExit("--deterministic needs CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                             "in the environment BEFORE python starts")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("deterministic mode on", flush=True)

    print(f"encoding {len(df):,} sequences from column '{args.seq_col}'", flush=True)
    codes = encode_sequences(df[args.seq_col].astype(str).values, max_len)

    cells = build_cells(dim, args)
    if args.cells:
        want = set(args.cells)
        cells = [c for c in cells if f"{c[0]}/{c[1]}" in want]
        if not cells:
            raise SystemExit(f"--cells matched nothing; available: "
                             f"{[f'{c[0]}/{c[1]}' for c in build_cells(dim, args)]}")
        print(f"restricted to cells: {[f'{c[0]}/{c[1]}' for c in cells]}")

    rows = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if args.deterministic:
            torch.cuda.manual_seed_all(seed)

        splits = [i[keep[i]] for i in grouped_split(groups, seed)]
        print(f"\nseed {seed}: train {len(splits[0]):,}  val {len(splits[1]):,}  "
              f"test {len(splits[2]):,}", flush=True)

        cross = None
        if has_cross is not None:
            m = has_cross[splits[2]]                 # positional mask into test preds
            sub = df.iloc[splits[2][m]]

            def rep_r(c1, c2):
                """Reliability on the tiles that actually have BOTH replicates.

                Some tiles carry only one pEF replicate, and an unmasked
                pearsonr returns NaN there, silently poisoning the ceiling and
                every percentage divided by it. cd43_model.py hit exactly this
                and its ceil_pEF had to be patched afterwards (the JSON still
                carries the NaN columns as *_old); masking here reproduces the
                patched value to six decimals. The cross-target replicates are
                non-null by construction of has_cross, so this is a no-op for
                them and the two ceilings stay on the published convention.
                """
                ok = sub[c1].notna().values & sub[c2].notna().values
                return float(pearsonr(sub[c1][ok], sub[c2][ok])[0]), int(ok.sum())

            r_tr_reps, n_tr = rep_r(*args.train_reps)
            r_cr_reps, n_cr = rep_r(*args.cross_reps)
            # Spearman-Brown reliability of a 2-replicate mean, then sqrt for the
            # correlation ceiling.
            ceil_tr = float(np.sqrt(2 * r_tr_reps / (1 + r_tr_reps)))
            ceil_cr = float(np.sqrt(2 * r_cr_reps / (1 + r_cr_reps)))
            r_meas = float(pearsonr(sub[args.target], sub[args.cross_target])[0])
            cap = r_meas / (ceil_tr * ceil_cr)
            cross = {"mask": m, "sub": sub, "ceil_train": ceil_tr,
                     "ceil_cross": ceil_cr, "r_meas": r_meas, "cap": cap,
                     "n_rep_train": n_tr, "n_rep_cross": n_cr}
            print(f"  cross subset {m.sum():,} tiles | ceil {args.target} "
                  f"{ceil_tr:.4f} (n={n_tr:,}) | ceil {args.cross_target} "
                  f"{ceil_cr:.4f} (n={n_cr:,}) | r_meas {r_meas:+.4f} | "
                  f"analytic cap {cap*100:.1f}% of the {args.cross_target} "
                  f"ceiling", flush=True)

        for rname, aname, which, build in cells:
            # Fresh loaders per cell. A DataLoader generator advances its state
            # across epochs, so cells sharing one loader object would each see a
            # different shuffle order and data order would confound the paired
            # contrast. Re-seeding per cell makes all six cells see one order.
            tr, va, te = make_loaders(which, args, args.emb, codes, lens, max_len,
                                      splits, y_all, device, seed)
            torch.manual_seed(seed)
            model = build()
            res, tp, ty = fit(model, tr, va, te, device,
                              epochs=args.epochs, patience=args.patience)
            hit_col = "Hit_PGK" if args.target == "Avg_PGK" else "Hit_pEF"
            res.update(shortlist_metrics(tp, splits[2], df, k=args.topk,
                                         hit_col=hit_col, score_col=args.target))
            res.update(representation=rname, arch=aname, seed=seed,
                       params=count_params(model), target=args.target,
                       split="homology" if args.clusters else "protein")

            if cross is not None:
                # tp comes from a shuffle=False test loader, so tp[i] is the
                # prediction for splits[2][i]; the cross subset is therefore a
                # positional mask, not a second forward pass. Taking it this way
                # also avoids any batch-composition difference between the two
                # scorings.
                p = tp[cross["mask"]]
                s = cross["sub"]
                r_cross = float(pearsonr(p, s[args.cross_target])[0])
                r_within = float(pearsonr(p, s[args.target])[0])
                res.update(
                    cross_target=args.cross_target,
                    n_cross=int(cross["mask"].sum()),
                    ceil_train=cross["ceil_train"], ceil_cross=cross["ceil_cross"],
                    r_meas_train_cross=cross["r_meas"], cap=cross["cap"],
                    n_rep_train=cross["n_rep_train"],
                    n_rep_cross=cross["n_rep_cross"],
                    r_cross=r_cross,
                    rho_cross=float(spearmanr(p, s[args.cross_target])[0]),
                    pct_ceil_cross=r_cross / cross["ceil_cross"],
                    # the same model on the same tiles against its OWN target, so
                    # the within/cross pair is apples-to-apples on one subset
                    r_within_crosssubset=r_within,
                    pct_ceil_within_crosssubset=r_within / cross["ceil_train"],
                )

            rows.append(res)
            line = (f"  {rname:12s} {aname:9s} params={res['params']:>8,}  "
                    f"test r={res['test_r']:.4f}  val r={res['val_r']:.4f}  "
                    f"gap={res['train_r']-res['test_r']:+.3f}  "
                    f"ep={res['best_epoch']:>3d}")
            if cross is not None:
                line += (f"  | cross r={res['r_cross']:+.4f} "
                         f"({res['pct_ceil_cross']*100:5.1f}% ceil)")
            print(line, flush=True)

    out = pd.DataFrame(rows)
    out.to_json(args.out, orient="records", indent=2)
    print(f"\nwrote {args.out}")

    # --- QC gate declared before the run: did every cell converge inside --epochs?
    print("\n=== QC: mean best_epoch per cell (gate: must be < 90 of "
          f"{args.epochs} max) ===")
    ep = out.groupby(["representation", "arch"])["best_epoch"].agg(["mean", "max"])
    print(ep)
    if (ep["mean"] >= 90).any():
        print("QC FAIL: a cell did not converge inside the epoch cap; raise --epochs "
              "and re-run ALL cells to keep the pairing.")
    else:
        print("QC PASS")

    print("\n=== mean test r over seeds ===")
    print(out.groupby(["representation", "arch"])["test_r"].agg(["mean", "std", "count"]))

    print("\n=== paired per-seed deltas (test r) ===")
    piv = out.pivot_table(index="seed", columns=["representation", "arch"], values="test_r")
    primary = (("R1_esm480", "MotifCNN"), ("R2_embed480", "MotifCNN"))
    contrasts = [
        primary,
        (("R1_esm480", "MotifCNN"), ("R3_onehot20", "MotifCNN")),
        (("R2_embed480", "MotifCNN"), ("R3_onehot20", "MotifCNN")),
        (("R1_esm480", "MotifCNN"), ("R1_esm480", "MeanPool")),
        (("R2_embed480", "MotifCNN"), ("R2_embed480", "MeanPool")),
        (("R3_onehot20", "MotifCNN"), ("R3_onehot20", "MeanPool")),
    ]
    for i, (a, b) in enumerate(contrasts):
        if a not in piv.columns or b not in piv.columns:
            continue
        d = piv[a] - piv[b]
        tag = "PRIMARY" if i == 0 else "secondary"
        print(f"  [{tag:9s}] {a[0]}/{a[1]} - {b[0]}/{b[1]}: "
              f"delta r = {d.mean():+.4f}  sd {d.std(ddof=1):.4f}  n={len(d)}")
    if args.cross_target:
        print(f"\n=== cross-context: scored on '{args.cross_target}', "
              f"trained on '{args.target}' (% of the {args.cross_target} "
              f"ceiling) ===")
        g = out.groupby(["representation", "arch"])["pct_ceil_cross"]
        print((g.agg(["mean", "std", "count"]) * [100, 100, 1]).round(2))
        print(f"\n  analytic cap, same seeds: "
              f"{out.groupby('seed')['cap'].first().mean()*100:.2f}% "
              f"(sd {out.groupby('seed')['cap'].first().std(ddof=1)*100:.2f})")
        print("\n  paired per-seed cross-context deltas (ceiling points):")
        pc = out.pivot_table(index="seed", columns=["representation", "arch"],
                             values="pct_ceil_cross") * 100
        for a, b in contrasts:
            if a not in pc.columns or b not in pc.columns:
                continue
            d = pc[a] - pc[b]
            print(f"    {a[0]}/{a[1]} - {b[0]}/{b[1]}: {d.mean():+.2f} pts  "
                  f"sd {d.std(ddof=1):.2f}  n={len(d)}  "
                  f"same sign {max((d>0).sum(), (d<0).sum())}/{len(d)}")
        print("\n  Reading rule declared with the design: if the three "
              "representations land within a few ceiling points of each other on "
              "the cross target while differing by far more on the training "
              "target, then representation is not what limits transfer -- the "
              "training objective is.")

    print("\nFormal TOST / BH and the %-noise-ceiling conversion are in "
          "repr_ablation_stats.py -- this block is descriptive only.")


if __name__ == "__main__":
    main()

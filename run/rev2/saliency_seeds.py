"""Multi-seed version of saliency_sanity.py.

saliency_sanity.py ran each randomisation control exactly once, against a real
model that swings 15.8-26.2% between training runs. Section 4.5 therefore had
to concede that the quantitative claim ("1.8-2.5x after subtracting the right
control") has no spread. This script closes that: the same three model states,
the same two counting conventions, over several seeds, each seed with its own
grouped split -- so the controls get the same seed-to-seed variance the real
model is already reported with.

Seed 0 reproduces saliency_sanity.py exactly and is kept as a check.

Usage:
  python saliency_seeds.py --seeds 0 1 2 3 4 --out ../run/rev2/saliency_seeds.csv
"""
import argparse
import json
import os
import re
import sys

# Must be set before torch initialises CUDA, or cuBLAS keeps its non-
# deterministic workspace and --deterministic silently does nothing.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr

sys.path.insert(0, "/home/dmin/ai&stem_cell/cnn_AI&stemcell")
from cnn_model import MotifCNN
from train_cnn import TileDataset, grouped_split, run_epoch

ROOT = "/home/dmin/ai&stem_cell"
EMB = f"{ROOT}/run/esm2_480_res"
PAT = re.compile(r"[VILMF]K.E")
N_EPOCH, PERMS = 7, 2000


def mass(sal, start, end):
    s = np.clip(sal, 0, None)
    tot = s.sum()
    return s[start:end].sum() / tot if tot > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", required=True)
    ap.add_argument("--deterministic", action="store_true",
                    help="pin cuDNN, cuBLAS and the DataLoader so a rerun of "
                         "the same seed reproduces bit for bit")
    args = ap.parse_args()
    if args.deterministic:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise SystemExit("--deterministic needs CUBLAS_WORKSPACE_CONFIG "
                             "set before CUDA initialises")
        print("deterministic mode: cudnn.deterministic, benchmark off, "
              "CUBLAS_WORKSPACE_CONFIG=:4096:8, DataLoader generator pinned",
              flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(EMB + ".json"))
    dim, L = meta["dim"], meta["max_len"]
    df = pd.read_csv(f"{ROOT}/run/rev2/master.csv", low_memory=False)
    y = df.Avg_pEF.astype(np.float32).values
    cl = pd.read_csv(f"{ROOT}/run/protein_clusters.tsv", sep="\t",
                     header=None, names=["rep", "member"])
    cmap = dict(zip(cl["member"], cl["rep"]))
    groups = np.array([cmap.get(g, g) for g in df.protein.astype(str)])
    mm = np.memmap(EMB + ".dat", dtype=np.float16, mode="r",
                   shape=(meta["n"], L, dim))

    rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        tr_i, va_i, te_i = grouped_split(groups, seed)
        test = df.iloc[te_i]
        hits = test[test.Hit_pEF == "Hit"]
        with_motif = [(i, s) for i, s in zip(hits.index, hits.sequence)
                      if PAT.search(str(s))]
        print(f"\n=== seed {seed}: test {len(te_i):,} | hits {len(hits):,} | "
              f"{len(with_motif)} carry {PAT.pattern} ===", flush=True)

        y_perm = y.copy()
        y_perm[tr_i] = rng.permutation(y[tr_i])      # only the training labels

        def build(state):
            if args.deterministic:
                torch.manual_seed(seed)
                np.random.seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True, warn_only=True)
            else:
                torch.manual_seed(seed); np.random.seed(seed)
            model = MotifCNN(dim, pooling="maxattn").to(device)
            if state == "random_init":
                return model, float("nan")
            target = y if state == "trained" else y_perm
            kw = {}
            if args.deterministic:
                g = torch.Generator()
                g.manual_seed(seed)
                kw["generator"] = g
                kw["worker_init_fn"] = lambda wid: np.random.seed(seed * 1000 + wid)
            tr = torch.utils.data.DataLoader(
                TileDataset(EMB, tr_i, target[tr_i]), batch_size=256,
                shuffle=True, num_workers=2, pin_memory=(device == "cuda"), **kw)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                    weight_decay=1e-2)
            lf = nn.MSELoss()
            for _ in range(N_EPOCH):
                run_epoch(model, tr, device, opt, lf)
            te = torch.utils.data.DataLoader(
                TileDataset(EMB, te_i, y[te_i]), batch_size=256,
                num_workers=2, pin_memory=(device == "cuda"))
            _, p, _ = run_epoch(model, te, device, None, lf)
            return model, float(pearsonr(p, y[te_i])[0])

        for state in ("trained", "label_perm", "random_init"):
            model, test_r = build(state)
            model.eval()
            A, B, wB, sal_cache = [], [], [], []
            for ridx, seq in with_motif:
                x = torch.from_numpy(
                    np.asarray(mm[ridx], dtype=np.float32))[None].to(device)
                msk = torch.ones(1, L, dtype=torch.bool, device=device)
                sal = model.residue_saliency(x, msk)[0].cpu().numpy()
                sal_cache.append(sal)
                m0 = PAT.search(seq)
                A.append(mass(sal, m0.start(), m0.end()))
                cov = np.zeros(L, bool)
                for m2 in PAT.finditer(seq):
                    cov[m2.start():m2.end()] = True
                sp = np.clip(sal, 0, None); tot = sp.sum()
                B.append(sp[cov].sum() / tot if tot > 0 else np.nan)
                wB.append(int(cov.sum()))
            A = np.array([v for v in A if v == v])
            B = np.array([v for v in B if v == v])
            wb = float(np.mean(wB))
            baseA, baseB = 4 / L, wb / L
            null = np.empty(PERMS)
            for p_ in range(PERMS):
                st = rng.integers(0, L - 4, size=len(sal_cache))
                null[p_] = np.nanmean(
                    [mass(s, a, a + 4) for s, a in zip(sal_cache, st)])
            p_pos = (1 + (null >= A.mean()).sum()) / (PERMS + 1)
            rows.append(dict(seed=seed, state=state, test_r=test_r,
                             n_tiles=len(A), massA=100 * A.mean(),
                             baselineA=100 * baseA, enrichA=A.mean() / baseA,
                             massB=100 * B.mean(), baselineB=100 * baseB,
                             enrichB=B.mean() / baseB,
                             null_pos=100 * null.mean(),
                             p_position=float(p_pos)))
            print(f"  {state:12s} test_r {test_r:+.4f}  massA "
                  f"{100*A.mean():5.2f}%  {A.mean()/baseA:4.2f}x  "
                  f"null {100*null.mean():4.2f}%  p {p_pos:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)

    t = pd.DataFrame(rows)
    t.to_csv(args.out, index=False)
    print("\n" + "=" * 70)
    print("=== mass A (%) by state, over seeds ===")
    g = t.groupby("state").massA.agg(["mean", "std", "min", "max", "count"])
    print(g.round(2).to_string())
    print("\n=== enrichment over the 5.0% length baseline ===")
    print(t.groupby("state").enrichA.agg(["mean", "std", "min", "max"])
          .round(3).to_string())
    print("\n=== learned component: trained minus each control, paired by seed ===")
    piv = t.pivot_table(index="seed", columns="state", values="massA")
    for ctrl in ("label_perm", "random_init"):
        d = piv["trained"] - piv[ctrl]
        ratio = piv["trained"] / piv[ctrl]
        print(f"  vs {ctrl:12s} diff {d.mean():+6.2f} pt (sd {d.std(ddof=1):.2f}, "
              f"n={len(d)}, neg in {(d<0).sum()})   ratio {ratio.mean():.2f}x "
              f"(sd {ratio.std(ddof=1):.2f})")
    print(f"\nwrote {args.out}")
    print("DONE_SALIENCY_SEEDS")


if __name__ == "__main__":
    main()

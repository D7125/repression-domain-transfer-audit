"""Does the model ADD selection value to a lookup table you already have?

cd43_topk.py asked "model or measurement, which ranks better" on 2 seeds. That
is the wrong question for a lab that already holds Avg_pEF for every tile: the
decision is not which single source to trust, it is whether training a model
buys anything on top of the table. So this run adds a fourth ranker --

  blend_z   mean of z(Avg_pEF) and z(pred_cd43)   <- the question
  meas_Avg_pEF  the rTetR measurement used directly   <- the comparator
  pred_cd43 MotifCNN trained on the dCas9 target      <- secondary
  pred_pEF  MotifCNN trained on the rTetR target      <- the transfer that fails

-- and runs 10 seeds instead of 2, because the pilot's two seeds disagreed at
small k by more than tile sampling can explain.

PRE-COMMITTED ANALYSIS PLAN (fixed before this script was first run):
  Primary     blend_z - meas_Avg_pEF, precision@50, paired by seed,
              CONFIRMATORY SEEDS 2-9 ONLY (0-1 are the pilot already seen).
  Secondary   pred_cd43 - meas_Avg_pEF @50; both contrasts at k in
              {20,25,100,200}; Benjamini-Hochberg within the secondary family.
  MBRE        +10 precision points (1 extra real hit per 10 wells).
  k=20        DESCRIPTIVE ONLY. One tile = 5.0 pp there and the pilot per-seed
              SD of the difference is 38.9 pp, so detecting 10 pp would need
              ~230 seeds. No inference is drawn at k=20 no matter what it shows.

Usage:
  python cd43_topk2.py --seeds 0 1 2 3 4 5 6 7 8 9 --n 10 --out ../run/rev2/cd43_topk2
"""
import argparse
import json
import os
import sys

# Must be set before torch initialises CUDA, or cuBLAS keeps its non-
# deterministic workspace and --deterministic silently does nothing.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, "/home/dmin/ai&stem_cell/cnn_AI&stemcell")
from cnn_model import MotifCNN, count_params
from train_cnn import TileDataset, grouped_split, run_epoch

ROOT = "/home/dmin/ai&stem_cell"
EMB = f"{ROOT}/run/esm2_480_res"
KS = [20, 25, 50, 100, 200, 500, 1000]
PRIMARY_K = 50


def loader_for(idx, y, batch, workers, device, shuffle=False, seed=None):
    """seed is not None => the shuffle order and every worker's RNG are pinned.

    Without this the DataLoader's shuffle draws from the global RNG, whose state
    depends on how much randomness model construction happened to consume, and
    each worker process reseeds itself from entropy.
    """
    ds = TileDataset(EMB, idx, y[idx])
    kw = {}
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        kw["generator"] = g
        kw["worker_init_fn"] = lambda wid: np.random.seed(seed * 1000 + wid)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
        pin_memory=(device == "cuda"), **kw)


def pin_training(seed):
    """Everything that makes two runs of the same seed differ on a GPU."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: an op with no deterministic kernel warns instead of raising,
    # so the run still finishes and the empirical rerun check is what decides.
    torch.use_deterministic_algorithms(True, warn_only=True)


def z(v):
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def topk_table(sub, rankers, base):
    """precision@k and enrichment over the base rate, per ranker."""
    rows = []
    hit = sub.cd43_hit_both.values.astype(bool)
    for name, score in rankers.items():
        order = np.argsort(-np.asarray(score, dtype=float))
        for k in KS:
            if k > len(sub):
                continue
            p = hit[order[:k]].mean()
            rows.append(dict(ranker=name, k=k, n_hit=int(hit[order[:k]].sum()),
                             precision=float(p), enrichment=float(p / base)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--deterministic", action="store_true",
                    help="pin cuDNN, cuBLAS and the DataLoader so a rerun of "
                         "the same seed reproduces bit for bit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.deterministic:
        print("deterministic mode: cudnn.deterministic, benchmark off, "
              "CUBLAS_WORKSPACE_CONFIG=:4096:8, DataLoader generator pinned",
              flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = json.load(open(EMB + ".json"))["dim"]
    df = pd.read_csv(f"{ROOT}/run/rev2/master.csv", low_memory=False)

    y_pef = df.Avg_pEF.astype(np.float32).values
    y_cd43 = df.cd43.astype(np.float32).values
    has = (df.cd43.notna().values & df.cd43_br1.notna().values
           & df.cd43_br2.notna().values)

    cl = pd.read_csv(f"{ROOT}/run/protein_clusters.tsv", sep="\t",
                     header=None, names=["rep", "member"])
    cmap = dict(zip(cl["member"], cl["rep"]))
    groups = np.array([cmap.get(g, g) for g in df.protein.astype(str)])
    print(f"{len(df):,} tiles | {has.sum():,} with CD43 in both bioreps",
          flush=True)
    print(f"primary endpoint: precision@{PRIMARY_K}, blend_z vs meas_Avg_pEF, "
          f"confirmatory seeds 2-9", flush=True)

    keep = ["protein", "tile_num", "tile_start", "tile_end", "seq", "Avg_pEF",
            "Avg_FLAG", "Hit_FLAG", "Hit_pEF", "cd43", "cd43_br1", "cd43_br2",
            "cd43_hit_both"]
    tk, preds_all = [], []
    for seed in args.seeds:
        tr_i, va_i, te_i = grouped_split(groups, seed)
        tr_c = tr_i[has[tr_i]]
        te_c = te_i[has[te_i]]
        sub = df.iloc[te_c][keep].copy().reset_index(drop=True)
        base = float(sub.cd43_hit_both.astype(bool).mean())
        print(f"\n=== seed {seed}: train {len(tr_i):,} test {len(te_c):,} | "
              f"base CD43 hit rate {base*100:.2f}% ===", flush=True)

        out = {}
        for tgt, y, tri in (("pEF", y_pef, tr_i), ("cd43", y_cd43, tr_c)):
            ls = seed if args.deterministic else None
            tr = loader_for(tri, y, args.batch, args.workers, device, True,
                            seed=ls)
            te = loader_for(te_c, y, args.batch, args.workers, device, seed=ls)
            if args.deterministic:
                pin_training(seed)
            else:
                torch.manual_seed(seed); np.random.seed(seed)
            model = MotifCNN(dim, pooling="maxattn").to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                    weight_decay=args.wd)
            lf = nn.MSELoss()
            for ep in range(args.n):
                run_epoch(model, tr, device, opt, lf)
            _, p, _ = run_epoch(model, te, device, None, lf)
            out[tgt] = np.asarray(p, dtype=float)
            print(f"  trained on {tgt} ({count_params(model):,} params): "
                  f"r vs cd43 = {pearsonr(out[tgt], sub.cd43)[0]:+.4f}  "
                  f"rho = {spearmanr(out[tgt], sub.cd43)[0]:+.4f}", flush=True)

        # The blend: both sources standardised on this test set, then averaged.
        # Same construction the report used for the held-out-biorep table.
        blend = (z(sub.Avg_pEF.values) + z(out["cd43"])) / 2.0

        sub["pred_cd43"] = out["cd43"]
        sub["pred_pEF"] = out["pEF"]
        sub["blend_z"] = blend
        sub["seed"] = seed
        preds_all.append(sub)

        rankers = {"blend_z": blend, "pred_cd43": out["cd43"],
                   "pred_pEF": out["pEF"], "meas_Avg_pEF": sub.Avg_pEF.values}
        rows = topk_table(sub, rankers, base)
        for r in rows:
            r["seed"] = seed
            r["base_rate"] = base
        tk += rows
        for r in rows:
            if r["k"] in (20, PRIMARY_K):
                print(f"  {r['ranker']:>13s} @{r['k']:<5d} "
                      f"{r['n_hit']:4d}/{r['k']:<5d} = {r['precision']*100:5.1f}%  "
                      f"{r['enrichment']:4.2f}x base", flush=True)

        pd.concat(preds_all).to_csv(f"{args.out}_preds.csv", index=False)
        pd.DataFrame(tk).to_csv(f"{args.out}.csv", index=False)

    t = pd.DataFrame(tk)
    print("\n=== precision@k averaged over ALL seeds (descriptive) ===")
    piv = t.pivot_table(index="k", columns="ranker", values="precision") * 100
    print(piv.round(1).to_string())

    print(f"\n=== PRIMARY: blend_z - meas_Avg_pEF @{PRIMARY_K}, "
          f"paired by seed, confirmatory seeds 2-9 ===")
    w = t.pivot_table(index=["seed", "k"], columns="ranker",
                      values="precision").reset_index()
    conf = w[(w.k == PRIMARY_K) & (w.seed >= 2)]
    if len(conf):
        dd = (conf.blend_z - conf.meas_Avg_pEF).values * 100
        m, sd, n = dd.mean(), dd.std(ddof=1), len(dd)
        se = sd / np.sqrt(n) if n > 1 else float("nan")
        print(f"  per-seed diffs (pp): {np.round(dd, 1)}")
        print(f"  mean {m:+.2f} pp   sd {sd:.2f}   n {n}   "
              f"95% CI [{m-1.96*se:+.2f}, {m+1.96*se:+.2f}]   "
              f"positive in {int((dd>0).sum())}/{n}")
    print(f"\nbase rate {t.base_rate.mean()*100:.2f}%")
    print(f"wrote {args.out}.csv and {args.out}_preds.csv")
    print("DONE_CD43_TOPK2")


if __name__ == "__main__":
    main()

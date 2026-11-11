"""What a ranking is actually worth when you have to pick k effectors.

cd43_model.py answered "does the sequence carry CD43 signal" with correlations.
A lab does not spend a correlation; it spends k wells. So this run saves the
per-tile predictions that cd43_model.py threw away and converts them into the
only number that decides an experiment: of the top k tiles a ranker hands you,
how many are real CD43 hits in both bioreps.

Four rankers on exactly the same test tiles (unseen homology clusters, CD43
scored in both bioreps):

  pred_cd43   MotifCNN trained on the dCas9 target        (the one to use)
  pred_pEF    MotifCNN trained on the rTetR target        (the transfer that fails)
  Avg_pEF     the rTetR measurement used directly         (what a catalog does)
  random      the base rate on the same tiles             (what you beat or don't)

Usage:
  python cd43_topk.py --seeds 0 1 --n 10 --out ../run/rev2/cd43_topk
"""
import argparse
import json
import sys

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
KS = [25, 50, 100, 200, 500, 1000]


def loader_for(idx, y, batch, workers, device, shuffle=False):
    ds = TileDataset(EMB, idx, y[idx])
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
        pin_memory=(device == "cuda"))


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
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

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
            tr = loader_for(tri, y, args.batch, args.workers, device, True)
            te = loader_for(te_c, y, args.batch, args.workers, device)
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

        sub["pred_cd43"] = out["cd43"]
        sub["pred_pEF"] = out["pEF"]
        sub["seed"] = seed
        preds_all.append(sub)

        rankers = {"pred_cd43": out["cd43"], "pred_pEF": out["pEF"],
                   "meas_Avg_pEF": sub.Avg_pEF.values}
        rows = topk_table(sub, rankers, base)
        for r in rows:
            r["seed"] = seed
            r["base_rate"] = base
        tk += rows
        for r in rows:
            print(f"  {r['ranker']:>13s} @{r['k']:<5d} "
                  f"{r['n_hit']:4d}/{r['k']:<5d} = {r['precision']*100:5.1f}%  "
                  f"{r['enrichment']:4.2f}x base", flush=True)

        pd.concat(preds_all).to_csv(f"{args.out}_preds.csv", index=False)
        pd.DataFrame(tk).to_csv(f"{args.out}.csv", index=False)

    t = pd.DataFrame(tk)
    print("\n=== precision@k averaged over seeds (% of top k that are real "
          "CD43 hits in both bioreps) ===")
    piv = t.pivot_table(index="k", columns="ranker", values="precision") * 100
    print(piv.round(1).to_string())
    print("\n=== enrichment over the base rate on the same tiles ===")
    print(t.pivot_table(index="k", columns="ranker",
                        values="enrichment").round(2).to_string())
    print(f"\nbase rate {t.base_rate.mean()*100:.2f}%")
    print(f"wrote {args.out}.csv and {args.out}_preds.csv")
    print("DONE_CD43_TOPK")


if __name__ == "__main__":
    main()

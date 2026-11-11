"""
The honest classification task: Hit vs Not Hit WITHIN the confirmation screen.

Why this exists. train_classify.py defines NO as `Hit_pEF != "Hit"`, which
pools the 94,526 tiles whose Hit_pEF is NaN into the negative class. NaN means
"not tried in the confirmation screen", not "tried and negative": of the 56,172
tiles in that NO class, 55,186 (98.2%) were never tested and only 986 are a
measured "Not Hit". So the reported AUROC/AUPRC measure "confirmed hit vs
mostly-untested tile", and Avg_pEF alone -- the regression arm's own target --
separates those two classes at AUROC 0.9967, far above the CNN's 0.8646.

Here the negative class is only the measured one. Two label sets:
  tested_all   YES Hit_pEF == "Hit" (6,566)  NO Hit_pEF == "Not Hit" (2,156)
  tested_expr  the same, plus Hit_FLAG == "Hit" on BOTH (3,783 / 986)
YES is now the MAJORITY class (75.3% / 79.3%), because the confirmation screen
was enriched for strong candidates: median Avg_pEF is 2.52 for Hit, 1.51 for
Not Hit, and -0.06 for the untested bulk. The question therefore flips to
"which confirmation candidates fail", so AUPRC is reported in both directions
and the Not-Hit direction is the informative one.

Avg_pEF is scored on the identical folds as the baseline that matters: if you
have run the primary screen you already hold that number for free, so a
sequence model has to beat it to be worth anything downstream.

5-fold GroupKFold over homology clusters, 15% of training groups held out as
validation for the decision threshold, fixed epochs (no early stopping, per
the project rule that early-stopping variance alone is comparable to the
effects being measured).

Usage (do NOT use python -I: that drops the script dir and cnn_model fails):
  python classify_confirmed.py --emb ../run/esm2_480_res --csv ../run/data.csv \
      --clusters ../run/protein_clusters.tsv --seeds 0 1 2 \
      --out ../run/cls_confirmed.json
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold

from cnn_model import MotifCNN, MeanPoolMLP, count_params
from train_cnn import TileDataset
from train_classify import run


LABEL_SETS = {
    "tested_all":  False,   # no expression filter
    "tested_expr": True,    # Hit_FLAG == "Hit" on both classes
}


def metrics(scores, y, thr=None):
    """AUPRC both ways: with YES in the majority the Not-Hit direction is the
    one that carries information, and its base rate is 1 - mean(y)."""
    out = {
        "auroc": float(roc_auc_score(y, scores)),
        "auprc_yes": float(average_precision_score(y, scores)),
        "auprc_no": float(average_precision_score(1 - y, -scores)),
        "base_yes": float(np.mean(y)),
        "base_no": float(1 - np.mean(y)),
        "n": int(len(y)),
    }
    out["lift_yes"] = out["auprc_yes"] / out["base_yes"]
    out["lift_no"] = out["auprc_no"] / out["base_no"]
    if thr is not None:
        pred = (scores >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0)
        out.update(precision=float(p), recall=float(r), f1=float(f1),
                   balanced_acc=float(balanced_accuracy_score(y, pred)),
                   threshold=float(thr))
    return out


def pick_threshold(scores, y):
    """Max balanced accuracy on VALIDATION -- not F1, because with a 75% base
    rate F1 on the majority class is maximised by predicting almost all YES."""
    best_thr, best = float(np.median(scores)), -1.0
    for t in np.quantile(scores, np.linspace(0.02, 0.98, 97)):
        b = balanced_accuracy_score(y, (scores >= t).astype(int))
        if b > best:
            best, best_thr = b, float(t)
    return best_thr


def train_one(name, dim, emb, tr_i, va_i, te_i, y, args, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    build = {"MotifCNN": lambda: MotifCNN(dim, pooling="maxattn"),
             "MeanPoolMLP": lambda: MeanPoolMLP(dim)}[name]
    model = build().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=args.wd)

    loaders = []
    for idx, shuf in ((tr_i, True), (va_i, False), (te_i, False)):
        ds = TileDataset(emb, idx, y[idx])
        loaders.append(torch.utils.data.DataLoader(
            ds, batch_size=args.batch, shuffle=shuf,
            num_workers=args.workers, pin_memory=(device == "cuda")))
    tr, va, te = loaders

    pw = torch.tensor((1 - y[tr_i]).sum() / max(y[tr_i].sum(), 1.0), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    for _ in range(args.fixed_epochs):
        run(model, tr, device, opt, loss_fn)

    _, vp, vy = run(model, va, device, None, loss_fn)
    thr = pick_threshold(1 / (1 + np.exp(-vp)), vy)
    _, tp, ty = run(model, te, device, None, loss_fn)
    res = metrics(1 / (1 + np.exp(-tp)), ty, thr)
    res.update(model=name, params=count_params(model))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--group-col", default="protein")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fixed-epochs", type=int, default=14)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--arms", nargs="+", default=["MotifCNN", "MeanPoolMLP"])
    ap.add_argument("--label-sets", nargs="+", default=list(LABEL_SETS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = json.load(open(args.emb + ".json"))["dim"]
    df = pd.read_csv(args.csv, low_memory=False)

    cl = pd.read_csv(args.clusters, sep="\t", header=None, names=["rep", "member"])
    cmap = dict(zip(cl["member"], cl["rep"]))
    groups_all = np.array([cmap.get(g, g) for g in df[args.group_col].astype(str)])

    rows = []
    for ls in args.label_sets:
        expr = LABEL_SETS[ls]
        yes = df.Hit_pEF == "Hit"
        no = df.Hit_pEF == "Not Hit"
        if expr:
            yes = yes & (df.Hit_FLAG == "Hit")
            no = no & (df.Hit_FLAG == "Hit")
        keep = np.where((yes | no).values)[0]
        y = yes.astype(np.float32).values
        g = groups_all[keep]
        print(f"\n######## {ls}: {len(keep):,} tiles | {int(yes.sum()):,} YES | "
              f"{int(no.sum()):,} NO | base YES {y[keep].mean():.4f} | "
              f"{len(np.unique(g)):,} clusters", flush=True)

        gkf = GroupKFold(n_splits=args.folds)
        for fold, (tr_rel, te_rel) in enumerate(gkf.split(keep, y[keep], g)):
            te_i = keep[te_rel]
            # Avg_pEF baseline: no training, so it is fold- and seed-free
            base = metrics(df.Avg_pEF.values[te_i], y[te_i])
            base.update(model="Avg_pEF", params=0, seed=-1, fold=fold, labels=ls)
            rows.append(base)
            print(f"  [{ls} fold {fold}] n_test {len(te_i):,} "
                  f"base_yes {base['base_yes']:.3f} | Avg_pEF AUROC "
                  f"{base['auroc']:.4f} AUPRC_no {base['auprc_no']:.4f} "
                  f"(lift {base['lift_no']:.2f}x)", flush=True)

            for seed in args.seeds:
                # carve validation out of the training groups, by group
                tr_groups = np.unique(g[tr_rel])
                rng = np.random.default_rng(1000 * seed + fold)
                rng.shuffle(tr_groups)
                n_va = max(1, int(0.15 * len(tr_groups)))
                va_set = set(tr_groups[:n_va])
                in_va = np.isin(g[tr_rel], list(va_set))
                va_i, tr_i = keep[tr_rel[in_va]], keep[tr_rel[~in_va]]

                for name in args.arms:
                    res = train_one(name, dim, args.emb, tr_i, va_i, te_i,
                                    y, args, seed, device)
                    res.update(seed=seed, fold=fold, labels=ls,
                               n_train=len(tr_i), n_val=len(va_i))
                    rows.append(res)
                    print(f"    {name:12s} seed {seed} AUROC {res['auroc']:.4f}  "
                          f"AUPRC_no {res['auprc_no']:.4f} (lift {res['lift_no']:.2f}x)  "
                          f"bal_acc {res['balanced_acc']:.4f}", flush=True)

        pd.DataFrame(rows).to_json(args.out, orient="records", indent=2)

    out = pd.DataFrame(rows)
    out.to_json(args.out, orient="records", indent=2)
    cols = ["auroc", "auprc_no", "lift_no", "auprc_yes", "balanced_acc"]
    print("\n=== mean over folds x seeds ===")
    print(out.groupby(["labels", "model"])[cols].mean().round(4).to_string())
    print("\n=== sd ===")
    print(out.groupby(["labels", "model"])[cols].std(ddof=1).round(4).to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

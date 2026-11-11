"""
Yes/no model: is this tile a confirmed repression domain?

Why this exists. The three arms reported so far are REGRESSION on Avg_pEF, so
they only have r / rho / MSE. "Accuracy" and "precision" are not defined for
them, and precision@50 is precision over a RANKING, not over a classifier's
decisions. The original report does have a yes/no arm (AUROC 0.882, AUPRC
0.563 at esm640) but only for mean-pooling, so MotifCNN has never been
measured on the task the deliverable actually is: hand a lab a shortlist.

Label definition is copied from the original report so the numbers compare:
  YES = Hit pEF == "Hit"                      (the paper's two-screen rule)
  NO  = not a hit AND Hit FLAG == "Hit"       (well-expressed non-hits only)
Dropping poorly-expressed non-hits matters: the paper warns low expression
produces false negatives, so keeping them would put real repressors in the
NO class. This yields 62,738 tiles, 6,566 YES -- a 1:8.6 ratio.

Every metric is reported on TRAIN, VAL and TEST. The train/test gap is the
point: MotifCNN has 105k parameters against ~45k training tiles whose
neighbours share 70 of 80 residues, so the effective sample is far smaller
than the tile count and overfitting is the first thing to rule out.

Threshold for the hard metrics (accuracy/precision/recall/F1) is chosen to
maximise F1 on VALIDATION, then applied unchanged to train and test. Picking
it on test would inflate every number.

Usage:
  python train_classify.py --emb esm2_480_res --csv data.csv \
      --clusters protein_clusters.tsv --seeds 0 1 2 --out cls
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support, accuracy_score)

from cnn_model import MotifCNN, MeanPoolMLP, AttentionPoolMLP, count_params
from train_cnn import TileDataset, grouped_split


def run(model, loader, device, opt=None, loss_fn=None):
    train = opt is not None
    model.train(train)
    tot, n, ps, ys = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            out = model(x, mask)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(y)
            n += len(y)
            ps.append(out.detach().cpu().numpy()); ys.append(y.detach().cpu().numpy())
    return tot / n, np.concatenate(ps), np.concatenate(ys)


def hard_metrics(logits, y, thr):
    pred = (1 / (1 + np.exp(-logits)) >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary",
                                                  zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision": p,
            "recall": r, "f1": f1}


def soft_metrics(logits, y):
    pr = 1 / (1 + np.exp(-logits))
    return {"auroc": roc_auc_score(y, pr), "auprc": average_precision_score(y, pr),
            "base_rate": float(np.mean(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--group-col", default="protein")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--symmetric", action="store_true",
                    help="apply the Hit_FLAG expression filter to YES as well "
                         "as NO, removing the Avg_FLAG shortcut")
    ap.add_argument("--fixed-epochs", type=int, default=0,
                    help="train exactly N epochs with no early stopping and "
                         "report the final-epoch model alongside the best-val "
                         "one; use this when comparing label definitions, so "
                         "early-stopping variance does not dominate")
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--wd", type=float, default=1e-2,
                    help="AdamW weight decay; 1e-2 is the pipeline default")
    ap.add_argument("--out", default="cls.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = json.load(open(args.emb + ".json"))["dim"]
    df = pd.read_csv(args.csv)

    # Label definition. The default reproduces the original report exactly:
    # the expression filter Hit_FLAG == "Hit" is applied to NO only, so YES
    # keeps 2,783 tiles that would have been dropped had they been non-hits.
    # That asymmetry is a shortcut -- Avg_FLAG alone separates the classes at
    # AUC 0.310, and 30.7% of YES tiles sit below the NO class's Avg_FLAG
    # floor. --symmetric applies the same filter to both classes (AUC 0.475).
    # Tightening YES removes those tiles from `keep` entirely; it cannot move
    # them into NO, which requires Hit_pEF != "Hit".
    if args.symmetric:
        yes = (df.Hit_pEF == "Hit") & (df.Hit_FLAG == "Hit")
    else:
        yes = (df.Hit_pEF == "Hit")
    no = (df.Hit_pEF != "Hit") & (df.Hit_FLAG == "Hit")
    label_tag = "symmetric" if args.symmetric else "asymmetric"
    keep = (yes | no).values
    y = yes.astype(np.float32).values
    print(f"label set [{label_tag}]: {int(keep.sum()):,} tiles | "
          f"{int(yes.sum()):,} YES | {int(no.sum()):,} NO | "
          f"ratio 1:{no.sum()/yes.sum():.1f}")

    groups = df[args.group_col].astype(str).values
    if args.clusters:
        cl = pd.read_csv(args.clusters, sep="\t", header=None, names=["rep", "member"])
        m = dict(zip(cl["member"], cl["rep"]))
        groups = np.array([m.get(g, g) for g in groups])
        print(f"splitting by {len(np.unique(groups)):,} homology clusters")
    else:
        print(f"splitting by {len(np.unique(groups)):,} proteins (paralogs leak)")

    rows = []
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        tr_i, va_i, te_i = grouped_split(groups, seed)
        tr_i, va_i, te_i = (i[keep[i]] for i in (tr_i, va_i, te_i))
        print(f"\nseed {seed}: train {len(tr_i):,} ({y[tr_i].mean()*100:.1f}% YES)  "
              f"val {len(va_i):,}  test {len(te_i):,} ({y[te_i].mean()*100:.1f}% YES)")

        loaders = []
        for idx, shuf in ((tr_i, True), (va_i, False), (te_i, False)):
            ds = TileDataset(args.emb, idx, y[idx])
            loaders.append(torch.utils.data.DataLoader(
                ds, batch_size=args.batch, shuffle=shuf,
                num_workers=args.workers, pin_memory=(device == "cuda")))
        tr, va, te = loaders
        # YES is rare, so weight it by the NO:YES ratio, as the report did
        pw = torch.tensor((1 - y[tr_i]).sum() / y[tr_i].sum(), device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

        arms = {"MeanPoolMLP": lambda: MeanPoolMLP(dim),
                "AttnPoolMLP": lambda: AttentionPoolMLP(dim),
                "MotifCNN": lambda: MotifCNN(dim, pooling="maxattn")}
        if args.arms:
            arms = {k: v for k, v in arms.items() if k in args.arms}

        for name, build in arms.items():
            torch.manual_seed(seed)
            model = build().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=args.wd)
            n_ep = args.fixed_epochs or args.epochs
            best, best_state, bad, best_ep = -np.inf, None, 0, 0
            traj = []
            for ep in range(1, n_ep + 1):
                run(model, tr, device, opt, loss_fn)
                _, vp, vy = run(model, va, device, None, loss_fn)
                auprc = average_precision_score(vy, 1 / (1 + np.exp(-vp)))
                traj.append(round(float(auprc), 5))
                if auprc > best + 1e-5:
                    best, bad, best_ep = auprc, 0, ep
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= args.patience and not args.fixed_epochs:
                        break
            final_state = {k: v.detach().clone()
                           for k, v in model.state_dict().items()}

            def evaluate(state, select, ep_used):
                model.load_state_dict(state)
                _, vp, vy = run(model, va, device, None, loss_fn)
                # threshold chosen on VALIDATION only
                best_thr, best_f1 = 0.5, -1
                for t in np.linspace(0.05, 0.95, 91):
                    f1 = hard_metrics(vp, vy, t)["f1"]
                    if f1 > best_f1:
                        best_f1, best_thr = f1, t
                res = {"model": name, "seed": seed, "wd": args.wd,
                       "labels": label_tag, "select": select,
                       "params": count_params(model), "epochs_run": len(traj),
                       "best_epoch": ep_used, "threshold": float(best_thr)}
                for split, loader in (("train", tr), ("val", va), ("test", te)):
                    _, lp, ly = run(model, loader, device, None, loss_fn)
                    for k, v in soft_metrics(lp, ly).items():
                        res[f"{split}_{k}"] = float(v)
                    for k, v in hard_metrics(lp, ly, best_thr).items():
                        res[f"{split}_{k}"] = float(v)
                res["val_traj"] = traj
                return res

            # with --fixed-epochs the final-epoch model is the comparable one;
            # best-val is kept beside it to show what the selection is worth
            variants = [(best_state, "best_val", best_ep)]
            if args.fixed_epochs:
                variants.append((final_state, f"fixed{n_ep}", n_ep))

            for state, select, ep_used in variants:
                res = evaluate(state, select, ep_used)
                rows.append(res)
                print(f"  {name:12s} [{select}] params={res['params']:>8,} "
                      f"thr={res['threshold']:.2f} ep={ep_used}")
                print(f"    AUROC  train {res['train_auroc']:.4f}  val {res['val_auroc']:.4f}  "
                      f"test {res['test_auroc']:.4f}   gap {res['train_auroc']-res['test_auroc']:+.4f}")
                print(f"    AUPRC  train {res['train_auprc']:.4f}  val {res['val_auprc']:.4f}  "
                      f"test {res['test_auprc']:.4f}   gap {res['train_auprc']-res['test_auprc']:+.4f}")
                print(f"    test: acc {res['test_accuracy']:.4f}  prec {res['test_precision']:.4f}  "
                      f"rec {res['test_recall']:.4f}  F1 {res['test_f1']:.4f}  "
                      f"(nen {res['test_base_rate']:.4f})")

    out = pd.DataFrame(rows)
    out.to_json(args.out, orient="records", indent=2)
    cols = ["test_auroc", "test_auprc", "test_accuracy", "test_precision",
            "test_recall", "test_f1"]
    g = ["model", "select"]
    print(f"\n=== mean over {out.seed.nunique()} seeds (test) ===")
    print(out.groupby(g)[cols].mean().round(4).to_string())
    print("\n=== sd over seeds (test) -- compare this to any effect claimed ===")
    print(out.groupby(g)[cols].std(ddof=1).round(4).to_string())
    print("\n=== overfitting: train minus test ===")
    for (m, sel), s in out.groupby(g):
        print(f"  {m:12s} [{sel:9s}] AUROC {(s.train_auroc-s.test_auroc).mean():+.4f}  "
              f"AUPRC {(s.train_auprc-s.test_auprc).mean():+.4f}  "
              f"F1 {(s.train_f1-s.test_f1).mean():+.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

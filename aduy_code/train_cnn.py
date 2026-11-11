"""
Train MotifCNN vs the existing mean-pool MLP on the SAME splits and seeds.

Why paired: the report's seed changes both the split and the init, so the
+-0.02 seed sd mixes two sources of variance. Comparing marginal means
across layouts is underpowered -- that is why 480 vs 640 looked ambiguous.
Here both models see an identical split per seed, so the per-seed DIFFERENCE
is the statistic, and its sd is much smaller than either model's sd alone.

Usage:
  python train_cnn.py --emb esm2_640_res --csv data.csv \
      --target Avg_pEF --group-col protein --seeds 0 1 2

Grouped split: pass --clusters clusters.tsv (mmseqs easy-cluster output,
two columns: representative, member) to split by homology cluster instead
of by protein. Expect test r to DROP -- that is the honest number, because
KLF4 in train / KLF7 in test is leakage through paralogy.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

from cnn_model import MotifCNN, MeanPoolMLP, AttentionPoolMLP, count_params


# ---------------------------------------------------------------- data

class TileDataset(torch.utils.data.Dataset):
    """Reads per-residue embeddings lazily from the memmap -- never loads 10 GB."""

    def __init__(self, emb_prefix, idx, y):
        meta = json.load(open(emb_prefix + ".json"))
        self.path, self.meta = emb_prefix + ".dat", meta
        self.shape = (meta["n"], meta["max_len"], meta["dim"])
        self.lens = np.load(emb_prefix + ".len.npy")
        self.idx = np.asarray(idx)
        self.y = np.asarray(y, dtype=np.float32)
        self._mm = None  # opened per worker, memmaps are not fork-safe

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.float16, mode="r", shape=self.shape)
        row = int(self.idx[i])
        x = torch.from_numpy(np.asarray(self._mm[row], dtype=np.float32))
        L = int(self.lens[row])
        mask = torch.zeros(self.shape[1], dtype=torch.bool)
        mask[:L] = True
        return x, mask, torch.tensor(self.y[i])


def grouped_split(groups, seed, frac=(0.70, 0.15, 0.15)):
    """70/15/15 by GROUP (protein or homology cluster), never by tile."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n1 = int(len(uniq) * frac[0])
    n2 = int(len(uniq) * (frac[0] + frac[1]))
    sets = (set(uniq[:n1]), set(uniq[n1:n2]), set(uniq[n2:]))
    return [np.where(np.isin(groups, list(s)))[0] for s in sets]


# ---------------------------------------------------------------- train

def run_epoch(model, loader, device, opt=None, loss_fn=None):
    train = opt is not None
    model.train(train)
    tot, n, preds, ys = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            out = model(x, mask)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item() * len(y)
            n += len(y)
            preds.append(out.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
    return tot / n, np.concatenate(preds), np.concatenate(ys)


def fit(model, tr, va, te, device, lr=1e-3, wd=1e-2, epochs=100, patience=10):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    best, best_state, bad, best_ep = np.inf, None, 0, 0

    for ep in range(1, epochs + 1):
        run_epoch(model, tr, device, opt, loss_fn)
        vl, vp, vy = run_epoch(model, va, device, None, loss_fn)
        if vl < best - 1e-5:
            best, bad, best_ep = vl, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    # Train r on the SAME frozen weights, so the train/test gap measures
    # overfitting rather than mixing in a different model state. Tiles
    # overlap 70/80 residues, so the effective sample size is far below the
    # tile count and this gap is the number to watch.
    trl, trp, trty = run_epoch(model, tr, device, None, loss_fn)
    _, vp, vy = run_epoch(model, va, device, None, loss_fn)
    tl, tp, ty = run_epoch(model, te, device, None, loss_fn)
    return {
        "train_r": float(pearsonr(trp, trty)[0]),
        "train_mse": float(trl),
        "val_r": float(pearsonr(vp, vy)[0]),
        "test_r": float(pearsonr(tp, ty)[0]),
        "test_spearman": float(spearmanr(tp, ty)[0]),
        "test_mse": float(tl),
        "best_epoch": best_ep,
    }, tp, ty


def shortlist_metrics(pred, idx, df, k=50, hit_col="Hit_pEF", score_col="Avg_pEF"):
    """The deliverable is a shortlist a lab would test, not a correlation.

    precision@k : of the k highest-predicted test tiles, how many are
                  confirmed hits? Scored against the Hit pEF label, which
                  comes from the higher-coverage confirmation screen, so a
                  noisy training target is being graded by a cleaner label.
    within-protein Spearman : can the model say WHERE in a protein it has
                  never seen the repression domain sits? That is the actual
                  question, and a global r can look fine while this is ~0.
    """
    sub = df.iloc[idx].copy()
    sub["pred"] = pred
    out = {}
    top = sub.nlargest(k, "pred")
    is_hit = (top[hit_col] == "Hit").sum()
    out[f"precision@{k}"] = float(is_hit / k)
    out["base_hit_rate"] = float((sub[hit_col] == "Hit").mean())

    # Same ranking, stricter grading. Hit_pEF == "Hit" alone counts 2,783
    # poorly-expressed tiles as confirmed hits; the paper warns low expression
    # produces false calls. Grading the SAME predictions both ways shows how
    # much of the shortlist claim rests on tiles that never expressed well.
    if "Hit_FLAG" in sub.columns:
        s_top = (top[hit_col] == "Hit") & (top["Hit_FLAG"] == "Hit")
        s_all = (sub[hit_col] == "Hit") & (sub["Hit_FLAG"] == "Hit")
        out[f"precision@{k}_strict"] = float(s_top.sum() / k)
        out["base_hit_rate_strict"] = float(s_all.mean())
        out[f"top{k}_hit_wellexpressed"] = (
            float(s_top.sum() / is_hit) if is_hit else float("nan"))

    rhos = []
    for _, g in sub.groupby("protein"):
        if len(g) >= 5 and g[score_col].nunique() > 1:
            rhos.append(spearmanr(g.pred, g[score_col])[0])
    rhos = [r for r in rhos if r == r]
    out["within_protein_spearman_median"] = float(np.median(rhos)) if rhos else float("nan")
    out["within_protein_n"] = len(rhos)
    # does it beat ranking by nothing? a protein-level null
    out["within_protein_frac_positive"] = (
        float(np.mean([r > 0 for r in rhos])) if rhos else float("nan"))
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="prefix from extract_per_residue.py")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="Avg_pEF")
    ap.add_argument("--group-col", default="protein",
                    help="column holding the protein identity (check your data.csv)")
    ap.add_argument("--clusters", default=None,
                    help="optional mmseqs TSV: rep<TAB>member, splits by cluster")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--proj-dim", type=int, default=64)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--kernels", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--pooling", default="maxattn",
                    choices=["max", "mean", "maxmean", "attn", "maxattn"],
                    help="pooling for MotifCNN; 'maxattn' = max-pool + learned attention")
    ap.add_argument("--attn-heads", type=int, default=1)
    ap.add_argument("--arms", nargs="+", default=None,
                    help="subset of: MeanPoolMLP AttnPoolMLP MotifCNN")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--topk", type=int, default=50,
                    help="k for precision@k on the test shortlist")
    ap.add_argument("--out", default="cnn_results.json")
    ap.add_argument("--deterministic", action="store_true",
                    help="pin cuDNN, cuBLAS and the DataLoader so a rerun of "
                         "the same seed reproduces bit for bit. Matters more "
                         "here than elsewhere because this arm early-stops: a "
                         "drifting trajectory stops at a different epoch, so "
                         "the drift compounds instead of averaging out.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(args.emb + ".json"))
    dim = meta["dim"]

    df = pd.read_csv(args.csv)
    if args.group_col not in df.columns:
        raise SystemExit(f"--group-col '{args.group_col}' not found. columns = {list(df.columns)}")
    keep = df[args.target].notna().values
    y_all = df[args.target].values.astype(np.float32)
    groups = df[args.group_col].astype(str).values

    if args.clusters:
        cl = pd.read_csv(args.clusters, sep="\t", header=None, names=["rep", "member"])
        m = dict(zip(cl["member"], cl["rep"]))
        missing = sum(g not in m for g in np.unique(groups))
        if missing:
            print(f"warning: {missing} proteins absent from cluster file, kept as own cluster")
        groups = np.array([m.get(g, g) for g in groups])
        print(f"splitting by {len(np.unique(groups)):,} homology clusters")
    else:
        print(f"splitting by {len(np.unique(groups)):,} proteins "
              f"(NOTE: paralogs leak -- rerun with --clusters)")

    if args.deterministic:
        import os
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise SystemExit(
                "--deterministic needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in the "
                "environment BEFORE python starts; cuBLAS reads it at CUDA init, "
                "so setting it here would be too late and silently do nothing.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("deterministic mode on", flush=True)

    rows = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if args.deterministic:
            torch.cuda.manual_seed_all(seed)
        tr_i, va_i, te_i = grouped_split(groups, seed)
        tr_i = tr_i[keep[tr_i]]
        va_i = va_i[keep[va_i]]
        te_i = te_i[keep[te_i]]
        print(f"\nseed {seed}: train {len(tr_i):,}  val {len(va_i):,}  test {len(te_i):,}")

        loaders = []
        for idx, shuf in ((tr_i, True), (va_i, False), (te_i, False)):
            ds = TileDataset(args.emb, idx, y_all[idx])
            kw = {}
            if args.deterministic:
                # The shuffle order otherwise draws from the global RNG, whose
                # state depends on how much randomness model construction
                # happened to consume; workers otherwise reseed from entropy.
                g = torch.Generator()
                g.manual_seed(seed)
                kw["generator"] = g
                kw["worker_init_fn"] = (
                    lambda wid, s=seed: np.random.seed(s * 1000 + wid))
            loaders.append(torch.utils.data.DataLoader(
                ds, batch_size=args.batch, shuffle=shuf,
                num_workers=args.workers, pin_memory=(device == "cuda"), **kw))
        tr, va, te = loaders

        arms = {
            # control: the report's existing head, same loader/split/seed
            "MeanPoolMLP": lambda: MeanPoolMLP(dim),
            # cheapest upgrade: learned pooling, no convolution
            "AttnPoolMLP": lambda: AttentionPoolMLP(dim, n_heads=args.attn_heads),
            # motif scanner + learned pooling
            "MotifCNN": lambda: MotifCNN(dim, proj_dim=args.proj_dim,
                                         channels=args.channels,
                                         kernels=tuple(args.kernels),
                                         pooling=args.pooling,
                                         attn_heads=args.attn_heads),
        }
        if args.arms:
            arms = {k: v for k, v in arms.items() if k in args.arms}

        for name, build in arms.items():
            torch.manual_seed(seed)
            model = build()
            res, tp, ty = fit(model, tr, va, te, device,
                              epochs=args.epochs, patience=args.patience)
            hit_col = "Hit_PGK" if args.target == "Avg_PGK" else "Hit_pEF"
            res.update(shortlist_metrics(tp, te_i, df, k=args.topk,
                                         hit_col=hit_col, score_col=args.target))
            res.update(model=name, seed=seed, params=count_params(model),
                       split="homology" if args.clusters else "protein")
            rows.append(res)
            print(f"  {name:12s} params={res['params']:>8,}  "
                  f"train r={res['train_r']:.3f}  val r={res['val_r']:.3f}  "
                  f"test r={res['test_r']:.3f}  "
                  f"gap={res['train_r']-res['test_r']:+.3f}  "
                  f"rho={res['test_spearman']:.3f}  ep={res['best_epoch']}")
            print(f"               precision@{args.topk}={res[f'precision@{args.topk}']:.3f} "
                  f"(base rate {res['base_hit_rate']:.3f})  "
                  f"within-protein rho={res['within_protein_spearman_median']:.3f} "
                  f"on {res['within_protein_n']} proteins "
                  f"({res['within_protein_frac_positive']*100:.0f}% positive)")

    out = pd.DataFrame(rows)
    out.to_json(args.out, orient="records", indent=2)

    print("\n=== mean over seeds ===")
    print(out.groupby("model")[["test_r", "test_spearman", "test_mse"]].agg(["mean", "std"]))

    # the statistic that actually matters: the PAIRED per-seed difference
    piv = out.pivot(index="seed", columns="model", values="test_r")
    if "MeanPoolMLP" in piv.columns:
        print("\n=== paired per-seed delta vs MeanPoolMLP control ===")
        for col in piv.columns:
            if col == "MeanPoolMLP":
                continue
            d = piv[col] - piv["MeanPoolMLP"]
            sd = d.std(ddof=1) if len(d) > 1 else float("nan")
            verdict = "REAL" if (len(d) > 1 and d.mean() > 2 * sd) else "not significant"
            print(f"  {col:14s} delta r = {d.mean():+.4f}  sd {sd:.4f}  "
                  f"per seed {list(np.round(d.values, 4))}  -> {verdict}")
        print("rule: only call it a gain if mean delta > 2x its sd across seeds.")


if __name__ == "__main__":
    main()

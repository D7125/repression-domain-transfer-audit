"""
Independent validation of the regression arm against the confirmation screen.

Why this exists. `Hit_pEF` is exactly `Avg_pEF_retest` thresholded -- one cut
reproduces 100.0% of the labels, AUROC 1.0000 -- so the confirmation screen's
real content is a CONTINUOUS second measurement of 8,722 tiles, not a binary
label. Binarising it threw that away, and pooling the 94,526 untested tiles
into the negative class turned what was left into "confirmed hit vs mostly
untested tile" (classify_confirmed.py). The asset the pipeline has been
wasting is an independent experimental replicate, and this is what it is for:
showing the model predicts the REPRODUCIBLE part of the signal rather than the
primary screen's own noise.

What is computed, all on the IDENTICAL set of tiles -- test-cluster tiles that
have a retest value -- because the confirmation screen is enriched for strong
candidates (median Avg_pEF 2.52 for Hit, 1.51 for Not Hit, -0.06 for the
untested bulk) and range restriction makes correlations on different subsets
incomparable:

  r(pred, Avg_pEF)         same screen the model was trained on
  r(pred, Avg_pEF_retest)  the independent replicate       <- the new number
  r(Avg_pEF, retest)       experimental reproducibility, the bar to read against

If r(pred, retest) tracks r(pred, Avg_pEF), the model is fitting signal. If it
falls away, the model is partly fitting primary-screen noise.

Noise ceilings are recomputed on that same restricted subset, never reused
from the all-tile figure (0.7567) in appendix A.5:
  rel_primary  = Spearman-Brown on r(R1_pEF, R2_pEF): 2r / (1 + r)
  ceil_primary = sqrt(rel_primary)
  rel_retest   = r(Avg_pEF, retest)^2 / rel_primary   -- assumes both measure
                 the same latent value with uncorrelated errors; stated, not
                 proven, so read ceil_retest as an estimate
  ceil_retest  = sqrt(rel_retest)

Test-set discipline, which is the whole point of the rerun: nothing selects on
the retest values or on the test clusters -- they are touched exactly once, at
the end.

Epoch choice runs in TWO passes, because early stopping picks the epoch per run
and its variance then contaminates the statistic: stopping epochs across three
seeds were 3, 12 and 3 for MotifCNN, and on this pipeline early-stopping
variance alone has produced effects comparable to those being measured. Fixed
epochs do not remove the choice, they move it -- from a per-run random variable
to one constant chosen in advance. The constant is chosen on validation:

  pass A  --traj-epochs E   train E epochs with no early stopping, record val
                            MSE per epoch, write the trajectory, and NEVER
                            build a test loader. Pick N* at the minimum of the
                            val curve averaged over seeds (per model, since the
                            optimum differs by architecture).
  pass B  --fixed-epochs N* train exactly N* epochs, then score test once.

Note for reading pass B: the headline comparison is PAIRED off one set of
predictions -- same model, two targets -- so a wrong N moves both correlations
the same way and the %-of-ceiling difference is robust to it. What needs the
fixed N is the absolute level of r(pred, retest).

Usage (do NOT use python -I: it drops the script dir and cnn_model fails):
  python retest_validate.py --emb ../run/esm2_480_res --csv ../run/data.csv \
      --clusters ../run/protein_clusters.tsv --seeds 0 1 2 \
      --traj-epochs 30 --out ../run/retest_traj.json          # pass A
  python retest_validate.py ... --fixed-epochs 6 \
      --out ../run/retest_validate_fixed.json                 # pass B
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

from cnn_model import MotifCNN, MeanPoolMLP, AttentionPoolMLP, count_params
from train_cnn import TileDataset, grouped_split, run_epoch


def loader_for(emb, idx, y, batch, workers, device, shuffle=False):
    ds = TileDataset(emb, idx, y[idx])
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
        pin_memory=(device == "cuda"))


def predict(model, loader, device):
    _, p, _ = run_epoch(model, loader, device, None, nn.MSELoss())
    return p


def reliability(r_split):
    """Spearman-Brown for a 2-item mean: Avg_pEF is the mean of R1 and R2."""
    return 2 * r_split / (1 + r_split)


def subset_ceilings(sub):
    """Noise ceilings on exactly the tiles being scored."""
    out = {}
    m = sub.R1_pEF.notna() & sub.R2_pEF.notna()
    r_split = float(pearsonr(sub.R1_pEF[m], sub.R2_pEF[m])[0])
    rel_p = reliability(r_split)
    r_pr = float(pearsonr(sub.Avg_pEF, sub.Avg_pEF_retest)[0])
    out["n_replicate_pairs"] = int(m.sum())
    out["r_R1_R2"] = r_split
    out["rel_primary"] = float(rel_p)
    out["ceil_primary"] = float(np.sqrt(rel_p))
    out["r_primary_retest"] = r_pr
    # both screens measure the same latent value => r = sqrt(rel1 * rel2)
    rel_r = min(1.0, r_pr ** 2 / rel_p)
    out["rel_retest_derived"] = float(rel_r)
    out["ceil_retest"] = float(np.sqrt(rel_r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--group-col", default="protein")
    ap.add_argument("--target", default="Avg_pEF")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--traj-epochs", type=int, default=0,
                    help="pass A: train this many epochs with no early "
                         "stopping, record the val MSE trajectory, and build "
                         "no test loader at all")
    ap.add_argument("--fixed-epochs", type=int, default=0,
                    help="pass B: train exactly this many epochs, no early "
                         "stopping, then score test once")
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--arms", nargs="+", default=["MotifCNN", "MeanPoolMLP"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = json.load(open(args.emb + ".json"))["dim"]
    df = pd.read_csv(args.csv, low_memory=False)
    y = df[args.target].astype(np.float32).values
    has_rt = df.Avg_pEF_retest.notna().values

    cl = pd.read_csv(args.clusters, sep="\t", header=None, names=["rep", "member"])
    cmap = dict(zip(cl["member"], cl["rep"]))
    groups = np.array([cmap.get(g, g) for g in df[args.group_col].astype(str)])
    print(f"{len(df):,} tiles | {has_rt.sum():,} with retest | "
          f"{len(np.unique(groups)):,} homology clusters", flush=True)

    builders = {"MotifCNN": lambda: MotifCNN(dim, pooling="maxattn"),
                "AttnPoolMLP": lambda: AttentionPoolMLP(dim),
                "MeanPoolMLP": lambda: MeanPoolMLP(dim)}

    traj_mode = args.traj_epochs > 0
    rows = []
    for seed in args.seeds:
        tr_i, va_i, te_i = grouped_split(groups, seed)
        tr = loader_for(args.emb, tr_i, y, args.batch, args.workers, device, True)
        va = loader_for(args.emb, va_i, y, args.batch, args.workers, device)

        if traj_mode:
            # pass A reads nothing downstream of validation: no test loader is
            # built, no ceiling is computed, no retest value is read
            print(f"\n=== seed {seed} [pass A, chon N tren val]: "
                  f"train {len(tr_i):,} val {len(va_i):,} ===", flush=True)
        else:
            te_rt = te_i[has_rt[te_i]]
            sub = df.iloc[te_rt]
            ceil = subset_ceilings(sub)
            print(f"\n=== seed {seed}: train {len(tr_i):,} val {len(va_i):,} "
                  f"test {len(te_i):,} (retest {len(te_rt):,}) ===")
            print(f"  tran nhieu tren dung tap nay: r(R1,R2) {ceil['r_R1_R2']:.4f}"
                  f" -> ceil_primary {ceil['ceil_primary']:.4f} | "
                  f"r(so cap, retest) {ceil['r_primary_retest']:.4f} -> "
                  f"ceil_retest {ceil['ceil_retest']:.4f}", flush=True)
            te_all = loader_for(args.emb, te_i, y, args.batch, args.workers, device)
            te_sub = loader_for(args.emb, te_rt, y, args.batch, args.workers, device)

        for name in args.arms:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = builders[name]().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                    weight_decay=args.wd)
            loss_fn = nn.MSELoss()

            if traj_mode:
                for ep in range(1, args.traj_epochs + 1):
                    run_epoch(model, tr, device, opt, loss_fn)
                    vl, _, _ = run_epoch(model, va, device, None, loss_fn)
                    rows.append({"model": name, "seed": seed, "epoch": ep,
                                 "val_mse": float(vl)})
                    print(f"  {name:12s} ep {ep:3d}  val MSE {vl:.5f}", flush=True)
                pd.DataFrame(rows).to_json(args.out, orient="records", indent=2)
                continue

            best, best_state, bad, best_ep = np.inf, None, 0, 0
            if args.fixed_epochs:
                # N is a constant chosen in advance on the pass-A val curves,
                # so it contributes no per-run variance
                for _ in range(args.fixed_epochs):
                    run_epoch(model, tr, device, opt, loss_fn)
                best_ep, select = args.fixed_epochs, f"fixed{args.fixed_epochs}"
            else:
                for ep in range(1, args.epochs + 1):
                    run_epoch(model, tr, device, opt, loss_fn)
                    vl, _, _ = run_epoch(model, va, device, None, loss_fn)
                    if vl < best - 1e-5:   # selection sees VAL MSE only
                        best, bad, best_ep = vl, 0, ep
                        best_state = {k: v.detach().clone()
                                      for k, v in model.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= args.patience:
                            break
                model.load_state_dict(best_state)
                select = "early_stop"

            # test clusters touched once, here
            p_all = predict(model, te_all, device)
            p_sub = predict(model, te_sub, device)
            prim = sub[args.target].values
            rt = sub.Avg_pEF_retest.values

            res = {"model": name, "seed": seed, "params": count_params(model),
                   "select": select, "best_epoch": best_ep, "n_test": len(te_i),
                   "n_test_retest": len(te_rt),
                   "r_pred_primary_alltest": float(pearsonr(p_all, y[te_i])[0]),
                   "r_pred_primary": float(pearsonr(p_sub, prim)[0]),
                   "r_pred_retest": float(pearsonr(p_sub, rt)[0]),
                   "rho_pred_retest": float(spearmanr(p_sub, rt)[0])}
            res.update(ceil)
            res["pct_ceil_primary"] = res["r_pred_primary"] / res["ceil_primary"]
            res["pct_ceil_retest"] = res["r_pred_retest"] / res["ceil_retest"]
            res["retest_minus_primary"] = res["r_pred_retest"] - res["r_pred_primary"]
            rows.append(res)
            print(f"  {name:12s} ep={best_ep:3d}  "
                  f"r(pred,so cap) all-test {res['r_pred_primary_alltest']:.4f} | "
                  f"tren tap retest {res['r_pred_primary']:.4f}  "
                  f"r(pred,retest) {res['r_pred_retest']:.4f}  "
                  f"hieu {res['retest_minus_primary']:+.4f}  "
                  f"({res['pct_ceil_retest']*100:.1f}% tran retest)", flush=True)
            pd.DataFrame(rows).to_json(args.out, orient="records", indent=2)

    out = pd.DataFrame(rows)
    out.to_json(args.out, orient="records", indent=2)

    if traj_mode:
        print("\n=== val MSE trung binh qua seed, theo epoch ===")
        piv = out.pivot_table(index="epoch", columns="model", values="val_mse")
        print(piv.round(5).to_string())
        print("\n=== N* = cuc tieu cua duong cong val trung binh ===")
        for m in piv.columns:
            n_star = int(piv[m].idxmin())
            per_seed = [int(g.loc[g.val_mse.idxmin(), "epoch"])
                        for _, g in out[out.model == m].groupby("seed")]
            print(f"  {m:12s} N* = {n_star:3d}  (val MSE {piv[m].min():.5f})"
                  f"  | cuc tieu theo tung seed: {per_seed}")
        print(f"\nwrote {args.out}")
        return

    cols = ["r_pred_primary_alltest", "r_pred_primary", "r_pred_retest",
            "retest_minus_primary", "pct_ceil_primary", "pct_ceil_retest"]
    print("\n=== mean over seeds ===")
    print(out.groupby("model")[cols].mean().round(4).to_string())
    print("\n=== sd over seeds ===")
    print(out.groupby("model")[cols].std(ddof=1).round(4).to_string())
    print("\n=== paired per-seed: r(pred,retest) - r(pred,so cap) ===")
    for m, s in out.groupby("model"):
        d = s.retest_minus_primary
        print(f"  {m:12s} {d.mean():+.4f}  sd {d.std(ddof=1):.4f}  "
              f"|hieu|/sd {abs(d.mean())/d.std(ddof=1):.1f}" if d.std(ddof=1) > 0
              else f"  {m:12s} {d.mean():+.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

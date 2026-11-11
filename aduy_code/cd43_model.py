"""
The missing arm: train and score the sequence model on the CD43 (dCas9) target.

Why this is the decisive run. The paper trains only on Avg_pEF (rTetR) and
validates against Avg_pEF_batchRetest -- which is the SAME assay (rTetR, pEF
reporter, K562, AAVS1), so what that validation licenses is "the model does not
fit one run's private noise", not "the model generalises across context".
Meanwhile rho(Avg_pEF, cd43) = 0.092 at tile level. Put together, those two
imply a ceiling on any rTetR-trained model used to pick a dCas9 effector, and
the paper never measures it. This script measures it, three ways on ONE set of
tiles (test-cluster tiles scored in both CD43 bioreps):

  r(pred_pEF,  Avg_pEF)   the published number, as a control
  r(pred_pEF,  cd43)      what an rTetR-trained ranker is worth at dCas9
  r(pred_cd43, cd43)      whether the sequence carries CD43 signal AT ALL
  r(Avg_pEF,   cd43)      the measurement-to-measurement bound to read against

Every correlation is also reported as a fraction of the target's own noise
ceiling, recomputed on that same subset (Spearman-Brown on the two biological
replicates), because CD43 is far noisier than pEF and raw coefficients across
the two targets are not comparable.

Two passes, as in retest_validate.py: --mode traj picks N on the validation
curve and builds no test loader; --mode fixed trains exactly N and scores test
once.

Usage:
  python cd43_model.py --mode traj  --epochs 20 --seeds 0 1 2 --out ../run/rev2/cd43_traj.json
  python cd43_model.py --mode fixed --n 7      --seeds 0 1 2 --out ../run/rev2/cd43_fixed.json
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

from cnn_model import MotifCNN, count_params
from train_cnn import TileDataset, grouped_split, run_epoch

ROOT = "/home/dmin/ai&stem_cell"
EMB = f"{ROOT}/run/esm2_480_res"


def loader_for(idx, y, batch, workers, device, shuffle=False, seed=None):
    """seed is not None => shuffle order and worker RNGs are pinned. Without it
    the shuffle draws from the global RNG, whose state depends on how much
    randomness model construction happened to consume."""
    ds = TileDataset(EMB, idx, y[idx])
    kw = {}
    if seed is not None:
        g = torch.Generator(); g.manual_seed(seed)
        kw["generator"] = g
        kw["worker_init_fn"] = lambda wid, s=seed: np.random.seed(s * 1000 + wid)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
        pin_memory=(device == "cuda"), **kw)


def sb(r):
    """Spearman-Brown reliability of a 2-replicate mean."""
    return 2 * r / (1 + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["traj", "fixed"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--deterministic", action="store_true",
                    help="pin cuDNN, cuBLAS and the DataLoader; needs "
                         "CUBLAS_WORKSPACE_CONFIG=:4096:8 already in the env")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.deterministic:
        import os
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise SystemExit("--deterministic needs CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                             "set BEFORE python starts; cuBLAS reads it at CUDA init.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("deterministic mode on", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = json.load(open(EMB + ".json"))["dim"]
    df = pd.read_csv(f"{ROOT}/run/rev2/master.csv", low_memory=False)

    y_pef = df.Avg_pEF.astype(np.float32).values
    y_cd43 = df.cd43.astype(np.float32).values
    has_cd43 = df.cd43.notna().values & df.cd43_br1.notna().values & df.cd43_br2.notna().values

    cl = pd.read_csv(f"{ROOT}/run/protein_clusters.tsv", sep="\t",
                     header=None, names=["rep", "member"])
    cmap = dict(zip(cl["member"], cl["rep"]))
    groups = np.array([cmap.get(g, g) for g in df.protein.astype(str)])
    print(f"{len(df):,} tiles | {has_cd43.sum():,} with CD43 in both bioreps | "
          f"{len(np.unique(groups)):,} homology clusters", flush=True)

    rows = []
    for seed in args.seeds:
        tr_i, va_i, te_i = grouped_split(groups, seed)
        # the CD43 arm can only train where the target exists
        tr_c = tr_i[has_cd43[tr_i]]
        va_c = va_i[has_cd43[va_i]]
        te_c = te_i[has_cd43[te_i]]

        if args.mode == "traj":
            print(f"\n=== seed {seed} [pass A]: train {len(tr_c):,} "
                  f"val {len(va_c):,} (no test loader built) ===", flush=True)
            tr = loader_for(tr_c, y_cd43, args.batch, args.workers, device, True,
                        seed=(seed if args.deterministic else None))
            va = loader_for(va_c, y_cd43, args.batch, args.workers, device,
                        seed=(seed if args.deterministic else None))
            torch.manual_seed(seed); np.random.seed(seed)
            if args.deterministic:
                torch.cuda.manual_seed_all(seed)
            model = MotifCNN(dim, pooling="maxattn").to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=args.wd)
            lf = nn.MSELoss()
            for ep in range(1, args.epochs + 1):
                run_epoch(model, tr, device, opt, lf)
                vl, _, _ = run_epoch(model, va, device, None, lf)
                rows.append({"target": "cd43", "seed": seed, "epoch": ep,
                             "val_mse": float(vl)})
                print(f"  cd43 ep {ep:3d}  val MSE {vl:.5f}", flush=True)
                pd.DataFrame(rows).to_json(args.out, orient="records", indent=2)
            continue

        sub = df.iloc[te_c]
        # ceilings on exactly the tiles being scored
        r_pef_reps = float(pearsonr(sub.R1_pEF, sub.R2_pEF)[0])
        r_cd43_reps = float(pearsonr(sub.cd43_br1, sub.cd43_br2)[0])
        ceil_pef = float(np.sqrt(sb(r_pef_reps)))
        ceil_cd43 = float(np.sqrt(sb(r_cd43_reps)))
        r_meas = float(pearsonr(sub.Avg_pEF, sub.cd43)[0])
        rho_meas = float(spearmanr(sub.Avg_pEF, sub.cd43)[0])
        print(f"\n=== seed {seed}: train {len(tr_i):,} val {len(va_i):,} "
              f"test {len(te_c):,} scored in both CD43 bioreps ===")
        print(f"  ceiling pEF  r(R1,R2)={r_pef_reps:.4f} -> {ceil_pef:.4f}")
        print(f"  ceiling CD43 r(BR1,BR2)={r_cd43_reps:.4f} -> {ceil_cd43:.4f}")
        print(f"  r(Avg_pEF, cd43) = {r_meas:.4f}   rho = {rho_meas:.4f}",
              flush=True)

        te_loader_pef = loader_for(te_c, y_pef, args.batch, args.workers, device)
        te_loader_cd = loader_for(te_c, y_cd43, args.batch, args.workers, device)

        preds = {}
        for tgt, y, tri, vai, te_loader in (
                ("pEF", y_pef, tr_i, va_i, te_loader_pef),
                ("cd43", y_cd43, tr_c, va_c, te_loader_cd)):
            tr = loader_for(tri, y, args.batch, args.workers, device, True,
                        seed=(seed if args.deterministic else None))
            torch.manual_seed(seed); np.random.seed(seed)
            if args.deterministic:
                torch.cuda.manual_seed_all(seed)
            model = MotifCNN(dim, pooling="maxattn").to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=args.wd)
            lf = nn.MSELoss()
            for _ in range(args.n):
                run_epoch(model, tr, device, opt, lf)
            _, p, _ = run_epoch(model, te_loader, device, None, lf)
            preds[tgt] = p
            print(f"  trained on {tgt}: done ({count_params(model):,} params)",
                  flush=True)

        res = {"seed": seed, "n_test": int(len(te_c)), "n_epochs": args.n,
               "r_pEF_reps": r_pef_reps, "r_cd43_reps": r_cd43_reps,
               "ceil_pEF": ceil_pef, "ceil_cd43": ceil_cd43,
               "r_meas_pEF_cd43": r_meas, "rho_meas_pEF_cd43": rho_meas}
        for tgt in ("pEF", "cd43"):
            p = preds[tgt]
            res[f"r_{tgt}model_pEF"] = float(pearsonr(p, sub.Avg_pEF)[0])
            res[f"r_{tgt}model_cd43"] = float(pearsonr(p, sub.cd43)[0])
            res[f"rho_{tgt}model_cd43"] = float(spearmanr(p, sub.cd43)[0])
        res["pct_ceil_pEFmodel_pEF"] = res["r_pEFmodel_pEF"] / ceil_pef
        res["pct_ceil_pEFmodel_cd43"] = res["r_pEFmodel_cd43"] / ceil_cd43
        res["pct_ceil_cd43model_cd43"] = res["r_cd43model_cd43"] / ceil_cd43
        rows.append(res)
        print(f"  r(pred_pEF , pEF ) {res['r_pEFmodel_pEF']:+.4f} "
              f"= {res['pct_ceil_pEFmodel_pEF']*100:5.1f}% of its ceiling")
        print(f"  r(pred_pEF , cd43) {res['r_pEFmodel_cd43']:+.4f} "
              f"= {res['pct_ceil_pEFmodel_cd43']*100:5.1f}% of CD43 ceiling")
        print(f"  r(pred_cd43, cd43) {res['r_cd43model_cd43']:+.4f} "
              f"= {res['pct_ceil_cd43model_cd43']*100:5.1f}% of CD43 ceiling",
              flush=True)
        pd.DataFrame(rows).to_json(args.out, orient="records", indent=2)

    out = pd.DataFrame(rows)
    out.to_json(args.out, orient="records", indent=2)
    if args.mode == "traj":
        piv = out.pivot_table(index="epoch", values="val_mse")
        print("\n=== val MSE averaged over seeds ===")
        print(piv.round(5).to_string())
        print(f"\nN* = {int(piv.val_mse.idxmin())}")
    else:
        cols = [c for c in out.columns if c.startswith(("r_", "rho_", "pct_"))]
        print("\n=== mean over seeds ===")
        print(out[cols].mean().round(4).to_string())
        print("\n=== sd over seeds ===")
        print(out[cols].std(ddof=1).round(4).to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

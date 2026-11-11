"""
B8: resolve the ceiling-ratio difference with seeds instead of leaving it open,
and state the result as an equivalence test rather than "not resolvable".

Section 3.2's headline is that the model reaches the same fraction of its
noise ceiling against the training screen (53.24%) and against the independent
retest (51.68%), difference -1.56 points at 1.25x the seed sd on THREE seeds.
Limitations says more seeds would settle it and no new data is needed. This
runs 12 and settles it, then reports TOST against a pre-stated equivalence
margin of +-5 percentage points of ceiling -- a margin chosen because the
paper's own claim is that the two ratios are interchangeable to within the
difference between fitting signal and fitting one screen's noise, and a 5-point
gap is well below the 24-point gap (76% -> 53%) that range restriction alone
produces in this pipeline.

Ceilings are MEASURED, not derived: the workbook carries both replicates of
the confirmation screen (R1/R2_pEF_batchRetest), so rel_retest comes from
Spearman-Brown directly. The derived value is reported alongside, since the
two disagree by 0.03 and the direction tells which assumption breaks.
"""
import argparse, json, os

# Must be set before torch initialises CUDA, or cuBLAS keeps its non-
# deterministic workspace and --deterministic silently does nothing.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import pearsonr, t as tdist

import sys
sys.path.insert(0, "/home/dmin/ai&stem_cell/cnn_AI&stemcell")
from cnn_model import MotifCNN
from train_cnn import TileDataset, grouped_split, run_epoch

ROOT = "/home/dmin/ai&stem_cell"
EMB = f"{ROOT}/run/esm2_480_res"
MARGIN = 5.0      # percentage points of ceiling, stated before the run
sb = lambda r: 2 * r / (1 + r)


def loader(idx, y, batch, workers, device, shuffle=False, seed=None):
    """seed is not None => the shuffle order and every worker's RNG are pinned.

    Without this the shuffle draws from the global RNG, whose state depends on
    how much randomness model construction happened to consume, so pinning only
    torch.manual_seed still drifts between runs of the same seed.
    """
    kw = {}
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        kw["generator"] = g
        kw["worker_init_fn"] = lambda wid: np.random.seed(seed * 1000 + wid)
    return torch.utils.data.DataLoader(
        TileDataset(EMB, idx, y[idx]), batch_size=batch, shuffle=shuffle,
        num_workers=workers, pin_memory=(device == "cuda"), **kw)


def pin_training(seed):
    """Everything that makes two runs of the same seed differ on a GPU."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def tost(diff, margin):
    """Two one-sided tests for |mean| < margin. Returns (p, CI90)."""
    n = len(diff)
    m, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
    se = sd / np.sqrt(n)
    t1 = (m + margin) / se          # H0: mean <= -margin
    t2 = (margin - m) / se          # H0: mean >= +margin
    p = max(tdist.sf(t1, n - 1), tdist.sf(t2, n - 1))
    half = tdist.ppf(0.95, n - 1) * se
    return m, sd, se, p, (m - half, m + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(range(12)))
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=f"{ROOT}/run/rev2/retest_seeds.json")
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
    dim = json.load(open(EMB + ".json"))["dim"]
    df = pd.read_csv(f"{ROOT}/run/rev2/master.csv", low_memory=False)
    y = df.Avg_pEF.astype(np.float32).values
    has_rt = df.Avg_pEF_retest.notna().values

    cl = pd.read_csv(f"{ROOT}/run/protein_clusters.tsv", sep="\t",
                     header=None, names=["rep", "member"])
    cmap = dict(zip(cl["member"], cl["rep"]))
    groups = np.array([cmap.get(g, g) for g in df.protein.astype(str)])

    rows = []
    for seed in args.seeds:
        tr_i, va_i, te_i = grouped_split(groups, seed)
        te_rt = te_i[has_rt[te_i]]
        sub = df.iloc[te_rt]

        r_prim_reps = float(pearsonr(sub.R1_pEF, sub.R2_pEF)[0])
        rp = sub[sub.R1_pEF_retest.notna() & sub.R2_pEF_retest.notna()]
        r_rt_reps = float(pearsonr(rp.R1_pEF_retest, rp.R2_pEF_retest)[0])
        ceil_p = float(np.sqrt(sb(r_prim_reps)))
        ceil_r_meas = float(np.sqrt(sb(r_rt_reps)))
        r_pr = float(pearsonr(sub.Avg_pEF, sub.Avg_pEF_retest)[0])
        ceil_r_der = float(np.sqrt(min(1.0, r_pr ** 2 / sb(r_prim_reps))))

        if args.deterministic:
            pin_training(seed)
            tr = loader(tr_i, y, args.batch, args.workers, device, True,
                        seed=seed)
            te = loader(te_rt, y, args.batch, args.workers, device)
        else:
            # legacy order, kept so the pre-patch run can still be reproduced
            # as closely as a non-deterministic run can be
            tr = loader(tr_i, y, args.batch, args.workers, device, True)
            te = loader(te_rt, y, args.batch, args.workers, device)
            torch.manual_seed(seed); np.random.seed(seed)
        model = MotifCNN(dim, pooling="maxattn").to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        lf = nn.MSELoss()
        for _ in range(args.n):
            run_epoch(model, tr, device, opt, lf)
        _, p, _ = run_epoch(model, te, device, None, lf)

        r_p = float(pearsonr(p, sub.Avg_pEF)[0])
        r_r = float(pearsonr(p, sub.Avg_pEF_retest)[0])
        row = dict(seed=seed, n_test_retest=int(len(te_rt)),
                   n_retest_reps=int(len(rp)),
                   r_prim_reps=r_prim_reps, r_retest_reps=r_rt_reps,
                   ceil_primary=ceil_p, ceil_retest_measured=ceil_r_meas,
                   ceil_retest_derived=ceil_r_der,
                   r_pred_primary=r_p, r_pred_retest=r_r,
                   pct_ceil_primary=100 * r_p / ceil_p,
                   pct_ceil_retest=100 * r_r / ceil_r_meas)
        row["diff_pct"] = row["pct_ceil_retest"] - row["pct_ceil_primary"]
        rows.append(row)
        print(f"  seed {seed:2d}  n={len(te_rt):5,}  ceil {ceil_p:.4f}/"
              f"{ceil_r_meas:.4f}  r {r_p:.4f}/{r_r:.4f}  "
              f"%ceil {row['pct_ceil_primary']:5.2f}/{row['pct_ceil_retest']:5.2f}"
              f"  diff {row['diff_pct']:+.2f}", flush=True)
        pd.DataFrame(rows).to_json(args.out, orient="records", indent=2)

    out = pd.DataFrame(rows)
    out.to_json(args.out, orient="records", indent=2)
    d = out.diff_pct.values
    m, sd, se, p, ci = tost(d, MARGIN)
    print("\n" + "=" * 70)
    print(f"seeds                      : {len(d)}")
    print(f"%ceiling vs training screen: {out.pct_ceil_primary.mean():.2f}"
          f" (sd {out.pct_ceil_primary.std(ddof=1):.2f})")
    print(f"%ceiling vs independent    : {out.pct_ceil_retest.mean():.2f}"
          f" (sd {out.pct_ceil_retest.std(ddof=1):.2f})")
    print(f"paired difference          : {m:+.2f} points (sd {sd:.2f}, "
          f"{abs(m)/sd:.2f}x sd, SE {se:.2f})")
    print(f"90% CI of the difference   : [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    print(f"TOST vs +-{MARGIN:.0f} points        : p = {p:.4g}"
          f"  -> {'EQUIVALENT' if p < 0.05 else 'not equivalent'}")
    print(f"measured retest ceiling    : {out.ceil_retest_measured.mean():.4f}"
          f" (sd {out.ceil_retest_measured.std(ddof=1):.4f})")
    print(f"derived retest ceiling     : {out.ceil_retest_derived.mean():.4f}"
          f" (sd {out.ceil_retest_derived.std(ddof=1):.4f})")
    # The paper quotes a two-sided test of "the difference is zero" alongside
    # the equivalence test, so print both from the same vector.
    tv = m / se
    print(f"two-sided t({len(d)-1})            : t = {tv:+.3f}, "
          f"p = {2 * tdist.sf(abs(tv), len(d) - 1):.4g}")
    print(f"negative in                : {int((d < 0).sum())}/{len(d)} seeds")
    print(f"\nwrote {args.out}")
    print("DONE_RETEST_SEEDS")


if __name__ == "__main__":
    main()

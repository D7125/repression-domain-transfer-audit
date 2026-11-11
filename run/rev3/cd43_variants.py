"""One runner for the three referee items that need retraining.

  M10 (Q2)  does the conclusion depend on the ESM-2 size? Only 35M (D=480) was
            ever run, yet the paper says "pooling is worth more than model
            size" -- a claim about a dimension never varied. --emb points at a
            different embedding tensor; run/rev3/esm2_640 is ESM-2 150M.
  M4  (Q2)  the CD43 labels average two bioreps whose hit calls agree only
            18.9% of the time. --label br1|br2 trains on ONE replicate and
            scores the other, which is the version that cannot be an artefact
            of averaging two differently-scaled replicates.
  M5  (Mock) does the homology threshold matter? --clusters points at a
            different MMseqs2 partition; run/rev3/clu/ has 20/30/40%, all made
            with the same -c 0.5 --cov-mode 1 so only identity varies.

The scored quantity is always the same: r on the held-out clusters, and that r
as a fraction of the target's own noise ceiling recomputed on the same tiles.
That is the only form in which numbers from different targets and different
splits are comparable.

Scoring the reverse direction too (--both-arms) costs one extra training run
per seed and keeps the 2x2 complete under every variant.

Run (deterministic needs the env var set before python starts):
  CUBLAS_WORKSPACE_CONFIG=:4096:8 ~/miniconda3/envs/machine_deep_learning/bin/python \
      cd43_variants.py --tag esm640 --emb ../esm2_640 --seeds 0 1 2 ... --deterministic
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr

ROOT = "/home/dmin/ai&stem_cell"
sys.path.insert(0, f"{ROOT}/cnn_AI&stemcell")
from cnn_model import MotifCNN                      # noqa: E402
from train_cnn import TileDataset, grouped_split, run_epoch   # noqa: E402


def sb(r):
    return 2 * r / (1 + r)


def ceil_of(r):
    """Noise ceiling = sqrt of the Spearman-Brown reliability of a 2-rep mean."""
    return float(np.sqrt(sb(r))) if r > 0 else float("nan")


def loader_for(emb, idx, y, batch, workers, device, shuffle=False, seed=None):
    ds = TileDataset(emb, idx, y[idx])
    # A final training batch of exactly 1 makes BatchNorm raise. Drop it ONLY in
    # that case: any other run keeps every tile, so numbers already computed
    # stay bit-identical instead of shifting under a blanket drop_last=True.
    kw = {"drop_last": True} if (shuffle and len(idx) % batch == 1) else {}
    if seed is not None:
        g = torch.Generator(); g.manual_seed(seed)
        kw["generator"] = g
        kw["worker_init_fn"] = lambda wid, s=seed: np.random.seed(s * 1000 + wid)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
        pin_memory=(device == "cuda"), **kw)


def train_one(emb, dim, tr, va_unused, y, batch, workers, device, seed, epochs,
              deterministic):
    torch.manual_seed(seed); np.random.seed(seed)
    if deterministic:
        torch.cuda.manual_seed_all(seed)
    model = MotifCNN(dim, pooling="maxattn").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lf = nn.MSELoss()
    ld = loader_for(emb, tr, y, batch, workers, device, True,
                    seed=(seed if deterministic else None))
    for _ in range(epochs):
        run_epoch(model, ld, device, opt, lf)
    return model


def predict(model, emb, idx, y, batch, workers, device):
    ld = loader_for(emb, idx, y, batch, workers, device)
    _, p, _ = run_epoch(model, ld, device, None, nn.MSELoss())
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="name for this variant")
    ap.add_argument("--emb", default=f"{ROOT}/run/esm2_480_res")
    ap.add_argument("--clusters", default=f"{ROOT}/run/protein_clusters.tsv")
    ap.add_argument("--label", default="avg", choices=["avg", "br1", "br2"],
                    help="CD43 target: the 2-rep mean, or one replicate alone")
    ap.add_argument("--score-against", default="same",
                    choices=["same", "other_rep"],
                    help="other_rep scores br1-trained models on br2 and vice versa")
    ap.add_argument("--restrict-hitboth", action="store_true",
                    help="keep only tiles called hit in BOTH CD43 replicates. "
                         "CONFOUNDED: it also cuts the training set 25x "
                         "(72,000 -> 2,800), so a low score measures "
                         "underfitting, not label quality. Use "
                         "--restrict-hitboth-test instead.")
    ap.add_argument("--restrict-hitboth-test", action="store_true",
                    help="train on ALL usable tiles, evaluate only on tiles "
                         "called hit in BOTH replicates. This isolates label "
                         "quality from training-set size.")
    ap.add_argument("--both-arms", action="store_true",
                    help="also train the pEF arm, to keep the 2x2 complete")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--n", type=int, default=10, help="epochs, fixed")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.deterministic:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise SystemExit("--deterministic needs CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                             "set BEFORE python starts")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = json.load(open(args.emb + ".json"))["dim"]
    df = pd.read_csv(f"{ROOT}/run/rev2/master.csv", low_memory=False)

    y_pef = df.Avg_pEF.astype(np.float32).values
    lab = {"avg": "cd43", "br1": "cd43_br1", "br2": "cd43_br2"}[args.label]
    y_cd43 = df[lab].astype(np.float32).values
    other = {"br1": "cd43_br2", "br2": "cd43_br1"}.get(args.label, "cd43")
    y_score = df[other].astype(np.float32).values \
        if args.score_against == "other_rep" else y_cd43

    have = np.array((df.cd43.notna() & df.cd43_br1.notna()
                     & df.cd43_br2.notna()).values, dtype=bool, copy=True)
    have &= np.isfinite(y_cd43) & np.isfinite(y_score)
    if args.restrict_hitboth:
        have &= df.cd43_hit_both.fillna(False).astype(bool).values
    # Evaluated-on mask, applied to the TEST fold only, so the training set keeps
    # its full size and the score answers "are the agreed-on labels easier to
    # predict?" rather than "what happens with 25x less training data?".
    eval_ok = np.ones(len(df), dtype=bool)
    if args.restrict_hitboth_test:
        eval_ok = np.array(df.cd43_hit_both.fillna(False).astype(bool).values,
                           dtype=bool, copy=True)

    cl = pd.read_csv(args.clusters, sep="\t", header=None, names=["rep", "member"])
    cmap = dict(zip(cl["member"], cl["rep"]))
    groups = np.array([cmap.get(g, g) for g in df.protein.astype(str)])

    print(f"tag={args.tag} emb=D{dim} label={args.label} "
          f"score={'other rep' if args.score_against=='other_rep' else 'same'} "
          f"hitboth={args.restrict_hitboth} "
          f"hitboth_test_only={args.restrict_hitboth_test}", flush=True)
    print(f"{len(df):,} tiles | {have.sum():,} usable | "
          f"{len(np.unique(groups)):,} clusters", flush=True)

    rows = []
    for seed in args.seeds:
        t0 = time.time()
        tr_i, va_i, te_i = grouped_split(groups, seed)
        tr_c, te_c = tr_i[have[tr_i]], te_i[have[te_i] & eval_ok[te_i]]
        if len(te_c) < 200:
            print(f"  seed {seed}: only {len(te_c)} test tiles, skipped", flush=True)
            continue

        m_cd43 = train_one(args.emb, dim, tr_c, None, y_cd43, args.batch,
                           args.workers, device, seed, args.n, args.deterministic)
        p_cd43 = predict(m_cd43, args.emb, te_c, y_score, args.batch,
                         args.workers, device)

        r1 = df.cd43_br1.values[te_c].astype(float)
        r2 = df.cd43_br2.values[te_c].astype(float)
        rep_cd43 = pearsonr(r1, r2)[0]
        # The ceiling depends on what is being scored. Against the 2-replicate
        # mean it is sqrt(Spearman-Brown reliability). Against ONE replicate it
        # is sqrt(r between replicates) -- a single draw carries its own noise,
        # so even a perfect predictor of the truth cannot exceed that.
        c_cd43 = (float(np.sqrt(rep_cd43)) if args.score_against == "other_rep"
                  else ceil_of(rep_cd43))
        rec = dict(tag=args.tag, seed=seed, n_test=len(te_c), dim=dim,
                   label=args.label, rep_cd43=float(rep_cd43), ceil_cd43=c_cd43,
                   r_cd43model_cd43=float(pearsonr(p_cd43, y_score[te_c])[0]))
        rec["pct_ceil_cd43model_cd43"] = rec["r_cd43model_cd43"] / c_cd43

        if args.both_arms:
            # The published run trains the pEF arm on EVERY training tile, not
            # only the CD43-covered ones: the pEF target exists everywhere, and
            # restricting it would handicap the arm the comparison is against.
            tr_p = tr_i[np.isfinite(y_pef[tr_i])]
            m_pef = train_one(args.emb, dim, tr_p, None, y_pef, args.batch,
                              args.workers, device, seed, args.n,
                              args.deterministic)
            p_pef = predict(m_pef, args.emb, te_c, y_pef, args.batch,
                            args.workers, device)
            q1 = df.R1_pEF.values[te_c].astype(float)
            q2 = df.R2_pEF.values[te_c].astype(float)
            ok = np.isfinite(q1) & np.isfinite(q2)
            c_pef = ceil_of(pearsonr(q1[ok], q2[ok])[0])
            rec["ceil_pEF"] = c_pef
            rec["r_pEFmodel_pEF"] = float(pearsonr(p_pef[ok], y_pef[te_c][ok])[0])
            rec["pct_ceil_pEFmodel_pEF"] = rec["r_pEFmodel_pEF"] / c_pef
            # the two off-diagonal cells
            p_pef_on_cd43 = predict(m_pef, args.emb, te_c, y_score, args.batch,
                                    args.workers, device)
            rec["r_pEFmodel_cd43"] = float(pearsonr(p_pef_on_cd43, y_score[te_c])[0])
            rec["pct_ceil_pEFmodel_cd43"] = rec["r_pEFmodel_cd43"] / c_cd43
            p_cd43_on_pef = predict(m_cd43, args.emb, te_c, y_pef, args.batch,
                                    args.workers, device)
            rec["r_cd43model_pEF"] = float(pearsonr(p_cd43_on_pef[ok],
                                                    y_pef[te_c][ok])[0])
            rec["pct_ceil_cd43model_pEF"] = rec["r_cd43model_pEF"] / c_pef

        rows.append(rec)
        msg = (f"  seed {seed}: n={len(te_c):,} ceil={c_cd43:.4f} "
               f"CD43->CD43 {100*rec['pct_ceil_cd43model_cd43']:.1f}%")
        if args.both_arms:
            msg += (f" | pEF->pEF {100*rec['pct_ceil_pEFmodel_pEF']:.1f}%"
                    f" pEF->CD43 {100*rec['pct_ceil_pEFmodel_cd43']:.1f}%"
                    f" CD43->pEF {100*rec['pct_ceil_cd43model_pEF']:.1f}%")
        print(msg + f"  [{time.time()-t0:.0f}s]", flush=True)
        out = args.out or f"{ROOT}/run/rev3/var_{args.tag}.json"
        json.dump(rows, open(out, "w"), indent=1)

    if rows:
        t = pd.DataFrame(rows)
        print(f"\n=== {args.tag}: mean over {len(t)} seeds ===")
        for c in [c for c in t.columns if c.startswith("pct_ceil")]:
            print(f"  {c:28s} {100*t[c].mean():6.2f}%  sd {100*t[c].std(ddof=1):5.2f}")
    print("DONE_CD43_VARIANTS")


if __name__ == "__main__":
    main()

"""
Does MotifCNN's attention land on the motifs we proved are real?

The patch doc's Task 0b wanted saliency validated against the paper's
deletion-scan "necessary sequences". That table is not in MOESM5, so that
route is closed. But there is an internal reference that costs nothing: the
three motifs that survived the composition-shuffled control --

    WRPW        (TLE)        absolute enrichment, 6 domains
    [VILMF]K.E  (SUMO)       5.0x real vs 1.3x shuffled, ~2800 hit tiles
    PL[DN]L.    (CtBP refined) 2.6x real vs 0.7x shuffled

If the model's attention peaks on those positions in tiles it never trained
on, it found real biology rather than a composition shortcut.

Design, and why each piece is needed:
  * tiles come from the TEST split only, so the model has never seen them
  * the statistic is the share of saliency mass inside the motif window,
    compared with the window's share of the tile (a 4-aa motif in an 80-aa
    tile is 5% by length, so 5% is the no-signal baseline)
  * two nulls, because beating 5% could happen for boring reasons:
      - position null: the same tiles, with the window slid to a random
        offset. Controls for attention having a positional bias, e.g.
        preferring the middle of the tile.
      - sequence null: hit tiles WITHOUT the motif, scored at the positions
        where the motif would match if it were there. Controls for the motif
        sitting in generally salient neighbourhoods.

Usage:
  python saliency_vs_motif.py --emb esm2_480_res --csv data.csv \
      --clusters protein_clusters.tsv --seed 0 --out saliency
"""

import argparse
import json
import re

import numpy as np
import pandas as pd
import torch

from cnn_model import MotifCNN
from train_cnn import TileDataset, grouped_split, fit

MOTIFS = {
    "WRPW": r"WRPW",
    "SUMO_psiKxE": r"[VILMF]K.E",
    "CtBP_refined": r"PL[DN]L.",
}


def saliency_mass_in_window(sal, start, end):
    """Share of (non-negative) saliency mass falling inside [start, end)."""
    s = np.clip(sal, 0, None)
    total = s.sum()
    if total <= 0:
        return np.nan
    return s[start:end].sum() / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--group-col", default="protein")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--out", default="saliency")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(args.emb + ".json"))
    dim, L = meta["dim"], meta["max_len"]

    df = pd.read_csv(args.csv)
    y = df.Avg_pEF.values.astype(np.float32)
    groups = df[args.group_col].astype(str).values
    if args.clusters:
        cl = pd.read_csv(args.clusters, sep="\t", header=None, names=["rep", "member"])
        m = dict(zip(cl["member"], cl["rep"]))
        groups = np.array([m.get(g, g) for g in groups])
        print(f"splitting by {len(np.unique(groups)):,} homology clusters")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tr_i, va_i, te_i = grouped_split(groups, args.seed)
    print(f"train {len(tr_i):,}  val {len(va_i):,}  test {len(te_i):,}")

    loaders = []
    for idx, shuf in ((tr_i, True), (va_i, False), (te_i, False)):
        ds = TileDataset(args.emb, idx, y[idx])
        loaders.append(torch.utils.data.DataLoader(
            ds, batch_size=args.batch, shuffle=shuf,
            num_workers=args.workers, pin_memory=(device == "cuda")))
    tr, va, te = loaders

    model = MotifCNN(dim, pooling="maxattn")
    res, _, _ = fit(model, tr, va, te, device,
                    epochs=args.epochs, patience=args.patience)
    print(f"MotifCNN trained: test r = {res['test_r']:.4f}  "
          f"(epoch {res['best_epoch']})")
    model.eval()

    # ---------------- saliency on test tiles ------------------------------
    mm = np.memmap(args.emb + ".dat", dtype=np.float16, mode="r",
                   shape=(meta["n"], L, dim))
    test = df.iloc[te_i]
    hits = test[test.Hit_pEF == "Hit"]
    print(f"\n{len(hits):,} hit tiles in the test split")

    rows = []
    for name, pat in MOTIFS.items():
        rx = re.compile(pat)
        with_motif, without = [], []
        for ridx, seq in zip(hits.index, hits.sequence):
            (with_motif if rx.search(seq) else without).append((ridx, seq))
        print(f"\n--- {name} ({pat}) ---")
        print(f"  test hit tiles with motif: {len(with_motif)} | without: {len(without)}")
        if len(with_motif) < 5:
            print("  too few to test; skipped")
            continue

        # Two ways to measure, because the motif can match more than once per
        # tile (psi-K-x-E averages 1.24 matches, up to 7). Reporting only the
        # first match understates the signal if attention lands on another
        # one; reporting the union raises both the observed mass AND the
        # baseline, so the two are not comparable and both are given.
        #   A: first match only, window = its 4 aa      -> baseline 4/80
        #   B: union of every match                      -> baseline union/80
        real, widths = [], []
        realB, widthsB = [], []
        for ridx, seq in with_motif:
            mt = rx.search(seq)
            x = torch.from_numpy(np.asarray(mm[ridx], dtype=np.float32))[None].to(device)
            msk = torch.ones(1, L, dtype=torch.bool, device=device)
            sal = model.residue_saliency(x, msk)[0].cpu().numpy()
            real.append(saliency_mass_in_window(sal, mt.start(), mt.end()))
            widths.append(mt.end() - mt.start())
            cov = np.zeros(L, bool)
            for m2 in rx.finditer(seq):
                cov[m2.start():m2.end()] = True
            sp = np.clip(sal, 0, None)
            tt = sp.sum()
            realB.append(sp[cov].sum() / tt if tt > 0 else np.nan)
            widthsB.append(int(cov.sum()))
        realB = np.array([v for v in realB if v == v]); wB = float(np.mean(widthsB))
        real = np.array([r for r in real if r == r])
        w = float(np.mean(widths))
        baseline = w / L
        print(f"  motif width {w:.1f} aa -> no-signal baseline {baseline*100:.2f}% of mass")
        print(f"  observed mean mass in window = {real.mean()*100:.2f}% "
              f"(median {np.median(real)*100:.2f}%)")
        print(f"  enrichment over length share = {real.mean()/baseline:.2f}x")
        print(f"  --- cach B: hop MOI match ---")
        print(f"  so match/tile: tb {np.mean([len(list(rx.finditer(sq))) for _, sq in with_motif]):.3f}")
        print(f"  hop cua so {wB:.2f} aa -> baseline {wB/L*100:.2f}%")
        print(f"  mass trong hop = {realB.mean()*100:.2f}% -> enrichment {realB.mean()/(wB/L):.2f}x")

        # Cache each tile's saliency once: the permutation below re-uses it,
        # so the model is run len(with_motif) times rather than
        # len(with_motif) * perms times.
        sal_cache = []
        for ridx, seq in with_motif:
            x = torch.from_numpy(np.asarray(mm[ridx], dtype=np.float32))[None].to(device)
            msk = torch.ones(1, L, dtype=torch.bool, device=device)
            sal_cache.append(model.residue_saliency(x, msk)[0].cpu().numpy())

        # null 1: same tiles, window slid to a random legal offset. Every
        # permutation must slide the window on EVERY tile and take the mean,
        # so the null has the same averaging as the observed statistic. (An
        # earlier version used a single tile per permutation, which inflated
        # the null's variance and made the p-value meaningless.)
        null_pos = np.empty(args.perms)
        wi = int(w)
        for p in range(args.perms):
            starts = rng.integers(0, L - wi, size=len(sal_cache))
            null_pos[p] = np.nanmean([
                saliency_mass_in_window(s, st, st + wi)
                for s, st in zip(sal_cache, starts)])
        p_pos = (1 + (null_pos >= real.mean()).sum()) / (args.perms + 1)

        # null 2: hit tiles without the motif, scored where it would match
        if without:
            sub = [without[i] for i in rng.choice(len(without),
                                                  min(len(with_motif), len(without)),
                                                  replace=False)]
            seq_null = []
            for ridx, seq in sub:
                x = torch.from_numpy(np.asarray(mm[ridx], dtype=np.float32))[None].to(device)
                msk = torch.ones(1, L, dtype=torch.bool, device=device)
                sal = model.residue_saliency(x, msk)[0].cpu().numpy()
                st = int(rng.integers(0, L - int(w)))
                seq_null.append(saliency_mass_in_window(sal, st, st + int(w)))
            seq_null = np.array([v for v in seq_null if v == v])
        else:
            seq_null = np.array([np.nan])

        print(f"  null 1 (window slid at random): mean {np.nanmean(null_pos)*100:.2f}%, "
              f"sd {np.nanstd(null_pos)*100:.2f}, p = {p_pos:.4f}")
        print(f"  null 2 (motif-free hit tiles):  mean {np.nanmean(seq_null)*100:.2f}%")

        # Per-tile comparison against null 2: no permutation needed, and it
        # uses the full distribution rather than two means.
        from scipy.stats import mannwhitneyu
        if len(seq_null) > 5 and np.isfinite(seq_null).any():
            u, p_mw = mannwhitneyu(real, seq_null[np.isfinite(seq_null)],
                                   alternative="greater")
            print(f"  Mann-Whitney, per tile vs null 2: p = {p_mw:.3e}  "
                  f"({len(real)} vs {int(np.isfinite(seq_null).sum())} tiles)")
        else:
            p_mw = float("nan")
        verdict = ("attention concentrates on the motif"
                   if p_pos < 0.05 and real.mean() > baseline
                   else "no concentration beyond position/length effects")
        print(f"  -> {verdict}")
        rows.append({"motif": name, "pattern": pat, "n_tiles": len(real),
                     "width": w, "baseline": baseline,
                     "mean_mass": float(real.mean()),
                     "enrichment": float(real.mean() / baseline),
                     "null_pos_mean": float(np.nanmean(null_pos)),
                     "p_position": float(p_pos), "p_mannwhitney": float(p_mw),
                     "null_seq_mean": float(np.nanmean(seq_null))})

    out = pd.DataFrame(rows)
    out.to_csv(f"{args.out}_motif_saliency.csv", index=False)
    print(f"\nwrote {args.out}_motif_saliency.csv")
    print(f"model test r = {res['test_r']:.4f}")


if __name__ == "__main__":
    main()

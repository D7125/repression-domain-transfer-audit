"""
B4: the two sanity checks the interpretability literature treats as mandatory.

Section 3.3 already runs two position nulls, which is more than most papers do.
What it does not run are the controls from Adebayo et al. (2018): a saliency
map must change when the model stops being the trained model. Two versions:

  label randomisation  -- retrain, identically, on a permutation of Avg_pEF.
                          The architecture, the ESM features and the training
                          procedure are unchanged; only the function being
                          learned is destroyed.
  weight randomisation -- score the SAME tiles with an untrained MotifCNN.

If motif-window saliency survives either, the map is reading the gradient
geometry or the ESM embedding rather than anything the model learned, and the
"attention lands on SUMO sites" claim has no support. Both controls are free:
no new data, one extra training run.

Reported for every model state: mass in the window by both counting
conventions (first match / union of matches), the length baseline, and the
sliding-window position null on the same tiles.
"""
import json, re, sys
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import pearsonr

sys.path.insert(0, "/home/dmin/ai&stem_cell/cnn_AI&stemcell")
from cnn_model import MotifCNN
from train_cnn import TileDataset, grouped_split, run_epoch

ROOT = "/home/dmin/ai&stem_cell"
EMB = f"{ROOT}/run/esm2_480_res"
PAT = re.compile(r"[VILMF]K.E")
N_EPOCH, SEED, PERMS = 7, 0, 2000


def mass(sal, start, end):
    s = np.clip(sal, 0, None); tot = s.sum()
    return s[start:end].sum() / tot if tot > 0 else np.nan


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(EMB + ".json")); dim, L = meta["dim"], meta["max_len"]
    df = pd.read_csv(f"{ROOT}/run/rev2/master.csv", low_memory=False)
    y = df.Avg_pEF.astype(np.float32).values
    cl = pd.read_csv(f"{ROOT}/run/protein_clusters.tsv", sep="\t",
                     header=None, names=["rep", "member"])
    cmap = dict(zip(cl["member"], cl["rep"]))
    groups = np.array([cmap.get(g, g) for g in df.protein.astype(str)])

    rng = np.random.default_rng(SEED)
    tr_i, va_i, te_i = grouped_split(groups, SEED)
    test = df.iloc[te_i]
    hits = test[test.Hit_pEF == "Hit"]
    with_motif = [(i, s) for i, s in zip(hits.index, hits.sequence)
                  if PAT.search(str(s))]
    print(f"test split {len(te_i):,} tiles | {len(hits):,} hits | "
          f"{len(with_motif)} carry {PAT.pattern}", flush=True)

    y_perm = y.copy()
    y_perm[tr_i] = rng.permutation(y[tr_i])      # only the training labels

    def build(state):
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = MotifCNN(dim, pooling="maxattn").to(device)
        if state == "random_init":
            return model, float("nan")
        target = y if state == "trained" else y_perm
        tr = torch.utils.data.DataLoader(
            TileDataset(EMB, tr_i, target[tr_i]), batch_size=256,
            shuffle=True, num_workers=2, pin_memory=(device == "cuda"))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        lf = nn.MSELoss()
        for _ in range(N_EPOCH):
            run_epoch(model, tr, device, opt, lf)
        te = torch.utils.data.DataLoader(
            TileDataset(EMB, te_i, y[te_i]), batch_size=256,
            num_workers=2, pin_memory=(device == "cuda"))
        _, p, _ = run_epoch(model, te, device, None, lf)
        return model, float(pearsonr(p, y[te_i])[0])

    mm = np.memmap(EMB + ".dat", dtype=np.float16, mode="r",
                   shape=(meta["n"], L, dim))
    rows = []
    for state in ("trained", "label_perm", "random_init"):
        model, test_r = build(state)
        model.eval()
        A, B, wB, sal_cache = [], [], [], []
        for ridx, seq in with_motif:
            x = torch.from_numpy(np.asarray(mm[ridx], dtype=np.float32))[None].to(device)
            msk = torch.ones(1, L, dtype=torch.bool, device=device)
            sal = model.residue_saliency(x, msk)[0].cpu().numpy()
            sal_cache.append(sal)
            m0 = PAT.search(seq)
            A.append(mass(sal, m0.start(), m0.end()))
            cov = np.zeros(L, bool)
            for m2 in PAT.finditer(seq):
                cov[m2.start():m2.end()] = True
            sp = np.clip(sal, 0, None); tot = sp.sum()
            B.append(sp[cov].sum() / tot if tot > 0 else np.nan)
            wB.append(int(cov.sum()))
        A = np.array([v for v in A if v == v]); B = np.array([v for v in B if v == v])
        wb = float(np.mean(wB)); baseA, baseB = 4 / L, wb / L
        null = np.empty(PERMS)
        for p_ in range(PERMS):
            st = rng.integers(0, L - 4, size=len(sal_cache))
            null[p_] = np.nanmean([mass(s, a, a + 4) for s, a in zip(sal_cache, st)])
        p_pos = (1 + (null >= A.mean()).sum()) / (PERMS + 1)
        row = dict(state=state, test_r=test_r, n_tiles=len(A),
                   massA=100 * A.mean(), baselineA=100 * baseA,
                   enrichA=A.mean() / baseA,
                   massB=100 * B.mean(), baselineB=100 * baseB,
                   enrichB=B.mean() / baseB,
                   null_pos=100 * null.mean(), p_position=float(p_pos))
        rows.append(row)
        print(f"\n--- {state} ---")
        print(f"  test r on real Avg_pEF : {test_r:.4f}")
        print(f"  mass A (first match)   : {row['massA']:.2f}%  "
              f"baseline {row['baselineA']:.2f}%  -> {row['enrichA']:.2f}x")
        print(f"  mass B (union)         : {row['massB']:.2f}%  "
              f"baseline {row['baselineB']:.2f}%  -> {row['enrichB']:.2f}x")
        print(f"  position null          : {row['null_pos']:.2f}%   p = {p_pos:.4f}",
              flush=True)
        pd.DataFrame(rows).to_csv(f"{ROOT}/run/rev2/saliency_sanity.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv(f"{ROOT}/run/rev2/saliency_sanity.csv", index=False)
    print("\n" + "=" * 68)
    print(out[["state", "test_r", "massA", "enrichA", "massB", "enrichB",
               "p_position"]].round(3).to_string(index=False))
    print(f"\nwrote {ROOT}/run/rev2/saliency_sanity.csv")


if __name__ == "__main__":
    main()

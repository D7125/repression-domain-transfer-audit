"""
Task 11: cluster on the embedding the CNN LEARNED, not on raw ESM.

The patch doc makes this conditional on Task 9 succeeding, which it did
(MotifCNN beats mean-pooling by +0.063, 5.4x its sd). The question this asks
is sharper than the Task 5 clustering: raw ESM is optimised for masked-token
prediction, so clusters found in it reflect whatever ESM encodes. The CNN's
pooled layer is optimised for the repression target itself, so a cluster there
is a group of domains the model treats alike *for predicting repression*.

What is taken: the vector entering the final head, i.e. after the conv bank
and the pooling. For pooling="maxattn" that is 4 kernels x 32 channels x
(max + 1 attention head) = 256 dimensions. A forward hook grabs it without
touching cnn_model.py.

Three comparisons, each answering a different objection:
  1. AMI(CNN clusters, raw-ESM clusters) -- did learning change the grouping?
  2. AMI(CNN clusters, Pfam domain groups) -- did it move toward or away from
     known protein families? Raw ESM scored 0.584 at group level, i.e. it was
     essentially a family classifier. Lower here = the CNN reorganised away
     from families; higher = it leaned on them harder.
  3. robustness spread across CNN clusters vs across ESM clusters -- the
     practical test. The target was Avg_pEF (strength), NOT robustness, so if
     CNN clusters separate robustness better than ESM clusters do, that is
     structure the model found without being told to.

Trained on the training split only; embeddings are then taken for every tile,
including train. That is fine here: the clustering is descriptive, not a
held-out performance claim.

Usage:
  python cnn_embed_cluster.py --emb esm2_480_res --csv data.csv \
      --clusters protein_clusters.tsv --domains best_domains.csv \
      --mmseqs dom_clu_cluster.tsv --annotated pfam_domains_annotated.csv \
      --out cnnclu
"""

import argparse
import ast
import json

import numpy as np
import pandas as pd
import torch

from cnn_model import MotifCNN
from train_cnn import TileDataset, grouped_split, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--domains", required=True)
    ap.add_argument("--mmseqs", required=True)
    ap.add_argument("--annotated", default=None)
    ap.add_argument("--group-col", default="protein")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-cluster-size", type=int, default=15)
    ap.add_argument("--out", default="cnnclu")
    args = ap.parse_args()

    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import adjusted_mutual_info_score as ami

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(args.emb + ".json"))
    dim, L, N = meta["dim"], meta["max_len"], meta["n"]

    df = pd.read_csv(args.csv)
    y = df.Avg_pEF.values.astype(np.float32)
    groups = df[args.group_col].astype(str).values
    if args.clusters:
        cl = pd.read_csv(args.clusters, sep="\t", header=None, names=["rep", "member"])
        m = dict(zip(cl["member"], cl["rep"]))
        groups = np.array([m.get(g, g) for g in groups])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tr_i, va_i, te_i = grouped_split(groups, args.seed)
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
    print(f"MotifCNN trained: test r = {res['test_r']:.4f} "
          f"(epoch {res['best_epoch']})")
    model.eval()

    # forward hook on the head's first Linear: its input is the pooled vector
    grabbed = {}

    def hook(mod, inp, out):
        grabbed["z"] = inp[0].detach().cpu().numpy()

    h = model.head[0].register_forward_hook(hook)

    mm = np.memmap(args.emb + ".dat", dtype=np.float16, mode="r", shape=(N, L, dim))
    Z = None
    with torch.no_grad():
        for s in range(0, len(df), 1024):
            e = min(s + 1024, len(df))
            x = torch.from_numpy(np.asarray(mm[s:e], dtype=np.float32)).to(device)
            msk = torch.ones(x.shape[0], L, dtype=torch.bool, device=device)
            model(x, msk)
            z = grabbed["z"]
            if Z is None:
                Z = np.zeros((len(df), z.shape[1]), dtype=np.float32)
            Z[s:e] = z
            if (s // 1024) % 25 == 0:
                print(f"  embedded {s:,}/{len(df):,}", flush=True)
    h.remove()
    print(f"CNN embedding: {Z.shape}")
    np.save(f"{args.out}_tile_embed.npy", Z)

    # tiles -> domains, using the paper's strongest-tile rule
    dom = pd.read_csv(args.domains)
    hits = df[(df.Hit_pEF == "Hit") & df.Avg_pEF.notna()]
    peak = {}
    for gene, sub in hits.groupby("protein"):
        for _, r in sub.iterrows():
            peak.setdefault(gene, []).append((int(r.tile_num), r.name, r.Avg_pEF))
    rows = []
    for _, d in dom.iterrows():
        cand = [(t, i, v) for t, i, v in peak.get(d.protein, [])
                if d.first_tile <= t <= d.last_tile]
        if not cand:
            rows.append(None)
            continue
        rows.append(max(cand, key=lambda c: c[2])[1])
    ok = [i for i, r in enumerate(rows) if r is not None]
    dom = dom.iloc[ok].reset_index(drop=True)
    X = np.vstack([Z[rows[i]] for i in ok])
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    print(f"domain vectors from CNN embedding: {X.shape}")

    lab = HDBSCAN(min_cluster_size=args.min_cluster_size,
                  cluster_selection_method="leaf", copy=True).fit_predict(X)
    dom["cnn_cluster"] = lab
    n_cl = len([c for c in set(lab) if c >= 0])
    print(f"\nCNN-embedding clustering: {n_cl} clusters, "
          f"{(lab < 0).sum():,} noise ({(lab < 0).mean()*100:.0f}%)")
    print(f"  raw-ESM clustering for comparison: 8 clusters, 28% noise")

    base = dom.pgk_hit.mean()
    both = dom[(dom.cnn_cluster >= 0) & (dom.cluster >= 0)]
    print(f"\n1. AMI(CNN clusters, raw-ESM clusters) = "
          f"{ami(both.cnn_cluster, both.cluster):.4f}  (on {len(both):,} domains)")

    if args.annotated:
        an = pd.read_csv(args.annotated)[["domain_id", "pfam_group"]]
        dd = dom.merge(an, on="domain_id", how="left")
        g = dd[(dd.cnn_cluster >= 0) & dd.pfam_group.notna()]
        if len(g) > 20:
            print(f"2. AMI(CNN clusters, Pfam GROUP) = "
                  f"{ami(g.pfam_group.astype(str), g.cnn_cluster):.4f}  "
                  f"(on {len(g):,}); raw ESM scored 0.5839")

    print(f"\n3. robustness spread (background {base:.4f})")

    def spread(col, name):
        s = dom[dom[col] >= 0].groupby(col).pgk_hit.agg(["size", "mean"])
        s = s[s["size"] >= 10]
        if len(s) < 2:
            print(f"   {name}: fewer than 2 clusters of >=10, cannot compare")
            return
        print(f"   {name}: {len(s)} clusters >=10 domains, robustness "
              f"{s['mean'].min():.2f} to {s['mean'].max():.2f}, "
              f"sd {s['mean'].std():.4f}")
        return s

    s_cnn = spread("cnn_cluster", "CNN embedding")
    s_esm = spread("cluster", "raw ESM      ")
    print("   The target was Avg_pEF (strength), not robustness. A wider")
    print("   robustness spread in CNN clusters is structure found unasked.")

    if s_cnn is not None:
        print("\nCNN clusters, largest first:")
        tbl = dom[dom.cnn_cluster >= 0].groupby("cnn_cluster").agg(
            n=("pgk_hit", "size"), robust=("pgk_hit", "mean"),
            pEF=("max_pEF", "median"), tiles=("n_tiles", "mean"))
        tbl["example"] = [", ".join(dom[dom.cnn_cluster == c].protein.head(3))
                          for c in tbl.index]
        print(tbl.sort_values("n", ascending=False).head(12).round(3).to_string())

    dom.to_csv(f"{args.out}_domains.csv", index=False)
    print(f"\nwrote {args.out}_domains.csv, {args.out}_tile_embed.npy")
    print(f"model test r = {res['test_r']:.4f}")


if __name__ == "__main__":
    main()

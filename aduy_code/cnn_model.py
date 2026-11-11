"""
Motif-scanning CNN head for per-residue ESM-2 embeddings.

Design rationale (repression biology, DelRosso et al. 2023):
repression domains are driven by SHORT linear motifs --
  TLE-binding   WRPW        (4 aa)
  CtBP-binding  PL[DN]Lx    (5 aa)
  HP1-binding   PxVxL       (5 aa)
  SUMOylation   psi-K-x-E   (4 aa)
Mean-pooling over 80 residues dilutes a 4-residue signal ~20x.
A Conv1d bank followed by max-pooling is literally a learned motif detector:
each channel fires on its best-matching window anywhere in the tile.

Input : [B, L, D] per-residue embeddings (L = 80, D = 320/480/640/1280)
Output: [B] scalar  (regression)  or logit (classification)
"""

import torch
import torch.nn as nn


NEG_INF = -1e4  # fp16-safe stand-in for -inf


def masked_max(x, mask):
    """x: [B, C, L]  mask: [B, L] bool (True = real residue) -> [B, C]"""
    x = x.masked_fill(~mask.unsqueeze(1), NEG_INF)
    return x.max(dim=2).values


def masked_mean(x, mask):
    """x: [B, C, L]  mask: [B, L] bool -> [B, C]"""
    m = mask.unsqueeze(1).to(x.dtype)
    return (x * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)


class AttnPool(nn.Module):
    """
    Learned attention pooling: one query vector scores every residue, softmax
    over positions, then a weighted sum.

    Difference from max/mean pooling:
      mean  -> every residue weighted 1/L, a 4-aa motif diluted ~20x
      max   -> one winning position per channel, hard and non-differentiable
      attn  -> soft, learned weights; the SAME weights are the explanation,
               so the saliency map is the model, not a post-hoc estimate.

    Returns (pooled [B, C], weights [B, L]). Keep the weights -- plotting them
    against the paper's deletion-scan "necessary sequences" is the figure that
    shows the model found real biology.
    """

    def __init__(self, dim, hidden=None, n_heads=1, dropout=0.0):
        super().__init__()
        hidden = hidden or dim
        self.n_heads = n_heads
        self.score = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_heads, bias=False),
        )

    def forward(self, x, mask):
        """x: [B, L, C]  mask: [B, L] bool -> ([B, C*n_heads], [B, L] or [B, H, L])"""
        s = self.score(x)                                     # [B, L, H]
        s = s.masked_fill(~mask.unsqueeze(-1), NEG_INF)
        w = torch.softmax(s, dim=1)                           # over POSITIONS
        pooled = torch.einsum("blh,blc->bhc", w, x)           # [B, H, C]
        pooled = pooled.flatten(1)                            # [B, H*C]
        wout = w.squeeze(-1) if self.n_heads == 1 else w.permute(0, 2, 1)
        return pooled, wout


class AttentionPoolMLP(nn.Module):
    """
    Attention pooling with NO convolution -- the cheapest possible upgrade to
    the existing pipeline. Barely more parameters than MeanPoolMLP, but the
    pooling weights are learned instead of uniform.

    Run this as its own arm: if it already recovers most of the CNN's gain,
    the story is "pooling was the bottleneck", which is a cleaner result than
    "we added a bigger architecture".
    """

    def __init__(self, in_dim, attn_hidden=64, n_heads=1, head_units=64, dropout=0.2):
        super().__init__()
        self.pool = AttnPool(in_dim, hidden=attn_hidden, n_heads=n_heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(in_dim * n_heads, head_units),
            nn.BatchNorm1d(head_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_units, 1),
        )
        self.last_weights = None

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        pooled, w = self.pool(x, mask)
        self.last_weights = w.detach()
        return self.head(pooled).squeeze(-1)


class MotifCNN(nn.Module):
    def __init__(
        self,
        in_dim,                      # 320 / 480 / 640 / 1280
        proj_dim=64,                 # bottleneck before convolution
        channels=32,                 # filters PER kernel width
        kernels=(3, 5, 7, 9),        # window sizes, in amino acids
        head_units=64,
        dropout=0.3,
        pooling="maxmean",           # "max"|"mean"|"maxmean"|"attn"|"maxattn"
        attn_heads=1,
    ):
        super().__init__()
        self.pooling = pooling
        if pooling not in ("max", "mean", "maxmean", "attn", "maxattn"):
            raise ValueError(f"unknown pooling '{pooling}'")

        # 1) squeeze the big ESM dimension down before convolving.
        #    LayerNorm here replaces the per-feature standardisation the
        #    mean-pooled pipeline did with the training set's mean/sd.
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2) parallel conv bank -- several motif widths at once.
        #    padding = k // 2 with odd k keeps length L unchanged.
        self.convs = nn.ModuleList(
            [nn.Conv1d(proj_dim, channels, kernel_size=k, padding=k // 2) for k in kernels]
        )
        self.conv_act = nn.GELU()

        total_c = len(kernels) * channels
        self.attn = None
        if pooling in ("attn", "maxattn"):
            self.attn = AttnPool(total_c, hidden=max(32, total_c // 2),
                                 n_heads=attn_heads, dropout=dropout)
        mult = {"max": 1, "mean": 1, "maxmean": 2,
                "attn": attn_heads, "maxattn": 1 + attn_heads}[pooling]
        n_pooled = total_c * mult

        # 3) same shape of head as the existing MLP, for a fair comparison.
        self.head = nn.Sequential(
            nn.Linear(n_pooled, head_units),
            nn.BatchNorm1d(head_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_units, 1),
        )

    def forward(self, x, mask=None):
        """
        x    : [B, L, D] float
        mask : [B, L] bool, True = real residue. None -> all residues real.
        returns [B] (raw score; apply sigmoid outside for classification)
        """
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)

        h = self.proj(x)                      # [B, L, P]
        h = h * mask.unsqueeze(-1)            # zero out padding before conv
        h = h.transpose(1, 2)                 # [B, P, L]

        convs = torch.cat([self.conv_act(conv(h)) for conv in self.convs], dim=1)  # [B, totalC, L]

        feats = []
        if self.pooling in ("max", "maxmean", "maxattn"):
            feats.append(masked_max(convs, mask))
        if self.pooling in ("mean", "maxmean"):
            feats.append(masked_mean(convs, mask))
        if self.attn is not None:
            pooled, w = self.attn(convs.transpose(1, 2), mask)   # [B, L, totalC]
            self.last_weights = w.detach()
            feats.append(pooled)

        z = torch.cat(feats, dim=1)
        return self.head(z).squeeze(-1)

    @torch.no_grad()
    def residue_saliency(self, x, mask=None):
        """
        Which positions drove the prediction?
        Returns [B, L]: for every residue, the summed activation of the
        conv channels that actually won their max-pool at that position.
        Use it to check the model peaks on known-necessary sequences
        (e.g. the SUMOylation site in SP3's RD, the WRPW in HES family).
        """
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        h = self.proj(x) * mask.unsqueeze(-1)
        h = h.transpose(1, 2)
        sal = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        for conv in self.convs:
            c = self.conv_act(conv(h))                       # [B, C, L]
            c = c.masked_fill(~mask.unsqueeze(1), NEG_INF)
            argmax = c.argmax(dim=2)                         # [B, C]
            peak = c.max(dim=2).values.clamp(min=0)          # [B, C]
            sal.scatter_add_(1, argmax, peak)
        return sal


class MeanPoolMLP(nn.Module):
    """
    Reimplementation of the report's existing head, taking per-residue input
    and mean-pooling it internally. Keep this as the control arm: it must be
    trained on the SAME splits, seeds and loader as MotifCNN, otherwise the
    comparison is confounded.
    """

    def __init__(self, in_dim, hidden=(64, 64), dropout=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, mask=None):
        if mask is None:
            pooled = x.mean(dim=1)
        else:
            m = mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.net(pooled).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for dim in (320, 480, 640, 1280):
        m = MotifCNN(dim)
        print(f"MotifCNN  in_dim={dim:5d}  params={count_params(m):,}")
    print(f"MeanPoolMLP in_dim=640  params={count_params(MeanPoolMLP(640)):,}")

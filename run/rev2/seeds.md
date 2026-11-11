# Seeds, splits and determinism

Every split is 70/15/15 over the 1,013 MMseqs2 protein-level clusters, drawn by
`numpy.random.default_rng(seed)` in `train_cnn.grouped_split`. The same seed
also initialises the model, so arms compared with each other always share a
seed and the statistic is the per-seed difference.

## What "deterministic" means here

Four things pinned together; any one alone is not enough:

```python
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # BEFORE importing torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
g = torch.Generator(); g.manual_seed(seed)
DataLoader(..., generator=g, worker_init_fn=lambda w: np.random.seed(seed*1000+w))
```

The DataLoader half is the one people forget: shuffle order draws from the
global RNG, whose state depends on how much randomness model construction
happened to consume, so pinning only `torch.manual_seed` still drifts.

Each deterministic arm is verified by rerunning seed 0 alone and comparing every
column. `benchmark = False` disables conv autotuning, so deterministic runs are
slower.

## Arms

| Arm | Script | Seeds | Epochs | Deterministic | Seed-0 recheck |
|---|---|---|---|---|---|
| Regression, three architectures | `train_cnn.py --deterministic` | 0–9 | early stopping | yes | `cnn_det_verify0.json`, max abs diff 0 |
| Retest validation + TOST | `run/rev2/retest_seeds.py --deterministic` | 0–9 | 7 | yes | `retest_det_verify0.json`, max abs diff 0 |
| CD43 context transfer, pass A | `cd43_model.py --mode traj --deterministic` | 0–9 | 15 | yes | — (curve only) |
| CD43 context transfer, pass B | `cd43_model.py --mode fixed --deterministic` | 0–9 | 10 | yes | `cd43_fixed_verify0.json`, max abs diff 0 |
| precision@k + prediction dump | `run/rev2/cd43_topk2.py --deterministic` | 0–9 | 10 | yes | within `cd43_topk3.log` |
| Saliency randomisation controls | `run/rev2/saliency_seeds.py --deterministic` | 0–9 | 7 | yes | `saliency_det_verify0.csv`, max abs diff 0 |
| De-noising test | `run/rev2/denoise_test.py` | 0–9 | — | n/a, reads the dump | — |
| Composition baselines (ridge) | `run/rev2/composition_baseline.py --seeds 0..9` | 0–9 | closed form | n/a, no GPU | — |
| Classification on measured negatives | `classify_confirmed.py` | 0, 1, 2 | 14 | **no** | — |
| Saliency, four-run history | `saliency_vs_motif.py` | 0, 1, 2 + fixed-epoch run | 7 / early stop | **no** | — |
| Weight-decay sweep | `train_classify.py` | 0, 1 | 14 | **no** | — |

The three arms marked **no** are read as history or as sweeps below the noise
floor; no number carrying a conclusion in the report comes from them. This is
stated in the report's data-and-code table rather than left implicit.

## Pre-patch files kept for comparison

Deterministic runs did not overwrite the originals. Where both exist, the report
quotes the deterministic one and names the other:

| Deterministic | Pre-patch | Changed the conclusion? |
|---|---|---|
| `cnn_det_10seed.json` | `cnn_seed*.json` | no (direction held, magnitude 5.4x → 4.0x sd) |
| `retest_det10_final.json` | `retest_seeds_final.json` | **yes** — TOST at the pre-stated ±5 margin went p = 0.024 (equivalent) to p = 0.096 (not established) |
| `cd43_fixed_det10_final.json` | `cd43_fixed_final.json` (3 seeds) | no, but 3.7x sd became 1.9x sd on the model-vs-measurement difference |
| `cd43_topk3.csv` | `cd43_topk.csv` (2 seeds) | **yes** — the CD43 model leads at every k; the 2-seed table had the measurement ahead at k = 200 and 500 |
| `saliency_det10.csv` | `saliency_seeds_all.csv` | no, strengthened; the seed-0 control outlier (16.92%) was a non-determinism artifact and is gone |
| `denoise_test_10seed.csv` | `denoise_test.csv` (2 seeds) | no |
| `composition_baseline_10seed.csv` | `composition_baseline.csv` (3 seeds) | no |

Weight decay is 1e-2 and learning rate 1e-3 (AdamW) everywhere unless the arm
is the weight-decay sweep itself. The permutation nulls in `saliency_vs_motif.py`,
`run/rev2/saliency_sanity.py` and `run/rev2/saliency_seeds.py` use 2,000 draws
from `default_rng(seed)`, which is deterministic independently of the GPU.

# Repression-domain cross-context transfer audit

Analysis code for a re-analysis of two published high-throughput screens of
human transcriptional repression domains. It tests whether sequence models
trained on one recruitment assay (rTetR) transfer to another (dCas9 / CD43),
and whether motif, domain-family and promoter-robustness annotations carry
predictive information beyond amino-acid composition.

No new experimental data are generated here. Everything is computed from the
supplementary tables of the two source papers.

## Source data (not in this repository)

Download from the original publications and place under `data/`:

- DelRosso N, Tycko J, Suzuki P, et al. Large-scale mapping and mutagenesis of
  human transcriptional effector domains. *Nature* 616, 365–372 (2023).
  Supplementary file MOESM5.
- Tycko J, DelRosso N, Hess GT, et al. Development of compact transcriptional
  effectors using high-throughput measurements in diverse contexts.
  *Nature Biotechnology* 43, 1525–1538 (2025). Tables S1 and S2.

External resources: the `Pfam-A.hmm` profile database (see the note in
`environment.yml` on which file was used) and the ESM-2 checkpoint
`esm2_t12_35M_UR50D`, fetched automatically by `fair-esm`.

## Environment

```bash
conda env create -f environment.yml      # creates `machine_deep_learning`
conda activate machine_deep_learning
```

MMseqs2 and HMMER live in a separate env (`mmseqs`); versions are listed at
the bottom of `environment.yml`. Training was run on a single 8 GB GPU.

## Layout

| Path | Contents |
|---|---|
| `aduy_code/` | Data preparation, ESM-2 embedding extraction, the CNN and its training loop, and the original analyses (motif controls, saliency, variance decomposition, classification, retest validation) |
| `run/rev2/` | Deterministic 10-seed reruns, TOST, precision@k, saliency controls, composition baseline, attenuation correction, and `paper_numbers.py`, which recomputes every number quoted in the paper in one place |
| `run/rev3/` | Robustness round: label-definition and threshold variants, clustering-identity variants, larger ESM-2 model, and the analytic transfer bound |
| `run/effector2/` | Exact-sequence join of tiles to the CD43 screen and per-replicate hit thresholds |
| `figures/` | `make_paper_figures.py`, `make_fig_topk.py` |
| `run/**/*.json`, `*.csv` | Saved results, so the reported numbers can be checked without retraining |

## Reproducing the results

Order of operations, starting from the downloaded source tables:

1. `aduy_code/predata_v2.py`, `aduy_code/prepare_data.py` — build the tile table.
2. `aduy_code/make_fasta.py`, then MMseqs2 clustering at 30 % identity — protein-level clusters used for every train/test split.
3. `aduy_code/extract_per_residue.py` — ESM-2 per-residue embeddings (about 10 GB on disk; not stored here).
4. `aduy_code/build_master.py` — writes `run/rev2/master.csv`, the shared tile table (103,248 × 28) used by all later steps.
5. Run the arm of interest (table below).
6. `run/rev2/paper_numbers.py` — collects the reported numbers from the saved result files.

| Result | Script | Saved output |
|---|---|---|
| Variance decomposition (binomial GLMM) | `aduy_code/icc_variance.py` | `run/icc_variance.json` |
| Classification on measured negatives | `aduy_code/classify_confirmed.py` | `run/cls_confirmed.json` |
| Retest validation, 10 deterministic seeds, TOST | `run/rev2/retest_seeds.py --deterministic`, `run/rev2/retest_tost.py` | `run/rev2/retest_det10_final.json` |
| CD43-trained model, 10 deterministic seeds | `aduy_code/cd43_model.py --deterministic` | `run/rev2/cd43_fixed_det10_final.json` |
| Cross-context transfer table, precision@k | `run/rev2/seed10_consolidate.py`, `run/rev2/cd43_topk2.py --deterministic` | `run/rev2/cd43_transfer_10seed.json`, `run/rev2/precision_at_k_10seed.csv`, `run/rev2/cd43_topk3.csv` |
| Held-out replicate prediction | `run/rev2/denoise_test.py` | `run/rev2/denoise_test_10seed.csv` |
| Saliency randomisation controls | `run/rev2/saliency_seeds.py --deterministic`, `run/rev2/saliency_tests.py` | `run/rev2/saliency_det10.csv`, `run/rev2/saliency_det10_tests.csv` |
| Composition (ridge) baseline | `run/rev2/composition_baseline.py --seeds 0..9` | — |
| One-factor-at-a-time decomposition | `run/rev2/whatbreaks.py` | `run/rev2/whatbreaks.csv` |
| Flanking-sequence tests | `run/rev2/flanking.py`, `run/rev2/flanking_within.py` | `run/rev2/flanking_*.csv` |
| Domain-group tests with BH correction | `run/rev2/stats_rev.py` | `run/rev2/cd43_groups_bh.csv` |
| Attenuation correction | `run/rev2/attenuation.py` | — |
| Robustness variants | `run/rev3/run_variants.sh`, `run/rev3/summarise_variants.py` | `run/rev3/var_*.json` |
| Analytic transfer bound | `run/rev3/transfer_bound.py` | `run/rev3/bound.json` |

Seeds, splits and what "deterministic" means for each arm are documented in
`run/rev2/seeds.md`.

## Known limitations of this code release

- **Hard-coded paths.** Many scripts set an absolute project root
  (`/home/dmin/ai&stem_cell`). Edit the `ROOT` assignment at the top of each
  script before running.
- **Large intermediates are not included.** `run/rev2/master.csv` and the
  per-tile prediction tables are regenerated by the scripts above; the ESM-2
  embedding tensors likewise.
- **`classify_confirmed.py` is not fully deterministic** (cuDNN / DataLoader
  not pinned), so its per-fold numbers will not reproduce bit-for-bit.

## License

MIT — see `LICENSE`.

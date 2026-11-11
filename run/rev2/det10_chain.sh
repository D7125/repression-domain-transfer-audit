#!/usr/bin/env bash
# Deterministic 10-seed reruns of the two arms report v9 presents as "10 seeds"
# but which were produced before the determinism patch (2026-10-30).
# Sequential on purpose: one 8 GB GPU, ~9 GB system RAM, and the memmapped
# ESM-2 tensor is the I/O bottleneck.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate machine_deep_learning
export CUBLAS_WORKSPACE_CONFIG=:4096:8
R=/home/dmin/ai\&stem_cell/run/rev2
cd "$R"

echo "### [$(date +%T)] retest_seeds, 10 deterministic seeds"
python retest_seeds.py --deterministic --seeds 0 1 2 3 4 5 6 7 8 9 --n 7 \
  --out "$R/retest_det10.json" > "$R/retest_det10.log" 2>&1
tail -14 "$R/retest_det10.log"

echo "### [$(date +%T)] retest_seeds, seed 0 again (bit-identity check)"
python retest_seeds.py --deterministic --seeds 0 --n 7 \
  --out "$R/retest_det_verify0.json" > "$R/retest_det_verify0.log" 2>&1
tail -4 "$R/retest_det_verify0.log"

echo "### [$(date +%T)] saliency_seeds, 10 deterministic seeds x 3 model states"
python saliency_seeds.py --deterministic --seeds 0 1 2 3 4 5 6 7 8 9 \
  --out "$R/saliency_det10.csv" > "$R/saliency_det10.log" 2>&1
tail -12 "$R/saliency_det10.log"

echo "### [$(date +%T)] saliency_seeds, seed 0 again (bit-identity check)"
python saliency_seeds.py --deterministic --seeds 0 \
  --out "$R/saliency_det_verify0.csv" > "$R/saliency_det_verify0.log" 2>&1
tail -4 "$R/saliency_det_verify0.log"

echo "### [$(date +%T)] ALL_DET10_DONE"

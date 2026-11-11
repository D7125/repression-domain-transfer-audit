#!/bin/bash
set -u
PY=~/miniconda3/envs/machine_deep_learning/bin/python
R="/home/dmin/ai&stem_cell/run/rev2"
cd "/home/dmin/ai&stem_cell/cnn_AI&stemcell" || exit 1
echo "### [1/3] CD43 pass B, N=10"
$PY -u cd43_model.py --mode fixed --n 10 --seeds 0 1 2 --out "$R/cd43_fixed.json" > "$R/cd43_fixed.log" 2>&1
echo "### [1/3] rc=$?"
echo "### [2/3] retest seeds + TOST"
$PY -u -I "$R/retest_seeds.py" --seeds 0 1 2 3 4 5 6 7 8 9 > "$R/retest_seeds.log" 2>&1
echo "### [2/3] rc=$?"
echo "### [3/3] saliency sanity"
$PY -u -I "$R/saliency_sanity.py" > "$R/saliency_sanity.log" 2>&1
echo "### [3/3] rc=$?"
echo "### CHAIN COMPLETE"

#!/usr/bin/env bash
# ETTh1 evaluation with DeepMA top-3 layer + per-layer polynomial ridge.
#
# Training pipeline (see /root/DeepMA/train_top3_layers.py):
#   1. read top-3 layers per Monash dataset from layer_energy_top3.csv
#   2. slide windows over each .tsf series + per-window RevIN
#   3. for each top-3 layer, fit a ridge with polynomial features
#      [x, x^2, ..., x^p] (p = --order)
# Inference: same protocol as benchmark.py deepma route (per-window RevIN ->
# DeepMA decomposition -> per-layer polynomial ridge -> sum -> de-RevIN).
set -euo pipefail

cd "$(dirname "$0")/.."

# ---- prerequisites (must already exist) ------------------------------------
decomp_dir=/root/autodl-tmp/deepma_decomp_pow2
top3_csv=/root/DeepMA/layer_energy_top3.csv

# ---- benchmark CSV ---------------------------------------------------------
test_csv=/root/dataset/ETT-small/ETTh1.csv

# ---- shared hyper-params ---------------------------------------------------
seq_len=720
ridge=1e-4
max_per_series=200
results_csv=/root/DeepMA/results_top3_etth1.csv

# ---- sweeps ----------------------------------------------------------------
orders=(1 2 3)
pred_lens=(96 192 336 720)

for order in "${orders[@]}"; do
  for pred_len in "${pred_lens[@]}"; do
    echo "========== order=${order}  L_in=${seq_len}  L_out=${pred_len} =========="
    python -u train_top3_layers.py \
      --decomp_dir "$decomp_dir" \
      --top3_csv "$top3_csv" \
      --test_csv "$test_csv" \
      --L_in "$seq_len" \
      --L_out "$pred_len" \
      --order "$order" \
      --ridge "$ridge" \
      --max_per_series "$max_per_series" \
      --results_csv "$results_csv"
  done
done

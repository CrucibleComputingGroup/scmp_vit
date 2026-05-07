#!/bin/bash
# Overnight INT baseline chain — W8A8 / W4A8 × {QwT off, QwT on}, full N=50000.
# All 4 cells share the same QwT recipe (cross-seed gate τ=0.5, last_τ=0.8,
# lookahead_veto, n_calib_qwt=1024) used by sweep_2026-04-25/.
set -e
set -o pipefail

cd /home/allenjin/Projects/vit_sc/cls
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vit_sc
export XFORMERS_DISABLED=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p results/int_baseline_2026-05-03

echo "=== INT baseline overnight chain start: $(date) ==="
echo "    GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0)"

# Format: <w_bit> <a_bit> <qwt 0|1> <out_name>
configs=(
  "8 8 0 w8a8_qwtoff"
  "8 8 1 w8a8_qwton"
  "4 8 0 w4a8_qwtoff"
  "4 8 1 w4a8_qwton"
)

for cfg in "${configs[@]}"; do
  read w a q name <<< "$cfg"
  echo
  echo "=== ${name}  (W${w}A${a}, qwt=${q})  start: $(date) ==="
  python -u experiments/int_baseline.py \
    --w_bit "$w" --a_bit "$a" --qwt "$q" \
    --batch_size 64 \
    --n_calib_init 256 \
    --n_calib_qwt 1024 \
    --n_eval 50000 \
    --lookahead_veto \
    --out_json "results/int_baseline_2026-05-03/${name}.json"
  echo "=== ${name}  done:  $(date) ==="
done

echo
echo "=== ALL CELLS DONE: $(date) ==="

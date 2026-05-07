#!/bin/bash
# Add W6A6 plain + QwT cells to int_baseline_2026-05-03/. Same recipe as the
# overnight chain (cross-seed τ=0.5, last_τ=0.8, lookahead_veto, n_calib_qwt=1024).
set -e
set -o pipefail

cd /home/allenjin/Projects/vit_sc/cls
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vit_sc
export XFORMERS_DISABLED=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p results/int_baseline_2026-05-03

echo "=== W6A6 cells start: $(date) ==="

configs=(
  "6 6 0 w6a6_qwtoff"
  "6 6 1 w6a6_qwton"
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
echo "=== W6A6 ALL DONE: $(date) ==="

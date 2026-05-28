#!/bin/bash
# Push hetero adaptive harder under QwT comp, find config where
# savings > B (uniform sl=128, 50%) AND top1 >= 0.815.
set -eu
cd "$(dirname "$0")/.."
OUT=results/qwt_hh
mkdir -p $OUT

CALIB=128
EVAL=100

# Row: tag  levels                alpha  beta
CONFIGS=(
  "C_a04_b10   256,128,64,32      0.4  0.10"
  "C_a05_b20   256,128,64,32      0.5  0.20"
  "C_L5        256,128,64,32,16   0.3  0.05"
  "C_L5a       256,128,64,32,16   0.4  0.10"
  "C_gap       256,128,32         0.3  0.05"
  "C_skip256   128,64,32          0.3  0.05"
  "C_skip256a  128,64,32          0.4  0.10"
)

for row in "${CONFIGS[@]}"; do
  read -r TAG LEV A B <<< "$row"
  echo "=== $TAG  levels=$LEV α=$A β=$B ==="
  python cls/experiments/qwt_sc_compensation.py \
    --sc_config full_attn --n_calib $CALIB --n_eval $EVAL \
    --skip_baseline \
    --adaptive_mp 1 --adaptive_mp_levels "$LEV" \
    --adaptive_mp_ops qkv_proj,out_proj \
    --adaptive_mp_alpha $A --adaptive_mp_beta $B \
    --adaptive_mp_enable_pruning 0 \
    --vit_timestep 9 --vit_total_timesteps 10 \
    --out_json $OUT/${TAG}.json 2>&1 | grep -E "SUMMARY|comp|adaptive_mp" | tail -5
  # Also measure savings
  python experiments/measure_adaptive_savings.py --n 32 --batch_size 16 \
    --levels "$LEV" --ops qkv_proj,out_proj \
    --alpha $A --beta $B --timesteps 9/10 \
    --out $OUT/${TAG}.savings.json 2>&1 | grep -E "savings=" | tail -3
  echo
done

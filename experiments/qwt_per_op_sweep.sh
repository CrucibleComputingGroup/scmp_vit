#!/bin/bash
# Per-op differentiated α/β sweep. Want: savings > 50%, top1 ≈ 0.818.
set -eu
cd "$(dirname "$0")/.."
OUT=results/qwt_hh
mkdir -p $OUT

CALIB=128
EVAL=100
LEV=256,128,64,32

# Row: tag  qkv_α  qkv_β  out_α  out_β
CONFIGS=(
  "P1_qkv_aggr   0.4  0.10  0.3  0.05"
  "P2_qkv_v_aggr 0.5  0.15  0.3  0.05"
  "P3_out_aggr   0.3  0.05  0.4  0.10"
  "P4_both_mid   0.35 0.075 0.35 0.075"
  "P5_qkv_4_out_3 0.4 0.10  0.25 0.03"
)

for row in "${CONFIGS[@]}"; do
  read -r TAG QA QB OA OB <<< "$row"
  echo "=== $TAG  qkv=(α=$QA β=$QB) out=(α=$OA β=$OB) ==="
  python cls/experiments/qwt_sc_compensation.py \
    --sc_config full_attn --n_calib $CALIB --n_eval $EVAL \
    --skip_baseline \
    --adaptive_mp 1 --adaptive_mp_levels $LEV \
    --adaptive_mp_ops qkv_proj,out_proj \
    --adaptive_mp_alpha 0.3 --adaptive_mp_beta 0.05 \
    --adaptive_mp_op_alphas "qkv_proj=$QA,out_proj=$OA" \
    --adaptive_mp_op_betas  "qkv_proj=$QB,out_proj=$OB" \
    --adaptive_mp_enable_pruning 0 \
    --vit_timestep 9 --vit_total_timesteps 10 \
    --out_json $OUT/${TAG}.json 2>&1 | grep -E "SUMMARY|comp|adaptive_mp" | tail -3
  echo
done

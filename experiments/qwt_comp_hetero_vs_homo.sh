#!/bin/bash
# QwT comp head-to-head: homo uniform vs hetero adaptive, both with calibration.
set -eu
cd "$(dirname "$0")/.."
OUT=results/qwt_hh
mkdir -p $OUT

CALIB=256
EVAL=500

# Config A: uniform sl=256 (no MP) + comp -- baseline for what "comp alone"
python cls/experiments/qwt_sc_compensation.py \
  --sc_config full_attn --n_calib $CALIB --n_eval $EVAL \
  --skip_baseline \
  --out_json $OUT/A_uniform_256_comp.json 2>&1 | tail -10
echo

# Config B: static uniform sl=128 on proj + comp (homo 50% savings)
python cls/experiments/qwt_sc_compensation.py \
  --sc_config full_attn --n_calib $CALIB --n_eval $EVAL \
  --skip_baseline \
  --mp_levels 128 --mp_fractions 1.0 --mp_ops qkv_proj,out_proj \
  --out_json $OUT/B_uniform_128_comp.json 2>&1 | tail -10
echo

# Config C: adaptive [256,128,64,32] on proj + comp (hetero ~46% savings)
python cls/experiments/qwt_sc_compensation.py \
  --sc_config full_attn --n_calib $CALIB --n_eval $EVAL \
  --skip_baseline \
  --adaptive_mp 1 --adaptive_mp_levels 256,128,64,32 \
  --adaptive_mp_ops qkv_proj,out_proj \
  --adaptive_mp_alpha 0.3 --adaptive_mp_beta 0.05 \
  --adaptive_mp_enable_pruning 0 \
  --vit_timestep 9 --vit_total_timesteps 10 \
  --out_json $OUT/C_adaptive_L4_comp.json 2>&1 | tail -10
echo

# Config D: static uniform sl=64 on proj + comp (homo 75%)
python cls/experiments/qwt_sc_compensation.py \
  --sc_config full_attn --n_calib $CALIB --n_eval $EVAL \
  --skip_baseline \
  --mp_levels 64 --mp_fractions 1.0 --mp_ops qkv_proj,out_proj \
  --out_json $OUT/D_uniform_64_comp.json 2>&1 | tail -10
echo

# Config E: adaptive [128,64,32] on proj + comp (hetero ~71% savings)
python cls/experiments/qwt_sc_compensation.py \
  --sc_config full_attn --n_calib $CALIB --n_eval $EVAL \
  --skip_baseline \
  --adaptive_mp 1 --adaptive_mp_levels 128,64,32 \
  --adaptive_mp_ops qkv_proj,out_proj \
  --adaptive_mp_alpha 0.3 --adaptive_mp_beta 0.05 \
  --adaptive_mp_enable_pruning 0 \
  --vit_timestep 9 --vit_total_timesteps 10 \
  --out_json $OUT/E_adaptive_L3_comp.json 2>&1 | tail -10

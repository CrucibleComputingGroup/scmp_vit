#!/usr/bin/env bash
# avg96 driver: 1 search + 4 evals (uniform/mp × QwT off/on), single GPU, sequential.
# Mirrors sweep_int678_skip30.sh recipe with target_main_sl=96. Per-SL Sobol
# precision is auto-selected by sc_prec_for_stoc_len (ceil(log2(sl))) per
# dispatch bucket — `--sc_prec 8` here is just the fallback for non-MP ops.
set -euo pipefail

cd "$(dirname "$0")/.."   # → cls/
PROJ_ROOT=/home/allenjin/Projects/vit_sc

source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vit_sc
export XFORMERS_DISABLED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Match sweep_2026-04-25 settings exactly. Env-overridable for smoke runs:
#   N_EVAL=100 N_CALIB=64 SEARCH_N_SEARCH=8 SEARCH_MAX_ITERS=1 \
#   RES_DIR=results/smoke_avg96 LOG_DIR=logs/smoke_avg96 \
#     bash cls/experiments/avg96_driver.sh
N_CALIB="${N_CALIB:-1024}"
N_EVAL="${N_EVAL:-50000}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
COS_THR="${COS_THR:-0.5}"
LAST_COS_THR="${LAST_COS_THR:-0.8}"
CALIB_SEED_B="${CALIB_SEED_B:-2}"
COMP_SC_PREC="${COMP_SC_PREC:-8}"
RIDGE="${RIDGE:-1e-4}"
SEARCH_N_SEARCH="${SEARCH_N_SEARCH:-32}"
SEARCH_MAX_ITERS="${SEARCH_MAX_ITERS:-4}"
SEARCH_OP_MIN_LEVELS="${SEARCH_OP_MIN_LEVELS:-mlp_fc2=192}"
SEARCH_PROXY="${SEARCH_PROXY:-comp_residual}"
SEARCH_FIXED_OPS="${SEARCH_FIXED_OPS:-qk=96,av=96}"
SEARCH_SCHED_FILE="${SEARCH_SCHED_FILE:-skip_worst30/skip_worst30_p7.json}"
# QwT compensator capacity: default per-head (D=64). Set HEAD_ALIGNED_ARGS=""
# to disable and use full-width SCLinear (D=1024) instead.
HEAD_ALIGNED_ARGS="${HEAD_ALIGNED_ARGS---head_aligned_only --n_heads 16}"
LINEAR_MP_OPS="mlp_fc1,mlp_fc2,qkv_proj,out_proj"

RES_DIR="${RES_DIR:-results/sweep_2026-04-25}"
LOG_DIR="${LOG_DIR:-logs/sweep_2026-04-25}"
MAP_DIR="${MAP_DIR:-$RES_DIR/sl_maps}"
mkdir -p "$LOG_DIR" "$RES_DIR" "$MAP_DIR"

SCHED=$PROJ_ROOT/cls/sensitivity/$SEARCH_SCHED_FILE
MAP_AVG96=$MAP_DIR/avg96_mp_main96_sl_map.json

run_task() {
  local name=$1; shift
  local out=$1; shift
  local logf=$LOG_DIR/${name}.log
  if [[ -f $out ]]; then
    echo "[skip] $name (output exists: $out)"
    return 0
  fi
  echo "[run ] $name → $logf"
  echo "       cmd: $*" >>"$logf"
  "$@" >>"$logf" 2>&1
  echo "[done] $name (out=$out)"
}

# --- Task 1: MP search for target_main_sl=96 ---
run_task "search_avg96" "$MAP_AVG96" \
  python experiments/mp_budget_swap_search.py \
    --sc_config skip_worst30 \
    --target_main_sl 96 \
    --levels 64,96,128,192,256 \
    --search_ops proj,mlp_fc1,mlp_fc2 \
    ${SEARCH_FIXED_OPS:+--fixed_ops "$SEARCH_FIXED_OPS"} \
    --op_min_levels "$SEARCH_OP_MIN_LEVELS" \
    --proxy "$SEARCH_PROXY" \
    --init_mode uniform_repair --init_level 96 \
    --n_search $SEARCH_N_SEARCH --search_seed 1 \
    --max_iters $SEARCH_MAX_ITERS \
    --sc_prec 8 \
    --out_json "$MAP_AVG96"

# --- Task 2: avg96 uniform, QwT ON (gate — run first) ---
run_task "avg96_uniform_qwton" "$RES_DIR/avg96_uniform_qwton.json" \
  python experiments/qwt_sc_overnight.py \
    --sc_ops_per_block_json "$SCHED" \
    --sc_prec 8 \
    --batch_size $BATCH --workers $WORKERS \
    --mp_levels 96 --mp_fractions 1.0 \
    --mp_ops "$LINEAR_MP_OPS" \
    --qk_mp_levels 96 --qk_mp_fractions 1.0 \
    --av_mp_levels 96 --av_mp_fractions 1.0 \
    --n_calib $N_CALIB --n_eval $N_EVAL \
    --comp_mode sc --comp_sc_prec $COMP_SC_PREC \
    --ridge $RIDGE \
    --cos_threshold $COS_THR \
    --last_block_cos_threshold $LAST_COS_THR \
    --calib_seed_b $CALIB_SEED_B \
    --lookahead_veto \
    $HEAD_ALIGNED_ARGS \
    --skip_baseline \
    --out_json "$RES_DIR/avg96_uniform_qwton.json"

# --- Task 3: avg96 MP, QwT ON (gate — run second) ---
run_task "avg96_mp_qwton" "$RES_DIR/avg96_mp_qwton.json" \
  python experiments/eval_custom_sl_map_fpcomp.py \
    --sc_config skip_worst30 \
    --sl_map_json "$MAP_AVG96" \
    --sc_prec 8 \
    --n_calib $N_CALIB --n_eval $N_EVAL \
    --batch_size $BATCH --workers $WORKERS \
    --comp_mode sc --comp_sc_prec $COMP_SC_PREC \
    --ridge $RIDGE \
    --cos_threshold $COS_THR \
    --last_block_cos_threshold $LAST_COS_THR \
    --calib_seed_b $CALIB_SEED_B \
    --lookahead_veto \
    $HEAD_ALIGNED_ARGS \
    --out_json "$RES_DIR/avg96_mp_qwton.json"

# --- Task 4: avg96 uniform, QwT OFF ---
run_task "avg96_uniform_qwtoff" "$RES_DIR/avg96_uniform_qwtoff.json" \
  python eval.py --mode sc \
    --sc_ops_per_block_json "$SCHED" \
    --sc_prec 8 \
    --batch_size $BATCH --workers $WORKERS \
    --mp_levels 96 --mp_fractions 1.0 \
    --mp_ops "$LINEAR_MP_OPS" \
    --qk_mp_levels 96 --qk_mp_fractions 1.0 \
    --av_mp_levels 96 --av_mp_fractions 1.0 \
    --max_images $N_EVAL \
    --out_json "$RES_DIR/avg96_uniform_qwtoff.json"

# --- Task 5: avg96 MP, QwT OFF ---
run_task "avg96_mp_qwtoff" "$RES_DIR/avg96_mp_qwtoff.json" \
  python experiments/eval_custom_sl_map_fpcomp.py \
    --sc_config skip_worst30 \
    --sl_map_json "$MAP_AVG96" \
    --sc_prec 8 \
    --n_calib $N_CALIB --n_eval $N_EVAL \
    --batch_size $BATCH --workers $WORKERS \
    --comp_mode sc --comp_sc_prec $COMP_SC_PREC \
    --ridge $RIDGE \
    --cos_threshold $COS_THR \
    --last_block_cos_threshold $LAST_COS_THR \
    --calib_seed_b $CALIB_SEED_B \
    --lookahead_veto \
    --skip_qwt \
    --out_json "$RES_DIR/avg96_mp_qwtoff.json"

echo "[avg96] all 5 tasks complete"
ls -la "$RES_DIR/"avg96_*.json 2>/dev/null || true

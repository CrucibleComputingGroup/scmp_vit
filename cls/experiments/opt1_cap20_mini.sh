#!/usr/bin/env bash
# Option 1 verification: 5 QwT-on configs at r2=0.5 with --max_block_for_qwt 20.
# N=1000 for quick evaluation. Uses existing sl_maps from prod run
# (results/sweep_int678_k30/sl_maps/), so no re-search needed.
#
# Only runs on GPUs 0,1,2 — GPU 3 left alone (prod avg192_uniform_qwton still
# calibrating there).
set -euo pipefail

cd "$(dirname "$0")/.."
PROJ_ROOT=/home/allenjin/Projects/vit_sc
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vit_sc
export XFORMERS_DISABLED=1

RES_DIR=results/sweep_int678_k30_opt1_cap20_n1000
LOG_DIR=logs/sweep_int678_k30_opt1_cap20_n1000
mkdir -p "$RES_DIR" "$LOG_DIR"

# Reuse searched sl_maps from the prod run — no re-search needed.
MAP_P7=$PROJ_ROOT/cls/results/sweep_int678_k30/sl_maps/p7_mp_main128_sl_map.json
MAP_AVG192=$PROJ_ROOT/cls/results/sweep_int678_k30/sl_maps/avg192_mp_main192_sl_map.json
[[ -f $MAP_P7     ]] || { echo "[fatal] missing sl_map $MAP_P7"     >&2; exit 1; }
[[ -f $MAP_AVG192 ]] || { echo "[fatal] missing sl_map $MAP_AVG192" >&2; exit 1; }

N_CALIB=1024
N_EVAL=1000
BATCH=16
WORKERS=8
R2_THR=0.5
CAP=20          # max_block_for_qwt — block 23 can't fire anymore
COMP_SC_PREC=8
RIDGE=1e-4
SCHED_DIR=$PROJ_ROOT/cls/sensitivity/skip_worst30
LINEAR_MP_OPS="mlp_fc1,mlp_fc2,qkv_proj,out_proj"

# --- Task commands ---

cmd_p7_uniform_qwton() {
  echo python experiments/qwt_sc_overnight.py \
    --sc_ops_per_block_json "$SCHED_DIR/skip_worst30_p7.json" \
    --sc_prec 7 --batch_size "$BATCH" --workers "$WORKERS" \
    --n_calib "$N_CALIB" --n_eval "$N_EVAL" \
    --comp_mode sc --comp_sc_prec "$COMP_SC_PREC" \
    --ridge "$RIDGE" --r2_threshold "$R2_THR" \
    --max_block_for_qwt "$CAP" \
    --skip_baseline \
    --out_json "$RES_DIR/p7_uniform_qwton.json"
}

cmd_p8_uniform_qwton() {
  echo python experiments/qwt_sc_overnight.py \
    --sc_ops_per_block_json "$SCHED_DIR/skip_worst30_p8.json" \
    --sc_prec 8 --batch_size "$BATCH" --workers "$WORKERS" \
    --n_calib "$N_CALIB" --n_eval "$N_EVAL" \
    --comp_mode sc --comp_sc_prec "$COMP_SC_PREC" \
    --ridge "$RIDGE" --r2_threshold "$R2_THR" \
    --max_block_for_qwt "$CAP" \
    --skip_baseline \
    --out_json "$RES_DIR/p8_uniform_qwton.json"
}

cmd_avg192_uniform_qwton() {
  # SL=192 everywhere via SC early termination (single-level MP path).
  echo python experiments/qwt_sc_overnight.py \
    --sc_ops_per_block_json "$SCHED_DIR/skip_worst30_p8.json" \
    --sc_prec 8 --batch_size "$BATCH" --workers "$WORKERS" \
    --mp_levels 192 --mp_fractions 1.0 --mp_ops "$LINEAR_MP_OPS" \
    --qk_mp_levels 192 --qk_mp_fractions 1.0 \
    --av_mp_levels 192 --av_mp_fractions 1.0 \
    --n_calib "$N_CALIB" --n_eval "$N_EVAL" \
    --comp_mode sc --comp_sc_prec "$COMP_SC_PREC" \
    --ridge "$RIDGE" --r2_threshold "$R2_THR" \
    --max_block_for_qwt "$CAP" \
    --skip_baseline \
    --out_json "$RES_DIR/avg192_uniform_qwton.json"
}

cmd_p7_mp_qwton() {
  echo python experiments/eval_custom_sl_map_fpcomp.py \
    --sc_config skip_worst30 \
    --sl_map_json "$MAP_P7" \
    --sc_prec 8 \
    --n_calib "$N_CALIB" --n_eval "$N_EVAL" \
    --batch_size "$BATCH" --workers "$WORKERS" \
    --comp_mode sc --comp_sc_prec "$COMP_SC_PREC" \
    --ridge "$RIDGE" --r2_threshold "$R2_THR" \
    --max_block_for_qwt "$CAP" \
    --out_json "$RES_DIR/p7_mp_qwton.json"
}

cmd_avg192_mp_qwton() {
  echo python experiments/eval_custom_sl_map_fpcomp.py \
    --sc_config skip_worst30 \
    --sl_map_json "$MAP_AVG192" \
    --sc_prec 8 \
    --n_calib "$N_CALIB" --n_eval "$N_EVAL" \
    --batch_size "$BATCH" --workers "$WORKERS" \
    --comp_mode sc --comp_sc_prec "$COMP_SC_PREC" \
    --ridge "$RIDGE" --r2_threshold "$R2_THR" \
    --max_block_for_qwt "$CAP" \
    --out_json "$RES_DIR/avg192_mp_qwton.json"
}

# --- Task queue: (name, command-generator) ---
TASKS=(
  "p7_uniform_qwton|cmd_p7_uniform_qwton"
  "p8_uniform_qwton|cmd_p8_uniform_qwton"
  "avg192_uniform_qwton|cmd_avg192_uniform_qwton"
  "p7_mp_qwton|cmd_p7_mp_qwton"
  "avg192_mp_qwton|cmd_avg192_mp_qwton"
)

# --- Pool: 3 GPUs (0,1,2 only). GPU 3 is held by prod, so skip it. ---
GPUS=(0 1 2)
declare -A pid_on_gpu
declare -A gpu_of_pid
declare -A start_of_pid
declare -A out_of_pid
declare -A name_of_pid
active_pids=()

pick_free_gpu() {
  local g
  for g in "${GPUS[@]}"; do
    [[ -z ${pid_on_gpu[$g]:-} ]] && { echo "$g"; return 0; }
  done
  return 1
}

reap() {
  local new=()
  local p
  for p in "${active_pids[@]}"; do
    if kill -0 "$p" 2>/dev/null; then
      new+=("$p")
    else
      local g=${gpu_of_pid[$p]}
      local alive=$(( SECONDS - ${start_of_pid[$p]:-0} ))
      local out=${out_of_pid[$p]:-}
      local name=${name_of_pid[$p]:-?}
      if [[ -n $out && -f $out ]]; then
        echo "[reap] pid=$p gpu=$g $name finished (alive=${alive}s)"
      else
        echo "[reap] pid=$p gpu=$g $name FAILED after ${alive}s (no output)" >&2
      fi
      unset "pid_on_gpu[$g]" "gpu_of_pid[$p]" "start_of_pid[$p]" "out_of_pid[$p]" "name_of_pid[$p]"
    fi
  done
  active_pids=("${new[@]}")
}

for entry in "${TASKS[@]}"; do
  IFS='|' read -r name cmd_fn <<< "$entry"
  outj="$RES_DIR/${name}.json"
  logf="$LOG_DIR/${name}.log"
  if [[ -f $outj ]]; then
    echo "[skip] $outj exists"
    continue
  fi
  while true; do
    if (( ${#active_pids[@]} >= ${#GPUS[@]} )); then
      wait -n 2>/dev/null || true
      reap
    fi
    gpu=$(pick_free_gpu || true)
    if [[ -n $gpu ]]; then
      cmd=$("$cmd_fn")
      echo "[launch gpu=$gpu] $name"
      ( CUDA_VISIBLE_DEVICES="$gpu" eval "$cmd" ) </dev/null >"$logf" 2>&1 &
      pid=$!
      pid_on_gpu[$gpu]=$pid
      gpu_of_pid[$pid]=$gpu
      start_of_pid[$pid]=$SECONDS
      out_of_pid[$pid]=$outj
      name_of_pid[$pid]=$name
      active_pids+=("$pid")
      break
    fi
    wait -n 2>/dev/null || true
    reap
  done
done

echo "[drain] tail: ${active_pids[*]}"
while (( ${#active_pids[@]} > 0 )); do
  wait -n 2>/dev/null || true
  reap
done

echo "[done] all 5 runs complete"
ls -la "$RES_DIR/"

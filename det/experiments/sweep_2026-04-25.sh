#!/usr/bin/env bash
# det production sweep — port of cls/experiments/sweep_int678_skip30.sh.
# Cross-seed gate + head-aligned SC comp + uniform/MP-192 mode × QwT off/on
# across sc_prec ∈ {7,8} × K ∈ {20,30}.
#
# Smoke-test via env overrides:
#   N_EVAL=100 N_CALIB=64 \
#   RES_DIR=results/smoke_2026-04-25 LOG_DIR=logs/smoke_2026-04-25 \
#     bash det/experiments/sweep_2026-04-25.sh
#
# Production:
#   nohup env \
#     RES_DIR=results/sweep_2026-04-25 \
#     LOG_DIR=logs/sweep_2026-04-25 \
#     HEAD_ALIGNED=1 HEAD_ALIGNED_HEADS=16 \
#     bash det/experiments/sweep_2026-04-25.sh \
#     > logs/sweep_2026-04-25/sched.log 2>&1 &
#
# Notes
# -----
# * det has no skip_worst_*_int8.json — only int6/int7 sensitivity matrices
#   exist. We use skip_worst_K_int7.json as the schedule for sc_prec=8 too,
#   since at lower SC noise the cells worst at int7 stay the worst order
#   (just smaller absolute drops). Conservative substitution; flagged in
#   RESULTS.md.
# * det has no MP-budget swap search yet (cls's mp_budget_swap_search.py
#   has a known equal-MAC DP limitation; see docs/SC_COMP_ALGORITHM.md
#   §"MP-vs-uniform parity"). MP runs here use the simple uniform-192
#   trick — every linear/attn op gets stoc_len=192.
set -euo pipefail

cd "$(dirname "$0")/.."   # → det/
PROJ_ROOT=/home/allenjin/Projects/vit_sc
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate qwt_d2
module load gcc/13.2.0

# --- Env-overridable knobs (prod defaults). ---
N_CALIB="${N_CALIB:-1024}"
N_EVAL="${N_EVAL:-5000}"
COS_THR="${COS_THR:-0.5}"
LAST_COS_THR="${LAST_COS_THR:-0.8}"
CALIB_SEED="${CALIB_SEED:-1}"
CALIB_SEED_B="${CALIB_SEED_B:-2}"
COMP_SC_PREC="${COMP_SC_PREC:-8}"
RIDGE="${RIDGE:-1e-2}"
FWD_CHUNK="${FWD_CHUNK:-2}"
HEAD_ALIGNED="${HEAD_ALIGNED:-1}"
HEAD_ALIGNED_HEADS="${HEAD_ALIGNED_HEADS:-16}"

LINEAR_MP_OPS="mlp_fc1,mlp_fc2,qkv_proj,out_proj"
MP_LEVEL="${MP_LEVEL:-192}"  # uniform MP target stoc_len

LOG_DIR="${LOG_DIR:-logs/sweep_2026-04-25}"
RES_DIR="${RES_DIR:-results/sweep_2026-04-25}"
mkdir -p "$LOG_DIR" "$RES_DIR"

# Schedule file lookup. Currently: int7 schedules used for both sc_prec=7
# and sc_prec=8 (no int8 sensitivity sweep — see header note).
sched_for() {
  local sc_prec=$1 K=$2
  local sens_p=7
  [[ $sc_prec == 6 ]] && sens_p=6
  echo "sensitivity/skip/skip_worst_${K}_int${sens_p}.json"
}

build_qwt_cmd() {
  local sc_prec=$1 K=$2 mode=$3 qwt=$4 name=$5
  local sched; sched=$(sched_for "$sc_prec" "$K")
  local out_json="$RES_DIR/${name}.json"
  local args=(
    --sc_prec "$sc_prec"
    --sc_ops_per_block_json "$sched"
    --n_calib "$N_CALIB" --n_eval "$N_EVAL"
    --calib_seed "$CALIB_SEED" --calib_seed_b "$CALIB_SEED_B"
    --cos_threshold "$COS_THR"
    --last_block_cos_threshold "$LAST_COS_THR"
    --lookahead_veto
    --ridge "$RIDGE" --fwd_chunk "$FWD_CHUNK"
    --comp_mode sc --comp_sc_prec "$COMP_SC_PREC"
    --out_tag "$name"
    --out_json "$out_json"
  )
  if [[ -n "$HEAD_ALIGNED" ]]; then
    args+=(--head_aligned --n_heads "$HEAD_ALIGNED_HEADS")
  fi
  if [[ "$mode" == "mp192" ]]; then
    args+=(
      --mp_levels "$MP_LEVEL" --mp_fractions 1.0 --mp_ops "$LINEAR_MP_OPS"
      --qk_mp_levels "$MP_LEVEL" --qk_mp_fractions 1.0
      --av_mp_levels "$MP_LEVEL" --av_mp_fractions 1.0
    )
  fi
  echo python experiments/qwt_det_compensate.py "${args[@]}"
}

build_sc_cmd() {
  # raw SC (no QwT) — uses sc_eval.py
  local sc_prec=$1 K=$2 mode=$3 name=$4
  local sched; sched=$(sched_for "$sc_prec" "$K")
  local out_tag="$name"
  local args=(
    --sc_prec "$sc_prec"
    --sc_ops_per_block_json "$sched"
    --n-eval "$N_EVAL"
    --out-tag "$out_tag"
  )
  if [[ "$mode" == "mp192" ]]; then
    args+=(
      --mp_levels "$MP_LEVEL" --mp_fractions 1.0 --mp_ops "$LINEAR_MP_OPS"
      --qk_mp_levels "$MP_LEVEL" --qk_mp_fractions 1.0
      --av_mp_levels "$MP_LEVEL" --av_mp_fractions 1.0
    )
  fi
  echo python sc_eval.py "${args[@]}"
}

# --- Task definitions --------------------------------------------------------
TASK_NAMES=()
TASK_CMDS=()
TASK_OUT=()

add_task() {
  local name=$1 cmd=$2 out=$3
  TASK_NAMES+=("$name")
  TASK_CMDS+=("$cmd")
  TASK_OUT+=("$out")
}

# Eval cells: sc_prec ∈ {7,8} × K ∈ {20,30} × mode ∈ {uniform, mp192}
# × qwt ∈ {off, on} = 16 evals. The qwt=on cell already runs raw SC inside
# the same driver (`--skip_baseline` not set), so qwt=off is a sanity duplicate
# that we route through sc_eval.py to keep raw vs comp on a fully separate
# code path.
for sc_prec in 7 8; do
  for K in 20 30; do
    for mode in uniform mp192; do
      base="p${sc_prec}_K${K}_${mode}"
      # qwt=off → raw SC eval via sc_eval.py
      name_off="${base}_qwtoff"
      out_off="$RES_DIR/${name_off}/metrics.json"
      add_task "$name_off" "$(build_sc_cmd "$sc_prec" "$K" "$mode" "$name_off")" "$out_off"
      # qwt=on → cross-seed calibration + head-aligned comp eval
      name_on="${base}_qwton"
      out_on="$RES_DIR/${name_on}.json"
      add_task "$name_on" "$(build_qwt_cmd "$sc_prec" "$K" "$mode" on "$name_on")" "$out_on"
    done
  done
done

# --- Unified 4-GPU pool scheduler -------------------------------------------
# Same shape as cls/experiments/sweep_int678_skip30.sh's scheduler. Det has
# no search prereqs in this iteration (no MP-budget swap search yet), so the
# task graph is fully parallel.

MAX_PAR=4
declare -A gpu_of_pid
declare -A pid_on_gpu
declare -A start_time_of_pid
declare -A gpu_blocked
declare -A out_of_pid
declare -A name_of_pid
active_pids=()
done_mask=()

pick_free_gpu() {
  local g
  for g in 0 1 2 3; do
    if [[ -z ${pid_on_gpu[$g]:-} && -z ${gpu_blocked[$g]:-} ]]; then
      echo "$g"; return 0
    fi
  done
  return 1
}

launch_task() {
  local idx=$1 gpu=$2
  local name=${TASK_NAMES[$idx]}
  local cmd=${TASK_CMDS[$idx]}
  local out=${TASK_OUT[$idx]}
  local logf=$LOG_DIR/${name}.log
  echo "[launch gpu=$gpu] $name"
  ( CUDA_VISIBLE_DEVICES="$gpu" eval "$cmd" ) </dev/null >"$logf" 2>&1 &
  local new_pid=$!
  gpu_of_pid[$new_pid]=$gpu
  pid_on_gpu[$gpu]=$new_pid
  start_time_of_pid[$new_pid]=$SECONDS
  out_of_pid[$new_pid]=$out
  name_of_pid[$new_pid]=$name
  active_pids+=("$new_pid")
  done_mask[$idx]=1
}

reap_finished_pids() {
  local new_active=()
  local p
  for p in "${active_pids[@]}"; do
    if kill -0 "$p" 2>/dev/null; then
      new_active+=("$p")
    else
      local g=${gpu_of_pid[$p]}
      local alive=$(( SECONDS - ${start_time_of_pid[$p]:-0} ))
      local out=${out_of_pid[$p]:-}
      local name=${name_of_pid[$p]:-"?"}
      if [[ -n $out && -f $out ]]; then
        echo "[reap] pid=$p gpu=$g $name finished (alive=${alive}s)"
      elif (( alive < 60 )); then
        gpu_blocked[$g]=1
        echo "[reap] pid=$p gpu=$g $name died in ${alive}s no output — BLOCKING gpu=$g"
      else
        echo "[reap] pid=$p gpu=$g $name FAILED after ${alive}s" >&2
      fi
      unset "pid_on_gpu[$g]"
      unset "gpu_of_pid[$p]"
      unset "start_time_of_pid[$p]"
      unset "out_of_pid[$p]"
      unset "name_of_pid[$p]"
    fi
  done
  active_pids=("${new_active[@]}")
}

# Idempotent reruns: skip tasks whose output already exists.
for i in "${!TASK_NAMES[@]}"; do
  done_mask[$i]=0
  if [[ -f ${TASK_OUT[$i]} ]]; then
    echo "[skip] ${TASK_OUT[$i]} exists (task ${TASK_NAMES[$i]})"
    done_mask[$i]=1
  fi
done

while true; do
  pending=0
  for i in "${!TASK_NAMES[@]}"; do
    [[ ${done_mask[$i]} == 0 ]] && pending=1 && break
  done
  (( pending == 0 && ${#active_pids[@]} == 0 )) && break

  if (( ${#active_pids[@]} >= MAX_PAR )); then
    wait -n 2>/dev/null || true
    reap_finished_pids
    continue
  fi

  launched_any=0
  for i in "${!TASK_NAMES[@]}"; do
    [[ ${done_mask[$i]} != 0 ]] && continue
    gpu=$(pick_free_gpu) || break
    launch_task "$i" "$gpu"
    launched_any=1
    break
  done

  if (( launched_any == 0 )); then
    if (( ${#active_pids[@]} == 0 )); then
      echo "[fatal] no active workers and remaining tasks unlaunchable" >&2
      exit 1
    fi
    wait -n 2>/dev/null || true
    reap_finished_pids
  fi
done

echo "[sweep] all tasks complete"
ls -la "$RES_DIR/"

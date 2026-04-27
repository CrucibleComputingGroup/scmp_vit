#!/usr/bin/env bash
# det production sweep — port of cls/experiments/sweep_int678_skip30.sh
# matrix structure:
#   prec_tag ∈ {p7, avg192, p8} × mode ∈ {uniform, mp} × qwt ∈ {off, on}
#                                                = 10 evals (no p8_mp; same as cls)
# plus 2 MP-budget swap searches (one per mp tag) producing the sl_maps.
#
# Smoke-test via env overrides:
#   N_EVAL=10 N_CALIB=16 SEARCH_N_SEARCH=4 SEARCH_MAX_ITERS=2 \
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
# * det has no skip_worst_*_int8.json — we use skip_worst_K_int7.json for
#   sc_prec=8 (conservative — int7-derived ranking is over-cautious at int8).
# * det's MP-budget swap search lives at det/experiments/mp_budget_swap_search.py
#   (uses the shared algorithm core in sc_integration/mp_search.py). Equal-MAC
#   DP requires search ops to share the same per-(op, block) MAC; default
#   --search_ops mlp_fc1,mlp_fc2 (qkv_proj/out_proj/qk/av pinned via fixed_ops).
set -euo pipefail

cd "$(dirname "$0")/.."   # → det/
PROJ_ROOT=/home/allenjin/Projects/vit_sc
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate qwt_d2
module load gcc/13.2.0

# --- Env-overridable knobs (prod defaults). ---
N_CALIB="${N_CALIB:-64}"   # det production calib size; cls uses 1024
N_EVAL="${N_EVAL:-5000}"
EVAL_START_IDX="${EVAL_START_IDX:-0}"   # COCO val start index for eval slice;
                                         # use to extend a prior run incrementally
SIZE="${SIZE:-0}"          # input square size override (mult of 256); 0=default 1280
INCLUDE_MP="${INCLUDE_MP:-1}"   # 0 to skip mp eval cells AND mp searches
                                # (uniform-only sweep — useful on 1-GPU hosts)
COS_THR="${COS_THR:-0.5}"
LAST_COS_THR="${LAST_COS_THR:-0.8}"
CALIB_SEED="${CALIB_SEED:-1}"
CALIB_SEED_B="${CALIB_SEED_B:-2}"
COMP_SC_PREC="${COMP_SC_PREC:-8}"
RIDGE="${RIDGE:-1e-2}"
FWD_CHUNK="${FWD_CHUNK:-2}"
HEAD_ALIGNED="${HEAD_ALIGNED:-1}"
HEAD_ALIGNED_HEADS="${HEAD_ALIGNED_HEADS:-16}"
K_SCHEDULE="${K_SCHEDULE:-30}"

LINEAR_MP_OPS="mlp_fc1,mlp_fc2,qkv_proj,out_proj"
AVG192_LEVEL="${AVG192_LEVEL:-192}"

# MP-budget swap search knobs.
# - SEARCH_OPS adds qkv_proj alongside mlp_fc1/mlp_fc2 at det's 1280²
#   scale (mlp_fc1=55.4G, mlp_fc2=55.4G, qkv_proj=38G — all within
#   1.5× MAC, the relaxed-tolerance limit in mp_search.py). Adding qk/av
#   needs more work: their per-block MAC swings 250× across window
#   (0.23G) vs global (57.6G) attention blocks; equal-MAC DP can't
#   absorb that without a per-block unit filter or two-stage search.
#   out_proj (12.7G) is also outside the 1.5× tolerance vs mlp.
# - SEARCH_N_SEARCH=64 / SEARCH_MAX_ITERS=12 match cls's N=50k production
#   (the budget that made cls's `p7_mp` Δ jump from +1.14 → +3.37).
SEARCH_OPS="${SEARCH_OPS:-mlp_fc1,mlp_fc2,qkv_proj}"
SEARCH_LEVELS="${SEARCH_LEVELS:-64,96,128,192,256}"
SEARCH_PROXY="${SEARCH_PROXY:-comp_residual}"
SEARCH_N_SEARCH="${SEARCH_N_SEARCH:-64}"
SEARCH_MAX_ITERS="${SEARCH_MAX_ITERS:-12}"
SEARCH_SEED="${SEARCH_SEED:-1}"

LOG_DIR="${LOG_DIR:-logs/sweep_2026-04-25}"
RES_DIR="${RES_DIR:-results/sweep_2026-04-25}"
MAP_DIR="${MAP_DIR:-$RES_DIR/sl_maps}"
mkdir -p "$LOG_DIR" "$RES_DIR" "$MAP_DIR"

SCHED_INT7="sensitivity/skip/skip_worst_${K_SCHEDULE}_int7.json"

# --- Per-tag accessors ---------------------------------------------------------
# target main SL: p7=128 (sc_prec=7), avg192=192, p8=256 (sc_prec=8).
target_main_sl_for_tag() { case "$1" in p7) echo 128 ;; avg192) echo 192 ;; p8) echo 256 ;; esac; }
# fixed-op stoc_len for the search. qkv_proj joined SEARCH_OPS, but
# qk/av (heterogeneous window vs global MACs) and out_proj (4× outside
# 1.5× MAC tolerance) stay pinned at the target SL.
fixed_for_tag()          {
  local t=$(target_main_sl_for_tag "$1")
  echo "qk=${t},av=${t},out_proj=${t}"
}
sl_map_for_tag()         { case "$1" in p7) echo "$MAP_DIR/p7_mp_sl_map.json" ;; avg192) echo "$MAP_DIR/avg192_mp_sl_map.json" ;; esac; }

# --- Search command builder ---------------------------------------------------

build_search_cmd() {
  local tag=$1 out=$2
  local target; target=$(target_main_sl_for_tag "$tag")
  local fixed; fixed=$(fixed_for_tag "$tag")
  echo python experiments/mp_budget_swap_search.py \
    --sc_config "skip_worst${K_SCHEDULE}" \
    --target_main_sl "$target" \
    --levels "$SEARCH_LEVELS" \
    --search_ops "$SEARCH_OPS" \
    --fixed_ops "$fixed" \
    --proxy "$SEARCH_PROXY" \
    --init_mode uniform_repair --init_level "$target" \
    --n_search "$SEARCH_N_SEARCH" --search_seed "$SEARCH_SEED" \
    --max_iters "$SEARCH_MAX_ITERS" \
    --sc_prec 8 \
    --out_json "$out"
}

# --- Eval command builder -----------------------------------------------------

build_eval_cmd() {
  local tag=$1 mode=$2 qwt=$3 name=$4
  local out_json=$RES_DIR/${name}.json
  local ha_args=()
  if [[ -n "$HEAD_ALIGNED" ]]; then
    ha_args=(--head_aligned --n_heads "$HEAD_ALIGNED_HEADS")
  fi

  # Determine sc_prec / schedule / per-tag mode args.
  local sc_prec=8
  local sched=""
  local mode_args=()
  case "$tag" in
    p7) sc_prec=7; sched="$SCHED_INT7" ;;
    p8) sc_prec=8; sched="$SCHED_INT7" ;;
    avg192)
      sc_prec=8; sched="$SCHED_INT7"
      # Uniform-fractional MP at AVG192_LEVEL across all SC ops (only for
      # mode=uniform — the mp variant uses the search-derived sl_map).
      mode_args=(
        --mp_levels "$AVG192_LEVEL" --mp_fractions 1.0 --mp_ops "$LINEAR_MP_OPS"
        --qk_mp_levels "$AVG192_LEVEL" --qk_mp_fractions 1.0
        --av_mp_levels "$AVG192_LEVEL" --av_mp_fractions 1.0
      )
      ;;
  esac

  # MP variant: replace mode_args with the per-(op, block) sl_map.
  if [[ "$mode" == "mp" ]]; then
    local map; map=$(sl_map_for_tag "$tag")
    mode_args=(--sl_map_json "$map")
  fi

  if [[ "$qwt" == "off" ]]; then
    # Raw SC eval via sc_eval.py
    local args=(
      --sc_prec "$sc_prec"
      --sc_ops_per_block_json "$sched"
      --n-eval "$N_EVAL"
      --start_idx "$EVAL_START_IDX"
      --size "$SIZE"
      --out-tag "$name"
      --out_dir "$RES_DIR/$name"
      "${mode_args[@]}"
    )
    echo python sc_eval.py "${args[@]}"
  else
    # SC + cross-seed QwT comp via qwt_det_compensate.py
    local args=(
      --sc_prec "$sc_prec"
      --sc_ops_per_block_json "$sched"
      --n_calib "$N_CALIB" --n_eval "$N_EVAL"
      --eval_start_idx "$EVAL_START_IDX"
      --size "$SIZE"
      --calib_seed "$CALIB_SEED" --calib_seed_b "$CALIB_SEED_B"
      --cos_threshold "$COS_THR" --last_block_cos_threshold "$LAST_COS_THR"
      --lookahead_veto
      --ridge "$RIDGE" --fwd_chunk "$FWD_CHUNK"
      --comp_mode sc --comp_sc_prec "$COMP_SC_PREC"
      --out_tag "$name" --out_json "$out_json"
      "${mode_args[@]}" "${ha_args[@]}"
    )
    echo python experiments/qwt_det_compensate.py "${args[@]}"
  fi
}

# --- Task definitions --------------------------------------------------------
TASK_NAMES=()
TASK_CMDS=()
TASK_OUT=()
TASK_PREREQ=()

add_task() {
  TASK_NAMES+=("$1"); TASK_CMDS+=("$2"); TASK_OUT+=("$3"); TASK_PREREQ+=("$4")
}

# Searches first (their sl_maps are prereqs for mp evals).
MAP_P7=$(sl_map_for_tag p7)
MAP_AVG192=$(sl_map_for_tag avg192)
if [[ "$INCLUDE_MP" == "1" ]]; then
  [[ -f $MAP_P7 ]]     || add_task "search_p7"     "$(build_search_cmd p7 "$MAP_P7")"         "$MAP_P7" ""
  [[ -f $MAP_AVG192 ]] || add_task "search_avg192" "$(build_search_cmd avg192 "$MAP_AVG192")" "$MAP_AVG192" ""
fi

# Eval matrix. p8_mp omitted (same as cls — p8 has no mp variant).
# When INCLUDE_MP=0, only the uniform variants run (6 cells, no MP search).
if [[ "$INCLUDE_MP" == "1" ]]; then
  EVAL_RUNS=(
    "p7     uniform off"  "p7     uniform on"
    "p7     mp      off"  "p7     mp      on"
    "p8     uniform off"  "p8     uniform on"
    "avg192 uniform off"  "avg192 uniform on"
    "avg192 mp      off"  "avg192 mp      on"
  )
else
  EVAL_RUNS=(
    "p7     uniform off"  "p7     uniform on"
    "p8     uniform off"  "p8     uniform on"
    "avg192 uniform off"  "avg192 uniform on"
  )
fi
for entry in "${EVAL_RUNS[@]}"; do
  read -r tag mode qwt <<< "$entry"
  name="${tag}_${mode}_qwt${qwt}"
  if [[ "$qwt" == "off" ]]; then
    outj="$RES_DIR/${name}/metrics.json"
  else
    outj="$RES_DIR/${name}.json"
  fi
  prereq=""
  [[ "$mode" == "mp" ]] && prereq=$(sl_map_for_tag "$tag")
  add_task "$name" "$(build_eval_cmd "$tag" "$mode" "$qwt" "$name")" "$outj" "$prereq"
done

# --- Unified GPU pool scheduler ----------------------------------------------
# Default: 4-way parallel (overnight production). Set MAX_PAR=1 for
# serialized single-GPU runs (e.g. tuning, debugging, or honoring the
# "one GPU at a time" preference on a shared host).
MAX_PAR="${MAX_PAR:-4}"
# GPU_LIST: space-separated logical GPU indices to schedule on. Default
# uses all 4. Set GPU_LIST="2" (or "0 2", etc.) on a host where only some
# physical GPUs are available — avoids the auto-block penalty for
# CUDA-busy GPUs that the scheduler would otherwise try first.
GPU_LIST="${GPU_LIST:-0 1 2 3}"
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
  for g in $GPU_LIST; do
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
    prereq=${TASK_PREREQ[$i]}
    if [[ -n $prereq && ! -f $prereq ]]; then
      continue  # waiting for an upstream search's sl_map
    fi
    gpu=$(pick_free_gpu) || break
    launch_task "$i" "$gpu"
    launched_any=1
    break
  done

  if (( launched_any == 0 )); then
    if (( ${#active_pids[@]} == 0 )); then
      echo "[fatal] no active workers and remaining tasks blocked on prereqs:" >&2
      for i in "${!TASK_NAMES[@]}"; do
        [[ ${done_mask[$i]} == 0 ]] && echo "  - ${TASK_NAMES[$i]} needs ${TASK_PREREQ[$i]}" >&2
      done
      exit 1
    fi
    wait -n 2>/dev/null || true
    reap_finished_pids
  fi
done

echo "[sweep] all tasks complete"
ls -la "$RES_DIR/"

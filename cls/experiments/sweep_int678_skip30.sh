#!/usr/bin/env bash
# Sweep: int 7/8 + avg192 × {uniform, mp} × {QwT off, on} at K=30 (skip 30 worst).
# int 6 removed — below the SC accuracy floor at K=30 (top1 ≈ 0).
#
# MP variants use per-(op, block) scalar stoc_len maps from mp_budget_swap_search
# (budget-preserving pair swap, `comp_residual` local proxy). Both uniform-on and
# MP-on invoke the SAME SC comp factory (comp_mode=sc, comp_sc_prec=8,
# ridge=1e-4), so only the stoc_len map differs between them.
#
# Unified pool: the 2 searches and 10 eval runs all share one 4-GPU pool.
# MP evals carry a prereq on their sl_map file, so the scheduler launches
# uniform evals and searches first, and picks up MP evals as soon as their
# search finishes — keeping all 4 GPUs busy. Per-GPU exclusive-mode contention
# is handled via <60 s death detection (bad GPU is blacklisted).
#
# Smoke-test via env overrides:
#   N_EVAL=100 N_CALIB=64 SEARCH_N_SEARCH=8 SEARCH_MAX_ITERS=1 \
#   RES_DIR=results/smoke_int678_k30 LOG_DIR=logs/smoke_int678_k30 \
#     bash cls/experiments/sweep_int678_skip30.sh
# Or use cls/experiments/sweep_int678_skip30_smoke.sh as a thin wrapper.
set -euo pipefail

cd "$(dirname "$0")/.."
PROJ_ROOT=/home/allenjin/Projects/vit_sc
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vit_sc

export XFORMERS_DISABLED=1

# --- Env-overridable knobs (prod defaults; smoke launcher overrides to small). ---
# Gate = cross-seed cosine (see qwt_sc/compensation.py): fit W on two disjoint
# calib batches (seeds 1 + 2 by default), admit iff cos(W_A, W_B) > COS_THR.
# Replaces the earlier r²-based recipe that was the previous default here.
N_CALIB="${N_CALIB:-1024}"
N_EVAL="${N_EVAL:-50000}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
COS_THR="${COS_THR:-0.5}"
LAST_COS_THR="${LAST_COS_THR:-0.8}"
CALIB_SEED_B="${CALIB_SEED_B:-2}"
COMP_SC_PREC="${COMP_SC_PREC:-8}"
RIDGE="${RIDGE:-1e-4}"
LINEAR_MP_OPS="mlp_fc1,mlp_fc2,qkv_proj,out_proj"

SEARCH_LEVELS="${SEARCH_LEVELS:-64,96,128,192,256}"
SEARCH_OPS="${SEARCH_OPS:-proj,mlp_fc1,mlp_fc2}"
SEARCH_OP_MIN="${SEARCH_OP_MIN:-mlp_fc2=192}"
SEARCH_PROXY="${SEARCH_PROXY:-comp_residual}"
SEARCH_N_SEARCH="${SEARCH_N_SEARCH:-32}"
SEARCH_SEED="${SEARCH_SEED:-1}"
SEARCH_MAX_ITERS="${SEARCH_MAX_ITERS:-4}"
# FIXED_OPS pins op-level SL during search. Default pins qk/av to the
# target SL (legacy behavior — only proj/mlp_fc1/mlp_fc2 vary). Set to
# "AUTO" to keep the legacy behavior, "" to unpin (search varies qk/av too),
# or any explicit "qk=N,av=M" string. Special: AUTO is needed because we
# can only resolve the per-tag target inside build_search_cmd.
FIXED_OPS="${FIXED_OPS-AUTO}"
# HEAD_ALIGNED=1 routes the QwT comp through HeadAlignedSCLinear (per-head
# D=64, reuses _CFG_CACHE[(64, 8)] from the block QK pool — zero new SNG
# pool entries). Empty (default) uses full-width SCLinear at D=1024.
HEAD_ALIGNED="${HEAD_ALIGNED:-}"
HEAD_ALIGNED_HEADS="${HEAD_ALIGNED_HEADS:-16}"

LOG_DIR="${LOG_DIR:-logs/sweep_int678_k30}"
RES_DIR="${RES_DIR:-results/sweep_int678_k30}"
MAP_DIR="${MAP_DIR:-$RES_DIR/sl_maps}"
# CWD here is cls/ (thanks to `cd $(dirname "$0")/..`). Relative paths below
# are relative to cls/; absolute-path helpers go through $PROJ_ROOT.
mkdir -p "$LOG_DIR" "$RES_DIR" "$MAP_DIR" "$PROJ_ROOT/results"

SCHED_DIR=$PROJ_ROOT/cls/sensitivity/skip_worst30
[[ -d $SCHED_DIR ]] || python experiments/build_skip_worst20_json.py --k 30 --out_dir sensitivity/skip_worst30

# --- Phase 0: synthesize 5-op merged sensitivity file expected by the search. ---
# mp_budget_swap_search.py sets SENSITIVITY_JSON = HERE/"results"/"sensitivity_all_ops.json"
# where HERE resolves to vit_sc/ (project root) — NOT cls/. So we write to
# $PROJ_ROOT/results/sensitivity_all_ops.json. Source is the real-SC p8 grid
# at cls/sensitivity/sensitivity_per_operator_real_sc_p8.json (6 ops); we
# merge qkv_proj+out_proj → proj by max l2 per block to get the 5-op layout.
SENS_JSON=$PROJ_ROOT/results/sensitivity_all_ops.json
if [[ ! -f $SENS_JSON ]]; then
  echo "[setup] synthesizing $SENS_JSON from p8 real-SC per-operator sensitivity"
  SENS_SRC=$PROJ_ROOT/cls/sensitivity/sensitivity_per_operator_real_sc_p8.json \
  SENS_DST=$SENS_JSON \
  python - <<'PY'
import json, os
from pathlib import Path
src = Path(os.environ["SENS_SRC"])
dst = Path(os.environ["SENS_DST"])
with open(src) as f:
    data = json.load(f)
merged = {}
for r in data["grid"]:
    op_raw = r["op"]
    op = "proj" if op_raw in ("qkv_proj", "out_proj") else op_raw
    key = (op, int(r["block"]))
    prev = merged.get(key)
    if prev is None or float(r["l2"]) > float(prev["l2"]):
        merged[key] = {"op": op, "block": int(r["block"]), "l2": float(r["l2"])}
out = {
    "grid": sorted(merged.values(), key=lambda x: (x["op"], x["block"])),
    "source": str(src),
    "note": "5-op merged from 6-op p8 real-SC: qkv_proj+out_proj -> proj (max l2).",
}
dst.parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w") as f:
    json.dump(out, f, indent=2)
print(f"[setup] wrote {dst} with {len(merged)} entries")
PY
fi

# --- Task definitions ---------------------------------------------------------
# Each task: a name, a command emitter, and an optional prereq file that must
# exist before the task can launch. Searches have no prereq; MP evals require
# the corresponding sl_map produced by their search. Uniform evals have no prereq.

MAP_P7=$MAP_DIR/p7_mp_main128_sl_map.json
MAP_AVG192=$MAP_DIR/avg192_mp_main192_sl_map.json

target_main_sl_for_tag() { case "$1" in p7) echo 128 ;; avg192) echo 192 ;; esac; }
sens_p_for_tag()         { case "$1" in p7) echo 7 ;; p8|avg192) echo 8 ;; esac; }
sc_prec_for_uniform()    { case "$1" in p7) echo 7 ;; p8|avg192) echo 8 ;; esac; }
sl_map_for_tag()         { case "$1" in p7) echo "$MAP_P7" ;; avg192) echo "$MAP_AVG192" ;; esac; }

build_search_cmd() {
  local tag=$1 out=$2
  local target; target=$(target_main_sl_for_tag "$tag")
  local fixed=""
  if [[ "$FIXED_OPS" == "AUTO" ]]; then
    fixed="qk=${target},av=${target}"
  else
    fixed="$FIXED_OPS"
  fi
  local fixed_args=()
  [[ -n "$fixed" ]] && fixed_args=(--fixed_ops "$fixed")
  echo python experiments/mp_budget_swap_search.py \
    --sc_config skip_worst30 \
    --target_main_sl "$target" \
    --levels "$SEARCH_LEVELS" \
    --search_ops "$SEARCH_OPS" \
    "${fixed_args[@]}" \
    --op_min_levels "$SEARCH_OP_MIN" \
    --proxy "$SEARCH_PROXY" \
    --init_mode uniform_repair --init_level "$target" \
    --n_search "$SEARCH_N_SEARCH" --search_seed "$SEARCH_SEED" \
    --max_iters "$SEARCH_MAX_ITERS" \
    --sc_prec 8 \
    --out_json "$out"
}

build_eval_cmd() {
  local tag=$1 mode=$2 qwt=$3 name=$4
  local out_json=$RES_DIR/${name}.json
  local ha_args=()
  if [[ -n "$HEAD_ALIGNED" ]]; then
    ha_args=(--head_aligned_only --n_heads "$HEAD_ALIGNED_HEADS")
  fi

  if [[ $mode == mp ]]; then
    local map; map=$(sl_map_for_tag "$tag")
    local args=(
      --sc_config skip_worst30
      --sl_map_json "$map"
      --sc_prec 8
      --n_calib "$N_CALIB" --n_eval "$N_EVAL"
      --batch_size "$BATCH" --workers "$WORKERS"
      --comp_mode sc --comp_sc_prec "$COMP_SC_PREC"
      --ridge "$RIDGE"
      --cos_threshold "$COS_THR"
      --last_block_cos_threshold "$LAST_COS_THR"
      --calib_seed_b "$CALIB_SEED_B"
      --lookahead_veto
      --out_json "$out_json"
    )
    [[ $qwt == off ]] && args+=(--skip_qwt)
    [[ $qwt == on ]]  && args+=("${ha_args[@]}")
    echo python experiments/eval_custom_sl_map_fpcomp.py "${args[@]}"
    return
  fi

  # uniform paths
  local sp; sp=$(sens_p_for_tag "$tag")
  local sched=$SCHED_DIR/skip_worst30_p${sp}.json
  local sc_prec; sc_prec=$(sc_prec_for_uniform "$tag")
  local common=(
    --sc_ops_per_block_json "$sched"
    --sc_prec "$sc_prec"
    --batch_size "$BATCH" --workers "$WORKERS"
  )
  local uni_args=()
  if [[ $tag == avg192 ]]; then
    # SL=192 everywhere via SC early termination; --sc_prec is pow2-only so
    # routed through MP path with a single level [192].
    uni_args=(
      --mp_levels 192 --mp_fractions 1.0
      --mp_ops "$LINEAR_MP_OPS"
      --qk_mp_levels 192 --qk_mp_fractions 1.0
      --av_mp_levels 192 --av_mp_fractions 1.0
    )
  fi
  if [[ $qwt == on ]]; then
    echo python experiments/qwt_sc_overnight.py \
      "${common[@]}" "${uni_args[@]}" \
      --n_calib "$N_CALIB" --n_eval "$N_EVAL" \
      --comp_mode sc --comp_sc_prec "$COMP_SC_PREC" \
      --ridge "$RIDGE" \
      --cos_threshold "$COS_THR" \
      --last_block_cos_threshold "$LAST_COS_THR" \
      --calib_seed_b "$CALIB_SEED_B" \
      --lookahead_veto \
      "${ha_args[@]}" \
      --skip_baseline \
      --out_json "$out_json"
  else
    echo python eval.py --mode sc \
      "${common[@]}" "${uni_args[@]}" \
      --max_images "$N_EVAL" \
      --out_json "$out_json"
  fi
}

# Unified task queue. Format: three parallel arrays name / command / prereq.
TASK_NAMES=()
TASK_CMDS=()
TASK_PREREQ=()
TASK_OUT=()

add_task() {
  local name=$1 cmd=$2 prereq=$3 out=$4
  TASK_NAMES+=("$name")
  TASK_CMDS+=("$cmd")
  TASK_PREREQ+=("$prereq")
  TASK_OUT+=("$out")
}

# Search tasks (seed the pool so MP evals can proceed as soon as they finish).
[[ -f $MAP_P7 ]]     || add_task "search_p7"     "$(build_search_cmd p7 "$MAP_P7")"         ""  "$MAP_P7"
[[ -f $MAP_AVG192 ]] || add_task "search_avg192" "$(build_search_cmd avg192 "$MAP_AVG192")" ""  "$MAP_AVG192"

# Eval tasks. MP evals gate on their map.
EVAL_RUNS=(
  "p7     uniform off"  "p7     uniform on"
  "p7     mp      off"  "p7     mp      on"
  "p8     uniform off"  "p8     uniform on"
  "avg192 mp      off"  "avg192 mp      on"
  "avg192 uniform off"  "avg192 uniform on"
)
for entry in "${EVAL_RUNS[@]}"; do
  read -r tag mode qwt <<< "$entry"
  name="${tag}_${mode}_qwt${qwt}"
  outj=$RES_DIR/${name}.json
  prereq=""
  [[ $mode == mp ]] && prereq=$(sl_map_for_tag "$tag")
  add_task "$name" "$(build_eval_cmd "$tag" "$mode" "$qwt" "$name")" "$prereq" "$outj"
done

# --- Unified pool scheduler -------------------------------------------------

MAX_PAR=4
declare -A gpu_of_pid
declare -A pid_on_gpu
declare -A start_time_of_pid
declare -A gpu_blocked
declare -A out_of_pid      # pid -> expected output file (to tell success from contention)
declare -A name_of_pid     # pid -> task name (for logging)
active_pids=()
done_mask=()               # done_mask[i]=1 if task i already launched or skipped

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
  # Contention test: blacklist a GPU only when the worker (a) died quickly AND
  # (b) produced no output file. The prior heuristic blacklisted purely on
  # alive<60s, which false-positives short but successful runs (e.g. uniform
  # eval at small N_EVAL) as if they hit CUDA exclusive-mode contention.
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
        echo "[reap] pid=$p gpu=$g $name finished (alive=${alive}s, out=${out})"
      elif (( alive < 60 )); then
        gpu_blocked[$g]=1
        echo "[reap] pid=$p gpu=$g $name died in ${alive}s with no output — BLOCKING gpu=$g (contended)"
      else
        echo "[reap] pid=$p gpu=$g $name exited without output after ${alive}s (FAILED, gpu not blocked)" >&2
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

# Skip tasks whose output JSON already exists (idempotent reruns).
for i in "${!TASK_NAMES[@]}"; do
  done_mask[$i]=0
  if [[ -f ${TASK_OUT[$i]} ]]; then
    echo "[skip] ${TASK_OUT[$i]} exists (task ${TASK_NAMES[$i]})"
    done_mask[$i]=1
  fi
done

# Scheduler loop: launch any task whose prereq is satisfied onto any free GPU.
# When blocked (no eligible task + GPUs busy), wait for a pid and retry.
while true; do
  # Are we done?
  pending=0
  for i in "${!TASK_NAMES[@]}"; do
    [[ ${done_mask[$i]} == 0 ]] && pending=1 && break
  done
  (( pending == 0 && ${#active_pids[@]} == 0 )) && break

  # If slots full, wait for any pid to finish before trying to launch.
  if (( ${#active_pids[@]} >= MAX_PAR )); then
    wait -n 2>/dev/null || true
    reap_finished_pids
    continue
  fi

  # Try to launch the next eligible task.
  launched_any=0
  for i in "${!TASK_NAMES[@]}"; do
    [[ ${done_mask[$i]} != 0 ]] && continue
    prereq=${TASK_PREREQ[$i]}
    if [[ -n $prereq && ! -f $prereq ]]; then
      continue  # wait for search to finish
    fi
    gpu=$(pick_free_gpu) || break  # no healthy GPU; we'll wait below
    launch_task "$i" "$gpu"
    launched_any=1
    break
  done

  # If nothing launched (either all GPUs in use, or all pending have unmet prereqs),
  # wait for any pid to finish and retry. Avoid busy-loop.
  if (( launched_any == 0 )); then
    if (( ${#active_pids[@]} == 0 )); then
      # Stuck: remaining tasks all have unmet prereqs AND no pids running.
      # This means required searches failed. Report and exit.
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

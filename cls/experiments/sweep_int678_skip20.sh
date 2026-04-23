#!/usr/bin/env bash
# Overnight sweep: int 6/7/8 x {uniform, mp} x {w/, w/o QwT (r2>0.7)}
# Fixed schedule: skip_worst_20_of_144 per sensitivity-p.
# Runs 4 at a time on gl1807 (4x RTX Pro 6000).
#
# MP fractions: [0.15, 0.575, 0.225, 0.05] on levels [2T, T, T/2, T/4]
#   -> main_sl = T = 2^p exactly (overhead ignored; post-hoc corrected eff_sl
#      reported per QwT run using actual n_enabled_blocks from r2>0.7 gating).
set -euo pipefail

cd "$(dirname "$0")/.."
PROJ_ROOT=/home/allenjin/Projects/vit_sc
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vit_sc

export XFORMERS_DISABLED=1

mkdir -p logs/sweep_int678 results/sweep_int678
SCHED_DIR=$PROJ_ROOT/cls/sensitivity/skip_worst20
[[ -d $SCHED_DIR ]] || python experiments/build_skip_worst20_json.py

N_CALIB=1024
N_EVAL=1000
BATCH=16
WORKERS=8
R2_THR=0.5
COMP_SC_PREC=8
LINEAR_MP_OPS="mlp_fc1,mlp_fc2,qkv_proj,out_proj"

# (tag, mode, qwt). tag selects sensitivity file + sc_prec + MP levels/fractions.
# int 8 MP is dropped because it'd collapse to uniform when capped at SL-256.
# 'avg192' is an intermediate MP config (avg SL = 192) reusing p8 sensitivity.
RUNS=(
  "p6     uniform off"  "p6     uniform on"
  "p6     mp      off"  "p6     mp      on"
  "p7     uniform off"  "p7     uniform on"
  "p7     mp      off"  "p7     mp      on"
  "p8     uniform off"  "p8     uniform on"
  "avg192 mp      off"  "avg192 mp      on"
)

mp_levels_for_tag()    { case "$1" in p6) echo "128,64,32,16" ;; p7|avg192) echo "256,128,64,32" ;; p8) echo "256,128,64,32" ;; esac; }
mp_fractions_for_tag() { case "$1" in p6|p7) echo "0.15,0.575,0.225,0.05" ;; avg192) echo "0.575,0.3,0.075,0.05" ;; esac; }
sens_p_for_tag()       { case "$1" in p6) echo 6 ;; p7) echo 7 ;; p8|avg192) echo 8 ;; esac; }
# sc_prec: uniform → p; mp → ceil(log2(max_level)).
#   p6 uniform=6 / mp=7 (max 128);  p7 uniform=7 / mp=8 (max 256);
#   p8 uniform=8;                    avg192 mp=8 (max 256).
sc_prec_for() {
  local tag=$1 mode=$2
  if [[ $mode == uniform ]]; then
    case "$tag" in p6) echo 6 ;; p7) echo 7 ;; p8) echo 8 ;; esac
  else
    case "$tag" in p6) echo 7 ;; p7|avg192) echo 8 ;; esac
  fi
}

build_cmd() {
  local tag=$1 mode=$2 qwt=$3 gpu=$4 name=$5
  local sp; sp=$(sens_p_for_tag "$tag")
  local sched="$SCHED_DIR/skip_worst20_p${sp}.json"
  local sc_prec; sc_prec=$(sc_prec_for "$tag" "$mode")
  local common=(
    --sc_ops_per_block_json "$sched"
    --sc_prec "$sc_prec"
    --batch_size "$BATCH" --workers "$WORKERS"
  )
  local mp_args=()
  if [[ $mode == mp ]]; then
    local levels fracs
    levels=$(mp_levels_for_tag "$tag")
    fracs=$(mp_fractions_for_tag "$tag")
    mp_args=(
      --mp_levels "$levels" --mp_fractions "$fracs"
      --mp_ops "$LINEAR_MP_OPS"
      --qk_mp_levels "$levels" --qk_mp_fractions "$fracs"
      --av_mp_levels "$levels" --av_mp_fractions "$fracs"
    )
  fi
  if [[ $qwt == on ]]; then
    local out_json="results/sweep_int678/${name}.json"
    echo python experiments/qwt_sc_overnight.py \
      "${common[@]}" "${mp_args[@]}" \
      --n_calib "$N_CALIB" --n_eval "$N_EVAL" \
      --comp_mode sc --comp_sc_prec "$COMP_SC_PREC" \
      --r2_threshold "$R2_THR" --skip_baseline \
      --out_json "$out_json"
  else
    local out_json="results/sweep_int678/${name}.json"
    echo python eval.py --mode sc \
      "${common[@]}" "${mp_args[@]}" \
      --max_images "$N_EVAL" \
      --out_json "$out_json"
  fi
}

# Pool-based scheduling: as soon as ANY worker exits, launch the next pending
# one on the freed GPU. Keeps all 4 GPUs busy even when runs have wildly
# different durations (MP is 3-10x slower than uniform).
MAX_PAR=4
declare -A gpu_of_pid     # pid -> gpu index
declare -A pid_on_gpu     # gpu -> pid (reverse lookup)
declare -A start_time_of_pid   # pid -> SECONDS at launch
declare -A gpu_blocked    # gpu -> 1 if a job on it died in <60s (contended)
active_pids=()

launch_on_free_gpu() {
  local tag=$1 mode=$2 qwt=$3 name=$4 logf=$5
  # Pick the lowest-indexed GPU that has no active pid and is not blocked.
  local g pick=""
  for g in 0 1 2 3; do
    if [[ -z ${pid_on_gpu[$g]:-} && -z ${gpu_blocked[$g]:-} ]]; then
      pick=$g; break
    fi
  done
  if [[ -z $pick ]]; then
    echo "[wait] no healthy GPU free for $name" >&2
    return 1
  fi
  local cmd
  cmd=$(build_cmd "$tag" "$mode" "$qwt" "$pick" "$name")
  echo "[launch gpu=$pick] $name"
  ( CUDA_VISIBLE_DEVICES="$pick" eval "$cmd" ) </dev/null >"$logf" 2>&1 &
  local new_pid=$!
  gpu_of_pid[$new_pid]=$pick
  pid_on_gpu[$pick]=$new_pid
  start_time_of_pid[$new_pid]=$SECONDS
  active_pids+=("$new_pid")
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
      if (( alive < 60 )); then
        gpu_blocked[$g]=1
        echo "[reap] pid=$p gpu=$g died in ${alive}s — BLOCKING gpu=$g (contended)"
      else
        echo "[reap] pid=$p gpu=$g finished (alive=${alive}s)"
      fi
      unset "pid_on_gpu[$g]"
      unset "gpu_of_pid[$p]"
      unset "start_time_of_pid[$p]"
    fi
  done
  active_pids=("${new_active[@]}")
}

for entry in "${RUNS[@]}"; do
  read -r tag mode qwt <<< "$entry"
  name="${tag}_${mode}_qwt${qwt}"
  logf="logs/sweep_int678/${name}.log"
  outj="results/sweep_int678/${name}.json"
  if [[ -f $outj ]]; then
    echo "[skip] $outj exists"
    continue
  fi
  # Retry until we get a healthy GPU or run out of options.
  while true; do
    # If slots full, wait for any worker to exit then reap.
    if (( ${#active_pids[@]} >= MAX_PAR )); then
      wait -n 2>/dev/null || true
      reap_finished_pids
    fi
    if launch_on_free_gpu "$tag" "$mode" "$qwt" "$name" "$logf"; then
      break
    fi
    # launch failed (all healthy GPUs blocked or in use) → wait for any pid.
    if (( ${#active_pids[@]} == 0 )); then
      echo "[fatal] no healthy GPUs and no active workers; skipping $name" >&2
      break
    fi
    wait -n 2>/dev/null || true
    reap_finished_pids
  done
done

# Final drain.
echo "[drain] waiting on tail: ${active_pids[*]}"
while (( ${#active_pids[@]} > 0 )); do
  wait -n 2>/dev/null || true
  reap_finished_pids
done

echo "[sweep] all 12 runs complete"
ls -la results/sweep_int678/

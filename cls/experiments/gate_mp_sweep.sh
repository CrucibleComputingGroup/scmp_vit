#!/usr/bin/env bash
# Gate × MP-map sweep at N=1000.
#
# Goal: pick one (recipe, MP-map) combination where
#   (a) QwT-on >= QwT-off for every config, and
#   (b) MP >= uniform at matched main_sl (raw and compensated).
#
# Axes:
#   - 8 recipes = {r2 ∈ 0.4, 0.5} × {rmse_margin ∈ 0, 1.5} × {comp_mode ∈ sc, fp}
#     (all use lookahead_veto on, last_block_r2 = 0.9)
#   - 5 configs = {p7_uniform, p7_mp, p8_uniform, avg192_uniform, avg192_mp}
#   - MP map: {old (from sweep_int678_k30), new (regenerated with the bigger
#             search config: n_search=64, max_iters=12, pair_eval_topk=4,
#             lookahead_blocks=1; mlp_fc2 floor dropped for avg192)}
#
# Matrix: 3 uniform × 8 + 2 MP × 8 × 2 map-versions = 56 evals + 2 MP searches
#         + 2 raw-SC-on-new-maps (for the QwT-off baseline delta) = 60 tasks.
#
# All jobs share one 4-GPU pool. MP-new evals carry a prereq on their search.
#
# Smoke-test via env overrides:
#   N_EVAL=100 N_CALIB=128 SEARCH_N_SEARCH=8 SEARCH_MAX_ITERS=1 \
#   RES_DIR=results/gate_mp_smoke LOG_DIR=logs/gate_mp_smoke \
#     bash cls/experiments/gate_mp_sweep.sh

set -euo pipefail

cd "$(dirname "$0")/.."
PROJ_ROOT=/home/allenjin/Projects/vit_sc
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate vit_sc

export XFORMERS_DISABLED=1

# --- Env-overridable knobs (diagnostic defaults at N=1000) ---
N_CALIB="${N_CALIB:-1024}"
N_EVAL="${N_EVAL:-1000}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
COMP_SC_PREC="${COMP_SC_PREC:-8}"
RIDGE="${RIDGE:-1e-4}"
LAST_R2="${LAST_R2:-0.9}"

SEARCH_LEVELS="${SEARCH_LEVELS:-64,96,128,192,256}"
SEARCH_OPS="${SEARCH_OPS:-proj,mlp_fc1,mlp_fc2}"
SEARCH_PROXY="${SEARCH_PROXY:-comp_residual}"
SEARCH_N_SEARCH="${SEARCH_N_SEARCH:-64}"
SEARCH_MAX_ITERS="${SEARCH_MAX_ITERS:-12}"
SEARCH_SEED="${SEARCH_SEED:-1}"
SEARCH_PAIR_TOPK="${SEARCH_PAIR_TOPK:-4}"
SEARCH_LOOKAHEAD="${SEARCH_LOOKAHEAD:-1}"

LINEAR_MP_OPS="mlp_fc1,mlp_fc2,qkv_proj,out_proj"

LOG_DIR="${LOG_DIR:-logs/gate_mp_sweep_n1000}"
RES_DIR="${RES_DIR:-results/gate_mp_sweep_n1000}"
MAP_DIR="${MAP_DIR:-$RES_DIR/sl_maps}"
mkdir -p "$LOG_DIR" "$RES_DIR" "$MAP_DIR"

SCHED_DIR=$PROJ_ROOT/cls/sensitivity/skip_worst30
SENS_JSON=$PROJ_ROOT/results/sensitivity_all_ops.json
[[ -f $SENS_JSON ]] || { echo "[fatal] missing $SENS_JSON — run sweep_int678_skip30.sh once to synthesize it" >&2; exit 1; }
[[ -d $SCHED_DIR ]] || python experiments/build_skip_worst20_json.py --k 30 --out_dir sensitivity/skip_worst30

# --- Map paths ---
OLD_MAP_P7=$PROJ_ROOT/cls/results/sweep_int678_k30/sl_maps/p7_mp_main128_sl_map.json
OLD_MAP_AVG192=$PROJ_ROOT/cls/results/sweep_int678_k30/sl_maps/avg192_mp_main192_sl_map.json
NEW_MAP_P7=$MAP_DIR/p7_mp_new_main128_sl_map.json
NEW_MAP_AVG192=$MAP_DIR/avg192_mp_new_main192_sl_map.json

[[ -f $OLD_MAP_P7 ]]     || { echo "[fatal] missing $OLD_MAP_P7" >&2; exit 1; }
[[ -f $OLD_MAP_AVG192 ]] || { echo "[fatal] missing $OLD_MAP_AVG192" >&2; exit 1; }

# --- Recipe table (id : r2_threshold : rmse_margin : comp_mode) ---
RECIPES=(
  "r2p5_m0_sc:0.5:0.0:sc"
  "r2p5_m0_fp:0.5:0.0:fp"
  "r2p5_m15_sc:0.5:1.5:sc"
  "r2p5_m15_fp:0.5:1.5:fp"
  "r2p4_m0_sc:0.4:0.0:sc"
  "r2p4_m0_fp:0.4:0.0:fp"
  "r2p4_m15_sc:0.4:1.5:sc"
  "r2p4_m15_fp:0.4:1.5:fp"
)

# --- Helpers ---
target_main_sl_for_tag() { case "$1" in p7) echo 128 ;; avg192) echo 192 ;; esac; }
sens_p_for_tag()         { case "$1" in p7) echo 7 ;; p8|avg192) echo 8 ;; esac; }
sc_prec_for_uniform()    { case "$1" in p7) echo 7 ;; p8|avg192) echo 8 ;; esac; }

map_for() {
  # $1=tag (p7|avg192)  $2=ver (old|new)
  case "$1:$2" in
    p7:old)     echo "$OLD_MAP_P7" ;;
    p7:new)     echo "$NEW_MAP_P7" ;;
    avg192:old) echo "$OLD_MAP_AVG192" ;;
    avg192:new) echo "$NEW_MAP_AVG192" ;;
  esac
}

# --- Search command (new MP maps) ---
build_search_cmd() {
  local tag=$1 out=$2
  local target; target=$(target_main_sl_for_tag "$tag")
  # At target=192, the mlp_fc2=192 floor is a no-op that freezes 24 slots.
  # For p7 (target=128), keep it at 192 (shields a comp-hostile op).
  local op_min_arg=""
  if [[ $tag == p7 ]]; then
    op_min_arg="--op_min_levels mlp_fc2=192"
  fi
  echo python experiments/mp_budget_swap_search.py \
    --sc_config skip_worst30 \
    --target_main_sl "$target" \
    --levels "$SEARCH_LEVELS" \
    --search_ops "$SEARCH_OPS" \
    --fixed_ops "qk=${target},av=${target}" \
    $op_min_arg \
    --proxy "$SEARCH_PROXY" \
    --init_mode uniform_repair --init_level "$target" \
    --n_search "$SEARCH_N_SEARCH" --search_seed "$SEARCH_SEED" \
    --max_iters "$SEARCH_MAX_ITERS" \
    --pair_eval_topk "$SEARCH_PAIR_TOPK" \
    --lookahead_blocks "$SEARCH_LOOKAHEAD" \
    --sc_prec 8 \
    --out_json "$out"
}

# --- Eval command builders ---
build_mp_eval_cmd() {
  # $1=tag  $2=map_ver  $3=r2  $4=margin  $5=comp  $6=out_json
  local tag=$1 ver=$2 r2=$3 marg=$4 comp=$5 out=$6
  local map; map=$(map_for "$tag" "$ver")
  echo python experiments/eval_custom_sl_map_fpcomp.py \
    --sc_config skip_worst30 \
    --sl_map_json "$map" \
    --sc_prec 8 \
    --n_calib "$N_CALIB" --n_eval "$N_EVAL" \
    --batch_size "$BATCH" --workers "$WORKERS" \
    --comp_mode "$comp" --comp_sc_prec "$COMP_SC_PREC" \
    --ridge "$RIDGE" \
    --r2_threshold "$r2" \
    --last_block_r2_threshold "$LAST_R2" \
    --rmse_margin "$marg" \
    --lookahead_veto \
    --out_json "$out"
}

build_uniform_eval_cmd() {
  # $1=tag (p7|p8|avg192)  $2=r2  $3=margin  $4=comp  $5=out_json
  local tag=$1 r2=$2 marg=$3 comp=$4 out=$5
  local sp; sp=$(sens_p_for_tag "$tag")
  local sched=$SCHED_DIR/skip_worst30_p${sp}.json
  local sc_prec; sc_prec=$(sc_prec_for_uniform "$tag")
  local uni_args=""
  if [[ $tag == avg192 ]]; then
    # SL=192 via SC early-termination, routed through MP path w/ single level
    uni_args="--mp_levels 192 --mp_fractions 1.0 --mp_ops $LINEAR_MP_OPS --qk_mp_levels 192 --qk_mp_fractions 1.0 --av_mp_levels 192 --av_mp_fractions 1.0"
  fi
  echo python experiments/qwt_sc_overnight.py \
    --sc_ops_per_block_json "$sched" \
    --sc_prec "$sc_prec" \
    --batch_size "$BATCH" --workers "$WORKERS" \
    $uni_args \
    --n_calib "$N_CALIB" --n_eval "$N_EVAL" \
    --comp_mode "$comp" --comp_sc_prec "$COMP_SC_PREC" \
    --ridge "$RIDGE" \
    --r2_threshold "$r2" \
    --last_block_r2_threshold "$LAST_R2" \
    --rmse_margin "$marg" \
    --lookahead_veto \
    --skip_baseline \
    --out_json "$out"
}

build_raw_mp_cmd() {
  # Raw SC (QwT off) on an MP map, to get the QwT-on vs QwT-off delta.
  local tag=$1 ver=$2 out=$3
  local map; map=$(map_for "$tag" "$ver")
  echo python experiments/eval_custom_sl_map_fpcomp.py \
    --sc_config skip_worst30 \
    --sl_map_json "$map" \
    --sc_prec 8 \
    --n_calib "$N_CALIB" --n_eval "$N_EVAL" \
    --batch_size "$BATCH" --workers "$WORKERS" \
    --comp_mode sc --comp_sc_prec "$COMP_SC_PREC" \
    --ridge "$RIDGE" \
    --skip_qwt \
    --out_json "$out"
}

# --- Task queue (same structure as sweep_int678_skip30.sh) ---
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

# (1) MP searches (no prereq; seed the pool)
[[ -f $NEW_MAP_P7 ]]     || add_task "search_p7_new"     "$(build_search_cmd p7 "$NEW_MAP_P7")"         ""  "$NEW_MAP_P7"
[[ -f $NEW_MAP_AVG192 ]] || add_task "search_avg192_new" "$(build_search_cmd avg192 "$NEW_MAP_AVG192")" ""  "$NEW_MAP_AVG192"

# (2) Raw-SC (QwT off) on new MP maps (prereq on their search)
add_task "p7_mp_new_raw"     "$(build_raw_mp_cmd p7 new "$RES_DIR/p7_mp_new_raw.json")"         "$NEW_MAP_P7"     "$RES_DIR/p7_mp_new_raw.json"
add_task "avg192_mp_new_raw" "$(build_raw_mp_cmd avg192 new "$RES_DIR/avg192_mp_new_raw.json")" "$NEW_MAP_AVG192" "$RES_DIR/avg192_mp_new_raw.json"

# (3) Uniform × recipes (no prereq, no MP map needed)
for cfg in p7_uniform p8_uniform avg192_uniform; do
  tag=${cfg%_uniform}
  for spec in "${RECIPES[@]}"; do
    IFS=':' read -r rid r2 marg comp <<< "$spec"
    name="${cfg}_${rid}"
    outj=$RES_DIR/${name}.json
    cmd=$(build_uniform_eval_cmd "$tag" "$r2" "$marg" "$comp" "$outj")
    add_task "$name" "$cmd" "" "$outj"
  done
done

# (4) MP × map_ver × recipes
for cfg in p7_mp avg192_mp; do
  tag=${cfg%_mp}
  for ver in old new; do
    for spec in "${RECIPES[@]}"; do
      IFS=':' read -r rid r2 marg comp <<< "$spec"
      name="${cfg}_${ver}_${rid}"
      outj=$RES_DIR/${name}.json
      cmd=$(build_mp_eval_cmd "$tag" "$ver" "$r2" "$marg" "$comp" "$outj")
      local_prereq=""
      [[ $ver == new ]] && local_prereq=$(map_for "$tag" new)
      add_task "$name" "$cmd" "$local_prereq" "$outj"
    done
  done
done

echo "[sweep] queued ${#TASK_NAMES[@]} tasks total"

# --- Pool scheduler (adapted from sweep_int678_skip30.sh) ---
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
        echo "[reap] pid=$p gpu=$g $name died in ${alive}s with no output — BLOCKING gpu=$g (contended)"
      else
        echo "[reap] pid=$p gpu=$g $name FAILED after ${alive}s (gpu not blocked)" >&2
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

# Skip already-completed tasks (idempotent reruns)
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
      continue
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

# --- Emit compact summary CSV + table ---
python3 - "$RES_DIR" <<'PY'
import json, os, sys, csv
from pathlib import Path
rd = Path(sys.argv[1])
rows = []
for p in sorted(rd.glob("*.json")):
    try:
        d = json.load(open(p))
    except Exception as e:
        continue
    name = p.stem
    # Uniform runs (qwt_sc_overnight output): top-level 'top1'/'top5' OR nested; also 'summary'
    # MP runs (eval_custom_sl_map_fpcomp): results stored under "results".{sc_comp,sc_raw,fp}
    top1 = top5 = None
    if "results" in d:
        r = d["results"]
        if "sc_comp" in r: top1, top5 = r["sc_comp"].get("top1"), r["sc_comp"].get("top5")
        elif "sc_raw" in r: top1, top5 = r["sc_raw"].get("top1"), r["sc_raw"].get("top5")
    elif "summary" in d:
        s = d["summary"]
        top1 = s.get("sc_comp_top1") or s.get("sc_top1")
        top5 = s.get("sc_comp_top5") or s.get("sc_top5")
    else:
        top1 = d.get("top1")
        top5 = d.get("top5")
    rows.append({"name": name, "top1": top1, "top5": top5})

out = rd / "summary.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name","top1","top5"])
    w.writeheader(); w.writerows(rows)

# Pretty print
print(f"\n=== summary ({len(rows)} rows) ===")
print(f"{'name':70s}  {'top1':>7s}  {'top5':>7s}")
for r in rows:
    t1 = f"{r['top1']*100:.2f}" if isinstance(r['top1'], (int,float)) else "?"
    t5 = f"{r['top5']*100:.2f}" if isinstance(r['top5'], (int,float)) else "?"
    print(f"{r['name']:70s}  {t1:>7s}  {t5:>7s}")
print(f"\n[summary] wrote {out}")
PY

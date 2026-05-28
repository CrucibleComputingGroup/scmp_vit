#!/usr/bin/env bash
# Launch run_coco_val.py across 8 GPUs (one process per GPU, contiguous shards).
#
# Usage:
#   bash det/run_coco_val_8gpu.sh \
#       --sc_prec 7 \
#       --skip det/sensitivity/skip/skip_worst_20_int7.json \
#       --out_dir det/results/int7_skip20 \
#       --d2_datasets /home/azrsadmin/vit_sc/data \
#       --ckpt /path/to/eva_coco_det.pth
#
# Each GPU writes to <out_dir>/shard_<i>/. Tail logs in <out_dir>/shard_<i>.log.
# After all 8 finish, runs merge_shards.py to produce <out_dir>/metrics.json.
#
# Re-run the same command with --resume appended to continue from any
# crashed/cancelled shards.

set -euo pipefail

NUM_SHARDS="${NUM_SHARDS:-8}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PYTHON="${PYTHON:-/home/azrsadmin/miniconda3/envs/qwt_d2/bin/python}"

# Pull --out_dir out of args; pass everything else through to each shard.
OUT_DIR=""
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out_dir)    OUT_DIR="$2"; shift 2 ;;
    --out_dir=*)  OUT_DIR="${1#*=}"; shift ;;
    *)            PASS_ARGS+=("$1"); shift ;;
  esac
done
if [[ -z "$OUT_DIR" ]]; then
  echo "error: --out_dir is required" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
if [[ ${#GPU_ARR[@]} -ne $NUM_SHARDS ]]; then
  echo "error: GPUS=$GPUS has ${#GPU_ARR[@]} entries but NUM_SHARDS=$NUM_SHARDS" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/det/run_coco_val.py"
MERGE="$REPO_ROOT/det/merge_shards.py"

echo "[launcher] $NUM_SHARDS shards on GPUs [$GPUS] -> $OUT_DIR"
declare -a PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  GPU="${GPU_ARR[$i]}"
  SHARD_DIR="$OUT_DIR/shard_$i"
  LOG="$OUT_DIR/shard_$i.log"
  mkdir -p "$SHARD_DIR"
  echo "  shard $i  GPU=$GPU  -> $SHARD_DIR (log: $LOG)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT" \
      --num_shards "$NUM_SHARDS" --shard_id "$i" \
      --out_dir "$SHARD_DIR" \
      "${PASS_ARGS[@]}" \
      > "$LOG" 2>&1 &
  PIDS+=($!)
done

echo "[launcher] PIDs: ${PIDS[*]}"
echo "[launcher] tail -f $OUT_DIR/shard_*.log to monitor"

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "[launcher] pid $pid exited non-zero" >&2
    FAIL=1
  fi
done

if [[ $FAIL -ne 0 ]]; then
  echo "[launcher] one or more shards failed; not merging." >&2
  echo "[launcher] inspect $OUT_DIR/shard_*.log, then re-run with --resume." >&2
  exit 1
fi

echo "[launcher] all shards done; merging."
# Forward d2_datasets to merge if it was passed.
MERGE_ARGS=()
for ((i=0; i<${#PASS_ARGS[@]}; i++)); do
  if [[ "${PASS_ARGS[$i]}" == "--d2_datasets" ]]; then
    MERGE_ARGS+=("--d2_datasets" "${PASS_ARGS[$((i+1))]}")
  fi
done
"$PYTHON" "$MERGE" "$OUT_DIR" "${MERGE_ARGS[@]}"

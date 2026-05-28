#!/usr/bin/env bash
# Launch a detached tmux session that runs SC eval at int6, int7, int8
# sequentially, each on 8 GPUs (8-shard data-parallel via run_coco_val_8gpu.sh).
#
# Defaults baked into this script (rely on run_coco_val.py defaults):
#   --skip auto        -> det/sensitivity/skip/skip_worst_20_int<N>.json
#   --size 1024        -> square_pad + ResizeShortestEdge (beit pos embed)
#   soft-NMS ON        -> requires mmcv (install via
#                         third_party/QwT-SC/QwT-det-RepQ-ViT/eva1/install_mmcv.sh)
#
# Usage:
#   bash det/run_int_sweep_tmux.sh                # default: int6 7 8
#   PRECS="6 8"        bash det/run_int_sweep_tmux.sh
#   LENGTHS="96 192"   bash det/run_int_sweep_tmux.sh   # arbitrary lengths
#   SIZE=1280  bash det/run_int_sweep_tmux.sh   # override the 1024 default
#
# Env overrides:
#   SESSION       tmux session name (default coco_sc_sweep)
#   PRECS         space-sep int precisions  (default "6 7 8")
#   LENGTHS       space-sep stream lengths  (default empty, ignored if set)
#                 If set, PRECS is ignored.
#   SIZE          --size for run_coco_val.py (default 1024)
#   SKIP_PCT      --skip_pct (default 20)
#   RES_ROOT      output root (default <repo>/det/results)
#   D2_DATASETS   COCO root  (default /home/azrsadmin/vit_sc/data)
#   CKPT          EVA .pth   (default $D2_DATASETS/pretrained/eva_coco_det.pth)
#
# After launch:
#   tmux attach -t <SESSION>
#   tail -F <RES_ROOT>/<tag>/run.log
#   tmux kill-session -t <SESSION>

set -euo pipefail

SESSION="${SESSION:-coco_sc_sweep}"
PRECS="${PRECS:-6 7 8}"
LENGTHS="${LENGTHS:-}"
SIZE="${SIZE:-1024}"
SKIP_PCT="${SKIP_PCT:-20}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RES_ROOT="${RES_ROOT:-$REPO/det/results}"
D2_DATASETS="${D2_DATASETS:-/home/azrsadmin/vit_sc/data}"
CKPT="${CKPT:-$D2_DATASETS/pretrained/eva_coco_det.pth}"
LAUNCH="$REPO/det/run_coco_val_8gpu.sh"

# --- Outer: spawn tmux and exec inner. ---
if [[ -z "${TMUX:-}" ]]; then
  if ! command -v tmux >/dev/null; then
    echo "error: tmux not found" >&2; exit 1
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "error: tmux session '$SESSION' already exists." >&2
    echo "  attach: tmux attach -t $SESSION" >&2
    echo "  kill:   tmux kill-session -t $SESSION" >&2
    exit 2
  fi
  for f in "$LAUNCH" "$CKPT"; do
    [[ -e "$f" ]] || { echo "error: missing $f" >&2; exit 3; }
  done
  echo "[start] tmux session '$SESSION'"
  if [[ -n "$LENGTHS" ]]; then
    echo "  configs: lengths=[$LENGTHS]  size=$SIZE  skip_pct=$SKIP_PCT"
  else
    echo "  configs: int=[$PRECS]  size=$SIZE  skip_pct=$SKIP_PCT"
  fi
  echo "  res:  $RES_ROOT"
  echo "  ckpt: $CKPT"
  echo "  data: $D2_DATASETS"
  RES_ROOT="$RES_ROOT" D2_DATASETS="$D2_DATASETS" CKPT="$CKPT" \
  PRECS="$PRECS" LENGTHS="$LENGTHS" SIZE="$SIZE" SKIP_PCT="$SKIP_PCT" \
      tmux new-session -d -s "$SESSION" "bash '$0'; \
          echo; echo '=== ALL JOBS DONE — press enter to close session ==='; \
          read -r"
  echo "[OK] launched."
  echo "  attach: tmux attach -t $SESSION"
  exit 0
fi

# --- Inner (running inside tmux): run each config sequentially. ---
run_one() {
  local mode="$1"      # "prec" | "length"
  local val="$2"       # e.g. 7  or  96
  local tag
  if [[ "$mode" == "prec" ]]; then
    tag="int${val}_skip${SKIP_PCT}_sz${SIZE}"
  else
    tag="len${val}_skip${SKIP_PCT}_sz${SIZE}"
  fi
  local out_dir="$RES_ROOT/$tag"
  local log="$out_dir/run.log"
  mkdir -p "$out_dir"
  echo "=== $(date -Iseconds)  start $tag ===" | tee -a "$log"
  set +e
  if [[ "$mode" == "prec" ]]; then
    bash "$LAUNCH" \
        --sc_prec "$val" \
        --skip_pct "$SKIP_PCT" \
        --size "$SIZE" \
        --out_dir "$out_dir" \
        --d2_datasets "$D2_DATASETS" \
        --ckpt "$CKPT" 2>&1 | tee -a "$log"
  else
    bash "$LAUNCH" \
        --length "$val" \
        --skip_pct "$SKIP_PCT" \
        --size "$SIZE" \
        --out_dir "$out_dir" \
        --d2_datasets "$D2_DATASETS" \
        --ckpt "$CKPT" 2>&1 | tee -a "$log"
  fi
  local rc=${PIPESTATUS[0]}
  set -e
  echo "=== $(date -Iseconds)  end   $tag  rc=$rc ===" | tee -a "$log"
  return "$rc"
}

if [[ -n "$LENGTHS" ]]; then
  for L in $LENGTHS; do
    run_one length "$L" || echo "[warn] length=$L returned non-zero — proceeding."
  done
else
  for P in $PRECS; do
    run_one prec "$P" || echo "[warn] sc_prec=$P returned non-zero — proceeding."
  done
fi

echo "[done] all configs finished."

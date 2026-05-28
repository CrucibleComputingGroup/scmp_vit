#!/usr/bin/env bash
# Launch a detached tmux session that runs, sequentially:
#   1) int7 skip_worst_20 on 8 GPUs
#   2) int6 skip_worst_20 on 8 GPUs
#
# Each run uses run_coco_val_8gpu.sh (per-shard checkpoints, auto-merge).
# If int7 fails, int6 is still attempted (so a single bad shard doesn't
# block the second config — you can resume int7 separately later).
#
# Usage:
#   bash det/run_int7_int6_tmux.sh
#
# Env overrides (optional):
#   SESSION       tmux session name (default coco_sc)
#   RES_ROOT      output root (default <repo>/det/results)
#   D2_DATASETS   COCO root      (default /home/azrsadmin/vit_sc/data)
#   CKPT          EVA .pth path  (default <D2_DATASETS>/pretrained/eva_coco_det.pth)
#
# After launch:
#   tmux attach -t <SESSION>            # watch live
#   tail -F <RES_ROOT>/int{7,6}_skip20/run.log
#   tmux kill-session -t <SESSION>      # abort

set -euo pipefail

SESSION="${SESSION:-coco_sc}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RES_ROOT="${RES_ROOT:-$REPO/det/results}"
D2_DATASETS="${D2_DATASETS:-/home/azrsadmin/vit_sc/data}"
CKPT="${CKPT:-$D2_DATASETS/pretrained/eva_coco_det.pth}"

LAUNCH="$REPO/det/run_coco_val_8gpu.sh"
INT7_DIR="$RES_ROOT/int7_skip20"
INT6_DIR="$RES_ROOT/int6_skip20"
INT7_SKIP="$REPO/det/sensitivity/skip/skip_worst_20_int7.json"
INT6_SKIP="$REPO/det/sensitivity/skip/skip_worst_20_int6.json"

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
  for f in "$LAUNCH" "$INT7_SKIP" "$INT6_SKIP" "$CKPT"; do
    [[ -e "$f" ]] || { echo "error: missing $f" >&2; exit 3; }
  done
  mkdir -p "$INT7_DIR" "$INT6_DIR"

  echo "[start] tmux session '$SESSION'"
  echo "  int7 -> $INT7_DIR"
  echo "  int6 -> $INT6_DIR"
  echo "  ckpt: $CKPT"
  echo "  data: $D2_DATASETS"

  # Re-exec self inside tmux. Env vars propagate via tmux's environment.
  RES_ROOT="$RES_ROOT" D2_DATASETS="$D2_DATASETS" CKPT="$CKPT" \
      tmux new-session -d -s "$SESSION" "bash '$0'; \
          echo; echo '=== ALL JOBS DONE — press enter to close session ==='; \
          read -r"
  echo "[OK] launched."
  echo "  attach: tmux attach -t $SESSION"
  echo "  tail:   tail -F $INT7_DIR/run.log $INT6_DIR/run.log"
  exit 0
fi

# --- Inner (running inside tmux): execute both jobs sequentially. ---
run_one() {
  local sc_prec="$1" skip_json="$2" out_dir="$3"
  local log="$out_dir/run.log"
  echo "=== $(date -Iseconds)  start sc_prec=$sc_prec  out=$out_dir ===" \
      | tee -a "$log"
  set +e
  bash "$LAUNCH" \
      --sc_prec "$sc_prec" \
      --skip "$skip_json" \
      --out_dir "$out_dir" \
      --d2_datasets "$D2_DATASETS" \
      --ckpt "$CKPT" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "=== $(date -Iseconds)  end   sc_prec=$sc_prec  rc=$rc ===" \
      | tee -a "$log"
  return "$rc"
}

run_one 7 "$INT7_SKIP" "$INT7_DIR" || \
    echo "[warn] int7 returned non-zero — proceeding to int6 anyway."
run_one 6 "$INT6_SKIP" "$INT6_DIR" || \
    echo "[warn] int6 returned non-zero."

echo "[done] both runs finished."

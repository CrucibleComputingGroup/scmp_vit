#!/usr/bin/env bash
# Smoke-test the manual avg192_mp sl_map at n_eval=10 (with/without QwT).
# Intended to short-circuit search_avg192 (40h with the comp_residual proxy).
# Runs sequentially on a single GPU (CUDA_VISIBLE_DEVICES env or arg).
#
# Usage on gl1807 interactive shell:
#   bash experiments/_run_manual_avg192_n10.sh        # uses GPU 0
#   bash experiments/_run_manual_avg192_n10.sh 1      # picks GPU 1
set -euo pipefail
cd "$(dirname "$0")/.."   # → det/

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1   # surface print()s immediately under shell redirect

source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate qwt_d2
module load gcc/13.2.0

RES_DIR="results/manual_avg192_n10"
LOG_DIR="logs/manual_avg192_n10"
SL_MAP="$RES_DIR/sl_maps/avg192_mp_manual_sl_map.json"
mkdir -p "$RES_DIR" "$LOG_DIR"

[[ -f "$SL_MAP" ]] || { echo "[fatal] missing $SL_MAP"; exit 1; }

echo "=========================================="
echo "[$(date +%F\ %T)] GPU=$GPU  manual avg192_mp smoke (n_eval=10)"
echo "  sl_map: $SL_MAP"
echo "=========================================="

# --- Cell A: raw SC (no QwT) ------------------------------------------------
NAME_A="avg192_mp_manual_qwtoff"
LOG_A="$LOG_DIR/${NAME_A}.log"
OUT_A="$RES_DIR/${NAME_A}/metrics.json"
echo "[$(date +%T)] [A] launching $NAME_A → $LOG_A"
python sc_eval.py \
    --sc_prec 8 \
    --sl_map_json "$SL_MAP" \
    --n-eval 10 \
    --out-tag "$NAME_A" \
    --out_dir "$RES_DIR/$NAME_A" \
    > "$LOG_A" 2>&1
echo "[$(date +%T)] [A] done; tail:"
tail -25 "$LOG_A"
[[ -f "$OUT_A" ]] && { echo "[A] metrics: $OUT_A"; python -c "import json; m=json.load(open('$OUT_A')); print(json.dumps(m, indent=2))"; }

# --- Cell B: SC + cross-seed QwT (with QwT) ---------------------------------
NAME_B="avg192_mp_manual_qwton"
LOG_B="$LOG_DIR/${NAME_B}.log"
OUT_B="$RES_DIR/${NAME_B}.json"
echo
echo "[$(date +%T)] [B] launching $NAME_B → $LOG_B"
python experiments/qwt_det_compensate.py \
    --sc_prec 8 \
    --sl_map_json "$SL_MAP" \
    --n_calib 16 --n_eval 10 \
    --calib_seed 1 --calib_seed_b 2 \
    --cos_threshold 0.5 --last_block_cos_threshold 0.8 \
    --lookahead_veto \
    --comp_mode sc --comp_sc_prec 8 \
    --head_aligned --n_heads 16 \
    --ridge 1e-2 --fwd_chunk 2 \
    --out_tag "$NAME_B" \
    --out_json "$OUT_B" \
    > "$LOG_B" 2>&1
echo "[$(date +%T)] [B] done; tail:"
tail -25 "$LOG_B"
[[ -f "$OUT_B" ]] && { echo "[B] metrics: $OUT_B"; python -c "import json; m=json.load(open('$OUT_B')); print(json.dumps({k:m[k] for k in ('bbox_AP','segm_AP','admitted_blocks','config') if k in m}, indent=2, default=str))"; }

echo
echo "=========================================="
echo "[$(date +%F\ %T)] both cells complete"
echo "  A (raw SC):     $OUT_A"
echo "  B (SC + QwT):   $OUT_B"
echo "=========================================="

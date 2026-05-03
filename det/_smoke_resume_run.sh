#!/bin/bash
# Smoke test for checkpoint/resume in qwt_det_compensate.py.
# Phase 1: run, SIGINT after >= 20 eval images saved.
# Phase 2: resume, verify it skips done images and finishes.
set -uo pipefail
cd /home/yjrcs/SC_V/vit_sc/det
source /home/yjrcs/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2

OUT_TAG="_smoke_resume"
OUT_DIR="results/${OUT_TAG}"
LOG1="${OUT_DIR}/run1.log"
LOG2="${OUT_DIR}/run2.log"
mkdir -p "${OUT_DIR}"

CMD=(python -u experiments/qwt_det_compensate.py
     --sc_prec 7
     --sc_ops_per_block_json sensitivity/skip/skip_worst_30_int7.json
     --n_calib 2 --n_eval 30 --size 1024
     --save_every 10
     --skip_baseline
     --head_aligned --n_heads 16 --lookahead_veto
     --comp_sc_prec 8
     --cos_threshold 0.68 --last_block_cos_threshold 0.8
     --use_soft_nms --interp_type beit
     --out_tag "${OUT_TAG}"
     --out_json "${OUT_DIR}/run.json")

phase=$1
if [[ "${phase}" == "phase1" ]]; then
  echo "[smoke] phase 1 (no resume)"
  "${CMD[@]}" > "${LOG1}" 2>&1 &
  PID=$!
  echo "[smoke] pid=${PID}"
  echo "${PID}" > "${OUT_DIR}/run1.pid"
elif [[ "${phase}" == "phase2" ]]; then
  echo "[smoke] phase 2 (--resume)"
  "${CMD[@]}" --resume > "${LOG2}" 2>&1 &
  PID=$!
  echo "[smoke] pid=${PID}"
  echo "${PID}" > "${OUT_DIR}/run2.pid"
else
  echo "usage: $0 {phase1|phase2}"; exit 1
fi

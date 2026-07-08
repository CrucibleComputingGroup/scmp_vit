#!/bin/bash
# QwT len192 sw30 sz1024 n=5000 soft+beit at cos=0.40, n_calib=128.
# Launched directly on the interactive node (no sbatch). Resumable via
# checkpoint.pt under results/<out_tag>/sc_comp/. Always pass --resume:
# safe on first run (no checkpoint -> falls through), required on rerun.
set -uo pipefail
cd /home/yjrcs/SC_V/vit_sc/det
source /home/yjrcs/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2

OUT_TAG="qwt_len192_sw30_sz1024_n5000_soft_beit_cos040"
OUT_DIR="results/${OUT_TAG}"
mkdir -p "${OUT_DIR}" logs
LOG="${OUT_DIR}/run.log"

echo "[launch] $(date)  host=$(hostname)  OUT_TAG=${OUT_TAG}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

CMD=(python -u experiments/qwt_det_compensate.py
     --sc_prec 8 --stoc_len 192
     --sc_ops_per_block_json sensitivity/skip/skip_worst_30_len192.json
     --n_calib 128 --n_eval 5000 --size 1024
     --use_soft_nms --interp_type beit
     --head_aligned --n_heads 16 --lookahead_veto
     --comp_mode sc --comp_sc_prec 8 --fwd_chunk 2
     --cos_threshold 0.40 --last_block_cos_threshold 0.8
     --skip_baseline
     --save_every 50 --resume
     --out_tag "${OUT_TAG}"
     --out_json "${OUT_DIR}/run.json")

echo "[launch] cmd: ${CMD[*]}" | tee -a "${LOG}"
nohup "${CMD[@]}" >> "${LOG}" 2>&1 &
PID=$!
disown $PID
echo "${PID}" > "${OUT_DIR}/run.pid"
echo "[launch] pid=${PID}  log=${LOG}"

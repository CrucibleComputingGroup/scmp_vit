#!/bin/bash
#SBATCH --job-name=qwt_sw10
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=results/qwt_det/%x_%j.out
#SBATCH --error=results/qwt_det/%x_%j.out

# Usage:
#   sbatch --job-name=qwt_int7_sw10_n64  --export=ALL,SC_PREC=7,N_CALIB=64  sbatch_qwt_sw10.sh
#   sbatch --job-name=qwt_int7_sw10_n128 --export=ALL,SC_PREC=7,N_CALIB=128 sbatch_qwt_sw10.sh
#   sbatch --job-name=qwt_int6_sw10_n64  --export=ALL,SC_PREC=6,N_CALIB=64  sbatch_qwt_sw10.sh
#   sbatch --job-name=qwt_int6_sw10_n128 --export=ALL,SC_PREC=6,N_CALIB=128 sbatch_qwt_sw10.sh

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2
cd /home/yjrcs/SC_V/vit_sc/det

SC_PREC="${SC_PREC:?must set SC_PREC}"
N_CALIB="${N_CALIB:?must set N_CALIB}"
N_EVAL="${N_EVAL:-100}"
RIDGE="${RIDGE:-0.01}"
AVG_SC_DRAWS="${AVG_SC_DRAWS:-1}"
MIN_R2="${MIN_R2:-0.0}"
TAG_SUFFIX="${TAG_SUFFIX:-}"

TAG="qwt_int${SC_PREC}_sw10_n${N_CALIB}${TAG_SUFFIX}"
SCHED="results/sensitivity_int${SC_PREC}_n50/schedules/skip_worst_10.json"
OUT_DIR="results/qwt_det/${TAG}"
mkdir -p "$OUT_DIR"

echo "=== ${TAG}  sc_prec=${SC_PREC}  n_calib=${N_CALIB}  n_eval=${N_EVAL} ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

python -u experiments/qwt_det_compensate.py \
    --sc_prec "$SC_PREC" \
    --sc_ops_per_block_json "$SCHED" \
    --n_calib "$N_CALIB" --n_eval "$N_EVAL" \
    --ridge "$RIDGE" --min_r2 "$MIN_R2" \
    --fwd_chunk 2 --avg_sc_draws "$AVG_SC_DRAWS" \
    --skip_baseline \
    --out_tag "$TAG" \
    --out_json "${OUT_DIR}/run.json" \
    2>&1 | tee "${OUT_DIR}/log.txt"

echo "=== DONE ${TAG} ==="
grep -E "SC \+ QwT|bbox AP|r2=" "${OUT_DIR}/log.txt" | tail -10

#!/bin/bash
#SBATCH --job-name=ilp_n5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --output=logs/sbatch_%x_%j.out
#SBATCH --error=logs/sbatch_%x_%j.err
#
# n=5000 ILP-eval. Estimated ~20h MP runtime; 24h walltime + --resume in
# case of preemption. sc_eval saves checkpoint.pt every 50 imgs so a
# resubmit (with RESUME=1) picks up where the previous run was killed.
#
#   TAG=ilpA SL_MAP=sensitivity/sl_maps/p7_sw30l2_ilpA_top20floor.json \
#     sbatch experiments/submit_ilp_n5000.sbatch
#
# Resume:
#   sbatch --export=ALL,RESUME=1 ... <same args>
set -euo pipefail
cd /home/yjrcs/SC_V/vit_sc/det
mkdir -p logs results

: "${TAG:?must set TAG}"
: "${SL_MAP:?must set SL_MAP (relative to det/ or absolute)}"
: "${SC_PREC:=8}"
: "${SIZE:=1024}"
: "${N_EVAL:=5000}"
: "${SEED:=0}"

source /home/yjrcs/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2

OUT_TAG="p${SC_PREC}_${TAG}_n${N_EVAL}_seed${SEED}"
echo "[sbatch] job=$SLURM_JOB_ID node=$SLURMD_NODENAME TAG=$TAG SL_MAP=$SL_MAP  $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
md5sum "$SL_MAP"

EXTRA=()
if [[ "${RESUME:-0}" == "1" ]]; then EXTRA+=(--resume); fi

python -u sc_eval.py \
    --n-eval "$N_EVAL" \
    --sc_prec "$SC_PREC" \
    --seed "$SEED" \
    --size "$SIZE" \
    --sl_map_json "$SL_MAP" \
    --use_soft_nms \
    --interp_type beit \
    --save_every 50 \
    --out-tag "$OUT_TAG" \
    "${EXTRA[@]}"

echo "[sbatch] done $(date)"

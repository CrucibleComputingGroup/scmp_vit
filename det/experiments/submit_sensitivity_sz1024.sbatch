#!/bin/bash
#SBATCH --job-name=det_sens
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=nbleier_owned1
#SBATCH --output=sensitivity/logs/sbatch_%x_%j.out
#SBATCH --error=sensitivity/logs/sbatch_%x_%j.err
#
# Submit per-length:
#   LEN=256 sbatch --job-name=det_sens_int8   experiments/submit_sensitivity_sz1024.sbatch
#   LEN=192 sbatch --job-name=det_sens_len192 experiments/submit_sensitivity_sz1024.sbatch
#   LEN=96  sbatch --job-name=det_sens_len96  experiments/submit_sensitivity_sz1024.sbatch

set -euo pipefail
cd /home/yjrcs/SC_V/vit_sc/det
mkdir -p sensitivity/logs

: "${LEN:?must set LEN env var (e.g. LEN=256)}"
: "${N_EVAL:=100}"
: "${SIZE:=1024}"

echo "[sbatch] job=$SLURM_JOB_ID node=$SLURMD_NODENAME LEN=$LEN N_EVAL=$N_EVAL SIZE=$SIZE  $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /home/yjrcs/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2
module load gcc/13.2.0 2>/dev/null || true

python experiments/sensitivity_sweep.py \
    --length "$LEN" \
    --n_eval "$N_EVAL" \
    --size "$SIZE" \
    --resume

echo "[sbatch] done $(date)"

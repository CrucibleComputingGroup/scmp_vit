#!/bin/bash
#SBATCH --job-name=sc_n1k_sw30_int8_sz1280
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --output=logs/sbatch_%x_%j.out
#SBATCH --error=logs/sbatch_%x_%j.err
#
# SC int8 + skip_worst_30 @ size 1280, n_eval=1000, soft-NMS.
# Reproduces (at scale) the validated 10-image config: bbox AP 68.9.
# Uses run_coco_val.py (auto interp, the path verified post sc/->scmp_kernels
# migration). Resumable: writes checkpoint every save_every images.
#   sbatch det/experiments/submit_sc_n1000_sw30_int8_sz1280.sbatch
# To resume a timed-out/killed run:
#   sbatch --export=ALL,RESUME=1 det/experiments/submit_sc_n1000_sw30_int8_sz1280.sbatch
set -euo pipefail
cd /home/yjrcs/SC_V/vit_sc/det
mkdir -p logs results

SHARED=/scratch/nbleier_owned_root/nbleier_owned1/shared_data
OUT_DIR=results/sc_sw30_int8_sz1280_n1000

echo "[sbatch] job=$SLURM_JOB_ID node=$SLURMD_NODENAME $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /home/yjrcs/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2

EXTRA=()
if [[ "${RESUME:-0}" == "1" ]]; then EXTRA+=(--resume); fi

python -u run_coco_val.py \
    --sc_prec 8 \
    --skip sensitivity/skip/skip_worst_30_int8.json \
    --n_eval 1000 \
    --size 1280 \
    --save_every 50 \
    --out_dir "$OUT_DIR" \
    --d2_datasets "$SHARED" \
    --ckpt "$SHARED/pretrained/eva_coco_det.pth" \
    "${EXTRA[@]}"

echo "[sbatch] done $(date)"

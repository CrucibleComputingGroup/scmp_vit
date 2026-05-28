#!/bin/bash
#SBATCH --job-name=qwt_len192_n5k_cos040_c128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --output=logs/sbatch_%x_%j.out
#SBATCH --error=logs/sbatch_%x_%j.err
#
# QwT cross-seed cosine-gated SC-comp on EVA-ViTDet at len192 sw30 sz1024
# n=5000 soft+beit, τ=0.40 (last-block 0.8), n_calib=128.
#
# Mirrors the prior interactive-node nohup launch
# (experiments/_run_qwt_len192_n5000_cos040.sh). Resumable via the
# checkpoint qwt_det_compensate.py writes to <out_tag>/sc_comp/checkpoint.pt
# every --save_every images. --resume is ALWAYS passed: on first run
# (no checkpoint) it falls through; on retry it skips already-done
# images. Calib is NOT checkpointed — each restart redoes ~95min calib
# (n_calib=128, sz1024). Eval-phase progress survives.
#
# Why these knobs:
#   * cos_threshold=0.40 explores the permissive end of the gate.
#   * n_calib=128 (8× prod default 16) for a more accurate W ridge fit
#     under the lower τ.
#   * --skip_baseline because raw len192 sw30 sz1024 n5000 baseline is
#     already at results/sc_sw30_len192_sz1024_n5000_soft_beit
#     (bbox 61.69 / segm 51.85, eval_seconds=30525).
#   * head_aligned + lookahead_veto: prod recommendation from cos sweep.
#
# Expected wall: ~95min calib + ~9.6h eval = ~11h total on a fresh run.
set -euo pipefail
cd /home/yjrcs/SC_V/vit_sc/det
mkdir -p logs results

OUT_TAG="qwt_len192_sw30_sz1024_n5000_soft_beit_cos040"
OUT_DIR="results/${OUT_TAG}"
mkdir -p "${OUT_DIR}"

echo "[sbatch] job=$SLURM_JOB_ID node=$SLURMD_NODENAME OUT_TAG=$OUT_TAG  $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /home/yjrcs/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2

python -u experiments/qwt_det_compensate.py \
    --sc_prec 8 --stoc_len 192 \
    --sc_ops_per_block_json sensitivity/skip/skip_worst_30_len192.json \
    --n_calib 128 --n_eval 5000 --size 1024 \
    --use_soft_nms --interp_type beit \
    --head_aligned --n_heads 16 --lookahead_veto \
    --comp_mode sc --comp_sc_prec 8 --fwd_chunk 2 \
    --cos_threshold 0.40 --last_block_cos_threshold 0.8 \
    --skip_baseline \
    --save_every 50 --resume \
    --out_tag "${OUT_TAG}" \
    --out_json "${OUT_DIR}/run.json"

echo "[sbatch] done $(date)"

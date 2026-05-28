#!/bin/bash
#SBATCH --job-name=qwt_int7_n5k_cos068
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
# SC int7 sw30 sz1024 n=5000 soft+beit + QwT cross-seed cosine gate at τ=0.68.
#
# Why these knobs:
#   * cos_threshold=0.68 is the peak of the n=100 sweep (sweep range 0.30-0.75,
#     13 points). At 18/40 admitted blocks, sweet spot for SC int7. Plateau is
#     {0.675, 0.68, 0.685} so 0.68 is the stable middle.
#   * --skip_baseline because raw SC int7 sw30 n5000 soft+beit already in
#     results/sc_sw30_int7_sz1024_n5000_soft_beit (bbox 58.84 / segm 48.68)
#     from the prior 4-tag sbatch sweep.
#   * head_aligned + lookahead_veto: production rec from the cos sweep.
#   * n_calib=16 keeps calib short (~11 min). Larger calib could be tried if
#     the 5000-img n_calib effect is interesting.
#
# Expected wall: ~7.8h (calib ~11 min + 5000-img comp eval ~7.6h).
set -euo pipefail
cd /home/yjrcs/SC_V/vit_sc/det
mkdir -p logs results/qwt_int7_5000

OUT_TAG="qwt_int7_sw30_sz1024_n5000_soft_beit_cos068"

echo "[sbatch] job=$SLURM_JOB_ID node=$SLURMD_NODENAME OUT_TAG=$OUT_TAG  $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /home/yjrcs/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2

python -u experiments/qwt_det_compensate.py \
    --sc_prec 7 \
    --sc_ops_per_block_json sensitivity/skip/skip_worst_30_int7.json \
    --n_calib 16 --n_eval 5000 --size 1024 \
    --use_soft_nms --interp_type beit \
    --head_aligned --n_heads 16 --lookahead_veto \
    --comp_sc_prec 8 \
    --cos_threshold 0.68 --last_block_cos_threshold 0.8 \
    --skip_baseline \
    --out_tag "$OUT_TAG" \
    --out_json "results/qwt_int7_5000/${OUT_TAG}.json"

echo "[sbatch] done $(date)"

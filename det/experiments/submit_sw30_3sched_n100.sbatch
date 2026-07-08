#!/bin/bash
#SBATCH --job-name=sw30_3sched_n100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --output=logs/sbatch_%x_%j.out
#SBATCH --error=logs/sbatch_%x_%j.err
#
# Compare 3 sw30 schedules at n=100, sz=1024, sc_prec=7, soft_nms, beit interp.
#   1. canonical = sz1280 + bbox_ap_drop  (skip_worst_30_int7.json)
#   2. l2        = sz1024 + l2            (skip_worst_30_int7_l2.json)
#   3. apdrop    = sz1024 + bbox_ap_drop  (skip_worst_30_int7_apdrop.json)
# Verifies: (a) canonical reproduces historical n=100 ≈ 65.83, (b) l2/apdrop
# rankings give different mAP at large enough n to overcome image-noise.
set -euo pipefail
cd /home/yjrcs/SC_V/vit_sc/det
mkdir -p logs results

source /home/yjrcs/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2

echo "[sbatch] job=$SLURM_JOB_ID node=$SLURMD_NODENAME  $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for tag_path in "canonical:sensitivity/skip/skip_worst_30_int7.json" \
                "l2:sensitivity/skip/skip_worst_30_int7_l2.json" \
                "apdrop:sensitivity/skip/skip_worst_30_int7_apdrop.json"; do
    tag=${tag_path%%:*}
    path=${tag_path##*:}
    echo ""
    echo "=== sw30_${tag} ($(date)) ==="
    md5sum "$path"
    python -u sc_eval.py \
        --n-eval 100 \
        --sc_prec 7 \
        --seed 0 \
        --size 1024 \
        --sc_ops_per_block_json "$path" \
        --use_soft_nms \
        --interp_type beit \
        --save_every 50 \
        --out-tag "p7_sw30_${tag}_n100_seed0"
done

echo "[sbatch] done $(date)"

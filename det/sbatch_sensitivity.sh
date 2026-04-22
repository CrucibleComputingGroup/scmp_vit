#!/bin/bash
#SBATCH --job-name=sens_sc
#SBATCH --account=nbleier_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=20:00:00
#SBATCH --output=results/sbatch_sens_%x_%j.out
#SBATCH --error=results/sbatch_sens_%x_%j.out

# Usage:
#   sbatch --job-name=sens_int7 --export=ALL,SC_PREC=7 sbatch_sensitivity.sh
#   sbatch --job-name=sens_int6 --export=ALL,SC_PREC=6 sbatch_sensitivity.sh
#
# Env vars:
#   SC_PREC : int, SC precision (e.g. 6 or 7). Required.
#   N_EVAL  : images per combo (default 10).

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2
cd /home/yjrcs/SC_V/vit_sc/det

SC_PREC="${SC_PREC:?must set SC_PREC}"
N_EVAL="${N_EVAL:-10}"
# If N_EVAL != 10, use a separate dir so we don't collide with the n=10 sweep's resume state.
if [[ "$N_EVAL" == "10" ]]; then
    OUT_DIR="results/sensitivity_int${SC_PREC}"
else
    OUT_DIR="results/sensitivity_int${SC_PREC}_n${N_EVAL}"
fi
mkdir -p "$OUT_DIR"

echo "=== Sensitivity sweep  sc_prec=$SC_PREC  n_eval=$N_EVAL ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

# Single-GPU full sweep (blocks 0-40)
python -u experiments/sensitivity_sweep.py \
    --n_eval "$N_EVAL" --sc_prec "$SC_PREC" --block_start 0 --block_end 40 \
    --out_json "$OUT_DIR/sensitivity_all_ops.json" --resume \
    2>&1 | tee "$OUT_DIR/log_sweep.txt"

echo "=== Building skip_worst schedules K=10,20,30 ==="
python -u experiments/merge_and_build_skip_worst.py \
    --shards "$OUT_DIR/sensitivity_all_ops.json" \
    --merged "$OUT_DIR/sensitivity_all_ops.json" \
    --skip_ks 10 20 30 \
    --schedule_dir "$OUT_DIR/schedules"

echo "=== Sensitivity int${SC_PREC} DONE ==="

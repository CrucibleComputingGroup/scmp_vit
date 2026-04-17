#!/bin/bash
# Full pipeline: sensitivity sweep (4 GPU shards) → merge → skip_worst eval (4 configs)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vit_sc
cd "$(dirname "$0")"

N_EVAL=10
SC_PREC=8
SCHED_DIR=results/sensitivity/schedules

mkdir -p results/sensitivity/schedules

echo "========================================"
echo "Phase 1: Sensitivity sweep (4 GPU shards)"
echo "========================================"

CUDA_VISIBLE_DEVICES=0 python -u experiments/sensitivity_sweep.py \
    --n_eval $N_EVAL --sc_prec $SC_PREC --block_start 0  --block_end 10 \
    --out_json results/sensitivity/shard_0.json --resume \
    >> results/sensitivity/log_shard0.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u experiments/sensitivity_sweep.py \
    --n_eval $N_EVAL --sc_prec $SC_PREC --block_start 10 --block_end 20 \
    --out_json results/sensitivity/shard_1.json --resume \
    >> results/sensitivity/log_shard1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u experiments/sensitivity_sweep.py \
    --n_eval $N_EVAL --sc_prec $SC_PREC --block_start 20 --block_end 30 \
    --out_json results/sensitivity/shard_2.json --resume \
    >> results/sensitivity/log_shard2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u experiments/sensitivity_sweep.py \
    --n_eval $N_EVAL --sc_prec $SC_PREC --block_start 30 --block_end 40 \
    --out_json results/sensitivity/shard_3.json --resume \
    >> results/sensitivity/log_shard3.txt 2>&1 &

echo "  4 shards launched, waiting..."
wait
echo "  All shards done."

echo "========================================"
echo "Phase 2: Merge + build skip_worst schedules"
echo "========================================"

python -u experiments/merge_and_build_skip_worst.py \
    --shards results/sensitivity/shard_0.json results/sensitivity/shard_1.json \
             results/sensitivity/shard_2.json results/sensitivity/shard_3.json \
    --merged results/sensitivity/sensitivity_all_ops.json \
    --skip_ks 20 30 40 50 \
    --schedule_dir $SCHED_DIR

echo "========================================"
echo "Phase 3: Skip_worst eval (4 configs × 4 GPUs)"
echo "========================================"

CUDA_VISIBLE_DEVICES=0 python -u sc_eval.py --n-eval $N_EVAL --sc_prec $SC_PREC \
    --sc_ops_per_block_json $SCHED_DIR/skip_worst_20.json \
    --out-tag skip_worst_20 >> results/log_skip_worst_20.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u sc_eval.py --n-eval $N_EVAL --sc_prec $SC_PREC \
    --sc_ops_per_block_json $SCHED_DIR/skip_worst_30.json \
    --out-tag skip_worst_30 >> results/log_skip_worst_30.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u sc_eval.py --n-eval $N_EVAL --sc_prec $SC_PREC \
    --sc_ops_per_block_json $SCHED_DIR/skip_worst_40.json \
    --out-tag skip_worst_40 >> results/log_skip_worst_40.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u sc_eval.py --n-eval $N_EVAL --sc_prec $SC_PREC \
    --sc_ops_per_block_json $SCHED_DIR/skip_worst_50.json \
    --out-tag skip_worst_50 >> results/log_skip_worst_50.txt 2>&1 &

echo "  4 skip_worst evals launched, waiting..."
wait
echo "  All skip_worst evals done."

echo "========================================"
echo "Results summary"
echo "========================================"
for k in 20 30 40 50; do
    echo "--- skip_worst_$k ---"
    tail -3 results/log_skip_worst_$k.txt
done
echo "DONE"

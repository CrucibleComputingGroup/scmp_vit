#!/bin/bash
# INT7 Phase 2: merge sensitivity + build skip_worst schedules + eval K=10/20/30 on 3 GPUs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2
cd "$(dirname "$0")"

N_EVAL=100
SC_PREC=7
SENS_DIR=results/sensitivity_int7
SCHED_DIR=$SENS_DIR/schedules

mkdir -p $SCHED_DIR

echo "========================================"
echo "Phase 2a: build skip_worst schedules (K=10,20,30)"
echo "========================================"
python -u experiments/merge_and_build_skip_worst.py \
    --shards $SENS_DIR/sensitivity_all_ops.json \
    --merged $SENS_DIR/sensitivity_all_ops.json \
    --skip_ks 10 20 30 \
    --schedule_dir $SCHED_DIR

echo "========================================"
echo "Phase 2b: skip_worst eval K=10,20,30 on 3 GPUs (INT7, n=$N_EVAL)"
echo "========================================"

CUDA_VISIBLE_DEVICES=0 python -u sc_eval.py --n-eval $N_EVAL --sc_prec $SC_PREC \
    --sc_ops_per_block_json $SCHED_DIR/skip_worst_10.json \
    --out-tag skip_worst_10_int7 > results/log_skip_worst_10_int7.txt 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python -u sc_eval.py --n-eval $N_EVAL --sc_prec $SC_PREC \
    --sc_ops_per_block_json $SCHED_DIR/skip_worst_20.json \
    --out-tag skip_worst_20_int7 > results/log_skip_worst_20_int7.txt 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python -u sc_eval.py --n-eval $N_EVAL --sc_prec $SC_PREC \
    --sc_ops_per_block_json $SCHED_DIR/skip_worst_30.json \
    --out-tag skip_worst_30_int7 > results/log_skip_worst_30_int7.txt 2>&1 &
PID2=$!

echo "Launched skip_worst K=10,20,30 pids: $PID0 $PID1 $PID2"
wait $PID0 $PID1 $PID2
echo "All skip_worst evals done."

echo "========================================"
echo "Phase 2 summary"
echo "========================================"
for k in 10 20 30; do
    echo "--- skip_worst_${k}_int7 ---"
    tail -n 3 results/log_skip_worst_${k}_int7.txt
done
echo "DONE_PHASE2"

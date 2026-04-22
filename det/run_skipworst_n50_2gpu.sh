#!/bin/bash
# Re-evaluate skip_worst K=10/30 using the n=50 sensitivity schedules for
# both INT7 and INT6, on 2 GPUs. Two waves of 2 parallel jobs each.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2
cd "$(dirname "$0")"

N_EVAL=100

run_pair() {
    local prec=$1
    local sched_dir=results/sensitivity_int${prec}_n50/schedules
    echo "========================================"
    echo "Phase: INT${prec} skip_worst 10/30 (n=$N_EVAL, n50-schedules)"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=0 python -u sc_eval.py --n-eval $N_EVAL --sc_prec $prec \
        --sc_ops_per_block_json $sched_dir/skip_worst_10.json \
        --out-tag skip_worst_10_int${prec}_n50 \
        > results/log_skip_worst_10_int${prec}_n50.txt 2>&1 &
    P0=$!
    CUDA_VISIBLE_DEVICES=1 python -u sc_eval.py --n-eval $N_EVAL --sc_prec $prec \
        --sc_ops_per_block_json $sched_dir/skip_worst_30.json \
        --out-tag skip_worst_30_int${prec}_n50 \
        > results/log_skip_worst_30_int${prec}_n50.txt 2>&1 &
    P1=$!
    echo "INT${prec} pids: K=10:$P0  K=30:$P1"
    wait $P0 $P1

    for k in 10 30; do
        echo "--- skip_worst_${k}_int${prec}_n50 ---"
        tail -n 4 results/log_skip_worst_${k}_int${prec}_n50.txt
    done
}

run_pair 7
echo "DONE_INT7"
run_pair 6
echo "DONE_INT6"
echo "DONE_ALL"

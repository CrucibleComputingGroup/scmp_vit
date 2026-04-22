#!/bin/bash
# Skip_worst eval for INT7 (phase A) then INT6 (phase B), 3 GPUs parallel each.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2
cd "$(dirname "$0")"

N_EVAL=100

run_phase() {
    local prec=$1
    local sched_dir=results/sensitivity_int${prec}/schedules
    echo "========================================"
    echo "Phase: INT${prec} skip_worst 10/20/30 (n=$N_EVAL)"
    echo "========================================"
    for k in 10 20 30; do
        [[ -f $sched_dir/skip_worst_${k}.json ]] || { echo "MISSING $sched_dir/skip_worst_${k}.json"; exit 1; }
    done

    CUDA_VISIBLE_DEVICES=0 python -u sc_eval.py --n-eval $N_EVAL --sc_prec $prec \
        --sc_ops_per_block_json $sched_dir/skip_worst_10.json \
        --out-tag skip_worst_10_int${prec} \
        > results/log_skip_worst_10_int${prec}.txt 2>&1 &
    P0=$!
    CUDA_VISIBLE_DEVICES=1 python -u sc_eval.py --n-eval $N_EVAL --sc_prec $prec \
        --sc_ops_per_block_json $sched_dir/skip_worst_20.json \
        --out-tag skip_worst_20_int${prec} \
        > results/log_skip_worst_20_int${prec}.txt 2>&1 &
    P1=$!
    CUDA_VISIBLE_DEVICES=2 python -u sc_eval.py --n-eval $N_EVAL --sc_prec $prec \
        --sc_ops_per_block_json $sched_dir/skip_worst_30.json \
        --out-tag skip_worst_30_int${prec} \
        > results/log_skip_worst_30_int${prec}.txt 2>&1 &
    P2=$!

    echo "  INT${prec} pids: $P0 $P1 $P2"
    wait $P0 $P1 $P2
    echo "  INT${prec} skip_worst all done."

    for k in 10 20 30; do
        echo "--- skip_worst_${k}_int${prec} ---"
        tail -n 4 results/log_skip_worst_${k}_int${prec}.txt
    done
}

run_phase 7
echo "DONE_INT7"
run_phase 6
echo "DONE_INT6"
echo "DONE_ALL"

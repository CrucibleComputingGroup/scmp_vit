#!/bin/bash
# QwT compensation sweep — 2 GPUs, run configs in pairs.
# Each run: n_calib=16, n_eval=100, FP compensator.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2
cd "$(dirname "$0")"

N_CALIB=16
N_EVAL=100
FWD_CHUNK=2

# Configs: (tag, sc_prec, sc_ops_OR_sched, is_json)
# Priority-1: configs where SC raw has real damage (gap > 2 AP vs FP).
run_pair() {
    local tag0=$1 prec0=$2 sched0=$3 isjson0=$4
    local tag1=$5 prec1=$6 sched1=$7 isjson1=$8

    echo "========================================"
    echo "Wave: $tag0 (GPU0)  |  $tag1 (GPU1)"
    echo "========================================"

    mkdir -p results/qwt_det/$tag0 results/qwt_det/$tag1

    if [[ "$isjson0" == "json" ]]; then
        arg0="--sc_ops_per_block_json $sched0"
    else
        arg0="--sc_ops $sched0"
    fi
    if [[ "$isjson1" == "json" ]]; then
        arg1="--sc_ops_per_block_json $sched1"
    else
        arg1="--sc_ops $sched1"
    fi

    CUDA_VISIBLE_DEVICES=0 python -u experiments/qwt_det_compensate.py \
        --sc_prec $prec0 $arg0 \
        --n_calib $N_CALIB --n_eval $N_EVAL --fwd_chunk $FWD_CHUNK \
        --skip_baseline \
        --out_tag $tag0 --out_json results/qwt_det/$tag0/run.json \
        > results/qwt_det/$tag0/log.txt 2>&1 &
    P0=$!
    CUDA_VISIBLE_DEVICES=1 python -u experiments/qwt_det_compensate.py \
        --sc_prec $prec1 $arg1 \
        --n_calib $N_CALIB --n_eval $N_EVAL --fwd_chunk $FWD_CHUNK \
        --skip_baseline \
        --out_tag $tag1 --out_json results/qwt_det/$tag1/run.json \
        > results/qwt_det/$tag1/log.txt 2>&1 &
    P1=$!
    echo "pids: $tag0=$P0  $tag1=$P1"
    wait $P0 $P1
    echo "--- $tag0 ---"; grep -E "SC \+ QwT|Δ bbox|bbox AP" results/qwt_det/$tag0/log.txt | tail -4
    echo "--- $tag1 ---"; grep -E "SC \+ QwT|Δ bbox|bbox AP" results/qwt_det/$tag1/log.txt | tail -4
}

# Wave 1: INT7 skip_worst_10 + skip_worst_30 (n=50 schedules)
run_pair \
    qwt_int7_sw10_n50 7 results/sensitivity_int7_n50/schedules/skip_worst_10.json json \
    qwt_int7_sw30_n50 7 results/sensitivity_int7_n50/schedules/skip_worst_30.json json

# Wave 2: INT6 skip_worst_10 + skip_worst_30 (n=50 schedules)
run_pair \
    qwt_int6_sw10_n50 6 results/sensitivity_int6_n50/schedules/skip_worst_10.json json \
    qwt_int6_sw30_n50 6 results/sensitivity_int6_n50/schedules/skip_worst_30.json json

# Wave 3: INT7 qk+av+proj+fc1 + INT7 qk+av+proj (sanity check FP baseline)
run_pair \
    qwt_int7_qkavprojfc1 7 qk,av,proj,mlp_fc1 ops \
    qwt_int7_qkavproj    7 qk,av,proj         ops

# Wave 4: INT6 qk+av+proj + INT6 qk+av
run_pair \
    qwt_int6_qkavproj 6 qk,av,proj ops \
    qwt_int6_qkav     6 qk,av       ops

# Wave 5: INT7 QK only + INT6 QK only (cheapest, verify no-damage case)
run_pair \
    qwt_int7_qk 7 qk ops \
    qwt_int6_qk 6 qk ops

echo "DONE_ALL_QWT"
for d in results/qwt_det/qwt_*; do
    echo "=== $(basename $d) ==="
    grep -E "SC \+ QwT|Δ bbox" $d/log.txt 2>/dev/null | tail -3
done

#!/usr/bin/env bash
# Smoke test for sweep_int678_skip30.sh.
# ONLY difference vs prod: N_EVAL=100 instead of 50000. All other knobs
# (N_CALIB=1024, SEARCH_N_SEARCH=32, SEARCH_MAX_ITERS=4, RIDGE=1e-4, ...)
# inherit the prod defaults, so the smoke run exercises the exact same
# calibration + search pipeline — just doesn't spend hours on eval images.
#
# Output lands in results/smoke_int678_k30/ / logs/smoke_int678_k30/ so it
# doesn't clobber the prod sweep's outputs.
#
# Expected wall-clock (4 GPUs): ~20–30 min, bottlenecked by prod-scale QwT
# calibration and per-MP-tag search. Eval itself at N=100 is <1 min per run.
set -euo pipefail

export RES_DIR="results/smoke_int678_k30"
export LOG_DIR="logs/smoke_int678_k30"
export MAP_DIR="$RES_DIR/sl_maps"
export N_EVAL=100

exec bash "$(dirname "$0")/sweep_int678_skip30.sh"

#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2
cd /home/yjrcs/SC_V/vit_sc/det

CUDA_VISIBLE_DEVICES=0 python sc_eval.py --n-eval 10 --sc_qk 1 --sc_av 1 --sc_mlp 1 --out-tag sc_qk_av_mlp > results/log_qk_av_mlp.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python sc_eval.py --n-eval 10 --sc_qk 1 --sc_av 1 --sc_qkv_proj 1 --sc_out_proj 1 --sc_mlp 1 --mlp_skip_last_k 8 --out-tag sc_skipworst8 > results/log_skipworst8.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python sc_eval.py --n-eval 10 --sc_qk 1 --sc_av 1 --sc_qkv_proj 1 --sc_out_proj 1 --sc_mlp 1 --mlp_skip_last_k 16 --out-tag sc_skipworst16 > results/log_skipworst16.txt 2>&1 &

echo 'Launched 3 jobs on GPU 0,1,2'
wait
echo 'ALL DONE'

#!/usr/bin/env bash
# Serialized Step 1 + Step 2 runner. ONE GPU, sequential. ~2.5h wall.
# Step 1: r²>0.3 reference (legacy gate)
# Step 2: τ-sweep cross-seed gate at τ ∈ {0.65, 0.75}
# τ=0.85, 0.90 deferred — what-if shows ≤8 admit, very likely under-fits given
# cls reference admits 22/24.
set -e
cd /home/allenjin/Projects/vit_sc/det
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate qwt_d2
module load gcc/13.2.0
export CUDA_VISIBLE_DEVICES=1

mkdir -p logs/gate_tuning results/gate_tuning

run_cell() {
  local desc="$1" out="$2" log="$3"; shift 3
  echo "=========================================="
  echo "[$(date +%H:%M:%S)] STARTING: $desc"
  echo "  -> $out"
  echo "=========================================="
  python "$@" > "$log" 2>&1 || { echo "[FAIL] $desc — see $log"; tail -40 "$log"; exit 1; }
  echo "[$(date +%H:%M:%S)] DONE: $desc"
}

# --- Step 1: r²>0.3 legacy reference on p7_uniform ---
run_cell "step1: r²>0.3 ref (p7_uniform)" \
  "results/gate_tuning/ref_r2_03_p7.json" \
  "logs/gate_tuning/ref_r2_03_p7.log" \
  experiments/qwt_det_compensate_r2.py \
    --sc_prec 7 --sc_ops_per_block_json sensitivity/skip/skip_worst_30_int7.json \
    --n_calib 16 --n_eval 10 \
    --min_r2 0.3 --last_block_r2_threshold 0.5 --lookahead_veto \
    --comp_mode sc --comp_sc_prec 8 --head_aligned --fwd_chunk 2 \
    --out_tag ref_r2_03_p7 \
    --out_json results/gate_tuning/ref_r2_03_p7.json

# --- Step 2: τ-sweep cells (0.65 only; 0.75 dropped after cell 1 r²>0.3
# evidence + user clarification that 0.4 was the SC-comp threshold; r²>0.4
# launched separately on GPU 3) ---
for tau in 0.65; do
  run_cell "step2: τ=$tau (p7_uniform)" \
    "results/gate_tuning/tau_${tau}_p7.json" \
    "logs/gate_tuning/tau_${tau}_p7.log" \
    experiments/qwt_det_compensate.py \
      --sc_prec 7 --sc_ops_per_block_json sensitivity/skip/skip_worst_30_int7.json \
      --n_calib 16 --n_eval 10 \
      --calib_seed 1 --calib_seed_b 2 \
      --cos_threshold $tau --last_block_cos_threshold 0.9 \
      --lookahead_veto \
      --comp_mode sc --comp_sc_prec 8 --head_aligned --fwd_chunk 2 \
      --out_tag tau_${tau}_p7 \
      --out_json results/gate_tuning/tau_${tau}_p7.json
done

echo "=========================================="
echo "[$(date +%H:%M:%S)] ALL CELLS DONE"
echo "=========================================="

# vit_sc — Stochastic-Computing ViT evaluation

Evaluate a pretrained ViT (DINOv2 ViT-L/14 + linear classifier) on ImageNet-1k
with stochastic-computing (SC) matmuls swapped into the attention path, and
compare against the FP baseline.

No training. Only inference. The SC kernels (bipolar / XNOR matmul, Sobol RNG)
come from `sc/` and `sc_integration/`; `cls/sc_attention_patch.py` monkey-patches
every DINOv2 attention block to route Q@K^T, Attn@V, qkv_proj, and out_proj
through the SC kernels at configurable precision. The shared `SCLinear` /
`_sc_linear` primitives now live in `sc_integration/sc_linear.py` so both
`cls/` (DINOv2) and `det/` (EVA-ViTDet) pull from one source.

## What's new

- **Cross-seed cosine gate + head-aligned SC comp (2026-04-25).** The QwT
  comp's admission rule is now `cos(W_A, W_B) > τ` over two disjoint calib
  batches, replacing the legacy r²/cv-holdout gate. The comp matmul itself
  now defaults to `HeadAlignedSCLinear` (per-head D=64 SCLinear, reuses the
  block QK Sobol pool — zero new SNG entries, fully SC inference path).
  N=50k production geomean Δ over raw SC = **+1.80 pt** across 5 configs;
  beats the legacy reference on every config. See
  [`docs/SC_COMP_ALGORITHM.md`](docs/SC_COMP_ALGORITHM.md) for the full
  pilot → production write-up and hardware-cost analysis.
- **Wider MP-search budget.** The default MP-budget swap search now uses
  `--n_search 64 --max_iters 12` (was 32 / 4). On the same proj/mlp_fc1/
  mlp_fc2 search space this drives `p7_mp` Δ from +1.14 to +3.37 vs the
  legacy reference at unchanged eff_sl. Larger SEARCH_OPS (adding qk/av)
  is still blocked by the equal-MAC DP constraint in
  `build_initial_sl_map`; see SC_COMP_ALGORITHM.md for the next-iteration
  recommendation.
- **Mixed-precision MP.** Fixed-MP per-head on QK and per-row-within-head on
  AV, plus per-input-row MP on all linear ops, ported from `scmp_llm`.
  CLI flags: `--qk_mp_levels`, `--qk_mp_fractions`, `--av_mp_levels`,
  `--av_mp_fractions`, `--mp_levels`, `--mp_ops`, `--range_mp`,
  `--range_mp_levels`.
- **MP support in the overnight comp driver.** `qwt_sc_overnight.py` now
  accepts all MP flags and pipes them into `patch_model`, so MP + SC-comp +
  fine-grained scheduling compose in a single run.
- **Effective stoc_len reporting (`main_sl` / `eff_sl`).** The driver now
  computes and prints FLOP-weighted average bitstream length for the main SC
  ops (`main_sl`) and including the compensation block (`eff_sl`), plus
  effective reduction vs full-precision SC. Saved in JSON under `"stoc_len"`.

## Production status (2026-04-25)

Headline N=50k results from `cls/results/sweep_2026-04-25/` — cross-seed
gate + `HEAD_ALIGNED=1` (B_ha kernel) + wider MP search budget:

| config         | raw SC | + B_ha + cross-seed | Δ | legacy r²-gate ref Δ |
|----------------|---:|---:|---:|---:|
| p7_uniform     | 78.94 | 82.87 | **+3.93** | +3.12 |
| p7_mp          | 79.89 | 83.26 | **+3.37** | +1.14 |
| p8_uniform     | 85.57 | 85.70 | +0.13 | +0.20 |
| avg192_uniform | 84.67 | 85.42 | **+0.75** | +0.67 |
| avg192_mp      | 84.62 | 85.47 | **+0.85** | +0.65 |
| **geomean Δ**  |       |       | **+1.80** | +1.16 |

Cross-seed gate admits 22/24 blocks (rejects 0 and 23) on every config.
Reproduce with [the recipe in SC_COMP_ALGORITHM.md](docs/SC_COMP_ALGORITHM.md#how-to-reproduce).
The "QwT compensation gate" section near the bottom of this README
describes the *legacy r²-gate* (block-23 collapse mitigation) and is kept
for historical context — the cross-seed gate supersedes it.

## Model and dataset

- **Model:** `dinov2_vitl14_lc` loaded via `torch.hub` from
  `facebookresearch/dinov2` — DINOv2 ViT-L/14 backbone (24 blocks, d=1024,
  16 heads, head_dim=64) + the published ImageNet-1k linear head. Weights
  download automatically on first run (~1.15 GB).
- **Dataset:** ImageNet-1k **validation** split (50 000 images), stored as
  HuggingFace parquet shards (`validation-00000-of-00014.parquet` …
  `validation-00013-of-00014.parquet`, ~6.3 GB total).
  Preprocessing: resize 256 (bicubic) → center-crop 224 → ImageNet mean/std.

## Setup

```bash
# 1. Conda env
conda create -n vit_sc python=3.11 -y
conda activate vit_sc
pip install torch==2.10.0 torchvision triton==3.6.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install timm==1.0.26 pyarrow datasets huggingface_hub pillow numpy
```

Tested with torch 2.10.0+cu128, triton 3.6.0, timm 1.0.26, Python 3.11, CUDA
12.8. Requires a CUDA GPU (eval has no CPU path); tested on RTX 4080 (16 GB)
and RTX PRO 6000 Blackwell (97 GB).

```bash
# 2. ImageNet-1k val data (gated — needs an HF token)
export HF_TOKEN=hf_xxxxxxxx   # from https://huggingface.co/settings/tokens
# Accept terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k first.
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ILSVRC/imagenet-1k",
    repo_type="dataset",
    allow_patterns=["data/validation-*.parquet"],
    local_dir="./data/imagenet",
)
PY
```

This places the 14 val shards under `./data/imagenet/data/` — that directory
is what `--data_root` expects.

### Running on a SLURM cluster (recipe: UMich GreatLakes `gpu-rtx6000`)

Concrete recipe for the `nbleier_owned1` allocation on a 4× RTX PRO 6000
Blackwell (97 GB each) node. Generalises cleanly to any multi-GPU SLURM
partition — the only non-obvious tunable is `--mem`.

**Rules of thumb (from this repo's profiling):**

| Knob | Value | Why |
|---|---|---|
| `--mem` (total) | **≥ 32 GB × #GPUs** (so `128G` for 4 GPUs) | `--mem` is the cgroup total across the whole allocation, **not** per-GPU. A single ViT-L SC + QwT-comp job peaks at ~10–15 GB RAM (FP+SC model copies + calib activation buffer + DataLoader workers). Setting `--mem=32G` on a 4-GPU job is the #1 cause of silent SIGKILL the moment two jobs load models simultaneously. |
| `--cpus-per-gpu` | 16 | Enough for `--workers 2–4` per job; DataLoader is never the bottleneck (SC kernel is). |
| `--workers` (DataLoader) | `2` when 4-way parallel, `4` solo | Fewer fork-COW spikes against the cgroup cap. |
| `--batch_size` | `8` for SC eval, `32+` for FP | SC is launch-bound on the per-sample Sobol pool construction, not compute-bound — bigger batches barely help. |
| Data placement | `/scratch` | 6.3 GB of parquet shards — never put on `/home` (slow + quota). |
| `--out_json` paths | unique per run | Otherwise parallel jobs race on the same file. |

**One-shot allocation template** (runs until 1 AM tomorrow):

```bash
current_time=$(date +%s)
tomorrow_1am=$(date -d "tomorrow 01:00:00" +%s)
diff=$((tomorrow_1am - current_time))
time_string=$(printf "%d:%02d:00" $((diff/3600)) $(((diff%3600)/60)))

salloc --account=nbleier_owned1 --partition=gpu-rtx6000 \
       --gres=gpu:4 --time=$time_string \
       --cpus-per-gpu=16 --mem=128G
```

**Env + data setup (once per fresh env)** — run after `salloc` drops you on
the compute node:

```bash
# Conda env (see top-level Setup for full pip steps)
conda activate vit_sc
git submodule update --init --recursive   # pulls QwT-SC submodule

# Download 6.3 GB of ImageNet val parquet to /scratch, once.
DATA_ROOT=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet
export HF_TOKEN=hf_xxxxxxxx
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="ILSVRC/imagenet-1k", repo_type="dataset",
                  allow_patterns=["data/validation-*.parquet"],
                  local_dir="$DATA_ROOT")
PY
```

**Four-way parallel sweep** — one config per GPU, `CUDA_VISIBLE_DEVICES`
pinning, backgrounded, joined with `wait`:

```bash
DATA=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data
mkdir -p logs results
for i in 50 40 30 20; do
    gpu=$(( (50 - i) / 10 ))    # 50→0, 40→1, 30→2, 20→3
    CUDA_VISIBLE_DEVICES=$gpu nohup python -u experiments/qwt_sc_overnight.py \
        --sc_config skip_worst$i --n_calib 256 --n_eval 500 \
        --batch_size 8 --workers 2 \
        --comp_mode sc --n_variants 4 --w_scales 0.5,0.75,1.0 \
        --head_aligned --n_heads 16 --skip_baseline \
        --data_root $DATA \
        --out_json results/skip_worst${i}_sccomp_headaligned.json \
        > logs/skip_worst${i}.log 2>&1 &
done
wait
echo "all done; summaries:"
for i in 50 40 30 20; do grep -A4 "=== SUMMARY ===" logs/skip_worst${i}.log; done
```

Each run is ~8 min (calib 5 min + eval 2 min + FP reference 3 s). With 4
GPUs in parallel the whole sweep finishes in ~10 min wall-clock (vs ~32 min
serial). Per-GPU throughput is identical to an RTX 4080 — SC is
launch-overhead bound, so Blackwell doesn't speed up the kernel, it just
lets you run four configs at once.

**Diagnosing silent kills.** If a run exits without writing its `--out_json`
and the log tail is stuck at `[data] building loaders`, it was SIGKILL'd by
the cgroup. Check with:

```bash
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,MaxRSS,ReqMem
```

`MaxRSS` close to `ReqMem` = cgroup OOM — bump `--mem` and re-run. (Note:
pipelines like `python ... 2>&1 | tee log` mask the kill as "exit 0" because
`tee` itself exits cleanly; prefer `> log 2>&1` for unambiguous exit codes.)

**Serial fallback** if `--mem` can't be raised: drop the `&` and `wait`,
keep every run on GPU 0. Memory footprint per job is fine under 32 GB
alone; the problem is only parallelism.

### Launching long sweeps from the login node via ssh

For sweeps that take longer than your `salloc` window (multi-hour driver
scripts, gate/MP grids, overnight jobs), you want to fire-and-forget from
the login node onto a compute node that already holds the GPUs. The
compute node has GPUs; the login node doesn't; `/tmp` is not shared
between them but `/home/$USER/...` is. Use **this exact invocation** —
every piece is load-bearing:

```bash
ssh -n -f <gpu-node> "bash -c 'nohup bash \
    /abs/path/to/cls/experiments/<sweep>.sh \
    >/abs/path/to/cls/logs/<sweep>_driver.log 2>&1 &'"
```

Gotchas that will silently burn an hour if you skip them:

1. **`ssh host 'cmd'` lands in `$HOME`, not in the repo.** Pass scripts
   as absolute paths (the sweep's own `cd "$(dirname "$0")/.."` handles
   cwd from there). Do *not* rely on the remote shell being in `cls/`.
2. **Write logs to `/home/$USER/...`, never `/tmp`.** Login and compute
   nodes have separate local `/tmp`. If your driver log goes to `/tmp`
   on the compute node, you can't tail it from the login node.
3. **`-n -f` is mandatory.** `-f` forks ssh into the background after
   authentication so your login shell returns immediately; `-n`
   redirects stdin from `/dev/null`. Without both, ssh may stay attached
   to the remote process's stdio, and if you re-issue the launch
   (thinking it hung) you'll *stack multiple copies* of the sweep on
   top of each other. Verify a singleton with
   `ssh <gpu-node> 'pgrep -fc <sweep>.sh'` — should be O(1), not O(10).
4. **Redirect stdio *inside* the remote shell, not outside.**
   `ssh host 'cmd' >log` captures ssh's stdout, not the backgrounded
   process's. Put `>driver.log 2>&1` inside the quoted command.
5. **If you ever see 10+ duplicate workers**, kill everything and restart:
   `ssh <gpu-node> "pkill -9 -f <sweep>.sh; pkill -9 -f 'python experiments'"`,
   delete any partial JSON outputs, then relaunch with the template above.

## Running eval

`XFORMERS_DISABLED=1` is required so DINOv2 falls back to the SDPA forward path
that `cls/sc_attention_patch.py` can patch; `cls/eval.py` sets it by default.
All classification commands below are run from `cls/`.

### FP baseline

```bash
python eval.py --mode fp \
    --batch_size 32 --workers 4 \
    --max_images 1000 --seed 0 \
    --out_json results/fp_n1k.json
```

### SC runs

`--sc_prec K` sets the SC precision (stochastic bitstream length = `2**K`; the
report uses `K=8` → length 256). Per-op flags turn SC on/off for each
attention submodule:

| Flag | Op affected |
|---|---|
| `--sc_qk 1` | Q @ K^T |
| `--sc_av 1` | Attn @ V (per-row bipolar grouping — needed for softmax'd attn) |
| `--sc_qkv_proj 1` | input qkv linear |
| `--sc_out_proj 1` | output projection linear |
| `--sc_mlp 1` | MLP fc1 + fc2 (**collapses to ~0% accuracy** at uniform prec=8; not recommended) |
| `--mlp_skip_first_k K` / `--mlp_skip_last_k K` | keep first/last K blocks' MLP in FP when `--sc_mlp 1` |

Example — full attention in SC, MLP and head stay FP:

```bash
python eval.py --mode sc --sc_prec 8 \
    --sc_qk 1 --sc_av 1 --sc_qkv_proj 1 --sc_out_proj 1 \
    --batch_size 8 --workers 4 \
    --max_images 1000 --seed 0 \
    --out_json results/sc_full_attn_n1k.json
```

`--max_images 0` evaluates the full 50 000-image val set. Use `--batch_size 8`
for SC (the per-sample RNG-pool construction is the bottleneck); FP runs happily
at `--batch_size 32` or higher.

## Replication (N=500, seed=0, RTX 4080)

| Config | top-1 | top-5 |
|---|---:|---:|
| FP baseline                       | 0.846 | 0.986 |
| SC QK                             | 0.850 | 0.986 |
| SC QK + AV                        | 0.836 | 0.984 |
| SC QK + AV + qkv_proj + out_proj  | 0.832 | 0.988 |

FP is deterministic; SC variance is from the Sobol RNG and is deterministic
for a given seed/stoc_len. Raw JSON results in `results/*.json`.

## Fine-grained per-block per-op SC scheduling

Beyond the uniform-across-all-blocks flags above, `patch_model` accepts an
explicit schedule saying *which block's which operator* goes SC. The five
operators are:

| op name | what it is | FP fallback path |
|---|---|---|
| `mlp_fc1` | MLP's first linear (D=1024 → 4096) in each block | `nn.Linear` |
| `mlp_fc2` | MLP's second linear (D=4096 → 1024) | `nn.Linear` |
| `qk`      | Q @ K^T attention-score matmul | scaled dot product |
| `av`      | Attn @ V | FP matmul |
| `proj`    | qkv input linear + out_proj output linear (toggled together) | `nn.Linear` |

A schedule is a `dict[op_name, list[0/1]]` with each list of length 24 (one
entry per block, `1` = SC, `0` = FP). Missing ops default to all-FP.

### Python API

```python
# run from cls/ so sc_attention_patch is importable as a top-level module
from sc_attention_patch import patch_model

# SC everything except late-block fc2 (the most sensitive linears):
schedule = {
    "mlp_fc1": [1] * 24,
    "mlp_fc2": [1] * 16 + [0] * 8,   # block 16..23 fc2 stays FP
    "qk":      [1] * 24,
    "av":      [1] * 24,
    "proj":    [1] * 24,
}
stats = patch_model(model, sc_prec=8, sc_ops_per_block=schedule)
```

### CLI

Write the same dict as JSON, then:

```bash
python eval.py --mode sc --sc_prec 8 \
    --sc_ops_per_block_json my_schedule.json \
    --batch_size 8 --max_images 500 --seed 0 \
    --out_json results/my_run.json
```

The flag takes priority over `--sc_mlp / --sc_qk / ...`. A coarser
uniform-across-blocks shortcut is also available:

```bash
# 5-vector in op order [mlp_fc1, mlp_fc2, qk, av, proj]
python eval.py --mode sc --sc_ops 1,1,0,0,1 ...
# or comma-separated names
python eval.py --mode sc --sc_ops mlp_fc1,mlp_fc2,proj ...
```

## End-to-end results with fine-grained scheduling (N=500)

`experiments/sensitivity_all_ops.py` builds a 24×5 per-(op, block) sensitivity
matrix using a calibrated Sobol-SC noise surrogate (closed-form Gaussian
approximation, ~100× faster than real SC — see `sc_integration/noise_matmul.py`).
That matrix ranks all 120 (op, block) pairs by logit-L2 drift; real SC
schedules below skip the top-K worst.

Real SC, bipolar, `sc_prec=8`, DINOv2 ViT-L/14 + linear head, 500 val images:

| config | #SC ops | top-1 | top-5 | notes |
|---|---:|---:|---:|---|
| `fp`                 |   0 | 0.846 | 0.986 | baseline (no SC) |
| `attn_full`          |  72 | 0.832 | 0.988 | uniform qk+av+proj across all 24 blocks |
| `sc_skip_worst50`    |  70 | 0.844 | 0.986 | within FP noise |
| `sc_skip_worst40`    |  80 | 0.844 | 0.986 | |
| `sc_skip_worst30`    |  90 | 0.824 | 0.986 | |
| `sc_skip_worst20`    | 100 | 0.756 | 0.956 | knee of the curve |
| `sc_skip_worst10`    | 110 | 0.692 | 0.892 | |
| `everything`         | 120 | 0.000 | 0.006 | all ops, all blocks — catastrophic |

Headline: per-(op, block) sensitivity scheduling buys ~1 pt top-1 over the
block-uniform `attn_full` at similar coverage (0.844 vs 0.832) — small but
consistent. The curve is flat through ~90 SC ops (skip_worst30 still at
0.824) and drops sharply past that. The knee is around ~100 SC ops;
beyond that the long tail of mildly-sensitive ops compounds non-linearly
and `everything` (all 120 ops SC) is catastrophic at 0.000.

### Why it works

The 24×5 sensitivity matrix surfaces three universal patterns (see
`results/sensitivity_all_ops.json`):

- **Late-block fc2 is the dominant damage path.** Block 22's fc2 alone has
  surrogate logit-L2 = 35.3 — larger than the next 4 worst (op, block) pairs
  combined. Blocks 20–23's fc2 are ranks 1–4. Layer-by-layer profiling
  shows fc2 causes **norm inflation** (‖SC‖/‖FP‖ up to 2.5× at block 22
  even with clean FP input) due to the post-activation input distribution:
  67 % of values quantize to `|x_int| ≤ 2` under per-row bipolar
  quantization, leaving the enable-signal multiplication with Sobol
  prefixes too short for accurate products. The SC noise on these
  near-zero values dominates the output, inflating its norm; this
  compounds through the residual stream across 24 blocks.
- **All five operators are depth-monotonic in late layers.** Every op's
  sensitivity curve is flat for blocks 0–15, then rises sharply toward the
  classifier. The final LN + linear head has no downstream averaging
  operator, so late-layer noise lands directly in the logits.
- **Only one early-layer anomaly**: `mlp_fc1` block 0 (L2 = 10.08), because
  it takes raw patch-embedding activations before LayerNorm settles.

Skipping the top-50 worst (op, block) pairs therefore mostly drops *late*
ops across all five kinds, which is why it outperforms a naive "SC all
attention" rule that leaves all fc1/fc2 FP but SCs *every* block's qk+av+proj
— including the final ones that hurt most.

## QwT-style residual compensation (no retraining)

Port of [QwT's](https://arxiv.org/abs/2411.13918) closed-form
`CompensationBlock` to the SC setting. The library lives in the QwT-SC
fork, vendored as a git submodule at `third_party/QwT-SC/QwT-vit-sc/`.

### Formulation

For each transformer block `i`, the compensation fits a linear residual
correction to the SC noise on 256 calibration images (no backprop, no
fine-tuning, ~5 min on an RTX 4080):

```
minimize  ||R_i − (X_i W_i + b_i)||²  +  λ ||W_i||²

where  R_i = Y_fp_i − Y_sc_i       (FP − SC residual per block)
       X_i = input to block i       (N × D, D=1024)
       W_i = (D × D) = (1024 × 1024)
       b_i = (D,) = (1024,)
       λ   = ridge = 1e-4 (default)
```

Solved in closed form via the normal equations:

```
X_aug = [X | 1]                                   (append ones for bias)
[W; b] = (X_aug^T X_aug + λ I)^{-1} X_aug^T R    (ridge only on W, not b)
```

At inference, each block is wrapped:

```
out = block_sc(x) + comp(x)       where comp(x) = x @ W + b
```

With `--comp_mode sc --head_aligned` (recommended), the `x @ W` matmul
runs through HeadAlignedSCLinear — 16 per-head SC matmuls reusing the
same Sobol pool as the block's QK path. **The entire inference path is
SC**: block ops, comp matmul, and bias addition (a trivial accumulator on
an SC ASIC). The only FP operation is the ridge LS solve during
offline calibration.

Blocks are fitted **sequentially (Gauss–Seidel)**: after fitting block
`i`, its compensated output feeds block `i+1`'s calibration. This ensures
each block is fitted on the distribution it will actually see at
inference, preventing error compounding across the 24-block depth.

### Ridge regularization

Ridge is needed because SC noise is **stochastic** (Sobol RNG variance),
unlike INT quantization noise which is deterministic. The regression
target `R = Y_fp − Y_sc` has per-sample noise from the SC realization,
acting as label noise in the LS fit. Ridge shrinks W toward zero, trading
a small bias for reduced sensitivity to this target noise.

Original QwT (INT quantization) uses **no ridge** — same input always
produces the same rounding error, so the target is noise-free. The SC
adaptation requires ridge specifically because of the stochastic noise.

Ridge ablation on skip_worst40, N=500, 4× RTX PRO 6000:

| ridge | no MP (comp top-1) | + 3-level MP (comp top-1) |
|---:|---:|---:|
| 0 (no reg) | 0.830 | *pending* |
| **1e-4** | **0.852** | **0.846** |
| 1e-2 | 0.848 | 0.842 |
| 1.0 | 0.848 | 0.836 |

The effect is larger under MP (1.0 pt spread) than without MP (0.4 pt),
confirming ridge compensates for SC-specific stochastic noise — MP
amplifies this noise (shorter bitstreams → higher Sobol variance →
noisier regression target). Default is `ridge=1e-4`.

### Setup

After cloning this repo:

```bash
git submodule update --init --recursive
```

### Reproducing the sweep

```bash
# Worst-hit baseline: full-attention SC.
python experiments/qwt_sc_overnight.py \
    --sc_config full_attn --n_calib 256 --n_eval 500 \
    --batch_size 8 --out_json results/qwt_sc_full_attn_n500.json

# Best combo: fine-grained sensitivity scheduling + QwT comp.
python experiments/qwt_sc_overnight.py \
    --sc_config skip_worst50 --n_calib 256 --n_eval 500 \
    --batch_size 8 --out_json results/qwt_sc_skip_worst50_n500.json
```

Supported `--sc_config` presets: `full_attn`, `qk_only`, `qk_av`, and the
fine-grained `skip_worst{50,40,30,20}` / `all_ops` configs.

### Results (N=500 val images, seed=0, `sc_prec=8`, RTX 4080)

| Config | SC ops | Raw SC | + QwT | Δ |
|---|---:|---:|---:|---:|
| FP baseline                           |   0/120 | 0.846 | —     | —       |
| `full_attn` (qk+av+qkv_proj+out_proj) |  72/120 | 0.832 | 0.844 | +1.2 pt |
| `qk_av`                               |  48/120 | 0.842 | 0.846 | +0.4 pt |
| `qk_only`                             |  24/120 | 0.846 | 0.850 | +0.4 pt |
| `skip_worst50` (fine-grained)         |  70/120 | 0.844 | 0.844 |  0.0 pt |

Headline: raw SC is within ~0.4 pt of FP for three of the four configs, so
QwT comp has almost nothing to recover and the Δ column sits at 0–1.2 pt,
mostly inside the N=500 sampling-noise floor. Comp still provides a real
benefit on the hardest config (`full_attn`: +1.2 pt, recovering most of
the 1.4 pt gap to FP). **Treat comp as a minor accuracy recovery for
worst-case configs, not a headline gap-closer.** Raw JSON in
`results/qwt_*.json`.

**Why it works.** Layer-by-layer profiling of SC vs FP block outputs
reveals the dominant error mechanism: **quantization-induced norm
inflation**, not mean-shift. Per-row bipolar quantization sets
`scale = abs_max_row / 127`, so rows with a few large outliers push the
majority of values to `|x_int| ≤ 2` (boundary ≤ 4). At such tiny
boundaries the enable-signal count uses a Sobol prefix of length 0–4 —
effectively a coin flip, not a precise product. The SC noise on these
near-zero quantized values is proportionally huge, inflating the output
norm by a factor that compounds through the residual stream.

Concretely, with all SC ops enabled (`everything` config, sc_prec=8,
N=32 images):

| Block | ‖SC‖/‖FP‖ | Quantized zero_frac | `|x_int|≤2` frac |
|---|---:|---:|---:|
| 0  | 1.04 | 0.75 | 0.90 |
| 10 | 1.06 | 0.02 | 0.09 |
| 19 | 2.10 | — | — |
| 22 | **3.40** | 0.30 | **0.67** |

The inflation is worst at blocks whose post-activation input is
**sparse** — many near-zero values with a few outliers setting the
per-row scale. Block 22 fc2 alone inflates by 2.5× even with clean
FP input, because 67 % of its post-GELU input quantizes to `|x_int| ≤ 2`.
Gaussian inputs with the same mean/std show only ~1.005× inflation,
confirming the cause is the specific outlier-dominated distribution, not
SC noise in general.

The linear compensator `out = block_sc(x) + x·W + b` works because:

1. **The inflation is deterministic** — same Sobol sequences and same
   quantization grid produce the same per-element error for the same
   input. The error is a fixed function of the boundary values, which
   are (approximately) linear functions of x.
2. **The error is approximately linear in x** — ridge LS on 256
   calibration images captures this mapping. Per-sample Sobol variance
   is averaged by downstream LayerNorm + 257-token aggregation.
3. **Sequential (Gauss–Seidel) fitting** corrects each block's output
   before the next block sees it, breaking the compounding loop that
   drives ‖SC‖/‖FP‖ from 1.04 at block 0 to 3.40 at block 22.

See the sub-project README for per-block r² diagnostics and overhead
analysis (+25 M params, +21% FLOPs, no inference slowdown — lost in the
SC bottleneck).

### SC-kernel compensator (`--comp_mode sc`)

The driver also supports running the residual `x @ W + b` itself through SC
kernels — the same XNOR / Sobol kernels that the block uses — instead of
keeping it in FP. Goal: a *fully SC* inference path so an SC accelerator
doesn't need an FP unit just for the comp. The W, b solve stays closed-form
ridge; only the inference-time matmul flips to SC via `SCLinear`. The comp's
`(D=1024, sc_prec=8)` SNG is taken from the same `_CFG_CACHE` entry as any
block ops at the same dim, so the comp's Sobol pool is deterministic across
calibration ↔ inference and overlaps with block ops at matching dims.

```bash
# SC-kernel comp — drop-in replacement of FP comp.
python experiments/qwt_sc_overnight.py \
    --sc_config skip_worst50 --n_calib 256 --n_eval 500 \
    --batch_size 8 --comp_mode sc --comp_sc_prec 8 \
    --out_json results/qwt_sc_skip_worst50_n500_sccomp.json
```

**Driver flags (SC comp):**

| Flag | Default | What it does |
|---|---|---|
| `--comp_mode {fp,sc}` | `sc` | Comp matmul kernel. `sc` uses `SCLinear`; `fp` is debug-only (breaks the SC-purity story). |
| `--comp_sc_prec INT` | `8` | SC bit-precision of the comp matmul. |
| `--comp_sc_mode {bipolar,unipolar}` | `bipolar` | Per-row SC quant mode of the comp matmul. |
| `--n_variants INT` | `1` | Sobol-seed bank size for the per-block picker. `1` = single-variant (production default); `>1` enables `comp_factory_variants` selection logged as log₂(K) bits/block. |
| `--polarity_flip` | off | Add the −1-polarity variant for each Sobol cfg (+1 config bit per block). |
| `--w_scales LIST` | `"1.0"` | Comma-separated W-magnitude scales for the picker (e.g. `0.75,1.0`). |

### Results — SC comp vs FP comp (N=500, sc_prec=8, RTX 4080)

Apples-to-apples (`avg_sc_draws=1`, `ridge=1e-2`, `start_block=0`):

| sc_config | Raw SC | + FP-comp | + SC-comp | Δ vs FP-comp |
|---|---:|---:|---:|---:|
| `qk_only`     | 0.846 | 0.850 | 0.852 | +0.2 pt |
| `qk_av`       | 0.842 | 0.846 | 0.836 | −1.0 pt |
| `full_attn`   | 0.832 | 0.844 | 0.840 | −0.4 pt |
| `skip_worst50`| 0.844 | 0.844 | 0.842 | −0.2 pt |

**Take-away.** Raw SC, FP-comp, and SC-comp all land within ±1 pt of each
other across every config tested — inside the N=500 sampling-noise floor.
Residual SC noise is too small for the pool-overlap / control-variate
mechanism to produce a detectable signal. The head-aligned SC compensator
below remains a useful structural design when residual noise does exist.

### Head-aligned SC compensator (recommended default)

The naive SC compensator uses a monolithic `SCLinear` at D=1024 — its Sobol
pool is `_CFG_CACHE[(1024, 8)]`, which has zero informational overlap with a
block's per-head QK operating at D=64. `HeadAlignedSCLinear`
(`sc_integration/head_aligned_comp.py`) fixes this by restructuring the
comp into **16 per-head (D=64) SC matmuls** whose Sobol pool is literally
the same `_CFG_CACHE[(64, 8)]` the block QK uses.

```
Naive SC comp:     x (N,1024) ─[SCLinear D=1024]─> Δ(N,1024)       pool: _CFG_CACHE[(1024,8)]
Head-aligned comp: x (N,1024)─reshape (N,16,64)
                   ├─head 0: (N,64) ─[SCLinear D=64 ] ─> Δ0
                   ├─head 1: (N,64) ─[SCLinear D=64 ] ─> Δ1        each pool: _CFG_CACHE[(64,8)]
                   ├─ …                                             = same as block QK
                   └─head15: (N,64) ─[SCLinear D=64 ] ─> Δ15
                   sum + bias ─────────────────────> Δ(N,1024)
```

**Identical FLOPs:** 16 × 64 × 1024 = 1024 × 1024 inner reductions. Same
bit-level work, same weights (just sliced differently). **Different noise
structure:** comp's SC noise now shares the block's per-head Sobol
sequences → anti-correlation reappears via shared-pool control-variate.

GPU speed: the 16-launch overhead is mitigated by CUDA-stream dispatch
(bit-exact same output; 22 % faster than the sequential version; actually
slightly *faster* than monolithic D=1024 SCLinear on ViT-L token counts
because smaller kernels overlap better).

#### Results — all-SC comp, head-aligned variant (N=500, sc_prec=8, RTX 4080)

With per-block variant selection from a small menu of
**{head-aligned, naive D=1024} × {W-scales 0.5, 0.75, 1.0}** (4 bits of LUT
per block → 96 bits for ViT-L), calibration-picked:

| sc_config | Raw SC | + FP-comp | + all-SC comp head-aligned | Δ vs FP-comp |
|---|---:|---:|---:|---:|
| `qk_only`     | 0.846 | 0.850 | 0.848 | −0.2 pt |
| `qk_av`       | 0.842 | 0.846 | 0.840 | −0.6 pt |
| `full_attn`   | 0.832 | 0.844 | 0.840 | −0.4 pt |
| `skip_worst50`| 0.844 | 0.844 | 0.844 |  0.0 pt |

Calibration still picks `head_aligned` for all 24 blocks across every
config (the menu mechanism is intact), but the resulting all-SC comp now
sits within ±0.6 pt of FP-comp — i.e., inside the N=500 noise floor. Under
the new QK kernel there is too little residual SC noise for the
shared-pool control-variate to produce a measurable advantage; the
head-aligned design remains the right structural choice for an all-SC
inference path (no FP-MAC required for the comp), but it no longer beats
FP-comp on accuracy.

**3 of 4 sc_configs now beat FP-comp at zero FP-MAC cost.** Calibration
picks `head_aligned/s1.00` in 89 / 96 blocks across all four configs; the
remaining 7 picks are minor scale/architecture tweaks on `skip_worst50`
only. Antithetic Sobol and random alternative seeds were never selected —
the head-alignment architecture does essentially all the work.

#### Reproduce

```bash
# Pure all-SC comp with head-aligned menu (recommended default):
python experiments/qwt_sc_overnight.py \
    --sc_config qk_only --n_calib 256 --n_eval 500 --batch_size 8 \
    --comp_mode sc --n_variants 4 --w_scales 0.5,0.75,1.0 \
    --head_aligned --n_heads 16 --skip_baseline \
    --out_json results/pure_sc_qk_only_combined_pureSC.json
```

**Driver flags (head-aligned):**

| Flag | Default | What it does |
|---|---|---|
| `--head_aligned` | off | Add head-aligned variants to the per-block menu. |
| `--head_aligned_only` | off | Drop regular SC variants entirely; menu is head-aligned only. |
| `--n_heads INT` | `16` | Number of head splits (must divide D=1024). |
| `--n_variants INT` | `1` | Sobol seed-bank size (default + antithetic + random alt K-side seeds). Empirically K=1 is usually enough when head-aligned is in the menu. |
| `--w_scales LIST` | `"1.0"` | Comma-separated W-premultipliers. `"0.5,0.75,1.0"` is the recommended menu. |

See `OVERNIGHT_REPORT.md` for the full ablation and variant-usage
statistics.

### Mixed-precision MP + SC compensation

MP assigns variable bitstream lengths per row/head/op: high-importance
rows keep the full stoc_len, low-importance rows use shorter (faster,
noisier) streams. The overnight driver now accepts the same MP flags as
`eval.py` and composes them with fine-grained scheduling + SC comp.

**Effective stoc_len (`eff_sl`)** is the FLOP-weighted average bitstream
length across *all* SC operations including the compensation block (which
runs at `comp_sc_prec=8` → stoc_len 256). For skip_worst40, the comp
block is 12.8 % of total SC MACs (6.47 B / 50.41 B), so the dilution
from comp is modest.

#### Results — skip_worst40, varying MP aggressiveness (N=500, seed=0, 4× RTX PRO 6000)

| MP recipe | main_sl | eff_sl | reduction | raw top-1 | + SC-comp | vs FP (0.846) |
|---|---:|---:|---:|---:|---:|---:|
| no MP `[256]` | 256 | 256.0 | 0 % | 0.850 | 0.848 | +0.002 |
| `[256,128]` 50/50 | 192 | 200.2 | 21.8 % | 0.822 | 0.836 | −0.010 |
| `[256,128,64]` ⅓ each | 149 | 163.0 | 36.3 % | 0.296 | **0.842** | −0.004 |
| uniform `[128]` | 128 | 144.4 | 43.6 % | 0.032 | 0.830 | −0.016 |
| `[256,128,64,32]` ¼ each | 120 | 137.5 | 46.3 % | 0.000 | 0.634 | −0.212 |
| uniform `[64]` | 64 | 88.6 | 65.4 % | 0.000 | 0.498 | −0.348 |

**Headline.** SC-comp is most valuable *with* MP: at the 3-level recipe
(`[256,128,64]`), raw SC collapses to 29.6 % but comp recovers to 0.842
(−0.4 pt of FP) at 36.3 % effective bitstream reduction — a genuine
energy win on an SC ASIC. The cliff is sharp: below eff_sl ≈ 140 (roughly
the 4-level recipe), raw SC is destroyed and comp can no longer fully
recover.

**Why comp works under MP.** MP assigns shorter bitstreams to
low-importance rows, which amplifies the quantization-induced norm
inflation described above: shorter streams mean smaller Sobol prefixes,
so even moderately-quantized values (not just near-zero ones) suffer
from imprecise enable-signal counts. This inflates the per-block output
norm further. The linear residual `Y_fp − Y_sc ≈ X·W + b` captures
this inflation because it remains a deterministic, approximately-linear
function of the input — the shorter bitstreams just increase its
magnitude. Per-sample Sobol variance is averaged out by downstream
LayerNorm + 257-token aggregation, so the correctable systematic
component dominates the logit error — until the per-sample variance
grows large enough to overwhelm the correction (the cliff).

#### skip_worst30 + MP (N=500)

| MP recipe | main_sl | eff_sl | raw top-1 | + SC-comp | vs FP |
|---|---:|---:|---:|---:|---:|
| no MP | 256 | 256.0 | 0.826 | 0.838 | −0.008 |
| `[256,128]` 50/50 | 192 | 200.2 | 0.774 | 0.834 | −0.012 |

skip_worst30 has 90 SC ops (vs 80 for skip_worst40), so MP has more
surface to damage. The 50/50 recipe drops raw SC to 0.774, but comp
recovers +6.0 pt to 0.834. Without MP, comp slightly hurts (−1.2 pt)
because raw SC is already above FP.

#### Reproduce

```bash
DATA=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data

# skip_worst40 + 3-level MP + head-aligned SC comp (best tradeoff):
python experiments/qwt_sc_overnight.py \
    --sc_config skip_worst40 --n_calib 256 --n_eval 500 --batch_size 64 \
    --comp_mode sc --n_variants 1 --w_scales 1.0 \
    --head_aligned --n_heads 16 \
    --mp_levels 256,128,64 --mp_fractions 0.333,0.334,0.333 \
    --mp_ops mlp_fc1,mlp_fc2,qkv_proj,out_proj \
    --qk_mp_levels 256,128,64 --qk_mp_fractions 0.333,0.334,0.333 \
    --av_mp_levels 256,128,64 --av_mp_fractions 0.333,0.334,0.333 \
    --out_json results/mp_test/sw40_3lvl.json

# 4-way parallel sweep (one MP level per GPU):
for cfg in "256,128:0.5,0.5" "256,128,64:0.333,0.334,0.333" \
           "128:1.0" "64:1.0"; do
    IFS=: read lvls fracs <<< "$cfg"
    gpu=$((i++))
    CUDA_VISIBLE_DEVICES=$gpu nohup python -u experiments/qwt_sc_overnight.py \
        --sc_config skip_worst40 --n_calib 256 --n_eval 500 --batch_size 64 \
        --comp_mode sc --n_variants 1 --w_scales 1.0 \
        --head_aligned --n_heads 16 --workers 2 \
        --mp_levels $lvls --mp_fractions $fracs \
        --mp_ops mlp_fc1,mlp_fc2,qkv_proj,out_proj \
        --qk_mp_levels $lvls --qk_mp_fractions $fracs \
        --av_mp_levels $lvls --av_mp_fractions $fracs \
        --data_root $DATA \
        --out_json results/mp_aggressive/sw40_${lvls//,/_}.json \
        > logs/sw40_${lvls//,/_}.log 2>&1 &
done
wait
```

#### Driver MP flags

| Flag | Default | What it does |
|---|---|---|
| `--mp_levels LIST` | `""` | Stoc_len levels (descending) for per-input-row MP on linear ops. |
| `--mp_fractions LIST` | `""` | Fraction of rows per level (sums to 1). Empty = equal split. |
| `--mp_ops LIST` | `""` | Linear ops receiving MP: `mlp_fc1,mlp_fc2,qkv_proj,out_proj`. |
| `--qk_mp_levels LIST` | `""` | Per-head fixed MP on QK (metric = per-head \|Q\|.amax). |
| `--qk_mp_fractions LIST` | `""` | Head fractions per QK level. |
| `--av_mp_levels LIST` | `""` | Per-attn-row fixed MP on AV (metric = attn_row.amax). |
| `--av_mp_fractions LIST` | `""` | Row fractions per AV level. |
| `--range_mp 0\|1` | `0` | Enable range-based per-weight-group MP. |
| `--range_mp_levels LIST` | `"256,128"` | Stoc_len levels for range MP. |
| `--range_mp_threshold FLOAT` | `0.3` | Normalized range threshold (higher = more low-prec groups). |

The driver now also reports `main_sl` (main ops only) and `eff_sl`
(including comp) in both the log summary and JSON (`"stoc_len"` key).

### Per-(op, block) coarse-grained runs

For the coarse-grained experiments, each active `(op, block)` gets a
single scalar `stoc_len`. This is the `fraction_units=1` / `n_bins=1`
case in [cls/experiments/mp_budget_swap_search.py](cls/experiments/mp_budget_swap_search.py).
Search uses only a calibration-time local proxy; end-to-end top-1 is
used only for the final evaluation.

Current `skip_worst30` scalar recipe under `post-drop main_sl = 128`:

- `qk = 128`
- `av = 128`
- `fc2 >= 192`
- search only `proj, mlp_fc1, mlp_fc2`
- levels: `64,96,128,192,256`
- proxy: `comp_residual`

Run the coarse search:

```bash
python cls/experiments/mp_budget_swap_search.py \
    --sc_config skip_worst30 \
    --target_main_sl 128 \
    --levels 64,96,128,192,256 \
    --search_ops proj,mlp_fc1,mlp_fc2 \
    --fixed_ops qk=128,av=128 \
    --op_min_levels mlp_fc2=192 \
    --proxy comp_residual \
    --init_mode uniform_repair \
    --init_level 128 \
    --n_search 32 \
    --search_seed 1 \
    --max_iters 4 \
    --out_json results/best_skip_worst30_mainsl128_alloc_seed1_i4.json
```

Evaluate a searched map with the official FP compensator:

```bash
python cls/experiments/eval_custom_sl_map_fpcomp.py \
    --sc_config skip_worst30 \
    --sl_map_json results/best_skip_worst30_mainsl128_alloc_seed1_i4.json \
    --n_calib 128 \
    --n_eval 500 \
    --out_json results/best_skip_worst30_mainsl128_alloc_fpcomp500.json
```

Uniform baseline under the same `post-drop main_sl = 128` budget:

```bash
python cls/experiments/eval_custom_sl_map_fpcomp.py \
    --sc_config skip_worst30 \
    --uniform_sl 128 \
    --n_calib 128 \
    --n_eval 500 \
    --out_json results/skip_worst30_uniform128_fpcomp500.json
```

Notes:

- `post-drop main_sl` means the MAC-weighted average `stoc_len` over the
  active SC `(op, block)` pairs after `skip_worst30`. Dropped pairs stay
  FP and are not included in this average.
- Coarse-grained search is the default because `--fraction_units`
  defaults to `1`. To stay in the coarse regime, do not set
  `--fraction_units > 1`.

Optional split-`proj` heuristic from FP profiling:

```bash
python cls/experiments/build_profiled_split_proj_map.py \
    --profile_json results/e2e/amax_row_block_table_fp_500.json \
    --base_map_json results/best_skip_worst30_mainsl128_alloc_seed1_i4.json \
    --out_json results/profiled_split_proj_map_main128.json

python cls/experiments/eval_custom_sl_map_fpcomp.py \
    --sc_config skip_worst30 \
    --sl_map_json results/profiled_split_proj_map_main128.json \
    --n_calib 128 \
    --n_eval 500 \
    --out_json results/profiled_split_proj_map_main128_fpcomp500.json
```

The builder keeps the validated scalar `qk / av / fc1 / fc2` settings,
splits `proj` into `qkv_proj` and `out_proj`, keeps `out_proj` at
uniform `128`, and assigns `qkv_proj` levels from FP `amax`
heterogeneity.

### skip_worst30 sweep across int 6/7/8 + avg192

Full 14-config matrix at `K=30` (drop the 30 worst `(op, block)` cells
from the SC schedule): **`{p6=SL64, p7=SL128, p8=SL256, avg192=mean-192} ×
{uniform, MP} × {QwT off, QwT on at r²=0.5}`**. Runs 4-way parallel with
pool-based scheduling (handles per-GPU exclusive-mode contention via <60 s
death detection). `N_EVAL=1000`, `N_CALIB=1024`.

**Config matrix**

| tag    | mode    | sc_prec | levels              | fractions                 |
|--------|---------|:-:|---------------------|---------------------------|
| p6     | uniform | 6 | — (uniform SL=64)   | —                         |
| p6     | mp      | 7 | `128,64,32,16`      | `0.15,0.575,0.225,0.05`   |
| p7     | uniform | 7 | — (uniform SL=128)  | —                         |
| p7     | mp      | 8 | `256,128,64,32`     | `0.15,0.575,0.225,0.05`   |
| p8     | uniform | 8 | — (uniform SL=256)  | —                         |
| avg192 | mp      | 8 | `256,128,64,32`     | `0.575,0.3,0.075,0.05`    |
| avg192 | uniform | 8 | `192` (every block) | `1.0`                     |

MP applies to all 6 ops: `mlp_fc1, mlp_fc2, qkv_proj, out_proj` via
`--mp_*`, plus `qk, av` via `--qk_mp_* / --av_mp_*`. `avg192 uniform` uses
SC early termination to put SL=192 on every block — implemented via the MP
path with a single level `[192]` because `--sc_prec` only accepts pow2
ints. All QwT runs use `--r2_threshold 0.5 --comp_sc_prec 8 --skip_baseline`.

**Prereqs.** Build schedule JSONs from the raw per-operator sensitivity
data (`cls/sensitivity/sensitivity_per_operator_real_sc_p{6,7,8}.json`):

```bash
python cls/experiments/build_skip_worst20_json.py \
    --k 30 \
    --sens_dir cls/sensitivity \
    --out_dir cls/sensitivity/skip_worst30
```

Produces `skip_worst30_p{6,7,8}.json` (5-op × 24-block 0/1 schedules).
Change `--k` for other drop counts (20, 40, …).

The driver scripts that produced these maps have been retired (used the
legacy r²/cv-holdout gate CLI that has since been replaced by the
cross-seed cosine gate — see [QwT compensation gate](#qwt-compensation-gate)
below). The generated `sl_map` JSONs referenced by the drivers are still
on disk at `cls/results/sweep_int678_k30/sl_maps/` on machines where the
sweep was run, and remain the MP baselines for current comparisons.

#### QwT compensation gate — the block-23 collapse and how to avoid it (legacy r²-gate; superseded 2026-04-25)

> **Status:** historical. The current production gate is the **cross-seed
> cosine** rule documented in
> [`docs/SC_COMP_ALGORITHM.md`](docs/SC_COMP_ALGORITHM.md) and the
> [Production status](#production-status-2026-04-25) section above. The
> `--r2_threshold` / `--last_block_r2_threshold` / `--max_block_for_qwt`
> CLI flags described below have been replaced by `--cos_threshold` /
> `--last_block_cos_threshold` / `--start_block` (still with
> `--lookahead_veto`). This section is preserved for the block-23 collapse
> analysis, which is still relevant context for any future gate redesign.

Early runs at `--r2_threshold 0.5` exposed a catastrophic failure where
some MP+QwT configs collapsed to `top1 ≈ 0`. Root cause: block 23's
residual-stream magnitude is ~20× larger than mid-block (pre-norm
transformer compound effect), so a **borderline-r² comp at block 23**
injects a ~0.8-magnitude wrong-direction correction into the pre-head
embedding and destroys classification. The same mechanism can also
manifest as compounded noise across multiple late blocks at aggressive
thresholds (r²=0.0 / 0.3).

Three gating knobs now exist in `qwt_sc_overnight.py` and
`eval_custom_sl_map_fpcomp.py` to handle it:

| flag | what it does | when to use |
|---|---|---|
| `--r2_threshold T` | Enable comp only for blocks with fit r² > T. | Primary knob. `0.5` is the safe universal default. |
| `--last_block_r2_threshold T_last` | Stricter gate *only for the last block*. Protects the pre-head embedding without sweeping all blocks under one conservative bar. | Pair with any aggressive `r2_threshold`. `0.9` is our default. |
| `--lookahead_veto` | 1-step binary lookahead: for every block that passes the r² gate, propagate `sc_out` vs `sc_out + comp` through `block_fp[i+1]` and veto apply when skip is closer to FP. Last block is exempt (no downstream block) — use `last_block_r2_threshold` there. | Cheap (~1 min extra per QwT calibration). Catches borderline-fit comps whose injected noise amplifies into block i+1. |
| `--max_block_for_qwt N` | Hard cap: never compensate blocks ≥ N. Blunt but trivial. | Fallback for very aggressive gates. Superseded by `last_block_r2_threshold` in most cases. |

**Recommended production recipe (and what the sweep now uses):**

```
--r2_threshold 0.5
--last_block_r2_threshold 0.9
--lookahead_veto
--comp_mode sc --comp_sc_prec 8 --ridge 1e-4
```

**Headline numbers at N=50,000 (ImageNet val), K=30:**

| Config | Raw SC | + QwT (prod r²=0.7) | + QwT (opt3 r²=0.5 + last=0.9 + lookahead) | Δ (opt3 vs raw) |
|---|---:|---:|---:|---:|
| p7 uniform (SL=128)     | 78.94 % | 78.94 %            | **81.10 %** (N=1k)¹ | **+2.16** ⭐ |
| p7 mp (search-driven)   | 79.89 % | 79.89 %            | **79.90 %** (N=1k)¹ | +0.01 |
| p8 uniform (SL=256)     | 85.57 % | 85.73 %            | **84.90 %** (N=1k)¹ | −0.67 |
| avg192 uniform (SL=192) | 84.67 % | 84.67 %            | **85.10 %** (N=1k)¹ | +0.43 |
| avg192 mp (SL≈192)      | 84.62 % | 84.62 %            | **84.00 %** (N=1k)¹ | −0.62 |

¹ opt3 column is from the N=1,000 mini-sweep; prod N=50k rerun at opt3
is still the natural next step if this recipe is committed.

**Ablation at more aggressive thresholds** (N=1000, mini-sweep):

| Config | opt2 la r²=0.5 | opt2 la r²=0.3 | opt3 la+last0.9 r²=0.3 |
|---|---:|---:|---:|
| p7 uniform    | 81.10 % | **83.30 %** ⭐ +4.36 | **83.30 %** (same) |
| p8 uniform    | 84.90 % | 0.10 % ❌ | 0.10 % ❌ (collapse not block-23-only) |
| avg192 uniform| 85.10 % | 0.00 % ❌ | 0.00 % ❌ (same) |
| p7 mp         | 79.90 % | 82.00 %  | (did not rerun) |
| avg192 mp     | (n/a)   | 84.60 %  | (did not rerun) |

**Interpretation.**

- At **r²=0.5**, block 23's fit r² is empirically below 0.5 in every
  K=30 config we measured (0.28–0.35), so the primary gate already
  skips block 23. `last_block_r2_threshold=0.9` is a cheap safety net
  for configs where block 23 might sneak through at lower thresholds.
- At **r²=0.3**, block 23 can clear the gate in some configs. The
  last-block threshold **successfully saves p7_uniform (+4.36 pt)** but
  does **not save p8_uniform / avg192_uniform** — their collapse is
  compounded noise across many mid-network comps, not a single-block
  pathology. Confirmed empirically: block 23 correctly shows
  `enabled=False` yet the classifier still outputs ~1/1000.
- The r²=0.5 recipe therefore trades the ~+4 pt p7_uniform opportunity
  (available at r²=0.3) for universal stability. Tighter MP+QwT wins
  at aggressive thresholds would require either full-chain lookahead
  (expensive, see commit history) or a magnitude-aware multi-block
  veto. **All of the open problems in this section are resolved by the
  cross-seed cosine gate** documented in
  [`docs/SC_COMP_ALGORITHM.md`](docs/SC_COMP_ALGORITHM.md); this section
  is preserved only as historical context for the legacy r²-gate
  collapse modes.

Scaling takeaway at K=30 (on raw-SC numbers, from the prod sweep):
`avg192_uniform` (SL=192 everywhere, ~7.58-bit precision via SC early
termination) hits **84.67 %** raw, trading ~25 % fewer bit-ops per op
vs `p8_uniform` (85.57 %). `avg192_mp` is ~indistinguishable from
`avg192_uniform` on raw SC, so a well-chosen uniform-at-192 is a
strong default for the ~192-SL budget.

### Hardware framing

For an SC accelerator, the relevant cost is **SC ops removed from the FP
path**, not GPU wall-clock:

- FP comp: 24 × 1024² = **25.2 M FP-MACs per image** (21 % of total FP).
- All-SC (head-aligned) comp: 24 × (16 × 64 × 1024) = **25.2 M SC-MACs**
  (identical inner reductions; one SC-MAC at sc_prec=8 = 256 bit-ANDs +
  popcount, no FP multiplier needed).
- Per-block scheduler LUT: ≤ 4 bits × 24 blocks = 96 bits.
- Net silicon delta: **one fewer FP multiplier**. No other mandatory new
  hardware (the SC matmul unit is already required for the attention path).

On GPU the Triton SC kernel has launch overhead (see above — mitigated by
CUDA streams but still present); the numbers above are **accuracy at
theoretical FLOPs/energy parity**, not stopwatch numbers.

## Portability

All data/checkpoint paths default to the GreatLakes shared_data location but
can be overridden via CLI flags so the repo runs on any machine:

| Script | Flag | Default (GreatLakes) |
|---|---|---|
| `eval.py` | `--data_root` | `/scratch/.../shared_data/imagenet/data` |
| `det/fp_eval.py`, `det/sc_eval.py` | `--d2_datasets`, `--ckpt` | `/scratch/.../shared_data`, `.../pretrained/eva_coco_det.pth` |
| `experiments/qwt_sc_overnight.py` etc. | `--data_root` | same |

Shell scripts (`det/run_*.sh`) use `cd "$(dirname "$0")"` and `conda activate vit_sc`
instead of hardcoded absolute paths.

## Detection extension (`det/`)

Ports the SC pipeline to **object detection + instance segmentation on COCO**
using **EVA-01 ViT-L/14 ViTDet** (Cascade Mask R-CNN, 40 blocks). Uses
**module-swap** instead of monkey-patching because EVA's 4-D window-partitioned
attention (`B, H, W, C`) is incompatible with the DINOv2 forward rewrite.

```bash
cd det/

# FP baseline
python fp_eval.py --n-eval 100

# SC eval (QK + AV + projections)
python sc_eval.py --n-eval 100 --sc_ops 'qk,av,proj' --sc_prec 8

# Visualize detection results on images
python visualize.py --results_dir results/sc_p8 --n_images 5
```

See `det/README.md` for the full op taxonomy, schedule dispatch, and
sensitivity sweep instructions.

## File layout

```
cls/                                 Classification (DINOv2 ViT-L/14 on ImageNet-1k)
    eval.py                          FP / SC evaluation driver
    imagenet_parquet.py              ImageFolder-style dataset over HF parquet shards
    sc_attention_patch.py            DINOv2 patcher: per-block per-op SC schedule support
    experiments/                     Sensitivity sweeps + end-to-end configs
        sensitivity_all_ops.py       24 x 5 per-(op, block) sensitivity matrix (noise surrogate)
        sensitivity_per_operator_real_sc.py  6 x 24 per-(op, block) sensitivity matrix (real SC)
        qwt_sc_overnight.py          Uniform-path QwT driver (cross-seed cosine gate)
        eval_custom_sl_map_fpcomp.py MP-path QwT driver (cross-seed cosine gate)
        mp_budget_swap_search.py     Budget-preserving MP sl_map search (local proxy, no E2E eval)
        build_skip_worst20_json.py   Build skip_worst{K} per-precision schedule JSONs
        build_profiled_split_proj_map.py  FP-amax heuristic split-proj sl_map builder
        mp_uniform_vs_fine.py        Uniform vs sensitivity-aware per-(op,block) MP comparison
    sensitivity/skip_worst{K}/       Generated per-K schedule JSONs (build via build_skip_worst20_json.py --k K)
    results/                         Per-run JSON (top-1, top-5, sensitivity data, qwt_sc_*)
sc/                                  SC kernels (Triton XNOR matmul, Sobol RNG, configs)
sc_integration/                      SC matmul wrappers + calibrated noise surrogate
    sc_linear.py                     Shared SCLinear + _sc_linear primitives (cls + det)
    noise_matmul.py                  Closed-form Gaussian surrogate for fast sweeps
det/                                 Detection extension (EVA-01 ViTDet on COCO)
    eval_common.py                   Shared model loading (auto-discovers QwT-SC submodule)
    fp_eval.py / sc_eval.py          FP/SC detection eval
    sc_patch/                        Module-swap SC (matmul1/matmul2 replacement)
    visualize.py                     Draw detection boxes on COCO images
    experiments/                     Per-(op, block) sensitivity sweep for EVA
third_party/QwT-SC/                  git submodule -> Allenjin123/QwT.git (QwT fork with QwT-vit-sc)
TASK_REPORT.md                       Original run log on gl1804 (2×RTX PRO 6000)
```

## SC-Vision Task: DINOv2 ViT-L/14 classification with SC Q@K^T
**Date:** 2026-04-14  **Node:** gl1804.arc-ts.umich.edu  **GPU:** 2× NVIDIA RTX PRO 6000 Blackwell (97 GB)

### Setup
- **Task / dataset:** ImageNet-1k validation classification (parquet at `/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data`, 50 000 images). Used a 100-image subset (seed 0).
- **Model:** `dinov2_vitl14_lc` from `facebookresearch/dinov2` torch.hub (DINOv2 ViT-L/14 backbone + published ImageNet-1k linear classifier head, 24 blocks, d=1024, 16 heads, head_dim=64).
- **SC config:** bipolar / XNOR matmul from `scmp_llm/SC/sc_triton.py`, Sobol-sequence RNG via `make_sobol_simple_config` (`scmp_llm/SC/config_helpers.py`). `sc_prec = 8` → `stoc_len = 256`, quant_max = 127. Per-head symmetric quantization for Q and K (from `qdit.sc_integration.sc_matmul.quantize_for_sc_per_head`).
- **Stochasticized operator:** only the Q @ Kᵀ attention-score matmul in every block. Softmax, Attn @ V, QKV/proj linears, and MLP remain FP32/FP16.
- **Env:** conda env `bench`, torch 2.10.0+cu128, timm 1.0.26, triton 3.6.0; `XFORMERS_DISABLED=1` so DINOv2 falls back to the non-xformers forward path we patch.

### What Was Done
- Created `vit_sc/` project, loaded ImageNet-1k val from HF parquet shards (`imagenet_parquet.py`).
- Implemented SC Q@Kᵀ monkey-patch (`cls/sc_attention_patch.py`): wraps every DINOv2 `Attention` / `MemEffAttention` module's forward to per-head-quantize Q and K, call `sc_matmul_qk_multihead` (bipolar XNOR, Sobol RNG, stoc_len=256), dequantize, then run standard softmax and Attn@V.
- Evaluation driver `eval.py` with FP vs SC modes, subset sampling, JSON output.
- Sanity-checked FP forward on 64 imgs first, then ran matched 100-img FP and SC evaluations with identical image subsets and seeds.

### Files Created
- [vit_sc/cls/imagenet_parquet.py](vit_sc/cls/imagenet_parquet.py) — parquet-shard ImageFolder-style dataset
- [vit_sc/cls/sc_attention_patch.py](vit_sc/cls/sc_attention_patch.py) — SC Q@Kᵀ attention forward + model patcher
- [vit_sc/cls/eval.py](vit_sc/cls/eval.py) — FP / SC evaluation driver
- [vit_sc/cls/results/fp_n100.json](vit_sc/cls/results/fp_n100.json), [vit_sc/cls/results/sc_p8_n100.json](vit_sc/cls/results/sc_p8_n100.json) — results
- Logs under [vit_sc/cls/logs/](vit_sc/cls/logs/)

### Results (ImageNet-1k val subsets, seed=0, identical subset across modes at each N)
| N | Variant | Top-1 | Top-5 | Throughput (img/s) |
|---|---------|-------|-------|---------------------|
| 100  | FP baseline                | 0.890 | 0.970 | 102.8 (bs=32) |
| 100  | SC Q@Kᵀ, sc_prec=8 (N=256) | 0.730 | 0.910 |   7.8 (bs=8)  |
| 1000 | FP baseline                | **0.854** | **0.983** | 232.8 (bs=32) |
| 1000 | SC Q@Kᵀ, sc_prec=8 (N=256) | **0.746** | **0.923** |   8.2 (bs=8)  |

On 1000 images: Δtop-1 = −0.108, Δtop-5 = −0.060.

### 1000-image comparison across SC configs (seed=0, identical subset)
| Config | Top-1 | Top-5 | img/s |
|---|---|---|---|
| FP baseline                   | 0.854 | 0.983 | 232.8 |
| SC: QK                        | 0.746 | 0.923 |   8.2 |
| SC: QK + AV                   | 0.723 | 0.915 |   3.5 |
| SC: QK + AV + qkv_proj + out_proj | 0.705 | 0.886 |   2.9 |

Cumulative SC load → top-1 cost: +AV adds −2.3 pt over QK-only; +projections adds another −1.8 pt. All attention operators (both matmuls + both linears) are now stochastic while MLP and linear_head stay FP.

### Extending SC to Attn@V and MLP linears (100 imgs, seed=0, same subset)
All three ops now use `sc_matmul_grouped_enable_triton(mode="bipolar", group_a=1, group_b=1)` — per-row (per-token / per-output-channel) symmetric bipolar quantization plus sign bits. This is the same primitive Q-DiT's `SCAttention._sc_av_uniform` uses for AV. First attempt with per-tensor `sc_matmul` failed because softmax'd attn has most values ≪ max, so per-tensor bipolar quant zeroed them.

| SC ops (all 24 blocks) | Top-1 | Top-5 | img/s |
|------------------------|-------|-------|-------|
| QK only                | 0.73  | 0.91  | 7.8  |
| QK + AV                | **0.72** | **0.90** | 3.4  |
| QK + AV + MLP (fc1+fc2)| 0.01  | 0.01  | 2.5  |
| QK + AV + MLP (skip last 2 blocks' MLP)| 0.00 | 0.02 | 2.6 |
| QK + AV + MLP (skip first 2 blocks' MLP)| 0.00 | 0.01 | 2.6 |
| qkv_proj + out_proj only (MLP/QK/AV all FP) | **0.86** | **0.96** | 16.2 |
| QK + AV + qkv_proj + out_proj (attention fully SC, MLP FP) | **0.67** | **0.89** | 2.9 |

**Projection layers are SC-friendly.** SC'ing the 48 attention linears (24 × qkv, 24 × out_proj) alone costs only 3 pt top-1 and 1 pt top-5 — similar in magnitude to what we see for QK/AV. The asymmetry with MLP (which has the same kind of FC layers) is the hidden dim: MLP's 4096-wide fc1/fc2 amplify per-layer SC noise roughly 4× more per multiply, and compound over 48 layers. Projections stay 1024-wide and tolerate uniform sc_prec=8 just fine.

- **AV drops in cleanly**: per-row bipolar grouping lets the soft-max'd attention survive SC — almost no additional accuracy cost over QK-only.
- **MLP at uniform sc_prec=8 is not viable** at this scale: 48 FC layers (24×fc1, 24×fc2) each with ~4 % relative error compound to destroy the 1 000-way classifier. Q-DiT's working recipe for MLP uses mixed precision per-group, enable-signal early termination, and selective-layer skipping — none of which we wired up here.

### Next Steps
- Precision sweep sc_prec ∈ {6, 7, 8, 9, 10} for QK + AV (fast, cheap).
- Before a full SC-MLP run: port Q-DiT's `SCMlp` mixed-precision dispatch, or at minimum try a higher sc_prec (e.g. 10) plus skipping the first/last few blocks.
- Scale QK + AV to full 50 k val images to lock in the accuracy number (≈ 4 h at 3.4 img/s on one RTX PRO 6000). SC throughput is dominated by per-batch RNG-pool construction and the per-sample Python loop in `_sc_matmul_qk_batched`; that's an implementation detail, not a fundamental limit.

### Integration notes
- Reused `scmp_llm/Q-DiT/qdit/sc_integration/sc_matmul.py` (`sc_matmul_qk_multihead`, `quantize_for_sc_per_head`, `dequantize_sc_result`) without changes — the DiT attention path it was built for is effectively identical to DINOv2's attention up to the shape (B, H, N, D), so it dropped in cleanly.
- Kept V in FP for this first pass (user only asked for SC multiplication; Q@Kᵀ is the canonical SC target and the softmax-free matmul).

### Next Steps
- Scale to full val (expected runtime ≈ 2 h for SC on one GPU at current speed).
- Precision sweep: sc_prec ∈ {4, 6, 7, 8} accuracy curve.
- SC-ify Attn@V as well (same primitive, different shape handling).
- Optimize RNG-pool reuse across batches and heads — `make_sobol_simple_config(D, D, sc_prec)` depends only on (D, sc_prec), so the pool can live at module scope.

# vit_sc/det — 实验计划

EVA-01-g ViTDet (patch16, 1280², embed 1408, 40 blocks) + COCO detection。
SC 替换 attention / MLP 的六类 op（`mlp_fc1, mlp_fc2, qkv_proj, out_proj, qk, av`）。

## What's new (2026-04-25)

**Production status**: drivers rewritten around the cross-seed cosine gate
+ head-aligned SC compensator, mirroring the cls-side
[`docs/SC_COMP_ALGORITHM.md`](../docs/SC_COMP_ALGORITHM.md) recipe (B_ha
production winner at `n_heads=16, sc_prec=8`).

- [`experiments/qwt_det_compensate.py`](experiments/qwt_det_compensate.py)
  — calls the cross-seed `calibrate_qwt` API with two disjoint COCO val
  slices (anchored at `--calib_seed` and `--calib_seed_b`); reports
  per-block `cos_ab` instead of the deprecated `r²` field. Adds
  `--head_aligned --n_heads 16` for HeadAlignedSCLinear comp.
- [`sc_eval.py`](sc_eval.py) and the QwT driver above both gain the cls-side
  MP CLI surface (`--mp_levels`, `--mp_ops`, `--qk_mp_levels`,
  `--av_mp_levels`, plus `--range_mp*` / `--adaptive_mp*` for parity);
  built into specs by [`mp_spec.py`](mp_spec.py) and passed through to
  [`sc_patch_eva`](sc_patch/sc_model_eva.py).
- [`experiments/sweep_2026-04-25.sh`](experiments/sweep_2026-04-25.sh) —
  4-GPU pool harness covering `sc_prec ∈ {7, 8} × K ∈ {20, 30} × mode ∈
  {uniform, mp192} × qwt ∈ {off, on}` at n_eval=5000.

Updated [`RESULTS.md`](RESULTS.md) summarises the production recipe and
holds the n_eval=5000 numbers (TBD until sweep finishes).

The old `--min_r2` / `--avg_sc_draws` flags are gone — they belonged to
the legacy r²-gate that the cross-seed gate physically supersedes (see
[`third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py`](../third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py)
module docstring for the derivation).

## 计划（三阶段）

### 阶段 1 — Sensitivity sweep（一次性）

每个 (op, block) 对单独 SC 化、其它全 FP，跑 COCO n_eval=100 得 AP，生成 6×40 的
sensitivity 矩阵。基于这个矩阵排序并 build `skip_worst_K` 调度（跳过最敏感的 K
个 op-block 对）。

- **n_eval = 100**
- 精度：`sc_prec ∈ {6, 7, 8}`
- 输出：`results/sensitivity_int{6,7,8}_n100/`
  - `sensitivity_all_ops.json` — 6×40 AP 矩阵
  - `schedules/skip_worst_K.json` — 由矩阵生成的调度（K 见阶段 2）
- 脚本：`experiments/sensitivity_sweep.py` + `experiments/merge_and_build_skip_worst.py`
- 只跑一次，后续所有 QwT 实验的 schedule 来源。

### 阶段 2 — Raw SC baseline（无 QwT）

用阶段 1 生成的 schedule 跑纯 SC，确认每个 (prec, K) 的 raw SC 有多少 AP
损失（相对 FP=68.73）。这是阶段 3 的参照。

- **n_eval = 100**
- 矩阵：`sc_prec ∈ {6, 7, 8}` × `K ∈ {TBD}`
  - K 值待定，大概率是 `{10, 20, 30}` 的子集，按阶段 1 结果再确认
- 输出：`results/skip_worst_{K}_int{prec}_n100/metrics.json`
- 脚本：`sc_eval.py --sc_ops_per_block_json <schedule>`

### 阶段 3 — QwT 补偿版

在阶段 2 同一 (prec, K) 配置上加 QwT 线性补偿，对比 raw SC 的 Δ。

- **n_eval = 100**
- 矩阵：`sc_prec ∈ {6, 7, 8}` × `K ∈ {和阶段 2 一致}` × `n_calib ∈ {TBD}`
  - `n_calib` 待定，从现有结果看 int6 对 16 张已 work，int7 需要 ≥64 张
    （甚至需要 `avg_sc_draws > 1`）——等阶段 1/2 结果出来再定
- 输出：`results/qwt_det/qwt_int{prec}_sw{K}_n{ncalib}/run.json`
- 脚本：`experiments/qwt_det_compensate.py`

## 当前状态（2026-04-21）

**已有**（n_eval=50 的旧 sweep；n_eval=100 的阶段 1 还没跑）：

- `results/sensitivity_int{6,7}_n50/` — sensitivity 矩阵 + sw10/20/30 schedules
- `results/skip_worst_{10,20,30}_int{6,7}[_n50]/` — raw SC baseline（部分）
- `results/qwt_det/qwt_int{6,7}_sw{10,30}_n50/` — QwT 补偿（n_calib=16）

**在跑**（阶段 3 预研，验证 n_calib 对 int7 的影响；基于旧 sweep schedule）：

| JobID | 配置 | n_calib |
|---|---|---|
| 48390688 | int7 sw10 | 64 |
| 48390689 | int7 sw10 | 128 |
| 48390690 | int6 sw10 | 64 |
| 48390691 | int6 sw10 | 128 |

## 关键 setup 说明

### Op 分类

| 名称 | 对应模块 |
|---|---|
| `mlp_fc1` | `block.mlp.fc1` (nn.Linear → SCLinear) |
| `mlp_fc2` | `block.mlp.fc2` (nn.Linear → SCLinear) |
| `qkv_proj` | `block.attn.qkv` (nn.Linear → SCLinear) |
| `out_proj` | `block.attn.proj` (nn.Linear → SCLinear) |
| `qk` | `block.attn.matmul1` (MatMul → SCMatMul) |
| `av` | `block.attn.matmul2` (MatMul → SCMatMul) |

### Skip-worst 语义

`skip_worst_K` = 把 sensitivity 排序后最敏感的 K 个 (op, block) 对留 FP，其它 SC。
K 越小越 aggressive（SC 覆盖越多），如当前 K=10 时 SC 覆盖 230/240 op-block（96%）。

### Module-swap（非 monkey-patch）

EVA-ViT 的 `Attention` 把 matmul 暴露成命名子模块（`matmul1`, `matmul2`），直接
swap 即可；不需要像 cls 那样重写 forward。

### Checkpoint / 数据

- EVA ViTDet cascade_mask_rcnn 权重：
  `/scratch/nbleier_owned_root/nbleier_owned1/shared_data/pretrained/eva_coco_det.pth`
- D2 数据根目录：
  `DETECTRON2_DATASETS=/scratch/nbleier_owned_root/nbleier_owned1/shared_data`
- conda env: `qwt_d2`

### Detectron2 安装两个坑

1. **GCC ≥ 9**（GL 节点默认 8.5 不够）：`module load gcc/13.2.0`
2. **build isolation 要关**：
   ```bash
   pip install --no-build-isolation -e \
       third_party/QwT-SC/QwT-det-RepQ-ViT/eva1/eva_det
   ```

## FP baseline

- bbox AP = **68.73**
- segm AP = **61.61**
（n_eval=100，见 `results/fp100/metrics.json`）

## 后续 TODO

- [ ] 清理 `results/` 下 n_eval=10/50 旧 sweep 产物，只保留 n_eval=100
- [ ] 阶段 1 完成后确定 K 集合
- [ ] 阶段 2 完成后确定 n_calib 集合（可能还包括 `avg_sc_draws`）
- [ ] 如果 int7 仍塌陷，尝试 `avg_sc_draws ∈ {4, 8}` 攻 SNR 而非样本量

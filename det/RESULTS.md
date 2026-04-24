# vit_sc/det — 实验结果

本文件汇总 det 侧当前已完成的 QwT-SC 补偿实验结果，重点关注 `min_r2`
门限对 SC-int8 补偿器稳定性的影响。

项目规划、setup 与阶段划分见 [`README.md`](README.md)。

## Baselines（COCO val, n_eval=10）

同一份 n_eval=10 子集（`results/smoke/`）：

| 模型 | bbox AP | segm AP |
|---|---:|---:|
| FP 参考（EVA-ViTDet-g） | 69.33 | 60.40 |
| Raw SC（sw20_int6, sc_prec=6, no QwT） | 52.31 | 43.54 |

Raw SC 相对 FP 损失约 −17.0 bbox / −16.9 segm。

> 注：README 中记录的 FP@n_eval=100 是 bbox=68.73 / segm=61.61。本文所有实验
> 都在 n_eval=10 上跑，数字波动较大（≈±1–2 AP），结论应以 Δ 而非绝对值理解。

## 阶段 3 预研：`min_r2` sweep（SC-int8 compensator）

**固定**：`sc_prec=6`、schedule = `skip_worst_20_int6`（SC 覆盖 220/240
op-block，≈92%）、`n_calib=16`、`n_eval=10`、`ridge=0.01`、`comp_mode=sc`、
`comp_sc_prec=8`、`avg_sc_draws=1`。

**变量**：`min_r2` ∈ {0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}。

结果（`results/sw20_int6_qwt_sc_int8_n10[_r2_0_*]/run.json`）：

| min_r2 | enabled | bbox AP (comp) | segm AP (comp) | Δbbox vs raw SC |
|---:|:---:|---:|---:|---:|
| 0.0 | 40/40 |  0.00 |  0.00 | −52.31 |
| **0.2** | **39/40** | **73.21** | **59.43** | **+20.90** |
| 0.3 | 39/40 | 73.21 | 59.43 | +20.90 |
| 0.4 | 38/40 | 72.34 | 59.01 | +20.03 |
| 0.5 | 31/40 | 67.69 | 55.98 | +15.38 |
| 0.6 | 29/40 | 65.30 | 55.25 | +12.99 |
| 0.7 | 26/40 | 66.85 | 57.12 | +14.54 |
| 0.8 | 21/40 | 67.61 | 55.81 | +15.30 |

### 关键发现

1. **Block 0 是毒瘤。** 当 `min_r2=0.0`（全 40 个 block 都上 SC-int8
   补偿），bbox/segm 同时塌到 0.00。Block 0 的 calib r2 最低（0.164），
   其 SC-int8 量化的 `W,b` 不稳定，串在流水线最前端会把后续所有特征
   污染掉。只要把它禁掉（`min_r2 ≥ 0.2`，等价于 `enabled=39/40`），bbox
   立刻回弹到 73.21（甚至在这个 n_eval=10 子集上超过 FP 参考）。
2. **甜蜜点在 min_r2=0.2–0.3。** 这时只禁掉 block 0，其余 39 个 block
   的 SC-int8 补偿都有效。
3. **继续抬 min_r2 并非单调有益。** 0.4 仅比 0.2 略低；0.5 之后（禁掉
   ≥9 个 block，补偿覆盖 <80%）bbox 回落到 65–68 区间，收益被"少补偿"
   的副作用抵消。这说明大多数中等 r2 block 的补偿仍有正收益，不应因
   r2 低就一刀切。
4. **相对 raw SC 的净收益 ≈ +20 bbox / +16 segm**（min_r2=0.2 时），
   证明 QwT 线性补偿 + SC-int8 投影在 det 路径上可行，和 cls 侧的结论
   一致。

### 未完成

- `comp_sc_prec ∈ {6, 7}` 的 sweep：`results/sw20_int6_qwt_sc_int{6,7}_n10_r2_*/`
  目录已建但 run.json 为空，对应 `logs/r2_sweep/*.log` 最后一行停在
  `[calib] collecting first-block inputs from model_sc`（被 kill 或超时）。
  需要重跑以比较 int6/int7 vs int8 补偿器的精度-比特权衡。
- 所有数字都是 n_eval=10，噪声较大（bbox=73.21 > FP=69.33 本身就是
  小样本的 artifact）。阶段 3 正式结果需要 n_eval=100。

## 近期代码改动

- `experiments/qwt_det_compensate.py`：新增 `--d2_datasets`、`--ckpt`
  CLI 参数，把数据集和权重路径从硬编码提取为可覆盖项，方便在非
  GreatLakes 环境复跑。默认仍走 shared_data 路径，行为不变。

## 复现命令

```bash
# min_r2=0.2 甜蜜点（上表的最佳行）
python experiments/qwt_det_compensate.py \
    --sc_prec 6 \
    --sc_ops_per_block_json sensitivity/skip/skip_worst_20_int6.json \
    --n_calib 16 --n_eval 10 \
    --ridge 0.01 --min_r2 0.2 \
    --comp_mode sc --comp_sc_prec 8 --comp_sc_mode bipolar \
    --out_tag sw20_int6_qwt_sc_int8_n10_r2_0_2 \
    --out_json results/sw20_int6_qwt_sc_int8_n10_r2_0_2/run.json
```

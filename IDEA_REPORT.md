# Uniform vs Fine-Grained Mixed Precision Analysis

**Date:** 2026-04-16
**Config:** DINOv2 ViT-L/14 + linear head, ImageNet-1k val
**GPU:** RTX 4080 (16 GB)

## Research Question

At the same effective SC length (eff_sl), does sensitivity-aware
per-(op,block) mixed precision achieve higher accuracy than uniform?

## Complete Results

### skip_worst40 (80/120 SC ops)

#### Proportional vs Uniform (N=500)

| target | prop_msl | prop_raw | unif_msl | unif_raw | delta |
|--------|----------|----------|----------|----------|-------|
| 256 | 238.2 | 0.848 | 256.0 | 0.844 | +0.4 |
| 224 | 221.1 | 0.844 | 224.0 | 0.834 | +1.0 |
| 192 | 191.9 | 0.818 | 192.0 | 0.816 | +0.2 |
| 160 | 159.6 | **0.738** | 160.0 | 0.578 | **+16.0** |
| 128 | 136.0 | 0.056 | 128.0 | 0.032 | +2.4 |

#### N=2000 confirmation at target=192

prop=0.8380 unif=0.8320 delta=+0.6 pt (Z=0.51, p=0.31, **not significant**)

#### 2-level {256,128} strategies (N=500)

| target | uniform | sensitivity | inverse | random | proportional |
|--------|---------|-------------|---------|--------|--------------|
| 256 | 0.844 | 0.844 | 0.844 | 0.844 | 0.848 |
| 224 | 0.834 | 0.832 | 0.814 | 0.518 | 0.844 |
| 192 | 0.816 | 0.788 | 0.358 | 0.102 | 0.818 |
| 160 | 0.578 | 0.104 | 0.072 | 0.082 | 0.738 |
| 128 | 0.032 | 0.032 | 0.032 | 0.032 | 0.056 |

### skip_worst30 (90/120 SC ops)

#### All strategies (N=500)

| target | uniform | sensitivity | inverse | random | proportional |
|--------|---------|-------------|---------|--------|--------------|
| 256 | 0.824 | 0.824 | 0.824 | 0.824 | 0.832 |
| 224 | 0.820 | 0.800 | 0.616 | 0.616 | 0.820 |
| 192 | 0.752 | 0.640 | 0.150 | 0.174 | **0.790** |
| 160 | 0.272 | 0.034 | 0.004 | 0.042 | **0.548** |
| 128 | 0.002 | 0.002 | 0.002 | 0.002 | 0.010 |

#### N=2000 confirmation at target=192

prop=0.7885 unif=0.7715 delta=+1.7 pt (Z=1.30, p=0.097, **marginal**)

### Proportional allocations (MAC-weighted %)

**skip_worst40:**

| target | msl | allocation |
|--------|-----|------------|
| 256 | 238.2 | 60%@256, 25%@224, 14%@192, 1%@160 |
| 224 | 221.1 | 35%@256, 28%@224, 31%@192, 7%@160 |
| 192 | 191.9 | 3%@256, 32%@224, 31%@192, 31%@160, 4%@128 |
| 160 | 159.6 | 43%@160, 29%@128, 28%@192, 0.2%@224 |

**skip_worst30:**

| target | msl | allocation |
|--------|-----|------------|
| 192 | 192.4 | similar spread across {128..256} |
| 160 | 160.2 | similar spread, more weight at 128-160 |

## Statistical Assessment

At N=500, the 95% CI is +/-3.2 pt. At N=2000, +/-1.8 pt.

| Comparison | N=500 delta | N=2000 delta | Significant? |
|------------|-------------|--------------|-------------|
| sw40 prop vs unif @ 192 | +0.2 | +0.6 | No |
| sw30 prop vs unif @ 192 | +3.8 | +1.7 | Marginal (p=0.10) |
| sw40 prop vs unif @ 160 | +16.0 | (not run) | **Yes** (N=500 sufficient) |
| sw30 prop vs unif @ 160 | +27.6 | (not run) | **Yes** (N=500 sufficient) |

**Honest conclusion:** At moderate reduction (target=192), proportional is
+0.6 to +1.7 pt better -- positive trend but not statistically significant
at N=2000. At aggressive reduction (target=160), proportional is
**dramatically** better (+16 to +28 pt).

## What the Data Says About the Algorithm

### Finding 1: Sensitivity ordering is critical

At target=192 for skip_worst30 with coarse {256,128} levels:
- sensitivity: 0.640 (correct ordering)
- inverse: 0.150 (wrong ordering)
- random: 0.174

The 4x gap between sensitivity and inverse/random proves that WHICH ops
get high precision matters enormously. This is the foundational observation.

### Finding 2: Coarse splits kill accuracy

The 2-level {256,128} sensitivity strategy (0.640 for sw30) is WORSE than
uniform 192 (0.752). Having 50% of ops at stoc_len=128 is catastrophic even
for insensitive ops. The "weakest link" dominates.

### Finding 3: Fine-grained proportional avoids the cliff

Proportional with quantized levels {128,160,192,224,256} achieves 0.790 vs
uniform 0.752 (+3.8 pt at N=500) by keeping most ops at 160-224. Only ~4%
of MACs land at 128.

### Finding 4: The benefit concentrates near the cliff

- target >= 224: proportional ~= uniform (within noise)
- target = 192: proportional ~1-2 pt better (positive but small)
- target <= 160: proportional >> uniform (+16-28 pt, significant)

The cliff is at stoc_len ~160-192 for skip_worst30/40. Proportional
allocation protects sensitive ops above the cliff while uniform forces
everything below it.

## Proposed Algorithm: Sensitivity-Proportional Per-Block Allocation

### Core idea

Replace the current uniform-across-blocks MP policy with per-(op,block)
stoc_len assignment based on the sensitivity matrix.

### Algorithm

```python
def alloc_proportional(active_ops, target_main_sl, sens_map,
                       min_sl=64, max_sl=256):
    """Assign stoc_len proportional to sensitivity.
    
    1. Map sensitivity linearly to [min_sl, max_sl]
    2. Scale to hit target FLOP-weighted average
    3. Quantize to hardware-friendly levels
    """
    # Step 1: Linear mapping from sensitivity to stoc_len
    s_min = min(sens_map[k] for k in active_ops)
    s_max = max(sens_map[k] for k in active_ops)
    raw = {}
    for k in active_ops:
        frac = (sens_map[k] - s_min) / (s_max - s_min + 1e-8)
        raw[k] = min_sl + frac * (max_sl - min_sl)
    
    # Step 2: Scale to hit target average
    total_macs = sum(OP_MACS[op] for op, _ in active_ops)
    raw_avg = sum(OP_MACS[op] * raw[(op, bi)] for op, bi in active_ops) / total_macs
    scale = target_main_sl / raw_avg
    scaled = {k: clip(v * scale, min_sl, max_sl) for k, v in raw.items()}
    
    # Step 3: Quantize to hardware levels
    LEVELS = [64, 96, 128, 160, 192, 224, 256]
    quantized = {k: nearest(v, LEVELS) for k, v in scaled.items()}
    return quantized
```

### Patching mechanism

Each (op, block) gets its own `MPConfig([stoc_len], [1.0])` via
`patch_model_with_sl_map()` in `experiments/mp_uniform_vs_fine.py`.
This reuses the existing MP infrastructure -- no kernel changes needed.

For attention QK/AV: per-block `make_sc_attention_forward()` with
single-level `qk_mp_cfg` / `av_mp_cfg`.

For linear ops (proj, mlp_fc1, mlp_fc2): per-block `SCLinear` with
single-level `mp_cfg`.

### Hardware cost

- Per-block stoc_len register: 3 bits per (op, block) for 8 levels
- Total: 5 ops x 24 blocks x 3 bits = 360 bits (trivial LUT)
- No runtime classification overhead (static allocation from calibration)

## Recommendations for Large-Scale Investigation

### Experiments to run at higher N

1. **N=5000+ at target=192** for skip_worst30/40 to resolve the +1-2 pt
   signal (need N >= 5000 for +1.5 pt to be significant at 95%)

2. **N=5000+ at target=160** to precisely measure the cliff protection
   (clearly significant even at N=500, but worth pinning down)

3. **With QwT compensation** at target=192 -- comp amplifies MP effects
   (README shows comp recovers +50 pt from destroyed raw SC). The
   proportional advantage may be larger with comp.

4. **Additional sc_configs**: skip_worst50, skip_worst20, full_attn,
   all_ops to map the Pareto frontier

### Combined block + row-level MP

The current proportional (block-level) and existing row-level MP are
orthogonal. Test the combination:
- Block-level: per-(op,block) base stoc_len from proportional
- Row-level: per-row variation within each op centered on base_sl

Implementation: replace `MPConfig([base_sl], [1.0])` with
`MPConfig([base_sl+32, base_sl, base_sl-32], [0.33, 0.34, 0.33])`.

### Reproduce commands

```bash
# Full sweep for one config (raw SC, ~60 min per config on RTX 4080)
python experiments/mp_uniform_vs_fine.py \
    --sc_config skip_worst40 --sweep --n_eval 500 \
    --batch_size 8 --levels "256,128" \
    --out_dir results/mp_comparison

# Proportional-only sweep
python experiments/mp_uniform_vs_fine.py \
    --sc_config skip_worst40 --strategy proportional --sweep \
    --n_eval 500 --batch_size 8 --levels "256,128" \
    --out_dir results/mp_proportional

# Single comparison at higher N
python experiments/mp_uniform_vs_fine.py \
    --sc_config skip_worst30 --target_main_sl 192 \
    --strategy proportional --n_eval 5000 \
    --batch_size 8 --out_dir results/mp_n5000

# With comp (adds ~5 min calibration per run)
python experiments/mp_uniform_vs_fine.py \
    --sc_config skip_worst40 --target_main_sl 192 \
    --strategy proportional --n_eval 500 --with_comp \
    --out_dir results/mp_comp

# Sweep all configs overnight
python experiments/mp_uniform_vs_fine.py --sweep_all --n_eval 2000
```

## Files

- `experiments/mp_uniform_vs_fine.py` -- experiment driver
- `results/mp_comparison/skip_worst40_sweep.json` -- sw40 2-level sweep
- `results/mp_comparison/skip_worst30_sweep.json` -- sw30 2-level sweep
- `results/mp_proportional/skip_worst40_sweep.json` -- sw40 all strategies
- `results/mp_n2000/skip_worst30_sweep.json` -- sw30 N=2000
- `results/mp_n2000_sw40/skip_worst40_sweep.json` -- sw40 N=2000

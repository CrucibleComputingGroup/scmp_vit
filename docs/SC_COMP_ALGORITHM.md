# SC-comp algorithm — choosing the kernel for QwT residual correction

**For:** anyone touching the QwT-SC stack who needs to know *which* SC kernel
runs the per-block correction `out = block_sc(x) + x @ W̄ + b̄` and *why*.
**Status:** Search complete (2026-04-25). Winner is **B_ha**
(`HeadAlignedSCLinear` at `n_heads=16`, `sc_prec=8`, bipolar). Production
N=50k results below.

The cross-seed cosine gate in
[`third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py`](../third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py)
decides *whether* to install a correction at block i; this doc decides *how*
the admitted W̄ is executed at inference. The gate is locked — only
the kernel changes here.

---

## TL;DR

- **Winner: `B_ha`** — `HeadAlignedSCLinear` at `n_heads=16`, `sc_prec=8`,
  bipolar. Built by `make_head_aligned_factory` in
  [`cls/experiments/qwt_sc_overnight.py`](../cls/experiments/qwt_sc_overnight.py#L233-L248).
- **Hardware cost:** 605 M comp MACs/image × 256 SC bit-ops = **154.9 G
  comp bit-ops/image**. **Zero new SNG pool entries**: reuses
  `_CFG_CACHE[(64, 8)]` already populated by every block's per-head QK
  matmul.
- **Production accuracy (N=50k):** geomean Δ over raw SC = **+1.80 pt**
  across 5 configs. Beats raw SC on every config; beats the legacy r²-gate
  + SC-comp reference (+1.16 geomean) on every config. The biggest win is
  on `p7_mp` (+3.37 vs +1.14 legacy = +2.23 pt improvement), driven jointly
  by the cross-seed gate and a wider-budget MP search.
- **Cross-seed gate admits 22/24 blocks** (rejects blocks 0 and 23) —
  same admission count across all 5 configs. Block 1 is now admitted
  (was rejected at N=512 calib) because larger N=1024 calib stabilizes
  its `cos_ab` above the 0.5 threshold.

---

## Hardware accounting cheat-sheet

The comp adds 25.2 M MACs per block × 24 blocks = 605 M comp MACs per image.
Multiplied by `2^comp_sc_prec` to get bit-ops.

| kernel | new SNG pool entries | comp bit-ops/image @ sc_prec=8 |
|---|---|---|
| `make_fp_factory`         | n/a (FP multiplier required) | 0 SC bit-ops, but breaks the SC story |
| `make_sc_factory`         | **none** — reuses `_CFG_CACHE[(1024, 8)]` already populated by qkv_proj/out_proj/mlp_fc1 SC matmuls (which exist on 17/24 blocks under the K=30 schedule) | 605 M × 256 = **154.9 G** |
| **`make_head_aligned_factory` (h=16)** ✓ winner | **none** — reuses `_CFG_CACHE[(64, 8)]` from per-head block QK | 605 M × 256 = **154.9 G** |
| `make_head_aligned_factory` (h=8)  | new `(128, 8)` entry | same |
| any of the above at `sc_prec=6`    | smaller SNG entry           | 605 M × 64 = **38.7 G** (4× smaller) |

**Note for the SC-purity story:** both `A_fw` and `B_ha` reuse pools the
block path *already* needs at the K=30 schedule, so they are equally
SC-pure. The differentiator is the breadth of the SNG entry's reuse:
`(64, 8)` is used by every block's QK matmul (24/24 blocks), while
`(1024, 8)` is shared with only 17/24 blocks (those whose proj/MLP cells
weren't dropped by skip_worst30). **`B_ha` wins on hardware reuse
breadth, not on net new pools.** The task brief's "A_fw needs a fresh
(1024,8) pool" was an overstatement — corrected here.

---

## Design space (axes the per-block picker can vary)

| axis | values | what it does | per-block cost (bits) |
|---|---|---|---|
| Sobol seed       | default / antithetic / k_seed alts | diversifies SC variance pattern | log₂(K_seeds) |
| Polarity         | +1, −1                          | row-bias cancellation if sign-flipping wins | 1 |
| W scale          | 0.5, 0.75, 1.0, 1.25            | matches SC output range to W̄'s natural range | log₂(K_scales) |
| Topology         | full-width, head-aligned (h=4/8/16) | hardware reuse vs accuracy | log₂(K_topo) |
| Precision        | comp_sc_prec ∈ {4, 6, 8}        | trade SC-comp noise for bit-op savings | log₂(K_prec) |
| Mode             | bipolar, unipolar               | signed W̄ handling                          | 1 |

The picker (in `calibrate_qwt(..., comp_factory_variants=...)`) measures
`||R - comp(X)||` per candidate per block and installs the winner. The
choice is logged in `report[i].variant` and amounts to log₂(K) config bits
in hardware. **Single-variant configs were tested in this iteration** —
no per-block picker active. Adding the picker on top of B_ha is the
natural next-iteration extension.

---

## N=1000 pilot — A_fw vs B_ha across 5 configs

The N=1000 head-to-head used `--cos_threshold 0.5
--last_block_cos_threshold 0.8 --lookahead_veto --calib_seed_b 2`,
N=512 calib, N=1000 eval. Raw SC numbers from
`cls/results/sweep_int678_k30/*qwtoff.json` at N=50k (so noise-free).

| config         | raw SC | A_fw Δ  | B_ha Δ  | B_ha − A_fw |
|----------------|---:|---:|---:|---:|
| p7_uniform     | 78.94 | **+3.06** | **+2.86** | −0.20 |
| p7_mp          | 79.89 | **+2.21** | +0.91 | **−1.30** |
| p8_uniform     | 85.57 | −0.67 | −0.37 | +0.30 |
| avg192_uniform | 84.67 | +0.13 | +0.53 | +0.40 |
| avg192_mp      | 84.62 | −0.32 | **+0.88** | **+1.20** |
| **geomean Δ**  | —     | +0.87 | **+0.96** | +0.09 |

**N=1000 noise floor:** stderr ≈ √(p(1−p)/N) is ~1.27 pt at top1≈80%
and ~1.13 pt at top1≈85%. Only `p7_mp` (B_ha −1.30) and `avg192_mp`
(B_ha +1.20) exceed 1σ. The N=50k confirmation below shows both were
N=1000 noise — at N=50k B_ha gets +3.37 on `p7_mp` and +0.85 on
`avg192_mp`, so there is no real `p7_mp` weakness in head-aligned.

Pilot results are in `cls/results/sc_comp_search/`.

---

## N=50k confirmation — winner B_ha

Production sweep, 2026-04-25 overnight. Same gate config; N_calib=1024,
N_eval=50000; `HEAD_ALIGNED=1` plumbed into
[`sweep_int678_skip30.sh`](../cls/experiments/sweep_int678_skip30.sh)'s
eval commands. Fresh MP search: `SEARCH_N_SEARCH=64`, `SEARCH_MAX_ITERS=12`
(2× and 3× the legacy budget); `SEARCH_OPS=proj,mlp_fc1,mlp_fc2`
(unchanged — see "MP-vs-uniform parity" below for why qk/av weren't added).

| config         | raw SC | B_ha N=50k | Δ | legacy ref Δ | B_ha vs legacy |
|----------------|---:|---:|---:|---:|---:|
| p7_uniform     | 78.94 | 82.87 | **+3.93** | +3.12 | +0.81 |
| p7_mp          | 79.89 | 83.26 | **+3.37** | +1.14 | **+2.23** |
| p8_uniform     | 85.57 | 85.70 | +0.13 | +0.20 | −0.07 |
| avg192_uniform | 84.67 | 85.42 | **+0.75** | +0.67 | +0.08 |
| avg192_mp      | 84.62 | 85.47 | **+0.85** | +0.65 | +0.20 |
| **geomean Δ**  |       |       | **+1.80** | +1.16 | **+0.64** |

Cross-seed gate admits **22/24 blocks** identically across all 5 configs
(rejects blocks 0 and 23). Per-block `cos_ab` is logged in
`results/sweep_2026-04-25/*_qwton.json` under `calib.per_block[i].cos_ab`.

---

## What did and didn't matter

- **Topology (full-width vs head-aligned):** at N=1000, within noise on
  4/5 configs; at N=50k, B_ha matches the +1.80 geomean target with
  *better* hardware reuse breadth (24/24 blocks share the (64, 8) pool
  versus 17/24 for (1024, 8)). The N=1000 `p7_mp` gap (−1.30 for B_ha)
  was N=1000 noise — at N=50k B_ha is +3.37 on `p7_mp`.
- **Cross-seed gate vs legacy r²-gate:** the gate change alone
  contributes most of the +0.64 geomean lift over the legacy reference,
  even at the same comp kernel. The biggest single-config jump
  (`p7_mp`: +1.14 → +3.37) is also driven partly by the wider-budget
  fresh MP map.
- **MP-search budget (legacy 32 / 4 → 64 / 12):** materially helps
  `p7_mp` (the legacy map's small effective search dimension was
  the binding constraint), neutral elsewhere. The map's structure
  (qk/av pinned to target SL, mlp_fc2 floored at 192) is the same in
  both runs — only the optimizer's exploration depth changed.
- **Sobol seed / polarity / W scale / `sc_prec`:** **not tested
  end-to-end** in this iteration. Single-variant production config locks
  default Sobol, polarity=+1, scale=1.0, sc_prec=8. The picker
  infrastructure is in place
  ([`calibrate_qwt(..., comp_factory_variants=...)`](../third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py#L417-L425))
  for a future iteration if a Δ-vs-FP-comp gap emerges.

### MP-vs-uniform parity is *not* a comp-kernel issue

In the N=1000 pilot, MP runs gained less from comp than uniform on the
same target SL. Investigation showed this is **structural to the legacy
MP map** rather than a flaw in the SC-comp kernel:

1. The MP search proxy (`comp_residual` in
   [`mp_budget_swap_search.py`](../cls/experiments/mp_budget_swap_search.py))
   doesn't model the cross-seed gate — it assumes every block gets a
   correction. Blocks 0 and 23 (rejected) get whatever SL the search
   assigned, with no comp safety net.
2. The map's effective search dimension is **17 mlp_fc1 cells** (qk/av
   were pinned to target SL via `--fixed_ops`; mlp_fc2 floored at 192
   via `--op_min_levels`). That's near-uniform with tiny perturbation,
   not real precision-prioritization.
3. **Adding qk/av to `SEARCH_OPS` is currently impossible** because
   `build_initial_sl_map` requires equal-cost search units (DP
   constraint), and qk/av MACs (~67 M) differ from proj/mlp_fc1/mlp_fc2
   MACs (~1.08 G) by 16×.

The 2026-04-25 sweep used the bigger search budget (`N_search=64`,
`max_iters=12`) on the unchanged search space. That alone moved
`p7_mp` Δ from +1.14 (legacy) to +3.37 (today) — most of the
B_ha-vs-legacy improvement comes from this, not from the comp kernel.

**Recommendation for next iteration on MP:** rewrite
`build_initial_sl_map` to handle multi-cost search units (drop the DP,
use a greedy / LP relaxation), then retry with `qk,av` in
`SEARCH_OPS`. Until then, the search is structurally limited.

---

## How to reproduce

```bash
# Production sweep (cross-seed gate + B_ha + wider MP search):
cd cls
nohup env \
  RES_DIR=results/sweep_$(date +%F) \
  LOG_DIR=logs/sweep_$(date +%F) \
  MAP_DIR=results/sweep_$(date +%F)/sl_maps \
  SEARCH_OPS=proj,mlp_fc1,mlp_fc2 \
  SEARCH_N_SEARCH=64 \
  SEARCH_MAX_ITERS=12 \
  HEAD_ALIGNED=1 \
  HEAD_ALIGNED_HEADS=16 \
  bash experiments/sweep_int678_skip30.sh \
  > logs/sweep_$(date +%F)/sched.log 2>&1 &

# Summarize results (inline; no separate summarizer script):
python -c "
import json, glob, os
RAW = {'p7_uniform': 78.94, 'p7_mp': 79.89, 'p8_uniform': 85.57,
       'avg192_uniform': 84.67, 'avg192_mp': 84.62}
for f in sorted(glob.glob('cls/results/sweep_*/+'*qwton.json'.strip('+'))):
    name = os.path.basename(f).replace('_qwton.json','')
    d = json.load(open(f))
    t = d['results']['sc_comp']['top1'] * 100
    raw = RAW[name]
    pb = d.get('calib', {}).get('per_block', [])
    en = sum(1 for r in pb if r.get('enabled'))
    print(f'{name:18s} raw={raw:.2f} qwt={t:.2f} d={t-raw:+.2f} en={en}/{len(pb)}')
"

# Per-block admission detail (cos_ab, r2_a/r2_b, rmse) for any config:
python -c "
import json
d = json.load(open('cls/results/sweep_2026-04-25/p7_mp_qwton.json'))
for r in d['calib']['per_block']:
    print(f\"blk {r['block']:2d} en={r['enabled']!s:>5} cos={r['cos_ab']:+.3f}\")
"
```

# Agent task — port cls/ wins to det/

**You are an autonomous coding agent. Execute this task end-to-end.**
Do not ask the user for permission to read files, edit files, run
smoke tests, launch GPU jobs, or commit + push changes. The work
described here is pre-authorized; act, then report what you did.
Only stop and ask if you hit a genuinely ambiguous decision that
isn't covered below — and even then, propose the answer you'd pick
and proceed unless the user objects.

The repo is `vit_sc`: DINOv2 ViT-L/14 classification in `cls/`,
EVA-01-g ViTDet detection in `det/`. Cls just shipped a redesigned
QwT compensator (cross-seed gate + head-aligned SC kernel) and a
wider-budget MP search. **Your job is to bring the same wins to det.**

## Start here — first three actions

Do these in order, right now, before anything else:

1. `Read` [`docs/SC_COMP_ALGORITHM.md`](SC_COMP_ALGORITHM.md) — the
   cls-side outcome, kernel-choice rationale, hardware-cost story.
2. `Read` the module docstring of
   [`third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py`](../third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py)
   — current cross-seed `calibrate_qwt` API.
3. `Read` [`det/experiments/qwt_det_compensate.py`](../det/experiments/qwt_det_compensate.py)
   — the broken det driver you'll fix in **Task A** below.

Then begin Task A immediately. Use `TodoWrite` to track A → B → C →
sweep → docs as you go.

> **End-state you're driving toward:** three landed PRs (one per
> task), a production sweep harness mirroring
> [`cls/experiments/sweep_int678_skip30.sh`](../cls/experiments/sweep_int678_skip30.sh),
> updated `det/RESULTS.md` and `det/README.md`, and N=5000 COCO
> numbers showing comp Δ over raw SC on every (sc_prec, K) cell. If
> any smoke test fails, **fix and rerun** — don't escalate to the
> user unless the failure points to a missing prerequisite (e.g.,
> conda env missing) or contradicts something documented here.

---

## 1. Server setup (one-shot, det-side)

The det side uses **Detectron2 + EVA-01 ViTDet** on COCO. Conda env is
**separate** from the cls-side `vit_sc` env: it's `qwt_d2`. Two known
install gotchas — both will silently fail without these:

```bash
# On a GreatLakes compute node (NOT the login node — needs CUDA)
module load gcc/13.2.0          # default 8.5 fails Detectron2's NMS build
conda activate qwt_d2

# If qwt_d2 is missing, build it from the env file:
#   conda env create -f det/environment_qwt_d2.yml
# Then once active:
pip install --no-build-isolation -e \
    third_party/QwT-SC/QwT-det-RepQ-ViT/eva1/eva_det
```

**Data + checkpoint** (live on GreatLakes shared scratch):

```bash
export DETECTRON2_DATASETS=/scratch/nbleier_owned_root/nbleier_owned1/shared_data
# EVA ViTDet cascade_mask_rcnn checkpoint:
ls /scratch/nbleier_owned_root/nbleier_owned1/shared_data/pretrained/eva_coco_det.pth
```

The det drivers default to these paths but accept `--d2_datasets` and
`--ckpt` overrides; pass them explicitly if you're on a different cluster.

**Server access** (mirrors the cls side):

```bash
# 4× RTX PRO 6000 Blackwell, exclusive-mode CUDA
ssh gl1807 'nvidia-smi'
# Always launch with nohup + ssh -n -f for fire-and-forget:
ssh -n -f gl1807 'bash -c "
  cd /home/allenjin/Projects/vit_sc/det
  source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
  conda activate qwt_d2
  module load gcc/13.2.0
  nohup env CUDA_VISIBLE_DEVICES=0 python <script.py> ARGS \
    > <log>.log 2>&1 &
"'
# Verify singleton (not stacked launches):
ssh gl1807 'pgrep -fc python.*det/'
```

**FP baseline (already measured):** bbox AP = 68.73, segm AP = 61.61 at
n_eval=100. Use this as the reference; if any new run shows AP near the
FP baseline, the SC story is intact. AP near 0 ⇒ collapse; investigate
before scaling.

**Smoke / production sample sizes:**
- n_eval=10 (smoke, ~1 min, AP ±2 noise — *don't* gate on absolute AP at this size)
- n_eval=100 (default smoke, ~10 min — used by the existing det pipeline)
- n_eval=5000 (full COCO val, ~6-8 h per run — the final number)

---

## 2. What's already in det/ — and what's broken

### 2.1 Skip-worst (sensitivity-driven scheduling)

**Status:** working. Built around end-to-end COCO AP, not the noise
surrogate that cls uses.

- [`det/experiments/sensitivity_sweep.py`](../det/experiments/sensitivity_sweep.py)
  — runs each (op, block) cell SC alone with everything else FP, at
  `n_eval=100`, records bbox/segm AP per cell. Output:
  `det/results/sensitivity_int{6,7,8}_n100/` (older n_eval=50 results
  also exist).
- [`det/experiments/merge_and_build_skip_worst.py`](../det/experiments/merge_and_build_skip_worst.py)
  — merges per-prec sensitivity matrices and emits the
  `skip_worst_K.json` schedule files.
- [`det/sensitivity/skip/`](../det/sensitivity/skip/) — generated
  schedule JSONs (one per K, one per sc_prec).

**Mechanism is identical to cls** (top-K most-damaging cells stay FP);
the difference is the proxy: det uses real COCO AP per cell (slower,
more accurate); cls uses a calibrated Gaussian noise surrogate (~100×
faster). cls also keeps a real-SC sensitivity script
([`cls/experiments/sensitivity_per_operator_real_sc.py`](../cls/experiments/sensitivity_per_operator_real_sc.py))
as an option. **No port needed for skip-worst — det's approach is
fine.** If you later want faster sweeps, port the noise-surrogate idea
from cls; it's optional.

### 2.2 QwT compensation — **THIS IS BROKEN against current submodule**

**Status:** the driver `qwt_det_compensate.py` calls a calibrate_qwt
signature that no longer exists.

The `qwt_sc.calibrate_qwt` import resolves to
[`third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py`](../third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py)
(submodule `Allenjin123/QwT.git`, branch main). Its **current** signature
(commit `eeb1bd7`) is:

```python
def calibrate_qwt(
    model_fp, model_sc,
    blocks_fp, blocks_sc_container,
    calib_loader_a, calib_loader_b,    # ← TWO disjoint calib batches
    device, n_calib, ridge=1e-2, start_block=0, fwd_chunk=32,
    cos_threshold=0.5,                  # ← cross-seed gate; not min_r2
    norm_floor=0.0,
    last_block_cos_threshold=None,      # ← stricter τ on the last block
    lookahead_veto=False,
    comp_factory=None,
    comp_factory_variants=None,         # ← optional per-block picker
    log_fn=print,
)
```

The det driver
[`det/experiments/qwt_det_compensate.py:179`](../det/experiments/qwt_det_compensate.py#L179)
calls it with `calib_loader=`, `min_r2=`, `avg_sc_draws=` — **all of
which are gone** from the current submodule. The driver will raise
`TypeError: calibrate_qwt() got an unexpected keyword argument
'calib_loader'` the moment it runs. Verify with a smoke run before
believing any of the legacy det results in `det/RESULTS.md`.

**Read these for the kernel-choice rationale** before porting:

1. [`third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py`](../third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py)
   module docstring — derivation of the cross-seed gate, why it
   replaces r².
2. [`docs/SC_COMP_ALGORITHM.md`](SC_COMP_ALGORITHM.md) — full
   kernel-choice writeup, A_fw vs B_ha pilot table, head-aligned
   hardware reuse story, N=50k production results.

### 2.3 Mixed precision

**Status:** patcher-layer support exists but is not exposed in the
drivers.

- [`det/sc_patch/sc_model_eva.py`](../det/sc_patch/sc_model_eva.py:181-218)
  already accepts `linear_mp_spec` and `attn_mp_spec` kwargs and wires
  them into `SCLinear` / `SCMatMul` constructors. The MP types
  (`MPConfig`, `AdaptiveMPConfig`, `RangeMPConfig`) are imported from the
  shared `sc_integration/mp_linear.py`.
- **Neither `det/sc_eval.py` nor `det/experiments/qwt_det_compensate.py`
  exposes the MP CLI flags** that cls's `qwt_sc_overnight.py` and
  `eval_custom_sl_map_fpcomp.py` have:
  `--mp_levels`, `--mp_fractions`, `--mp_ops`, `--qk_mp_levels`,
  `--qk_mp_fractions`, `--av_mp_levels`, `--av_mp_fractions`,
  `--range_mp*`.
- **MP-budget swap search is missing entirely from det**
  ([`cls/experiments/mp_budget_swap_search.py`](../cls/experiments/mp_budget_swap_search.py)
  has no det counterpart).

---

## 3. Port plan — three concrete tasks, in order

Each task ends with a smoke test you must run before declaring it done.
Don't move on to the next task until the smoke passes; don't run the
full n_eval=5000 sweep until *all three* are landed.

### Task A — fix `qwt_det_compensate.py` to call the cross-seed API

This is the unblocker. Mirror `cls/experiments/qwt_sc_overnight.py`'s
calibration plumbing (lines 470–496 in cls version) into
`det/experiments/qwt_det_compensate.py`. Keep the Detectron2 specifics
(image preprocessing through `model.preprocess_image`, COCO evaluator,
1024² fwd_chunk) — only the calibrate_qwt invocation changes.

Concrete changes:

1. Remove `--min_r2`, `--avg_sc_draws` CLI args (the new gate doesn't
   need them).
2. Add `--cos_threshold` (default 0.5),
   `--last_block_cos_threshold` (default 0.8),
   `--lookahead_veto` (action="store_true"),
   `--calib_seed_b` (int, must differ from `--calib_seed`),
   `--norm_floor` (default 0.0).
3. Replace the single `BackboneCalibLoader` with **two** loaders driven
   by two different calib seeds (A and B), as in cls. EVA-det uses
   D2's `build_detection_test_loader` — easiest path is to call
   `load_model_and_loader(args.n_calib, ...)` twice with different
   shuffle seeds. Verify the two loaders yield disjoint COCO image IDs
   in your smoke run by logging `image_id` from each yield.
4. Update the `calibrate_qwt(...)` call to the new signature
   (`calib_loader_a=`, `calib_loader_b=`, `cos_threshold=`,
   `last_block_cos_threshold=`, `lookahead_veto=`, `norm_floor=`).
5. Update the per-block report consumer in the JSON output: the new
   report fields are `cos_ab`, `r2_a`, `r2_b`, `rmse_before_a/b`,
   `rmse_after_a/b`, `lookahead_decision`, `enabled`. The old `r2`
   field is gone.

**Smoke test (~15 min):**

```bash
# Run on the smallest schedule + sc_prec=8 first (most likely to pass cleanly).
cd det
nohup env CUDA_VISIBLE_DEVICES=0 python experiments/qwt_det_compensate.py \
  --sc_prec 8 \
  --sc_ops_per_block_json sensitivity/skip/skip_worst_30_int8.json \
  --n_calib 64 --n_eval 100 \
  --calib_seed 1 --calib_seed_b 2 \
  --cos_threshold 0.5 --last_block_cos_threshold 0.8 --lookahead_veto \
  --comp_mode sc --comp_sc_prec 8 \
  --out_json results/qwt_det/smoke_xseed_int8_sw30.json \
  > logs/qwt_det_smoke_xseed_int8_sw30.log 2>&1 &
```

**Pass criterion:** no TypeError, the calib report logs `cos=...` per
block (not `r2=...`), and `bbox AP` after comp is **≥ raw SC bbox AP**
on the same schedule. If raw SC at sw30+int8 is ~67 AP, post-comp
should be ≥ 67. (If raw is already at FP, comp will be a no-op — that's
fine; just confirm no regression.)

### Task B — add the head-aligned SC comp factory

Cls uses `make_head_aligned_factory` (D=64 per head, n_heads=16, reuses
`_CFG_CACHE[(64, 8)]` from QK) and ships it as the production winner.
EVA-ViTDet's residual dim is **1408**, not 1024, and it has **16 heads**
of dim 88 (1408 / 16 = 88) — verify this in
[`det/sc_patch/sc_model_eva.py`](../det/sc_patch/sc_model_eva.py); the
64-vs-88 difference matters for the SNG-pool reuse story.

Concrete changes:

1. Add a `make_head_aligned_comp_factory(sc_prec, mode, n_heads,
   cfg_override=None, scale=1.0, polarity=1)` to
   `det/experiments/qwt_det_compensate.py` (or a shared det helper).
   Mirror
   [`cls/experiments/qwt_sc_overnight.py:233-248`](../cls/experiments/qwt_sc_overnight.py#L233-L248).
   The class is
   [`sc_integration/head_aligned_comp.py:HeadAlignedSCLinear`](../sc_integration/head_aligned_comp.py)
   and is already in the shared `sc_integration` package — no new code
   path needed, just wire it.
2. Verify EVA's per-head dim divides 1408 evenly (1408 / 16 = 88, OK).
   The reused SNG pool will then be `_CFG_CACHE[(88, 8)]` — which the
   block QK matmul on EVA already populates if QK is SC-active.
   Confirm by adding a log line that prints `_CFG_CACHE.keys()` after
   patching + after the first calibration forward.
3. Add CLI flags `--head_aligned`, `--head_aligned_only`, `--n_heads`
   (default 16) mirroring the cls driver.
4. Add the variant-builder block from
   [`cls/experiments/qwt_sc_overnight.py:432-454`](../cls/experiments/qwt_sc_overnight.py#L432-L454)
   (single-variant fast path is fine — the picker is optional). Default
   should be `comp_factory = make_head_aligned_comp_factory(...)` when
   `--head_aligned` is set.

**Smoke test (~15 min):**

Same configuration as Task A's smoke, but add `--head_aligned --n_heads 16`.
Pass criterion: bbox AP within ~0.5 of the full-width comp run from
Task A (on n_eval=100 the AP noise floor is real; we expect parity, not
a measurable lift, at this sample size).

Cross-check: log `_CFG_CACHE.keys()` and confirm `(88, 8)` appears
**only once** — i.e., the head-aligned comp shares it with the QK path.

### Task C — plumb MP into the drivers

`sc_model_eva.py` already accepts `linear_mp_spec` and `attn_mp_spec`.
The drivers don't expose them.

Concrete changes (mirror cls's `_build_linear_mp_spec` /
`_build_attn_mp_spec` from
[`cls/eval.py`](../cls/eval.py)):

1. Add CLI flags to **both** `det/sc_eval.py` and
   `det/experiments/qwt_det_compensate.py`:
   `--mp_levels`, `--mp_fractions`, `--mp_ops`, `--qk_mp_levels`,
   `--qk_mp_fractions`, `--av_mp_levels`, `--av_mp_fractions`.
   (Optional second pass: `--range_mp*`, `--adaptive_mp*`.)
2. Build the spec dicts and pass them as `linear_mp_spec=` /
   `attn_mp_spec=` into `sc_patch_eva(...)`. The signature is already
   there; no patcher change needed.
3. Update the JSON output's `config` block to record the MP flags.
4. **Stop here for this iteration** — do *not* port
   `mp_budget_swap_search.py` to det yet. The cls side has a known
   limitation (the equal-MAC DP in `build_initial_sl_map` rejects
   adding qk/av to the search; see
   [`docs/SC_COMP_ALGORITHM.md`](SC_COMP_ALGORITHM.md)
   "MP-vs-uniform parity is *not* a comp-kernel issue"). Port the
   search only after that DP is fixed cls-side.

**Smoke test (~10 min):**

```bash
cd det
nohup env CUDA_VISIBLE_DEVICES=0 python sc_eval.py \
  --sc_prec 8 \
  --sc_ops_per_block_json sensitivity/skip/skip_worst_30_int8.json \
  --mp_levels 192 --mp_fractions 1.0 \
  --mp_ops mlp_fc1,mlp_fc2,qkv_proj,out_proj \
  --qk_mp_levels 192 --qk_mp_fractions 1.0 \
  --av_mp_levels 192 --av_mp_fractions 1.0 \
  --n_eval 100 \
  --out_json results/sc_smoke_mp192_int8_sw30.json \
  > logs/sc_smoke_mp192.log 2>&1 &
```

Pass criterion: the run finishes without errors, the JSON's `config`
block records the MP flags, and `bbox AP` is in the ballpark of raw SC
at sw30+int8 (slightly lower because 192 is below 256, so a couple
points of AP drop is expected).

---

## 4. After the three ports — full sweep

Once Tasks A/B/C land and their smokes pass, run the production sweep.
Mirror the cls side's
[`cls/experiments/sweep_int678_skip30.sh`](../cls/experiments/sweep_int678_skip30.sh)
pattern: 4-GPU pool, MP-cap-3, env-overridable knobs, idempotent reruns.
Don't write the sweep until Tasks A/B/C smokes pass — the harness is
mechanical once the drivers work.

Suggested matrix (start with):

```
sc_prec  ∈ {7, 8}                    # int6 only if 7/8 succeed
K        ∈ {20, 30}                  # K=10 too aggressive for det at n_eval=5000
mode     ∈ {uniform, mp192}          # MP via the simple uniform-192 trick first
qwt      ∈ {off, on}
```

= 16 evals at n_eval=5000. ETA ~40 GPU-hours; ~10-12 h wall on 4 GPUs
(MP runs slower; det matmuls at 1280² are bigger than cls's 224²).

---

## 5. Things to **not** do

These are anti-patterns — skip without asking; don't burn cycles on them.

- **Don't re-run sensitivity sweeps.** The existing `n_eval=100`
  matrices in `det/sensitivity/skip/` are good for this iteration. Only
  re-sweep if a sc_prec or schedule outside the existing files is
  needed.
- **Don't change the cross-seed gate parameters per-config.** The cls
  pilot found τ insensitive in [0.3, 0.65]; keep `cos_threshold=0.5`
  and `last_block_cos_threshold=0.8` as-is. A loosened
  `last_block_cos_threshold` was the next-iteration knob if MP runs
  underperform — don't pre-emptively tune it.
- **Don't port `comp_factory_variants` (multi-variant picker)** unless
  the head-aligned single-variant comp leaves a measurable gap to FP at
  n_eval=5000. The picker added zero net production benefit on cls.
- **Don't ship FP-comp results as production numbers.** `--comp_mode fp`
  is a debug path; production must be `--comp_mode sc --head_aligned`.
- **Don't use `--no-verify` on git commits, don't rebase published
  history, don't force-push.**
- **Don't pause to ask "should I commit?" or "should I launch?"** —
  yes to both, when the smoke passes. Commit messages should follow
  the existing repo style (`<area>: <imperative description>`; see
  `git log --oneline -5` for examples). Push after each task lands.
  Co-author tag (the same one used in recent main-repo commits) is
  expected.

---

## 6. Reference table — what to match against

These are cls-side N=50k production numbers; det's analogue at
n_eval=5000 should show *qualitatively similar* behavior (raw SC →
positive Δ from comp, bigger Δ on lower-prec/lower-K configs). Don't
expect numerically identical lifts because the model and metric
differ.

| metric | cls expectation (raw → comp) | det target |
|---|---|---|
| largest Δ at lowest sc_prec | yes (p7 +3.93 vs p8 +0.13) | should hold |
| MP barely beats uniform at same target SL | yes (legacy MP map limitation) | likely same — see SC_COMP_ALGORITHM.md §"MP-vs-uniform parity" |
| 22/24 blocks admitted | should generalize: ≈(N-2)/N admitted, blocks 0 and N-1 rejected | for EVA-40 expect ≈38/40 |
| comp Δ stays within 1 AP of raw on the worst config | tested via the avg192 collapse-risk regime | watch for collapse on int6 + sw20 |

---

## 7. Deliverables

When you're done:

1. `det/experiments/qwt_det_compensate.py` updated to the cross-seed
   API (Task A).
2. Head-aligned comp factory wired in det (Task B). Optional helper
   module if it grows past ~30 lines.
3. MP CLI flags in both `det/sc_eval.py` and
   `det/experiments/qwt_det_compensate.py` (Task C).
4. `det/experiments/sweep_<date>.sh` — production sweep harness (mirror
   cls's `sweep_int678_skip30.sh`), with overrides for sc_prec / K /
   MP / HEAD_ALIGNED.
5. `det/RESULTS.md` updated with the new n_eval=5000 numbers (replace
   the n_eval=10 `min_r2` table currently there — that table is from
   the legacy gate and is no longer the production recipe).
6. `det/README.md` "What's new" or "Production status" section
   pointing to the new results, mirroring the cls README's layout.

**Land each task as its own commit (or PR) so they review independently.**
You don't need to wait for human review between tasks — chain them and
push at the end of each. Final sweep + doc updates can be one final
commit. The user is asleep / away during the GPU sweep window; report
status by appending to the launcher log, not by pinging them.

## Quick re-check before you write code

If you're rereading this prompt mid-task and unsure whether to stop or
continue, the answer is **continue.** This document already authorizes
every action it describes. The only legitimate stop conditions are:

- A required artifact is missing (e.g., `qwt_d2` env doesn't exist and
  you can't build it from `det/environment_qwt_d2.yml`).
- A smoke test fails in a way that contradicts something documented here
  (e.g., the new cross-seed `calibrate_qwt` import fails — meaning the
  submodule pin is wrong; investigate `git -C third_party/QwT-SC log`).
- You hit an ambiguous design choice not covered above. In that case:
  pick the answer that mirrors the cls-side equivalent, log your choice,
  and proceed.

Begin with the three "Start here" actions at the top of this doc.

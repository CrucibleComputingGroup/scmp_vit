"""QwT-style closed-form linear compensation for SC accuracy loss.

Thin driver that couples the ``qwt_sc`` library (living in the QwT-SC fork
submodule at ``third_party/QwT-SC/QwT-vit-sc``) to the DINOv2 ViT-L/14 + SC
kernels in this repo. All of the reusable logic — CompensationBlock,
closed_form_ridge, sequential calibration — lives in the submodule.

Usage examples::

    # The worst-hit baseline (full attention SC):
    python experiments/qwt_sc_compensation.py \\
        --sc_config full_attn --n_calib 256 --n_eval 500 \\
        --out_json results/qwt_sc_full_attn_n500.json

    # State-of-the-art: fine-grained sensitivity scheduling + QwT comp:
    python experiments/qwt_sc_compensation.py \\
        --sc_config skip_worst50 --n_calib 256 --n_eval 500 \\
        --out_json results/qwt_sc_skip_worst50_n500.json
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]   # vit_sc/
CLS = HERE / "cls"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CLS))
# Library lives in the QwT-SC fork, vendored as a git submodule.
QWT_SC_LIB = HERE / "third_party" / "QwT-SC" / "QwT-vit-sc"
if not (QWT_SC_LIB / "qwt_sc").exists():
    raise RuntimeError(
        f"qwt_sc library not found at {QWT_SC_LIB}. Did you forget to run "
        f"`git submodule update --init --recursive`?"
    )
sys.path.insert(0, str(QWT_SC_LIB))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from eval import build_transform, load_model, seed_all
from imagenet_parquet import ImageNetParquetVal
from sc_attention_patch import patch_model, set_noise_model, SC_OP_NAMES, SCLinear

from qwt_sc import calibrate_qwt  # noqa: E402


def make_sc_comp_factory(sc_prec: int, mode: str, cfg_override=None):
    """Return a closure (W, b) -> SCLinear init'd from W, b.

    When ``cfg_override`` is None, SCLinear uses the cached default Sobol
    config (same as the block's D=D_in SC ops). When provided, the comp's SC
    matmul uses ``cfg_override`` — a different Sobol scrambling — to give the
    comp a *different* SC noise realization than the block. Per-block variant
    selection at calibration picks whichever realization correlates *least*
    with the block's residual.
    """
    def factory(W: torch.Tensor, b: torch.Tensor) -> nn.Module:
        D_in, D_out = W.shape
        layer = nn.Linear(D_in, D_out, bias=True)
        with torch.no_grad():
            layer.weight.copy_(W.t().contiguous())
            layer.bias.copy_(b)
        return SCLinear(layer, sc_prec=sc_prec, mode=mode,
                         cfg_override=cfg_override)
    return factory

SENSITIVITY_JSON = HERE / "results" / "sensitivity_all_ops.json"


def build_skip_worst_k_schedule(k: int, n_blocks: int = 24):
    """Re-create e2e_fine_grained.py's sc_skip_worst_k schedule.

    SC every (op, block) in {mlp_fc1, mlp_fc2, qk, av, proj} x n_blocks except
    the top-K worst (op, block) pairs by calibrated-noise surrogate L2.
    """
    with open(SENSITIVITY_JSON) as f:
        data = json.load(f)
    rows = sorted(data["grid"], key=lambda r: r["l2"], reverse=True)
    drop = {(r["op"], int(r["block"])) for r in rows[:k]}
    spec = {name: [1] * n_blocks for name in SC_OP_NAMES}
    for (op, bi) in drop:
        spec[op][bi] = 0
    return spec


@torch.no_grad()
def evaluate(model, loader, device, log_every=20):
    model.eval()
    n = top1 = top5 = 0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        _, p5 = logits.topk(5, 1)
        correct = p5.eq(y.unsqueeze(1))
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(1).sum().item()
        n += y.numel()
        if (i + 1) % log_every == 0:
            dt = time.time() - t0
            print(f"  [{i+1}] top1={top1/n:.4f} top5={top5/n:.4f} "
                  f"({n} imgs, {dt:.1f}s, {n/dt:.1f} img/s)", flush=True)
    dt = time.time() - t0
    return {"n": n, "top1": top1 / n, "top5": top5 / n,
            "elapsed_s": dt, "img_per_s": n / dt if dt > 0 else 0.0}


SC_PRESETS = {
    "full_attn": dict(sc_qk=True, sc_av=True, sc_qkv_proj=True, sc_out_proj=True,
                      sc_mlp=False),
    "qk_only":   dict(sc_qk=True, sc_av=False, sc_qkv_proj=False, sc_out_proj=False,
                      sc_mlp=False),
    "qk_av":     dict(sc_qk=True, sc_av=True, sc_qkv_proj=False, sc_out_proj=False,
                      sc_mlp=False),
    "full_everything": dict(sc_qk=True, sc_av=True, sc_qkv_proj=True,
                            sc_out_proj=True, sc_mlp=True),
    "mlp_only": dict(sc_qk=False, sc_av=False, sc_qkv_proj=False,
                     sc_out_proj=False, sc_mlp=True),
    "av_only":  dict(sc_qk=False, sc_av=True, sc_qkv_proj=False,
                     sc_out_proj=False, sc_mlp=False),
}

# Fine-grained per-(op, block) schedules from the sensitivity matrix. Values
# are K = number of worst (op, block) pairs left in FP. Match the README.
FINE_GRAINED_KS = {
    "skip_worst50": 50,   # 74.8% raw baseline (best known)
    "skip_worst40": 40,   # 74.2%
    "skip_worst30": 30,   # 68.8% (knee)
    "skip_worst20": 20,   # 43.8%
    "all_ops":       0,   # 0% (the "everything" config)
}


def build_dataset(data_root, seed, n_images):
    ds = ImageNetParquetVal(data_root, transform=build_transform(224))
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if n_images and n_images < len(ds):
        ds = Subset(ds, idx[:n_images])
    return ds


def get_blocks(model):
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        return model.backbone.blocks
    return model.blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data")
    ap.add_argument("--sc_prec", type=int, default=8)
    ap.add_argument("--sc_config",
                    choices=list(SC_PRESETS) + list(FINE_GRAINED_KS),
                    default="full_attn",
                    help=("preset name. SC_PRESETS={full_attn,qk_only,qk_av} or "
                          "FINE_GRAINED_KS={skip_worst50,40,30,20,all_ops}."))
    ap.add_argument("--n_calib", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--calib_seed", type=int, default=1)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--r2_threshold", type=float, default=0.0,
                    help="Skip QwT compensation in a block if its fit r2 is "
                         "<= this threshold. Default 0.0 = original behavior.")
    ap.add_argument("--start_block", type=int, default=0)
    ap.add_argument("--avg_sc_draws", type=int, default=1)
    ap.add_argument("--fwd_chunk", type=int, default=32)
    ap.add_argument("--skip_baseline", action="store_true",
                    help="skip the no-comp SC baseline eval (saves time)")
    ap.add_argument("--comp_mode", choices=["fp", "sc"], default="fp",
                    help="fp = original QwT-SC linear comp; sc = SC matmul comp "
                         "reusing the block's _CFG_CACHE Sobol pool.")
    ap.add_argument("--comp_sc_prec", type=int, default=8)
    ap.add_argument("--comp_sc_mode", choices=["bipolar", "unipolar"],
                    default="bipolar")
    ap.add_argument("--comp_refit_iters", type=int, default=0,
                    help="if >0 and comp_mode=sc, do K rounds of "
                         "noise-aware refit: subtract the comp's measured SC "
                         "noise δ from the LS target before refitting W.")
    ap.add_argument("--comp_variant_select", action="store_true",
                    help="per-block, try {default, antithetic-K} Sobol "
                         "configs for the comp and pick the lower-residual "
                         "one. 1 config-bit per block in hardware. NOTE: "
                         "value-complement antithetic gives only ~−0.15 "
                         "noise correlation in our kernel, so empirical gain "
                         "is small and not strictly Pareto.")
    ap.add_argument("--out_json", default="results/qwt_sc_pilot.json")
    # --- static uniform/quantile MP on linear ops (for head-to-head) ---
    ap.add_argument("--mp_levels", default="",
                    help="CSV stoc_len levels for static MPConfig. Single "
                         "value ⇒ uniform at that sl (e.g. '128').")
    ap.add_argument("--mp_fractions", default="",
                    help="CSV fractions for levels. Empty ⇒ equal split.")
    ap.add_argument("--mp_ops", default="qkv_proj,out_proj",
                    help="CSV op names the static MP applies to.")
    # --- per-token-row adaptive MP on linear ops (this session's extension) ---
    ap.add_argument("--adaptive_mp", type=int, default=0)
    ap.add_argument("--adaptive_mp_levels", default="",
                    help="e.g. '256,128,64,32'. Empty ⇒ adaptive disabled.")
    ap.add_argument("--adaptive_mp_alpha", type=float, default=0.3)
    ap.add_argument("--adaptive_mp_beta", type=float, default=0.05)
    ap.add_argument("--adaptive_mp_enable_pruning", type=int, default=0)
    ap.add_argument("--adaptive_mp_ops", default="qkv_proj,out_proj",
                    help="CSV op names for adaptive MP.")
    ap.add_argument("--adaptive_mp_op_alphas", default="",
                    help="Per-op α overrides, 'op1=α1,op2=α2,...'. Empty ⇒ "
                         "global α for all.")
    ap.add_argument("--adaptive_mp_op_betas", default="",
                    help="Per-op β overrides, same format as op_alphas.")
    ap.add_argument("--adaptive_mp_fractions", default="",
                    help="CSV target fractions per level (quantile mode); "
                         "overrides linear-threshold. e.g. '0.05,0.35,0.40,0.20' "
                         "for levels=[256,128,64,32]. Sums to 1.")
    # ---- Auto-MP calibration (zero-hyperparameter) ------------------------
    ap.add_argument("--auto_mp_calibrate", type=int, default=0,
                    help="When > 0, replace α/β/fractions with a learned "
                         "FreeBoundaryMPConfig: per-(block, op) k-1 boundaries "
                         "searched by oracle coord-descent on a calibration set "
                         "of this size. Integrates with QwT comp (single pass, "
                         "Gauss-Seidel). Ignores adaptive_mp_alpha/beta/fractions.")
    ap.add_argument("--auto_mp_n_candidates", type=int, default=10,
                    help="Candidates per 1D line search in oracle coord descent.")
    ap.add_argument("--auto_mp_outer_per_op", type=int, default=2,
                    help="Coord-descent rounds within one (block, op).")
    ap.add_argument("--auto_mp_outer_block", type=int, default=1,
                    help="Outer rounds over all ops in a block (handles "
                         "inter-op coupling; 1 is usually enough).")
    ap.add_argument("--auto_mp_budget_ratio", type=float, default=0.0,
                    help="When > 0, constrain auto-MP search to a target "
                         "weighted-stoc_len compute budget ratio relative to "
                         "the all-max-stoc_len baseline. Example: 0.50 means "
                         "search for the best boundaries at ~50% compute.")
    ap.add_argument("--auto_mp_raw_only", action="store_true",
                    help="Search auto-MP boundaries on raw SC error, do not "
                         "fit/install compensation, and evaluate the "
                         "post-calibration SC model directly.")
    ap.add_argument("--auto_mp_qk_fixed_stoc_len", type=int, default=0,
                    help="When > 0 and auto-MP is enabled, pin qk to this "
                         "uniform stoc_len across all blocks instead of "
                         "searching qk boundaries. Must be one of "
                         "--adaptive_mp_levels.")
    ap.add_argument("--sc_mlp_mode", choices=["bipolar", "unipolar"],
                    default="bipolar",
                    help="SC quant mode for MLP fc1/fc2 when sc_mlp=True.")
    ap.add_argument("--sc_proj_mode", choices=["bipolar", "unipolar"],
                    default="bipolar",
                    help="SC quant mode for qkv/out projections.")
    ap.add_argument("--mlp_fc_mask", choices=["both", "fc1", "fc2"],
                    default="both",
                    help="Which MLP linears get SC when sc_mlp=True.")
    ap.add_argument("--av_metric_reverse", type=int, default=0,
                    help="When 1, reverse the AV per-row metric so LOW "
                         "softmax-amax rows get higher sl. Assumption: "
                         "near-one-hot rows are quant-easy, near-uniform rows "
                         "need precision.")
    ap.add_argument("--av_metric_kind", default="entropy",
                    choices=["amax", "entropy", "ipr"],
                    help="AV per-row peakiness metric. Default: entropy. "
                         "'entropy' and 'ipr' natively encode flat→high-score; "
                         "amax needs --av_metric_reverse 1 for the same "
                         "direction.")
    ap.add_argument("--fc2_skip_last_k", type=int, default=0,
                    help="When >0 (requires sc_mlp=True), SC fc2 in all blocks "
                         "EXCEPT the last K. fc1 stays governed by mlp_fc_mask. "
                         "Bypasses preset's sc_mlp path and uses sc_ops_per_block.")
    ap.add_argument("--vit_timestep", type=int, default=9)
    ap.add_argument("--vit_total_timesteps", type=int, default=10)
    args = ap.parse_args()

    os.environ.setdefault("XFORMERS_DISABLED", "1")
    seed_all(args.calib_seed)
    device = torch.device("cuda")
    set_noise_model(False)
    from sc_attention_patch import set_av_metric_kind, set_av_metric_reverse
    set_av_metric_reverse(bool(args.av_metric_reverse))
    set_av_metric_kind(args.av_metric_kind)
    print(f"[av_metric] kind={args.av_metric_kind} "
          f"reverse={int(bool(args.av_metric_reverse))}", flush=True)

    print(f"[env] torch {torch.__version__} cuda {torch.version.cuda} "
          f"dev={torch.cuda.get_device_name(0)}", flush=True)

    fine_schedule = None
    preset = None
    if args.sc_config in SC_PRESETS:
        preset = SC_PRESETS[args.sc_config]
        cfg_desc = f"preset {preset}"
    else:
        k = FINE_GRAINED_KS[args.sc_config]
        fine_schedule = build_skip_worst_k_schedule(k=k)
        total_on = sum(sum(v) for v in fine_schedule.values())
        cfg_desc = f"fine_grained skip_worst_K={k} (SC ops: {total_on}/120)"
    print(f"[cfg] sc_config={args.sc_config} -> {cfg_desc}  "
          f"sc_prec={args.sc_prec}  n_calib={args.n_calib}  n_eval={args.n_eval}  "
          f"ridge={args.ridge}  start_block={args.start_block}  "
          f"avg_sc_draws={args.avg_sc_draws}", flush=True)

    print("[data] loading calibration subset", flush=True)
    calib_ds = build_dataset(args.data_root, args.calib_seed, args.n_calib)
    calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=False)
    print("[data] loading eval subset", flush=True)
    eval_ds = build_dataset(args.data_root, args.eval_seed, args.n_eval)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True,
                             drop_last=False)

    print("[model] loading FP reference model", flush=True)
    model_fp = load_model(device).eval()
    print("[model] loading SC-to-be-patched model", flush=True)
    model_sc = load_model(device).eval()

    # Build optional linear MP spec. adaptive and static can coexist on
    # DIFFERENT ops (e.g., qkv_proj+out_proj adaptive, mlp_fc1 uniform).
    # Same op in both → adaptive wins.
    linear_mp_spec: dict = {}
    # Attention MP spec — ops 'qk' / 'av' go here, not linear_mp_spec.
    # QK per-head: static MPConfig (only 16 candidates, quantile is fine).
    # AV per-row: static MPConfig with level_fractions from --adaptive_mp_fractions.
    attn_mp_spec: dict = {}
    _LINEAR_OP_SET = {"qkv_proj", "out_proj", "mlp_fc1", "mlp_fc2", "proj"}
    _ATTN_OP_SET = {"qk", "av"}
    auto_mp_on = bool(args.auto_mp_calibrate > 0)
    if args.auto_mp_raw_only and not auto_mp_on:
        raise ValueError("--auto_mp_raw_only requires --auto_mp_calibrate > 0")
    if args.adaptive_mp and args.adaptive_mp_levels:
        from sc_integration.mp_linear import (
            AdaptiveMPConfig, FreeBoundaryMPConfig,
        )
        from sc_integration.sc_linear import set_vit_timestep
        levels = [int(x) for x in args.adaptive_mp_levels.split(",") if x.strip()]
        def _one_hot_fractions(target_stoc_len: int):
            if int(target_stoc_len) not in levels:
                raise ValueError(
                    f"fixed stoc_len {target_stoc_len} not in adaptive_mp_levels={levels}")
            return [1.0 if sl == int(target_stoc_len) else 0.0 for sl in levels]
        # Parse per-op overrides like "qkv_proj=0.4,out_proj=0.3"
        def _parse_kv(s):
            out = {}
            for pair in s.split(","):
                pair = pair.strip()
                if not pair: continue
                k, v = pair.split("=")
                out[k.strip()] = float(v)
            return out
        op_alphas = _parse_kv(args.adaptive_mp_op_alphas)
        op_betas = _parse_kv(args.adaptive_mp_op_betas)
        op_params = {}
        for op in set(op_alphas) | set(op_betas):
            op_params[op] = (
                op_alphas.get(op, args.adaptive_mp_alpha),
                op_betas.get(op, args.adaptive_mp_beta),
            )
        target_fractions = None
        if args.adaptive_mp_fractions.strip():
            target_fractions = [float(x) for x in
                                args.adaptive_mp_fractions.split(",")
                                if x.strip()]
        if auto_mp_on:
            # Zero-hyperparameter path: ignore α/β/fractions; a single
            # FreeBoundaryMPConfig is shared across all ops and all blocks,
            # with per-(block, op) boundaries filled in by auto_calibrate_mp
            # after patch_model installs it into SC ops.
            if target_fractions is not None or op_params:
                print("[auto_mp] ignoring --adaptive_mp_alpha/beta/fractions "
                      "(auto_mp_calibrate is on)", flush=True)
            adap = FreeBoundaryMPConfig(
                stoc_len_levels=levels,
                enable_pruning=bool(args.adaptive_mp_enable_pruning))
        else:
            adap = AdaptiveMPConfig(
                stoc_len_levels=levels,
                alpha=args.adaptive_mp_alpha, beta=args.adaptive_mp_beta,
                enable_pruning=bool(args.adaptive_mp_enable_pruning),
                operator_params=op_params,
                target_fractions=target_fractions)
        # Attention ops: QK per-head (MPConfig quantile, H=16 is tiny so
        # static is fine) / AV per-row (AdaptiveMPConfig with target_fractions
        # — classifies thousands of softmax rows per forward). In auto mode
        # we route QK through the FreeBoundaryMPConfig too (H=16 works with
        # per-head boundaries).
        from sc_integration.mp_linear import MPConfig as _MPConfig
        qk_mp_static = None
        if not auto_mp_on and target_fractions is not None:
            qk_mp_static = _MPConfig(stoc_len_levels=levels,
                                     level_fractions=target_fractions)
        elif auto_mp_on and args.auto_mp_qk_fixed_stoc_len > 0:
            qk_mp_static = _MPConfig(
                stoc_len_levels=levels,
                level_fractions=_one_hot_fractions(
                    args.auto_mp_qk_fixed_stoc_len),
            )
        for op in args.adaptive_mp_ops.split(","):
            op = op.strip()
            if not op:
                continue
            if op == "qk":
                if auto_mp_on:
                    if qk_mp_static is not None:
                        attn_mp_spec["qk"] = qk_mp_static
                    else:
                        attn_mp_spec["qk"] = adap
                else:
                    if qk_mp_static is None:
                        qk_mp_static = _MPConfig(stoc_len_levels=levels)
                    attn_mp_spec["qk"] = qk_mp_static
            elif op == "av":
                attn_mp_spec["av"] = adap
            else:
                linear_mp_spec[op] = {"adaptive": adap}
        set_vit_timestep(args.vit_timestep, args.vit_total_timesteps)
        adap_ops = [op for op in args.adaptive_mp_ops.split(",") if op.strip()]
        if auto_mp_on:
            print(f"[auto_mp] levels={levels} ops={adap_ops} "
                  f"n_calib_search={args.auto_mp_calibrate} "
                  f"candidates={args.auto_mp_n_candidates} "
                  f"outer_per_op={args.auto_mp_outer_per_op} "
                  f"outer_block={args.auto_mp_outer_block} "
                  f"budget_ratio={args.auto_mp_budget_ratio:.3f} "
                  f"raw_only={int(args.auto_mp_raw_only)} "
                  f"qk_fixed={args.auto_mp_qk_fixed_stoc_len}",
                  flush=True)
        else:
            prog = args.vit_timestep / max(args.vit_total_timesteps - 1, 1)
            thr = min(args.adaptive_mp_alpha * prog + args.adaptive_mp_beta, 0.95)
            print(f"[adaptive_mp] levels={levels} α={args.adaptive_mp_alpha} "
                  f"β={args.adaptive_mp_beta} t={args.vit_timestep}/"
                  f"T={args.vit_total_timesteps} progress={prog:.2f} "
                  f"threshold={thr:.3f} ops={adap_ops}", flush=True)
    if args.mp_levels:
        from sc_integration.mp_linear import MPConfig
        s_levels = [int(x) for x in args.mp_levels.split(",") if x.strip()]
        s_fractions = ([float(x) for x in args.mp_fractions.split(",")
                        if x.strip()]
                       if args.mp_fractions else None)
        mp = MPConfig(stoc_len_levels=s_levels, level_fractions=s_fractions)
        s_ops = [op.strip() for op in args.mp_ops.split(",") if op.strip()]
        # Don't clobber adaptive entries; user can force with disjoint op sets.
        for op in s_ops:
            if op in linear_mp_spec:
                print(f"[static_mp] skip op {op!r} (already adaptive)",
                      flush=True)
                continue
            linear_mp_spec[op] = {"fixed": mp}
        print(f"[static_mp] levels={s_levels} fractions="
              f"{s_fractions or 'equal'} ops={s_ops}", flush=True)
    if not linear_mp_spec:
        linear_mp_spec = None

    # Custom per-block schedule: keep fc2 FP in last K blocks while still SC'ing
    # everything else. Only makes sense with a preset that implies full SC.
    custom_sched = None
    if args.fc2_skip_last_k > 0:
        n_blocks = 24  # DINOv2 ViT-L/14
        K = args.fc2_skip_last_k
        custom_sched = {
            "mlp_fc1": [1] * n_blocks,
            "mlp_fc2": [1] * (n_blocks - K) + [0] * K,
            "qk":      [1] * n_blocks,
            "av":      [1] * n_blocks,
            "proj":    [1] * n_blocks,
        }
        print(f"[sc] fc2 FP in last {K} blocks; fc1/attn/proj SC in all 24",
              flush=True)

    _attn_spec_arg = attn_mp_spec or None
    if custom_sched is not None:
        stats = patch_model(model_sc, sc_prec=args.sc_prec,
                            sc_mlp_mode=args.sc_mlp_mode,
                            sc_proj_mode=args.sc_proj_mode,
                            sc_ops_per_block=custom_sched,
                            linear_mp_spec=linear_mp_spec,
                            attn_mp_spec=_attn_spec_arg)
    elif preset is not None:
        print(f"[sc] patching model_sc with preset {preset}", flush=True)
        stats = patch_model(model_sc, sc_prec=args.sc_prec,
                            sc_mlp_mode=args.sc_mlp_mode,
                            sc_proj_mode=args.sc_proj_mode,
                            mlp_fc_mask=args.mlp_fc_mask,
                            linear_mp_spec=linear_mp_spec,
                            attn_mp_spec=_attn_spec_arg, **preset)
    else:
        print(f"[sc] patching model_sc with fine-grained schedule "
              f"({args.sc_config})", flush=True)
        stats = patch_model(model_sc, sc_prec=args.sc_prec,
                            sc_mlp_mode=args.sc_mlp_mode,
                            sc_proj_mode=args.sc_proj_mode,
                            mlp_fc_mask=args.mlp_fc_mask,
                            sc_ops_per_block=fine_schedule,
                            linear_mp_spec=linear_mp_spec,
                            attn_mp_spec=_attn_spec_arg)
    if attn_mp_spec:
        first = list(attn_mp_spec.values())[0]
        frac = getattr(first, "target_fractions",
                       getattr(first, "level_fractions", None))
        print(f"[attn_mp] ops={list(attn_mp_spec)} "
              f"levels={first.stoc_len_levels} fractions={frac}",
              flush=True)
    print(f"[sc] patched={stats}", flush=True)

    res_sc_raw = None
    if not args.skip_baseline:
        print("[eval] SC baseline (no compensation)", flush=True)
        seed_all(args.eval_seed)
        res_sc_raw = evaluate(model_sc, eval_loader, device)
        print("[result:sc_raw]", res_sc_raw, flush=True)

    comp_factory = None
    comp_factory_variants = None
    if auto_mp_on and args.auto_mp_raw_only:
        print("[comp] disabled (raw-only auto-MP)", flush=True)
    elif args.comp_mode == "sc":
        if args.comp_variant_select:
            # The comp matmul has D_in == D_out == 1024 for ViT-L (residual
            # stream). Build both candidate Sobol scramblings at that D and
            # let calibration pick per block.
            from config_helpers import make_sobol_simple_config, make_sobol_antithetic_config
            D = 1024
            cfg_default = make_sobol_simple_config(D, D, args.comp_sc_prec)
            cfg_anti    = make_sobol_antithetic_config(D, D, args.comp_sc_prec)
            comp_factory_variants = [
                ("default",    make_sc_comp_factory(args.comp_sc_prec, args.comp_sc_mode, cfg_override=cfg_default)),
                ("antithetic", make_sc_comp_factory(args.comp_sc_prec, args.comp_sc_mode, cfg_override=cfg_anti)),
            ]
            print(f"[comp] SC compensator: sc_prec={args.comp_sc_prec} "
                  f"mode={args.comp_sc_mode}  variants=2 (default, antithetic)",
                  flush=True)
        else:
            comp_factory = make_sc_comp_factory(
                sc_prec=args.comp_sc_prec, mode=args.comp_sc_mode)
            print(f"[comp] SC compensator: sc_prec={args.comp_sc_prec} "
                  f"mode={args.comp_sc_mode}", flush=True)
    else:
        print(f"[comp] FP compensator (baseline)", flush=True)

    t_cal = time.time()
    seed_all(args.calib_seed)
    blocks_fp = list(get_blocks(model_fp))
    blocks_sc_container = get_blocks(model_sc)
    if auto_mp_on:
        if args.auto_mp_raw_only:
            print("[calib] starting AUTO-MP calibration "
                  "(oracle boundary search on raw SC error, no comp)",
                  flush=True)
        else:
            print("[calib] starting AUTO-MP calibration (oracle boundary search "
                  "+ QwT comp, Gauss-Seidel)", flush=True)
        sys.path.insert(0, str(HERE / "sc"))
        from mp_auto_calibrator import auto_calibrate_mp  # noqa: E402
        auto_ops = []
        for op in args.adaptive_mp_ops.split(","):
            op = op.strip()
            if not op:
                continue
            if op == "qk" and args.auto_mp_qk_fixed_stoc_len > 0:
                continue
            auto_ops.append(op)
        calib_report = auto_calibrate_mp(
            model_fp=model_fp, model_sc=model_sc,
            blocks_fp=blocks_fp,
            blocks_sc_container=blocks_sc_container,
            calib_loader=calib_loader,
            device=device,
            n_calib=args.auto_mp_calibrate,
            cfg=adap,                       # the FreeBoundaryMPConfig
            ops_per_block=auto_ops,
            ridge=args.ridge,
            start_block=args.start_block,
            fwd_chunk=args.fwd_chunk,
            avg_sc_draws=args.avg_sc_draws,
            n_candidates=args.auto_mp_n_candidates,
            n_outer_per_op=args.auto_mp_outer_per_op,
            n_outer_block=args.auto_mp_outer_block,
            budget_ratio=(
                args.auto_mp_budget_ratio
                if args.auto_mp_budget_ratio > 0 else None),
            search_objective=(
                "raw_mse" if args.auto_mp_raw_only else "comp_residual"),
            comp_factory=comp_factory,
            comp_factory_variants=comp_factory_variants,
            comp_refit_iters=args.comp_refit_iters,
            fit_compensation=not args.auto_mp_raw_only,
            install_compensation=not args.auto_mp_raw_only,
        )
    else:
        print("[calib] starting QwT-style sequential calibration", flush=True)
        calib_report = calibrate_qwt(
            model_fp=model_fp, model_sc=model_sc,
            blocks_fp=blocks_fp,
            blocks_sc_container=blocks_sc_container,
            calib_loader=calib_loader,
            device=device,
            n_calib=args.n_calib,
            ridge=args.ridge,
            start_block=args.start_block,
            fwd_chunk=args.fwd_chunk,
            avg_sc_draws=args.avg_sc_draws,
            comp_factory=comp_factory,
            comp_factory_variants=comp_factory_variants,
            comp_refit_iters=args.comp_refit_iters,
            r2_threshold=args.r2_threshold,
        )
    calib_s = time.time() - t_cal
    print(f"[calib] done in {calib_s:.1f}s", flush=True)

    res_sc_post = None
    res_sc_comp = None
    if auto_mp_on and args.auto_mp_raw_only:
        print("[eval] SC after raw-only auto-MP", flush=True)
        seed_all(args.eval_seed)
        res_sc_post = evaluate(model_sc, eval_loader, device)
        print("[result:sc_post]", res_sc_post, flush=True)
    else:
        print("[eval] SC + QwT compensation", flush=True)
        seed_all(args.eval_seed)
        res_sc_comp = evaluate(model_sc, eval_loader, device)
        print("[result:sc_comp]", res_sc_comp, flush=True)

    print("[eval] FP reference", flush=True)
    seed_all(args.eval_seed)
    res_fp = evaluate(model_fp, eval_loader, device)
    print("[result:fp]", res_fp, flush=True)

    out = {
        "config": {
            "sc_config": args.sc_config, "sc_prec": args.sc_prec,
            "n_calib": args.n_calib, "n_eval": args.n_eval,
            "calib_seed": args.calib_seed, "eval_seed": args.eval_seed,
            "ridge": args.ridge, "start_block": args.start_block,
            "avg_sc_draws": args.avg_sc_draws,
            "batch_size": args.batch_size,
            "comp_mode": args.comp_mode,
            "comp_sc_prec": args.comp_sc_prec,
            "comp_sc_mode": args.comp_sc_mode,
            "comp_refit_iters": args.comp_refit_iters,
            "comp_variant_select": args.comp_variant_select,
            "r2_threshold": args.r2_threshold,
            "auto_mp_budget_ratio": args.auto_mp_budget_ratio,
            "auto_mp_raw_only": args.auto_mp_raw_only,
            "auto_mp_qk_fixed_stoc_len": args.auto_mp_qk_fixed_stoc_len,
        },
        "calib": {"per_block": calib_report, "elapsed_s": calib_s},
        "results": {
            "fp": res_fp,
            "sc_raw": res_sc_raw,
            "sc_post": res_sc_post,
            "sc_comp": res_sc_comp,
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out_json}", flush=True)

    print("\n=== SUMMARY ===")
    print(f"  FP                : top1={res_fp['top1']:.4f}")
    if res_sc_raw:
        print(f"  SC ({args.sc_config})  : top1={res_sc_raw['top1']:.4f}")
    if res_sc_post is not None:
        print(f"  SC after auto-MP  : top1={res_sc_post['top1']:.4f}")
        if res_sc_raw:
            delta = res_sc_post['top1'] - res_sc_raw['top1']
            gap = res_fp['top1'] - res_sc_raw['top1']
            closed = (delta / gap * 100.0) if gap > 0 else float('nan')
            print(f"  Gain over SC      : {delta:+.2f} pt  "
                  f"(closes {closed:.1f}% of FP-SC gap)")
    else:
        print(f"  SC + QwT comp     : top1={res_sc_comp['top1']:.4f}")
        if res_sc_raw:
            delta = res_sc_comp['top1'] - res_sc_raw['top1']
            gap = res_fp['top1'] - res_sc_raw['top1']
            closed = (delta / gap * 100.0) if gap > 0 else float('nan')
            print(f"  Gain over SC      : {delta:+.2f} pt  "
                  f"(closes {closed:.1f}% of FP-SC gap)")


if __name__ == "__main__":
    main()

"""SC eval on COCO with an EVA-ViTDet backbone.

CLI mirrors ``vit_sc/eval.py`` (DINOv2 classification). Three-tier dispatch:

  1. ``--sc_ops_per_block_json`` — highest priority, full per-(op, block) control.
  2. ``--sc_ops``               — comma-separated op names or 5-element 0/1 vector.
  3. Per-op int flags           — ``--sc_qk 1 --sc_av 0 ...`` (lowest priority).

MLP selection within tier 3:

  ``--sc_mlp 1`` enables MLP SC. Which blocks / fcs are on is further
  controlled by ``--mlp_skip_last_k``, ``--mlp_skip_first_k``,
  ``--mlp_fc_mask``, ``--mlp_sc_spec_json`` (same semantics as vit_sc).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

from eval_common import (load_model_and_loader, set_evaluator_output_dir,
                         checkpointed_eval, file_sha256)

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (_HERE, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from sc_patch import (  # noqa: E402
    sc_patch_eva,
    SC_OP_NAMES,
    normalize_sc_ops_per_block,
    count_sc_ops,
)
from mp_spec import (  # noqa: E402
    add_mp_args, build_attn_mp_spec, build_linear_mp_spec, mp_args_to_dict,
)

# Canonical 5-name vocabulary matching vit_sc's SC_OP_NAMES. "proj" expands
# to {"qkv_proj", "out_proj"} internally.
_VIT_SC_OP_NAMES = ("mlp_fc1", "mlp_fc2", "qk", "av", "proj")


def parse_sc_ops(spec: str) -> set[str]:
    """Parse ``--sc_ops`` into a set of canonical 6-element op names.

    Accepts:
      * comma-separated names from _VIT_SC_OP_NAMES (e.g. "qk,av,proj")
      * a 5-element 0/1 vector in _VIT_SC_OP_NAMES order (e.g. "1,0,1,0,0")
    "proj" expands to {"qkv_proj", "out_proj"}.
    """
    items = [s.strip() for s in spec.split(",") if s.strip()]
    if not items:
        return set()
    if len(items) == len(_VIT_SC_OP_NAMES) and all(x in ("0", "1") for x in items):
        names = {n for n, v in zip(_VIT_SC_OP_NAMES, items) if v == "1"}
    else:
        names = set(items)
    unknown = names - set(_VIT_SC_OP_NAMES)
    if unknown:
        raise SystemExit(f"unknown sc_ops: {sorted(unknown)} "
                         f"(valid: {_VIT_SC_OP_NAMES})")
    out = set()
    for n in names:
        if n == "proj":
            out.update(("qkv_proj", "out_proj"))
        else:
            out.add(n)
    return out


def build_schedule(args, n_blocks: int) -> dict:
    """Build a full per-(op, block) schedule dict from CLI args.

    Three-tier priority (highest first):
      1. ``--sc_ops_per_block_json``
      2. ``--sc_ops``
      3. per-op int flags + MLP-specific sub-flags
    """
    # --- Tier 1 ---
    if args.sc_ops_per_block_json:
        with open(args.sc_ops_per_block_json) as f:
            raw = json.load(f)
        return normalize_sc_ops_per_block(raw, n_blocks)

    # --- Tier 2 ---
    if args.sc_ops:
        ops = parse_sc_ops(args.sc_ops)
        sched = {op: [0] * n_blocks for op in SC_OP_NAMES}
        for op in ops:
            sched[op] = [1] * n_blocks
        return sched

    # --- Tier 3: per-op bools ---
    sched = {op: [0] * n_blocks for op in SC_OP_NAMES}
    if args.sc_qk:
        sched["qk"] = [1] * n_blocks
    if args.sc_av:
        sched["av"] = [1] * n_blocks
    if args.sc_qkv_proj:
        sched["qkv_proj"] = [1] * n_blocks
    if args.sc_out_proj:
        sched["out_proj"] = [1] * n_blocks

    if args.sc_mlp:
        # Determine which (block, fc) pairs go SC.
        if args.mlp_sc_spec_json:
            with open(args.mlp_sc_spec_json) as f:
                spec = json.load(f)  # list of [block_idx, "fc1"|"fc2"]
            for bi, fc in spec:
                sched[f"mlp_{fc}"][int(bi)] = 1
        else:
            skip = set()
            if args.mlp_skip_last_k > 0:
                skip |= set(range(n_blocks - args.mlp_skip_last_k, n_blocks))
            if args.mlp_skip_first_k > 0:
                skip |= set(range(args.mlp_skip_first_k))
            mask = args.mlp_fc_mask
            for i in range(n_blocks):
                if i in skip:
                    continue
                if mask in ("both", "fc1"):
                    sched["mlp_fc1"][i] = 1
                if mask in ("both", "fc2"):
                    sched["mlp_fc2"][i] = 1

    return sched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--sc_prec", type=int, default=8)
    p.add_argument("--stoc_len", type=int, default=0,
                   help="Override SC bitstream length for ALL ops (fast "
                        "uniform path). Default 0 = use 2**sc_prec. Use this "
                        "instead of --mp_levels L --mp_fractions 1.0 to avoid "
                        "the MP routing overhead (extra abs/amax + index_select"
                        " + index_copy_ per linear, plus a Python BH loop in "
                        "SCMatMul av).")

    # --- Tier 1: per-block JSON schedule ---
    p.add_argument("--sc_ops_per_block_json", default="",
                   help="Path to JSON {op: [0/1]*n_blocks}. Highest priority.")

    # --- Tier 2: --sc_ops string ---
    p.add_argument("--sc_ops", default="",
                   help="Comma-separated op names from "
                        "{mlp_fc1,mlp_fc2,qk,av,proj} OR a 5-element 0/1 "
                        "vector. Overrides per-op flags.")

    # --- Tier 3: per-op int flags (mirror vit_sc/eval.py) ---
    p.add_argument("--sc_qk",       type=int, default=1)
    p.add_argument("--sc_av",       type=int, default=0)
    p.add_argument("--sc_mlp",      type=int, default=0)
    p.add_argument("--sc_qkv_proj", type=int, default=0)
    p.add_argument("--sc_out_proj", type=int, default=0)

    # MLP sub-flags (active when --sc_mlp 1)
    p.add_argument("--mlp_skip_last_k",  type=int, default=0)
    p.add_argument("--mlp_skip_first_k", type=int, default=0)
    p.add_argument("--mlp_fc_mask", choices=["both", "fc1", "fc2"],
                   default="both")
    p.add_argument("--mlp_sc_spec_json", default="",
                   help="JSON list of [block_idx, fc_name] pairs. "
                        "Overrides skip_*_k / fc_mask when --sc_mlp 1.")

    # SC quantization modes (passed to sc_patch_eva)
    p.add_argument("--sc_mlp_mode",  choices=["bipolar", "unipolar"],
                   default="bipolar")
    p.add_argument("--sc_proj_mode", choices=["bipolar", "unipolar"],
                   default="bipolar")

    p.add_argument("--out-tag", type=str, default=None)
    p.add_argument("--mlp_chunk_d", type=int, default=0,
                   help="When > 0, route mlp_fc1 / mlp_fc2 SC matmul through "
                        "the chunked MLP-specialized kernel "
                        "(sc_matmul_enable_triton_mlp). Recommended: 64 — "
                        "evenly divides D_in for both fc1 (1408) and "
                        "fc2 (6144), and shrinks the cum_indicator table "
                        "from O(D) to O(chunk_d) so it fits in L2.")
    p.add_argument("--size", type=int, default=0,
                   help="Override input square size (multiple of 256). 0=default(1280).")
    p.add_argument("--start_idx", type=int, default=0,
                   help="COCO val start index. Default 0 evaluates [0, n_eval). "
                        "Use to incrementally extend a prior n=N1 run by another "
                        "N2 imgs: rerun with --start_idx N1 --n_eval N2 (and a "
                        "different --out_tag to avoid skipping).")
    p.add_argument("--d2_datasets", type=str, default="",
                   help="DETECTRON2_DATASETS root (default: GreatLakes shared_data).")
    p.add_argument("--ckpt", type=str, default="",
                   help="Path to EVA-ViTDet checkpoint (default: GreatLakes shared_data).")
    p.add_argument("--sl_map_json", default="",
                   help="Per-(op, block) stoc_len map from "
                        "experiments/mp_budget_swap_search.py. When set, "
                        "overrides --sc_ops_per_block_json and --mp_* flags; "
                        "patches each block individually via the sl_map.")
    p.add_argument("--out_dir", type=str, default="",
                   help="Output directory (containing metrics.json, predictions, "
                        "etc.). Default = det/results/<out-tag>. Use to redirect "
                        "into a sweep RES_DIR — e.g. results/sweep_x/<out-tag>.")
    p.add_argument("--use_soft_nms", action="store_true",
                   help="Enable mmcv linear soft NMS in CascadeROIHeads "
                        "(iou_thresh=0.3, sigma=0.5). Default off — matches "
                        "prior runs.")
    p.add_argument("--interp_type", choices=["", "vitdet", "beit"], default="",
                   help="ViT rel-pos interp type when q/k size != trained "
                        "(80x80 for EVA at 1280). 'beit' = cubic on log-spaced "
                        "grid (recommended for off-train res); '' / 'vitdet' "
                        "= linear interp (default).")
    p.add_argument("--resume", action="store_true",
                   help="If checkpoint.pt exists in --out_dir, skip already-"
                        "evaluated images and append new ones. cfg_sig must "
                        "match (sched/sl_map files are sha-checked).")
    p.add_argument("--save_every", type=int, default=50,
                   help="Checkpoint cadence (images). Also flushed on SIGINT.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for python `random` and torch — controls SC "
                        "LFSR seeds drawn by make_sobol_simple_config. Without "
                        "this, two runs of the same schedule give different "
                        "predictions (bitstreams diverge), so n=10 mAP swings "
                        "by ~10. Default 0 = deterministic.")
    add_mp_args(p)
    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tag = args.out_tag or (f"sc_p{args.sc_prec}_sz{args.size}" if args.size
                           else f"sc_p{args.sc_prec}")
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = Path(__file__).parent / out_dir
    else:
        out_dir = Path(__file__).parent / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1] Loading config + model + weights (size={args.size or 'default'}, "
          f"start_idx={args.start_idx})")
    model, test_loader, evaluator, subset = load_model_and_loader(
        args.n_eval, d2_datasets=args.d2_datasets, ckpt=args.ckpt,
        size=args.size, start_idx=args.start_idx,
        use_soft_nms=args.use_soft_nms,
        interp_type=args.interp_type)
    set_evaluator_output_dir(evaluator, out_dir)

    n_blocks = len(list(model.backbone.net.blocks))
    print(f"    n_blocks = {n_blocks}")

    if args.sl_map_json:
        # Per-(op, block) sl_map mode — bypass sc_patch_eva and use the
        # MP search's per-block patcher.
        with open(args.sl_map_json) as f:
            sl_map_d = json.load(f)
        sl_map = sl_map_d.get("sl_map", sl_map_d) if isinstance(sl_map_d, dict) else sl_map_d
        from experiments.mp_budget_swap_search import _patch_block_with_spec
        from sc_integration import mp_search as ms
        tier = f"sl_map_json={args.sl_map_json}"
        print(f"[2] Schedule (tier: {tier})")
        sched = {op: [int(bool(ms.entry_to_mpconfig(sl_map[op][bi])))
                      for bi in range(n_blocks)] for op in SC_OP_NAMES}
        stats_requested = count_sc_ops(sched)
        for op in SC_OP_NAMES:
            on, tot = stats_requested[op]
            print(f"    {op:<10s}  {on:3d}/{tot}")
        print(f"[3] Patching backbone per-block from sl_map "
              f"(sc_prec={args.sc_prec})")
        for bi in range(n_blocks):
            spec = ms.build_current_spec(sl_map, SC_OP_NAMES, bi)
            new_blk = _patch_block_with_spec(
                model.backbone.net.blocks[bi], args.sc_prec, spec)
            model.backbone.net.blocks[bi] = new_blk.cuda()
        stats = stats_requested
        linear_mp_spec = None
        attn_mp_spec = None
    else:
        sched = build_schedule(args, n_blocks)
        stats_requested = count_sc_ops(sched)
        tier = ("sc_ops_per_block_json" if args.sc_ops_per_block_json
                else "sc_ops" if args.sc_ops else "per-op flags")
        print(f"[2] Schedule (tier: {tier})")
        for op in SC_OP_NAMES:
            on, tot = stats_requested[op]
            print(f"    {op:<10s}  {on:3d}/{tot}")

        linear_mp_spec = build_linear_mp_spec(args) or None
        attn_mp_spec = build_attn_mp_spec(args) or None
        if linear_mp_spec:
            print(f"    linear_mp_spec ops={sorted(linear_mp_spec)}")
        if attn_mp_spec:
            print(f"    attn_mp_spec ops={sorted(attn_mp_spec)}")

        if args.stoc_len and (linear_mp_spec or attn_mp_spec):
            raise SystemExit("--stoc_len conflicts with --mp_levels / "
                             "--qk_mp_levels / --av_mp_levels (the whole "
                             "point of --stoc_len is to skip the MP path).")
        sl_eff = args.stoc_len or None  # None ⇒ sc_patch_eva uses 2**sc_prec
        print(f"[3] Patching backbone (sc_prec={args.sc_prec}, "
              f"stoc_len={sl_eff or 2**args.sc_prec}, "
              f"mlp_mode={args.sc_mlp_mode}, proj_mode={args.sc_proj_mode})")
        stats = sc_patch_eva(
            model,
            sc_prec=args.sc_prec,
            sc_ops_per_block=sched,
            mlp_mode=args.sc_mlp_mode,
            proj_mode=args.sc_proj_mode,
            linear_mp_spec=linear_mp_spec,
            attn_mp_spec=attn_mp_spec,
            mlp_chunk_d=args.mlp_chunk_d,
            stoc_len=sl_eff,
        )
        print(f"    swapped: {stats}")

    from scmp_kernels.sc import det_kernel_tuning

    cfg_sig = {
        "n_eval": args.n_eval,
        "start_idx": args.start_idx,
        "size": args.size,
        "sc_prec": args.sc_prec,
        "stoc_len": args.stoc_len,
        "schedule": {k: list(v) for k, v in sched.items()},
        "sc_ops_per_block_json_sha256": file_sha256(args.sc_ops_per_block_json),
        "sl_map_json_sha256": file_sha256(args.sl_map_json),
        "sc_mlp_mode": args.sc_mlp_mode,
        "sc_proj_mode": args.sc_proj_mode,
        "mlp_chunk_d": args.mlp_chunk_d,
        "use_soft_nms": bool(args.use_soft_nms),
        "interp_type": args.interp_type or "",
        "mp": mp_args_to_dict(args),
    }
    print(f"[4] Running inference on {args.n_eval} images "
          f"(resume={args.resume}, save_every={args.save_every})")
    t0 = time.time()
    with torch.no_grad(), det_kernel_tuning():
        results = checkpointed_eval(
            model, evaluator, subset, out_dir,
            cfg_sig=cfg_sig, save_every=args.save_every,
            resume=args.resume, phase="eval", size=args.size,
        )
    dt = time.time() - t0
    print(f"    done in {dt:.1f}s ({dt/max(args.n_eval, 1):.2f}s/img)")

    out = {
        "n_eval": args.n_eval,
        "sc_prec": args.sc_prec,
        "stoc_len": args.stoc_len or (1 << args.sc_prec),
        "use_soft_nms": bool(args.use_soft_nms),
        "interp_type": args.interp_type or "vitdet",
        "schedule_tier": tier,
        "sc_mlp_mode": args.sc_mlp_mode,
        "sc_proj_mode": args.sc_proj_mode,
        "mlp_chunk_d": args.mlp_chunk_d,
        "n_blocks": n_blocks,
        "schedule": {k: list(v) for k, v in sched.items()},
        "swapped": stats,
        "mp": mp_args_to_dict(args),
        "eval_seconds": dt,
        **results,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    for task, m in results.items():
        print(f"    [{task}] AP={m.get('AP',0):.2f}  "
              f"AP50={m.get('AP50',0):.2f}  AP75={m.get('AP75',0):.2f}")
    print(f"[OK] metrics -> {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()

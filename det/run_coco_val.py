"""Resumable COCO-val SC eval with precision + skip-schedule controls.

Runs the EVA-ViTDet backbone on COCO val images *in order*, supports SC at a
chosen precision with an optional per-(op, block) skip schedule, and
checkpoints predictions every N images so a crashed/cancelled run can be
resumed without redoing work.

Outputs in ``--out_dir``:
  * ``checkpoint.pt``      — periodic dump of evaluator predictions + done ids
  * ``metrics.json``       — final config + AP results (written on completion)
  * ``coco_instances_results.json`` — written by the COCO evaluator on completion

CLI essentials::

    # First run (full val set, SC int7 with 20-worst skipped):
    python det/run_coco_val.py \\
        --sc_prec 7 \\
        --skip det/sensitivity/skip/skip_worst_20_int7.json \\
        --out_dir det/results/int7_skip20 \\
        --d2_datasets <COCO_ROOT> --ckpt <EVA_PTH>

    # Resume after Ctrl-C / crash:
    python det/run_coco_val.py ... --out_dir det/results/int7_skip20 --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (_HERE, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_common import (  # noqa: E402
    CFG_PATH, load_model_and_loader, set_evaluator_output_dir,
)
from sc_patch import (  # noqa: E402
    SC_OP_NAMES, sc_patch_eva, normalize_sc_ops_per_block, count_sc_ops,
)


def _sha256(path: Path | None) -> str:
    if path is None:
        return ""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _all_ops_schedule(n_blocks: int) -> dict:
    return {op: [1] * n_blocks for op in SC_OP_NAMES}


def _load_skip_schedule(skip_path: str, n_blocks: int) -> dict:
    """Load a per-(op, block) 0/1 schedule. Compatible with
    ``--sc_ops_per_block_json`` in sc_eval.py: 1 = SC, 0 = FP.
    """
    with open(skip_path) as f:
        raw = json.load(f)
    return normalize_sc_ops_per_block(raw, n_blocks)


def _build_test_loader_for(items: list, base_name: str, sub_name: str):
    """Register ``items`` as ``sub_name`` and build a fresh test loader."""
    from detectron2.config import LazyConfig, instantiate
    from detectron2.data import DatasetCatalog, MetadataCatalog

    if sub_name in DatasetCatalog.list():
        DatasetCatalog.remove(sub_name)
        MetadataCatalog.remove(sub_name)
    DatasetCatalog.register(sub_name, lambda items=items: items)
    md = MetadataCatalog.get(base_name).as_dict()
    md.pop("name", None)
    MetadataCatalog.get(sub_name).set(**md)

    cfg = LazyConfig.load(CFG_PATH)
    cfg.dataloader.test.dataset.names = sub_name
    return instantiate(cfg.dataloader.test)


def _atomic_save(state: dict, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _config_signature(args, skip_sha: str) -> dict:
    return {
        "sc_prec": args.sc_prec,
        "stoc_len": args.stoc_len,
        "skip_json": str(args.skip) if args.skip else "",
        "skip_json_sha256": skip_sha,
        "n_eval": args.n_eval,
        "start_idx": args.start_idx,
        "sc_mlp_mode": args.sc_mlp_mode,
        "sc_proj_mode": args.sc_proj_mode,
        "size": args.size,
        "use_soft_nms": args.use_soft_nms,
    }


def _default_skip_for(stoc_len: int, default_skip_pct: int) -> Path | None:
    """Auto-locate a skip schedule for the given stoc_len.

    Looks for ``det/sensitivity/skip/skip_worst_<pct>_int<sc_prec>.json`` if
    ``stoc_len`` is a power of 2 (so a clean ``int<N>`` tag exists), or
    ``..._len<stoc_len>.json`` otherwise. Returns None if no file exists.
    """
    import math
    sc_prec_pow2 = (stoc_len & (stoc_len - 1)) == 0  # power of 2?
    if sc_prec_pow2:
        sc_prec = int(math.log2(stoc_len))
        tag = f"int{sc_prec}"
    else:
        tag = f"len{stoc_len}"
    candidate = (Path(__file__).resolve().parent / "sensitivity" / "skip"
                 / f"skip_worst_{default_skip_pct}_{tag}.json")
    return candidate if candidate.exists() else None


def main():
    import math
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--sc_prec", type=int, default=None,
                     help="SC precision in bits; stoc_len = 2**sc_prec. "
                          "Use 6/7/8 for int6/int7/int8.")
    grp.add_argument("--length", type=int, default=None,
                     help="Stochastic stream length directly; need not be a "
                          "power of 2 (e.g. 96, 192). Mutually exclusive "
                          "with --sc_prec.")
    p.add_argument("--skip", type=str, default="auto",
                   help="Path to per-(op, block) JSON schedule {op: [0/1]*n_blocks}. "
                        "1 = SC, 0 = FP. Same format as sensitivity/skip/*.json. "
                        "Default 'auto' resolves to "
                        "det/sensitivity/skip/skip_worst_<--skip_pct>_<tag>.json. "
                        "Pass 'none' to SC every (op, block).")
    p.add_argument("--skip_pct", type=int, default=20,
                   help="Used only when --skip=auto: which skip-worst-N file to "
                        "pick up (matches sensitivity/skip/skip_worst_<N>_*.json).")
    p.add_argument("--n_eval", type=int, default=-1,
                   help="# COCO val images to evaluate. -1 = full set (~5000). "
                        "Ignored when --num_shards is set.")
    p.add_argument("--start_idx", type=int, default=0,
                   help="Starting offset into the val set. "
                        "Ignored when --num_shards is set.")
    p.add_argument("--num_shards", type=int, default=0,
                   help="If >0, partition the full val set into this many "
                        "contiguous shards by index and run only --shard_id. "
                        "Overrides --start_idx / --n_eval.")
    p.add_argument("--shard_id", type=int, default=0,
                   help="Which shard (0..num_shards-1) this process runs.")
    p.add_argument("--cap_n_eval", type=int, default=0,
                   help="Smoke-test cap: if >0, truncate this run's n_eval "
                        "to at most this many images (applied AFTER shard "
                        "computation). Use to validate the pipeline cheaply.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output dir. Checkpoint + metrics land here.")
    p.add_argument("--resume", action="store_true",
                   help="If checkpoint.pt exists in --out_dir, skip already-"
                        "evaluated images and append new ones.")
    p.add_argument("--save_every", type=int, default=50,
                   help="Checkpoint cadence (images). Also saves on Ctrl-C.")
    p.add_argument("--d2_datasets", type=str, default="",
                   help="DETECTRON2_DATASETS root.")
    p.add_argument("--ckpt", type=str, default="",
                   help="Path to EVA-ViTDet checkpoint.")
    p.add_argument("--sc_mlp_mode", choices=["bipolar", "unipolar"],
                   default="bipolar")
    p.add_argument("--sc_proj_mode", choices=["bipolar", "unipolar"],
                   default="bipolar")
    p.add_argument("--size", type=int, default=1024,
                   help="square_pad + ResizeShortestEdge (multiple of 256). "
                        "Default 1024. Pass 1280 to skip the size override "
                        "and use the cfg default.")
    p.add_argument("--no_soft_nms", action="store_true",
                   help="Disable soft-NMS (default ON, requires mmcv).")
    args = p.parse_args()

    # Resolve --sc_prec / --length into both args.sc_prec + args.stoc_len so
    # downstream code (and the config signature) can rely on having both.
    if args.length is not None:
        args.stoc_len = int(args.length)
        args.sc_prec = max(1, int(math.ceil(math.log2(max(args.stoc_len, 2)))))
    else:
        args.stoc_len = 1 << args.sc_prec
    args.use_soft_nms = not args.no_soft_nms

    # Resolve --skip:
    #   "auto" -> det/sensitivity/skip/skip_worst_<pct>_<tag>.json (if exists)
    #   "none" -> SC every (op, block)
    #   else   -> literal path
    if args.skip == "auto":
        auto = _default_skip_for(args.stoc_len, args.skip_pct)
        if auto is not None:
            print(f"[skip=auto] using {auto}")
            args.skip = str(auto)
        else:
            print(f"[skip=auto] no skip_worst_{args.skip_pct}_* file found "
                  f"for stoc_len={args.stoc_len}; falling back to all-ops SC")
            args.skip = ""
    elif args.skip == "none":
        args.skip = ""

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"

    skip_path = Path(args.skip).resolve() if args.skip else None
    skip_sha = _sha256(skip_path)
    cfg_sig = _config_signature(args, skip_sha)

    # eval_common sets DETECTRON2_DATASETS via setdefault; do it here too so
    # DatasetCatalog.get() below works before load_model_and_loader runs.
    import os
    if args.d2_datasets:
        os.environ["DETECTRON2_DATASETS"] = args.d2_datasets
    elif "DETECTRON2_DATASETS" not in os.environ:
        from eval_common import _DEFAULT_D2_DATASETS
        os.environ["DETECTRON2_DATASETS"] = _DEFAULT_D2_DATASETS

    # --- Resolve full val length up front so we can default n_eval. ---
    print("[1] Loading model + full val subset")
    from detectron2.data import DatasetCatalog
    full_items = DatasetCatalog.get("coco_2017_val")
    full_len = len(full_items)
    if args.num_shards > 0:
        if not (0 <= args.shard_id < args.num_shards):
            sys.exit(f"--shard_id {args.shard_id} out of range "
                     f"[0, {args.num_shards})")
        base = full_len // args.num_shards
        rem = full_len % args.num_shards
        args.start_idx = args.shard_id * base + min(args.shard_id, rem)
        args.n_eval = base + (1 if args.shard_id < rem else 0)
        print(f"    sharding: shard {args.shard_id}/{args.num_shards} -> "
              f"[{args.start_idx}, {args.start_idx + args.n_eval})")
    if args.n_eval < 0 or args.n_eval > full_len - args.start_idx:
        args.n_eval = full_len - args.start_idx
    if args.cap_n_eval > 0 and args.n_eval > args.cap_n_eval:
        print(f"    cap_n_eval: {args.n_eval} -> {args.cap_n_eval}")
        args.n_eval = args.cap_n_eval
    cfg_sig["n_eval"] = args.n_eval
    cfg_sig["start_idx"] = args.start_idx
    cfg_sig["num_shards"] = args.num_shards
    cfg_sig["shard_id"] = args.shard_id if args.num_shards > 0 else -1
    print(f"    val total={full_len}  start_idx={args.start_idx}  "
          f"n_eval={args.n_eval}")

    model, _full_loader, evaluator, full_subset = load_model_and_loader(
        args.n_eval, d2_datasets=args.d2_datasets, ckpt=args.ckpt,
        start_idx=args.start_idx, size=args.size,
        use_soft_nms=args.use_soft_nms)
    print(f"    size={args.size}  use_soft_nms={args.use_soft_nms}")
    set_evaluator_output_dir(evaluator, out_dir)
    # COCOEvaluator initializes self._predictions only inside reset(); call it
    # here so .process() / our resume code can rely on the attribute existing.
    evaluator.reset()
    n_blocks = len(list(model.backbone.net.blocks))

    # --- Schedule ---
    if skip_path is not None:
        sched = _load_skip_schedule(str(skip_path), n_blocks)
        sched_src = f"skip={skip_path.name}"
    else:
        sched = _all_ops_schedule(n_blocks)
        sched_src = "all-ops SC"
    stats_req = count_sc_ops(sched)
    print(f"[2] SC schedule ({sched_src}, sc_prec={args.sc_prec})")
    for op in SC_OP_NAMES:
        on, tot = stats_req[op]
        print(f"    {op:<10s}  {on:3d}/{tot}")

    # --- Resume ---
    done_ids: set[int] = set()
    elapsed_prev = 0.0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        prev_sig = ck.get("config", {})
        if prev_sig != cfg_sig:
            print("[resume] config mismatch — refusing to resume.")
            print(f"    prev: {prev_sig}")
            print(f"    curr: {cfg_sig}")
            sys.exit(2)
        evaluator._predictions = list(ck.get("predictions", []))
        done_ids = {int(p["image_id"]) for p in evaluator._predictions}
        elapsed_prev = float(ck.get("elapsed_seconds", 0.0))
        print(f"[resume] {len(done_ids)} images already done "
              f"({elapsed_prev:.1f}s prior wall time)")
    elif ckpt_path.exists() and not args.resume:
        print(f"[warn] {ckpt_path} exists but --resume not set; "
              "starting fresh and OVERWRITING.")
        ckpt_path.unlink()

    remaining = [it for it in full_subset if int(it["image_id"]) not in done_ids]
    print(f"    remaining to evaluate: {len(remaining)}/{len(full_subset)}")

    # --- Patch model ---
    print(f"[3] Patching backbone (mlp_mode={args.sc_mlp_mode}, "
          f"proj_mode={args.sc_proj_mode})")
    stats_swapped = sc_patch_eva(
        model,
        sc_prec=args.sc_prec,
        stoc_len=args.stoc_len,
        sc_ops_per_block=sched,
        mlp_mode=args.sc_mlp_mode,
        proj_mode=args.sc_proj_mode,
    )
    print(f"    swapped: {stats_swapped}")

    # --- Inference loop with checkpointing ---
    t0 = time.time()  # used in elapsed accounting even if remaining is empty
    if remaining:
        sub_name = f"coco_2017_val_runord_{args.start_idx}_{args.n_eval}_rem{len(remaining)}"
        loader = _build_test_loader_for(remaining, "coco_2017_val", sub_name)

        from scmp_kernels.sc import det_kernel_tuning

        # Trap Ctrl-C: flush a checkpoint then re-raise.
        interrupted = {"flag": False}

        def _on_sigint(signum, frame):
            interrupted["flag"] = True
            print("\n[signal] SIGINT received; will checkpoint after current image",
                  flush=True)

        prev_sigint = signal.signal(signal.SIGINT, _on_sigint)

        def _flush(elapsed_now: float):
            state = {
                "config": cfg_sig,
                "schedule": {k: list(v) for k, v in sched.items()},
                "n_blocks": n_blocks,
                "predictions": list(evaluator._predictions),
                "image_ids_done": sorted({int(p["image_id"])
                                          for p in evaluator._predictions}),
                "elapsed_seconds": elapsed_prev + elapsed_now,
            }
            _atomic_save(state, ckpt_path)

        print(f"[4] Inference on {len(remaining)} images "
              f"(save_every={args.save_every})")
        t0 = time.time()  # reset for this-run timing
        n_done_this_run = 0
        try:
            with torch.no_grad(), det_kernel_tuning():
                from detectron2.evaluation.evaluator import inference_context
                with inference_context(model):
                    for inputs in loader:
                        outputs = model(inputs)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        evaluator.process(inputs, outputs)
                        n_done_this_run += 1
                        total_done = len(evaluator._predictions)
                        if n_done_this_run % args.save_every == 0:
                            _flush(time.time() - t0)
                            dt = time.time() - t0
                            ips = n_done_this_run / max(dt, 1e-6)
                            eta = (len(remaining) - n_done_this_run) / max(ips, 1e-6)
                            print(f"    [{total_done}/{len(full_subset)}]  "
                                  f"this-run {n_done_this_run}/{len(remaining)}  "
                                  f"{dt:.0f}s  {ips:.2f} img/s  "
                                  f"ETA {eta/60:.1f}min", flush=True)
                        if interrupted["flag"]:
                            break
        finally:
            # Save first, then restore the prior handler — if a SIGINT lands
            # mid-restore, we don't want it raising during the save.
            _flush(time.time() - t0)
            signal.signal(signal.SIGINT, prev_sigint)

        if interrupted["flag"]:
            print(f"[interrupted] saved checkpoint at {ckpt_path}. "
                  f"Re-run with --resume to continue.")
            sys.exit(130)
        print(f"    inference done in {time.time()-t0:.1f}s "
              f"(+{elapsed_prev:.1f}s prior)")
    else:
        print("[4] Nothing to evaluate; all images already done.")

    # --- Final COCO eval ---
    print("[5] Running COCO evaluation")
    results = evaluator.evaluate()
    out = {
        "config": cfg_sig,
        "n_blocks": n_blocks,
        "schedule": {k: list(v) for k, v in sched.items()},
        "schedule_source": sched_src,
        "swapped": stats_swapped,
        "n_predictions": len(evaluator._predictions),
        "elapsed_seconds_total": elapsed_prev + (time.time() - t0 if remaining else 0.0),
        **results,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    for task, m in results.items():
        print(f"    [{task}] AP={m.get('AP', 0):.2f}  "
              f"AP50={m.get('AP50', 0):.2f}  AP75={m.get('AP75', 0):.2f}")
    print(f"[OK] metrics -> {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()

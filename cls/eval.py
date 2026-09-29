"""Evaluate DINOv2 ViT-L/14 + linear head on ImageNet-1k val, FP vs SC Q@K^T."""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# cls/ holds eval/imagenet_parquet; vit_sc/model/ holds the SC integration,
# which imports the vendored scmp_kernels from vit_sc/kernels.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
from imagenet_parquet import ImageNetParquetVal
from model.sc_vit import (
    patch_model, parse_sc_ops, set_vit_timestep, load_adaptive_mp,
    read_mp_table_provenance, _resolve_mp_table_path,
    set_sc_defaults, set_qk_granularity,
    MPConfig, AdaptiveMPConfig, RangeMPConfig,
)
from model._bootstrap import kernels_provenance
from model.int_vit import patch_model_int, IntConfig


def build_transform(image_size=224):
    # DINOv2 eval recipe: resize 256 (shorter side), center crop 224,
    # ImageNet mean/std
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None, log_every=10,
             per_image_out=None):
    """Top-1/top-5 over ``loader``; optionally dump per-image correctness.

    ``per_image_out`` writes one uint8 per image (1 = top-1 correct) in loader
    order. At small n the marginal Wilson interval (+-2.6 pt at n=1000) is
    wider than the effects being measured, but every cell scores the SAME
    images in the SAME order, so the PAIRED difference is far tighter — and a
    paired test needs the per-image outcomes, which aggregates cannot recover.
    """
    model.eval()
    n = 0
    top1 = 0
    top5 = 0
    hits = []
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        _, pred5 = logits.topk(5, dim=1)
        correct = pred5.eq(y.unsqueeze(1))
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        if per_image_out is not None:
            hits.append(correct[:, 0].to(torch.uint8).cpu())
        n += y.numel()
        if (i + 1) % log_every == 0:
            dt = time.time() - t0
            print(f"  [{i+1}] top1={top1/n:.4f} top5={top5/n:.4f} "
                  f"({n} imgs, {dt:.1f}s, {n/dt:.1f} img/s)", flush=True)
        if max_batches is not None and (i + 1) >= max_batches:
            break
    dt = time.time() - t0
    if per_image_out is not None and hits:
        import numpy as _np
        _np.save(per_image_out, torch.cat(hits).numpy())
        print(f"[eval] per-image top-1 hits -> {per_image_out}", flush=True)
    return {
        "n": n,
        "top1": top1 / n,
        "top5": top5 / n,
        "elapsed_s": dt,
        "img_per_s": n / dt if dt > 0 else 0.0,
    }


def load_model(device):
    os.environ.setdefault("XFORMERS_DISABLED", "1")  # fall back to SDPA forward
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitl14_lc",
        source="github", trust_repo=True,
    )
    model = model.to(device).eval()
    return model


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_csv_ops(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _effective_stoc_len(args) -> int:
    """Uniform stream length the kernel will actually run.

    Mirrors ``model.sc_vit._resolve_stoc_len``: under ``--halve`` the grid
    ceiling is ``2**(sc_prec-1)``, so an unset (0) or oversized ``--stoc_len``
    resolves to the halved cap, NOT ``2**sc_prec``. Reporting the raw request
    instead would put a ``--halve 1`` run with no ``--stoc_len`` on the levels
    axis at 512 when the kernel ran 128 cycles / 256 levels.
    """
    cap = 2 ** (args.sc_prec - 1) if args.halve else 2 ** args.sc_prec
    return min(int(args.stoc_len) or cap, cap)


def _build_adaptive_cfg(args):
    """Build the shared AdaptiveMPConfig, or None when adaptive MP is off.

    The kernel's AdaptiveMPConfig is data-driven — a calibrated threshold table
    (``--adaptive_mp_table``) or fixed quantile targets
    (``--adaptive_mp_fractions``). The old closed-form ``alpha*progress+beta``
    threshold no longer exists in scmp_kernels, so one of the two is required.

    Memoized on ``args``: both ``_build_linear_mp_spec`` and
    ``_build_attn_mp_spec`` ask for it, and ONE shared instance is what we want
    anyway — dispatch is keyed by (operator, block) at call time, so a single
    config serves every op, and the table is parsed once.
    """
    if not bool(args.adaptive_mp):
        return None
    cached = getattr(args, "_adaptive_cfg", None)
    if cached is not None:
        return cached
    cfg = _build_adaptive_cfg_uncached(args)
    args._adaptive_cfg = cfg
    return cfg


def _build_adaptive_cfg_uncached(args):
    if args.adaptive_mp_table:
        # The table records the SC settings its thresholds were fitted on;
        # load_adaptive_mp raises on a mismatch rather than silently deploying
        # a halved ladder on an unhalved grid.
        return load_adaptive_mp(args.adaptive_mp_table,
                                expect_sc_prec=args.sc_prec,
                                expect_halve=bool(args.halve),
                                expect_chunk_d=args.chunk_d)
    levels = (_parse_csv_ints(args.adaptive_mp_levels)
              if args.adaptive_mp_levels else [])
    if not levels:
        return None
    fracs = (_parse_csv_floats(args.adaptive_mp_fractions)
             if args.adaptive_mp_fractions else None)
    if not fracs:
        raise SystemExit(
            "--adaptive_mp 1 needs either --adaptive_mp_table <wrapper.json> "
            "(calibrated thresholds) or --adaptive_mp_fractions (quantile "
            "targets). scmp_kernels removed the alpha/beta closed-form mode.")
    return AdaptiveMPConfig(
        stoc_len_levels=levels,
        target_fractions=fracs,
        enable_pruning=bool(args.adaptive_mp_enable_pruning))


def _build_attn_mp_spec(args) -> dict:
    """Assemble {"qk": MPConfig|AdaptiveMPConfig, "av": ...} from CLI.

    Priority: if ``--adaptive_mp 1`` and ``qk`` / ``av`` listed in
    ``--adaptive_mp_ops``, that op gets the shared ``AdaptiveMPConfig``.
    Otherwise falls back to the static ``--qk_mp_*`` / ``--av_mp_*`` flags.
    """
    spec = {}
    adaptive_ops = (set(_parse_csv_ops(args.adaptive_mp_ops))
                    if bool(args.adaptive_mp) else set())
    adaptive_cfg = _build_adaptive_cfg(args)

    # QK
    if "qk" in adaptive_ops and adaptive_cfg is not None:
        spec["qk"] = adaptive_cfg
    elif args.qk_mp_levels:
        spec["qk"] = MPConfig(
            stoc_len_levels=_parse_csv_ints(args.qk_mp_levels),
            level_fractions=(_parse_csv_floats(args.qk_mp_fractions)
                             if args.qk_mp_fractions else None),
        )
    # AV
    if "av" in adaptive_ops and adaptive_cfg is not None:
        spec["av"] = adaptive_cfg
    elif args.av_mp_levels:
        spec["av"] = MPConfig(
            stoc_len_levels=_parse_csv_ints(args.av_mp_levels),
            level_fractions=(_parse_csv_floats(args.av_mp_fractions)
                             if args.av_mp_fractions else None),
        )
    return spec


def _build_linear_mp_spec(args) -> dict:
    """Assemble {op_name: {"fixed": MPConfig, "adaptive": AdaptiveMPConfig,
    "range": RangeMPConfig, ...}} from CLI flags. Returns {} if nothing
    is requested.
    """
    fixed_levels = _parse_csv_ints(args.mp_levels) if args.mp_levels else []
    fixed_fracs = _parse_csv_floats(args.mp_fractions) if args.mp_fractions else None
    fixed_ops = _parse_csv_ops(args.mp_ops)
    range_on = bool(args.range_mp)
    range_levels = _parse_csv_ints(args.range_mp_levels) if range_on else []
    range_ops = _parse_csv_ops(args.range_mp_ops) or (fixed_ops if range_on else [])
    adaptive_on = bool(args.adaptive_mp)
    _raw_adaptive_ops = (_parse_csv_ops(args.adaptive_mp_ops)
                         or (fixed_ops if adaptive_on else []))
    # qk/av are routed through _build_attn_mp_spec, not here.
    adaptive_ops = [op for op in _raw_adaptive_ops if op not in ("qk", "av")]

    mp_fixed = (MPConfig(stoc_len_levels=fixed_levels,
                         level_fractions=fixed_fracs)
                if fixed_levels else None)
    mp_range = (RangeMPConfig(stoc_len_levels=range_levels,
                              base_threshold=args.range_mp_threshold)
                if range_on else None)
    mp_adaptive = _build_adaptive_cfg(args)

    if not mp_fixed and not mp_range and not mp_adaptive:
        return {}

    spec: dict = {}
    if mp_fixed:
        for op in fixed_ops:
            spec.setdefault(op, {})["fixed"] = mp_fixed
    if mp_adaptive:
        for op in adaptive_ops:
            spec.setdefault(op, {})["adaptive"] = mp_adaptive
    if mp_range:
        for op in range_ops:
            entry = spec.setdefault(op, {})
            entry["range"] = mp_range
            entry["range_group_size"] = args.range_mp_group_size
    return spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data",
                   help="Path to ImageNet-1k parquet shards.")
    p.add_argument("--mode", choices=["fp", "sc", "int"], required=True)
    p.add_argument("--int_bits", type=int, default=8,
                   help="INT-b fixed-point baseline width (--mode int). "
                        "W=A. scmp_llm's RTN protocol; scale scope matched to "
                        "SC's, so INT-b and halved SC at 2**bits effective "
                        "levels are directly comparable.")
    p.add_argument("--int_sym", type=int, default=1,
                   help="1 = symmetric (matches bipolar SC), 0 = asymmetric "
                        "with zero point. scmp_llm found asymm wins below "
                        "W6A6.")
    p.add_argument("--int_chunk", type=int, default=128,
                   help="Input-dim chunk for INT scales. Must equal the SC "
                        "run's --chunk_d for a matched comparison.")
    p.add_argument("--int_attn_scope", choices=["per_head", "per_row"],
                   default="per_head",
                   help="QK operand scale scope. per_head matches SC's "
                        "granularity (default); per_row reproduces scmp_llm's "
                        "literal behaviour, which is finer than its own SC.")
    p.add_argument("--sc_prec", type=int, default=8)
    p.add_argument("--stoc_len", type=int, default=0,
                   help="Uniform SC stream length (cycles). 0 = 2**sc_prec. "
                        "This is the compute-budget knob; sc_prec only sets "
                        "the quantization grid.")
    p.add_argument("--chunk_d", type=int, default=128,
                   help="Inner-dim chunking for SC linears. Keeps the "
                        "cum_indicator table in L2 for wide MLPs (ViT-L fc2 "
                        "has D_in=4096). 0 disables.")
    p.add_argument("--halve", type=int, default=None,
                   help="uSystolic/HUB bipolar cycle halving (stream runs at "
                        "2**(sc_prec-1)). Unset: inherited from "
                        "--adaptive_mp_table when one is given (its ladder "
                        "rungs live in halved space), else OFF — the repo "
                        "default, so every non-MP invocation keeps its meaning. "
                        "Passing it explicitly against a table that disagrees "
                        "is an error, not a silent reinterpretation.")
    p.add_argument("--sc_qk", type=int, default=1)
    p.add_argument("--sc_av", type=int, default=0)
    p.add_argument("--sc_mlp", type=int, default=0)
    p.add_argument("--mlp_skip_last_k", type=int, default=0)
    p.add_argument("--mlp_skip_first_k", type=int, default=0)
    p.add_argument("--sc_qkv_proj", type=int, default=0)
    p.add_argument("--sc_out_proj", type=int, default=0)
    p.add_argument("--sc_mlp_mode", choices=["bipolar", "unipolar"],
                   default="bipolar",
                   help="MLP SC quantization: bipolar=symmetric |max|, "
                        "unipolar=asymmetric zero-point (better for post-GELU)")
    p.add_argument("--sc_proj_mode", choices=["bipolar", "unipolar"],
                   default="bipolar",
                   help="qkv/out proj SC quant mode (default bipolar).")
    p.add_argument("--mlp_sc_spec_json", default="",
                   help="Path to JSON file with list of [block_idx, fc_name] "
                        "pairs specifying exactly which MLP linears go SC. "
                        "Overrides --mlp_skip_first_k / --mlp_skip_last_k. "
                        "Sets --sc_mlp 1 implicitly.")
    p.add_argument("--sc_ops", default="",
                   help="Comma-separated op names from "
                        "{mlp_fc1,mlp_fc2,qk,av,proj} OR a 5-element 0/1 "
                        "vector (same order, e.g. '1,0,0,1,0'). Overrides "
                        "per-op --sc_qk/--sc_av/etc flags.")
    # ---- Mixed precision (ported from ../scmp_llm) ----
    p.add_argument("--mp_levels", default="",
                   help="Comma-separated stoc_len levels (descending), e.g. "
                        "'256,128'. Enables fixed per-input-row MP on --mp_ops.")
    p.add_argument("--mp_fractions", default="",
                   help="Comma-separated fractions per level (sums to 1). "
                        "Empty = equal split.")
    p.add_argument("--mp_ops", default="",
                   help="Comma-separated linear ops that get fixed MP. "
                        "Valid: mlp_fc1,mlp_fc2,qkv_proj,out_proj,proj.")
    p.add_argument("--range_mp", type=int, default=0,
                   help="Enable range-based per-weight-group MP on --range_mp_ops.")
    p.add_argument("--range_mp_levels", default="256,128",
                   help="Comma-separated stoc_len levels (descending) for range MP.")
    p.add_argument("--range_mp_threshold", type=float, default=0.3,
                   help="Normalized range threshold. Higher = more groups "
                        "get lower precision.")
    p.add_argument("--range_mp_ops", default="",
                   help="Comma-separated linear ops that get range MP. "
                        "Empty ⇒ same as --mp_ops.")
    p.add_argument("--range_mp_group_size", type=int, default=0,
                   help="Output-rows per range-MP group. 0 = per-tensor.")
    # adaptive (timestep-aware) mixed precision on linear ops
    p.add_argument("--adaptive_mp", type=int, default=0,
                   help="Enable adaptive per-input-row MP (α·progress + β). "
                        "ViT has no native timestep, so (t, T) are driven via "
                        "--vit_timestep / --vit_total_timesteps.")
    p.add_argument("--adaptive_mp_levels", default="",
                   help="Comma-separated stoc_len levels (descending), e.g. "
                        "'256,64,0'. The '0' level enables pruning (requires "
                        "--adaptive_mp_enable_pruning=1). Ignored when "
                        "--adaptive_mp_table is given (the table carries them).")
    p.add_argument("--adaptive_mp_table", default="",
                   help="Path to a calibrated MP wrapper JSON "
                        '{"stoc_len_levels": [...], "threshold_table_path": '
                        '"<table>.json"} from calibrate_mp_thresholds.py. '
                        "This is the scmp_llm/scmp_cnn-compatible path and "
                        "carries the per-op ρ-selected dispatch metrics.")
    p.add_argument("--adaptive_mp_fractions", default="",
                   help="Quantile targets per level (sums to 1) — the "
                        "table-free adaptive path. Top frac[0] rows get "
                        "levels[0], etc.")
    p.add_argument("--adaptive_mp_enable_pruning", type=int, default=1,
                   help="Allow stoc_len=0 (row pruning).")
    p.add_argument("--adaptive_mp_ops", default="",
                   help="Comma-separated linear ops that get adaptive MP. "
                        "Empty ⇒ same as --mp_ops. Valid: "
                        "mlp_fc1,mlp_fc2,qkv_proj,out_proj,proj.")
    p.add_argument("--vit_timestep", type=int, default=0,
                   help="External timestep t for adaptive MP (ViT has no "
                        "native timestep). Reports at this (t, T) pair.")
    p.add_argument("--vit_total_timesteps", type=int, default=1,
                   help="External total timesteps T. (0,1) ⇒ progress=0 "
                        "(most conservative, equivalent to threshold=β).")
    # per-head / per-row MP on QK and AV (scmp_llm-compatible)
    p.add_argument("--protect_from_table", default="",
                   help="Apply the protected input channels recorded in a "
                        "calibration table WITHOUT per-row MP: uniform SC "
                        "everywhere, except those channels at "
                        "--protect_stoc_len. The control that separates "
                        "'protection is an MP feature' from 'protection is a "
                        "general SC fix'.")
    p.add_argument("--protect_stoc_len", type=int, default=0,
                   help="Stream length for protected channels under "
                        "--protect_from_table. 0 = the table's own value.")
    p.add_argument("--hybrid_config_json", default="",
                   help="Hybrid SC/INT schedule: {'schedule': {op: [backend per "
                        "block]}, 'int_bits': N, ...} with backends sc|int|fp. "
                        "Routes the listed (op, block) groups to INT RTN while "
                        "the rest stay on SC. Composes with --adaptive_mp.")
    p.add_argument("--exclude_calib_images", type=int, default=0,
                   help="1: drop the images --adaptive_mp_table was calibrated "
                        "on before drawing the eval subset. Needed for a "
                        "full-val run, where the calibrator's own "
                        "--eval_exclude_n cannot express the disjointness.")
    p.add_argument("--qk_granularity", default=None,
                   choices=["per_head", "per_row"],
                   help="Quantization scope for Q@K^T, and hence the qk MP "
                        "dispatch unit. Unset: inherit from --adaptive_mp_table "
                        "when it records one, else 'per_head' (the legacy "
                        "vit_sc setting). 'per_row' gives each QUERY ROW its "
                        "own range — the finer unit, and what makes per-row qk "
                        "MP exact.")
    p.add_argument("--qk_mp_levels", default="",
                   help="Per-head fixed MP on QK (metric = per-head |Q|.amax). "
                        "Comma-separated stoc_len levels, e.g. '256,128'.")
    p.add_argument("--qk_mp_fractions", default="",
                   help="Comma-separated head fractions per level.")
    p.add_argument("--av_mp_levels", default="",
                   help="Per-attn-row fixed MP on AV (metric = attn_row.amax). "
                        "Comma-separated stoc_len levels.")
    p.add_argument("--av_mp_fractions", default="",
                   help="Comma-separated row fractions per level.")
    p.add_argument("--sc_ops_per_block_json", default="",
                   help="Path to JSON file with per-op per-block vectors: "
                        '{"mlp_fc1":[0,1,...],"mlp_fc2":[...],"qk":[...],'
                        '"av":[...],"proj":[...]}. Each list has length '
                        "n_blocks (24 for DINOv2 ViT-L/14). Highest priority.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_images", type=int, default=0, help="0 = full val")
    p.add_argument("--skip_images", type=int, default=0,
                   help="Offset into the seeded shuffle before taking "
                        "--max_images. 0 (default) reproduces the historical "
                        "subset exactly. Use a non-zero value to score a window "
                        "DISJOINT from the reported first-N subset, e.g. for "
                        "config search that must not touch the report set.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", default="")
    args = p.parse_args()

    # Resolve --halve BEFORE anything reads it: it selects the quantization grid
    # for the whole run, gates the MP table provenance check, and feeds
    # set_sc_defaults(). Unset + an MP table => adopt the table's setting (the
    # only value at which its thresholds are valid); unset + no table => OFF.
    if args.halve is None:
        _prov = (read_mp_table_provenance(args.adaptive_mp_table)
                 if (args.adaptive_mp and args.adaptive_mp_table) else {})
        _inherited = _prov.get("halve_bipolar_stoc_len")
        args.halve = int(bool(_inherited)) if _inherited is not None else 0
        if _inherited is not None:
            print(f"[sc] --halve unset -> {bool(_inherited)} inherited from "
                  f"{args.adaptive_mp_table}", flush=True)

    # Q@K^T quantization scope. An MP table calibrated at one granularity does
    # not transfer to the other (different dispatch unit, different metric
    # pool, different MACs/unit in the budget), so a table that records one is
    # authoritative and disagreeing with it explicitly RAISES — the same
    # contract --halve already has.
    _qk_prov = (read_mp_table_provenance(args.adaptive_mp_table)
                if (args.adaptive_mp and args.adaptive_mp_table) else {})
    _qk_tab = _qk_prov.get("qk_granularity")
    if _qk_tab is not None:
        if args.qk_granularity is None:
            args.qk_granularity = _qk_tab
            print(f"[sc] --qk_granularity unset -> {_qk_tab} inherited from "
                  f"{args.adaptive_mp_table}", flush=True)
        elif args.qk_granularity != _qk_tab:
            raise SystemExit(
                f"[sc] MP table was calibrated with qk_granularity={_qk_tab!r} "
                f"but this run deploys {args.qk_granularity!r}. The qk MP unit "
                f"and its budget price differ between the two — recalibrate, or "
                f"drop --qk_granularity to inherit.")
    if args.qk_granularity is None:
        args.qk_granularity = "per_head"
    set_qk_granularity(args.qk_granularity)

    seed_all(args.seed)
    device = torch.device("cuda")

    print(f"[env] torch {torch.__version__}, cuda {torch.version.cuda}, "
          f"device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[env] scmp_kernels {kernels_provenance()}", flush=True)

    print(f"[data] loading parquet val from {args.data_root}", flush=True)
    ds = ImageNetParquetVal(args.data_root, transform=build_transform(224))
    print(f"[data] {len(ds)} images", flush=True)
    # Drop the images the MP thresholds were fitted on. The calibrator's own
    # --eval_exclude_n only keeps the two sets disjoint while the eval subset
    # is the first N of a known shuffle; on a FULL-val run it is not, so the
    # exclusion has to run from this side using the indices the table records.
    keep = list(range(len(ds)))
    if args.exclude_calib_images and args.adaptive_mp_table:
        import json as _json
        _w, _t, _s = _resolve_mp_table_path(args.adaptive_mp_table)
        if _t is not None:
            with open(_t) as _f:
                _ci = set(_json.load(_f).get("calib_indices") or [])
            if _ci:
                keep = [i for i in keep if i not in _ci]
                print(f"[data] excluded {len(_ci)} calibration images -> "
                      f"{len(keep)} eligible", flush=True)
            else:
                raise SystemExit(
                    f"[data] --exclude_calib_images was requested but "
                    f"{_t.name} records no calib_indices (calibrated before "
                    f"they were tracked). Re-calibrate, or drop the flag and "
                    f"state the overlap.")
    if args.max_images and args.max_images < len(keep):
        random.Random(args.seed).shuffle(keep)
        # --skip_images opens a window PAST the reported subset. The reported
        # eval set is shuffle(seed)[:N], so shuffle(seed)[:1000] is a strict
        # SUBSET of shuffle(seed)[:10000] -- tuning a config on the former and
        # reporting it on the latter is tuning on the report set. A search at
        # --skip_images N --max_images M instead scores shuffle(seed)[N:N+M],
        # which is disjoint from the first N by construction.
        keep = keep[args.skip_images: args.skip_images + args.max_images]
    # Fingerprint the EXACT image list, in loader order. Every paired test in
    # this repo compares per-image hit vectors POSITIONALLY, so two cells scored
    # on different images -- or the same images in a different order -- produce a
    # silently meaningless delta rather than an error. The list is easy to
    # perturb without noticing: --exclude_calib_images edits `keep` BEFORE the
    # shuffle, so one cell using it and another not shifts every index that
    # follows. cls/check_eval_sets.py refuses to report across mismatched
    # fingerprints.
    import hashlib as _hashlib
    _eval_sha = _hashlib.sha256(
        ",".join(map(str, keep)).encode()).hexdigest()[:16]
    print(f"[data] eval-set fingerprint {_eval_sha} "
          f"({len(keep)} images, seed={args.seed}, skip={args.skip_images})",
          flush=True)
    if len(keep) < len(ds):
        ds = Subset(ds, keep)
        print(f"[data] subset -> {len(ds)} images", flush=True)

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False,
    )

    print("[model] loading DINOv2 ViT-L/14 + linear head", flush=True)
    model = load_model(device)

    if args.mode == "int":
        cfg_int = IntConfig(args.int_bits, sym=bool(args.int_sym),
                            chunk_size=args.int_chunk,
                            attn_scope=args.int_attn_scope)
        stats = patch_model_int(model, bits=args.int_bits,
                                ops=(args.sc_ops or None),
                                sym=bool(args.int_sym),
                                chunk_size=args.int_chunk,
                                attn_scope=args.int_attn_scope)
        print(f"[int] {cfg_int.describe()}", flush=True)
        print(f"[int] ops={sorted(parse_sc_ops(args.sc_ops)) if args.sc_ops else 'all'} "
              f"patched={stats}", flush=True)

    if args.mode == "sc":
        mlp_spec = None
        sc_mlp_flag = bool(args.sc_mlp)
        if args.mlp_sc_spec_json:
            import json as _json
            with open(args.mlp_sc_spec_json) as _f:
                mlp_spec = [tuple(p) for p in _json.load(_f)]
            sc_mlp_flag = True
            print(f"[sc] mlp_sc_spec: {len(mlp_spec)} linears from "
                  f"{args.mlp_sc_spec_json}", flush=True)
        sc_ops_arg = args.sc_ops or None
        if sc_ops_arg:
            print(f"[sc] sc_ops={sorted(parse_sc_ops(sc_ops_arg))}", flush=True)
        per_block_arg = None
        if args.sc_ops_per_block_json:
            import json as _json
            with open(args.sc_ops_per_block_json) as _f:
                per_block_arg = _json.load(_f)
            print(f"[sc] sc_ops_per_block from {args.sc_ops_per_block_json}",
                  flush=True)

        linear_mp_spec = _build_linear_mp_spec(args)
        if linear_mp_spec:
            print(f"[sc] linear_mp_spec: {sorted(linear_mp_spec)}", flush=True)
        attn_mp_spec = _build_attn_mp_spec(args)
        if attn_mp_spec:
            print(f"[sc] attn_mp_spec: {sorted(attn_mp_spec)}", flush=True)
        if args.adaptive_mp:
            set_vit_timestep(args.vit_timestep, args.vit_total_timesteps)
            src = (args.adaptive_mp_table or
                   f"quantile fractions [{args.adaptive_mp_fractions}]")
            print(f"[sc] adaptive_mp t={args.vit_timestep}/"
                  f"T={args.vit_total_timesteps} source={src}", flush=True)

        # Hybrid SC/INT schedule: route the SC-fragile (op, block) groups to
        # INT instead of SC, leaving everything else on the SC budget. The
        # masked groups do NOT shrink the SC average — they leave it — so a
        # hybrid cell is only comparable to another cell carrying the SAME
        # mask, and is priced on the energy axis rather than iso-total-compute.
        hybrid_sched, hybrid_int_cfg = None, None
        if args.hybrid_config_json:
            import json as _json
            with open(args.hybrid_config_json) as _f:
                _hc = _json.load(_f)
            hybrid_sched = _hc["schedule"]
            hybrid_int_cfg = IntConfig(
                int(_hc.get("int_bits", 7)),
                sym=bool(_hc.get("int_sym", False)),
                chunk_size=int(_hc.get("chunk_size", args.int_chunk)),
                attn_scope=_hc.get("attn_scope", args.int_attn_scope))
            _n_int = sum(sum(1 for b in v if b == "int")
                         for v in hybrid_sched.values())
            print(f"[hybrid] {args.hybrid_config_json}: {_n_int} groups -> INT"
                  f" ({hybrid_int_cfg.tag()}), rest SC", flush=True)

        # Protected channels WITHOUT MP: reads the protected set straight out of
        # a calibration table so the control shares the EXACT channels the MP
        # cell protects, and differs from it only in whether per-row dispatch
        # runs at all.
        protect_arg = None
        if args.protect_from_table:
            import json as _json
            with open(args.protect_from_table) as _f:
                _pcj = _json.load(_f)
            _pc = _pcj.get("protected_channels")
            if not _pc:
                raise SystemExit(
                    f"[sc] {args.protect_from_table} carries no "
                    f"protected_channels block — calibrate with "
                    f"--protect-channel-frac > 0, or drop --protect_from_table.")
            _idx = {k: v for k, v in _pc["indices"].items()}
            protect_arg = (_idx, int(args.protect_stoc_len or _pc["stoc_len"]))
            print(f"[sc] protected channels (no MP) from "
                  f"{args.protect_from_table}: {len(_idx)} modules at "
                  f"stoc_len={protect_arg[1]}", flush=True)

        set_sc_defaults(halve_bipolar_stoc_len=bool(args.halve))
        stats = patch_model(
            model, sc_prec=args.sc_prec,
            sc_qk=bool(args.sc_qk), sc_av=bool(args.sc_av), sc_mlp=sc_mlp_flag,
            stoc_len=(args.stoc_len or None),
            chunk_d=args.chunk_d,
            mlp_skip_last_k=args.mlp_skip_last_k,
            mlp_skip_first_k=args.mlp_skip_first_k,
            sc_qkv_proj=bool(args.sc_qkv_proj),
            sc_out_proj=bool(args.sc_out_proj),
            sc_mlp_mode=args.sc_mlp_mode,
            sc_proj_mode=args.sc_proj_mode,
            mlp_sc_spec=mlp_spec,
            sc_ops=sc_ops_arg,
            sc_ops_per_block=per_block_arg,
            linear_mp_spec=linear_mp_spec or None,
            attn_mp_spec=attn_mp_spec or None,
            hybrid_schedule=hybrid_sched,
            int_cfg=hybrid_int_cfg,
            protect_channels=protect_arg,
        )
        # Report the RESOLVED surface, not the raw flags: --sc_ops (and
        # --sc_ops_per_block_json) override the per-op booleans inside
        # patch_model, so echoing args.sc_av/args.sc_mlp here misreports
        # which ops actually went stochastic.
        eff = parse_sc_ops(sc_ops_arg) if sc_ops_arg else {
            name for name, on in (("qk", args.sc_qk), ("av", args.sc_av),
                                  ("proj", args.sc_qkv_proj or args.sc_out_proj))
            if on}
        if sc_ops_arg is None and sc_mlp_flag:
            eff |= {"mlp_fc1", "mlp_fc2"}
        if per_block_arg is not None:
            eff = {k for k, v in per_block_arg.items() if any(v)}
        print(f"[sc] sc_prec={args.sc_prec} "
              f"stoc_len={_effective_stoc_len(args)} "
              f"halve={bool(args.halve)} chunk_d={args.chunk_d} "
              f"ops={sorted(eff)} "
              f"mlp_mode={args.sc_mlp_mode} proj_mode={args.sc_proj_mode} "
              f"skip_first={args.mlp_skip_first_k} skip_last={args.mlp_skip_last_k} "
              f"patched={stats}",
              flush=True)

    print(f"[eval] mode={args.mode} bs={args.batch_size} "
          f"workers={args.workers}", flush=True)
    _pi = None
    if args.out_json:
        _pi = str(Path(args.out_json).with_name(
            Path(args.out_json).stem + "_hits.npy"))
    res = evaluate(model, loader, device, per_image_out=_pi)
    res["mode"] = args.mode
    # Self-describing enough for a collector to place the cell on the figure
    # without a side manifest: which arm it belongs to, and the two settings
    # (qk scope, hybrid mask) that make a cell comparable only to cells sharing
    # them.
    res["mp"] = bool(args.adaptive_mp) if args.mode == "sc" else None
    res["mp_table"] = (args.adaptive_mp_table or None) if args.mode == "sc" else None
    res["qk_granularity"] = args.qk_granularity if args.mode == "sc" else None
    res["hybrid_config"] = args.hybrid_config_json or None
    res["hybrid_frac"] = None
    if args.hybrid_config_json:
        import json as _json
        with open(args.hybrid_config_json) as _f:
            res["hybrid_frac"] = _json.load(_f).get("frac")
    res["sc_prec"] = args.sc_prec if args.mode == "sc" else None
    res["stoc_len"] = _effective_stoc_len(args) if args.mode == "sc" else None
    res["halve"] = bool(args.halve) if args.mode == "sc" else None
    res["int_bits"] = args.int_bits if args.mode == "int" else None
    res["int_sym"] = bool(args.int_sym) if args.mode == "int" else None
    # Comparison axis shared by INT and SC: number of representable levels.
    # Halved bipolar SC carries 2x its stream length (wu-hpca2022), so a
    # 128-cycle halved stream == int8's 256 levels.
    if args.mode == "int":
        res["levels"] = 1 << args.int_bits
    elif args.mode == "sc":
        sl = _effective_stoc_len(args)
        res["levels"] = 2 * sl if args.halve else sl
    # Which images this number is about. Cells whose fingerprints differ must
    # never be differenced against each other -- see the [data] note above.
    res["eval_set_sha"] = _eval_sha
    res["eval_seed"] = args.seed
    res["eval_skip"] = args.skip_images
    print("[result]", res, flush=True)

    if args.out_json:
        import json
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[result] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()

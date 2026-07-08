#!/usr/bin/env python
"""Det-side MP-budget swap search for EVA-01 ViTDet + COCO.

Thin driver around ``sc_integration.mp_search`` (the model-agnostic
algorithm core). Provides a :class:`DetBackend` that wires det's
schedule loaders, MAC tables, model + loader builders, and per-block
patching adapter into the shared search loop.

Usage::

    python det/experiments/mp_budget_swap_search.py \\
        --sc_config skip_worst30 \\
        --target_main_sl 192 \\
        --search_ops mlp_fc1,mlp_fc2 \\
        --fixed_ops qk=256,av=256,qkv_proj=192,out_proj=192 \\
        --levels 64,96,128,192,256 \\
        --n_search 16 --max_iters 4 \\
        --proxy comp_residual \\
        --sc_prec 8 \\
        --out_json sensitivity/sl_maps/avg192_mp.json

Equal-MAC DP requirement (same as cls): all searchable (op, block) units
must share the same MAC. On EVA-ViTDet at 1280²/patch16, ``mlp_fc1`` and
``mlp_fc2`` share 55.4 G MACs/block; ``qkv_proj`` (38 G) and ``out_proj``
(12.7 G) differ. Default ``--search_ops mlp_fc1,mlp_fc2`` and pin the
projs / qk / av to fixed levels via ``--fixed_ops``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_DET = _HERE.parent
_REPO = _DET.parent
for p in (_DET, _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_common import load_model_and_loader  # noqa: E402
from sc_patch import sc_patch_eva, SC_OP_NAMES  # noqa: E402
from sc_patch.sc_matmul import SCMatMul  # noqa: E402
from sc_integration.sc_linear import SCLinear  # noqa: E402
from sc_integration.mp_linear import MPConfig  # noqa: E402
from sc_integration import mp_search as ms  # noqa: E402


# --------------------------------------------------------------------------
# EVA-01 ViTDet MAC table.
#
# embed_dim=1408, num_heads=16, head_dim=88, mlp_hidden=6144,
# patch_size=16, img_size=1280, window_size=16 patches.
# Linear ops scale with #tokens (6400), independent of attention type
# (window vs global) — windowing reshapes the sequence but the per-token
# matmul shape is the same. qk/av MACs differ by attention type:
#   * window: 25 windows × 16h × 256 × 88 × 256 = 562 M per block
#   * global:           16h × 6400 × 88 × 6400  = 57.7 G per block
# --------------------------------------------------------------------------
_EVA_TOKENS = 80 * 80                        # 6400
_EVA_DIM = 1408
_EVA_HEADS = 16
_EVA_HEAD_DIM = _EVA_DIM // _EVA_HEADS       # 88
_EVA_MLP_HIDDEN = 6144

_EVA_LINEAR_MACS = {
    "mlp_fc1":  _EVA_TOKENS * _EVA_DIM * _EVA_MLP_HIDDEN,
    "mlp_fc2":  _EVA_TOKENS * _EVA_MLP_HIDDEN * _EVA_DIM,
    "qkv_proj": _EVA_TOKENS * _EVA_DIM * (3 * _EVA_DIM),
    "out_proj": _EVA_TOKENS * _EVA_DIM * _EVA_DIM,
}
# qk/av per block depend on attention type — see DetBackend.op_macs.
_EVA_QK_AV_GLOBAL = _EVA_HEADS * _EVA_TOKENS * _EVA_HEAD_DIM * _EVA_TOKENS
_EVA_QK_AV_WINDOW = 25 * _EVA_HEADS * 256 * _EVA_HEAD_DIM * 256
_EVA_COMP_MACS = _EVA_TOKENS * _EVA_DIM * _EVA_DIM       # comp matmul per block

# EVA-01 ViTDet config: global attention every 4th block (indices 3, 7, ..., 39).
# The list is the COMPLEMENT of window_block_indexes from
# third_party/QwT-SC/QwT-det-RepQ-ViT/eva1/eva_det/projects/ViTDet/configs/COCO/
# cascade_mask_rcnn_vitdet_eva.py.
_EVA_GLOBAL_BLOCKS = frozenset({3, 7, 11, 15, 19, 23, 27, 31, 35, 39})


_FINE_KS = {
    "skip_worst50": 50, "skip_worst40": 40, "skip_worst30": 30,
    "skip_worst20": 20, "skip_worst10": 10, "all_ops": 0,
}


def _load_skip_schedule(k: int, sc_prec: int, n_blocks: int = 40) -> dict[str, list[int]]:
    """Resolve det's existing skip_worst_K_int{6,7}.json schedules."""
    sens_p = 7 if sc_prec >= 7 else 6
    path = _DET / "sensitivity" / "skip" / f"skip_worst_{k}_int{sens_p}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing schedule {path}. Run det/sensitivity/build_skip.py for "
            f"sc_prec={sc_prec} (or pin to sc_prec ∈ {{6, 7}}).")
    with open(path) as f:
        spec = json.load(f)
    out: dict[str, list[int]] = {op: [0] * n_blocks for op in SC_OP_NAMES}
    for op, vec in spec.items():
        if op not in SC_OP_NAMES:
            continue
        out[op] = [int(x) for x in vec]
    return out


def _load_sensitivity_grid(sc_prec: int) -> dict[tuple[str, int], float]:
    """Load det's sensitivity matrix (bbox_ap_drop) as the per-(op, block)
    score. Falls back to int7 if the requested sc_prec wasn't swept."""
    sens_p = 7 if sc_prec >= 7 else 6
    path = _DET / "sensitivity" / f"sensitivity_int{sens_p}_n100.json"
    with open(path) as f:
        data = json.load(f)
    return {
        (r["op"], int(r["block"])): float(r.get("bbox_ap_drop", 0.0))
        for r in data["grid"]
    }


# --------------------------------------------------------------------------
# det per-block patch adapter
# --------------------------------------------------------------------------


def _patch_block_with_spec(block: nn.Module, sc_prec: int,
                           block_spec: dict) -> nn.Module:
    """Return a deepcopy of ``block`` with SC submodules installed per
    ``block_spec`` (dict[op_name, entry]). Mirror of
    cls/mp_budget_swap_search:patch_block_with_spec but for det's EVA
    Attention layout (matmul1/matmul2 swappable to SCMatMul, qkv/proj
    swappable to SCLinear)."""
    import copy as _copy
    blk = _copy.deepcopy(block)

    attn = getattr(blk, "attn", None)
    mlp = getattr(blk, "mlp", None)

    # Attn matmuls (qk, av)
    if attn is not None:
        qk_mp = ms.entry_to_mpconfig(block_spec.get("qk", 0))
        av_mp = ms.entry_to_mpconfig(block_spec.get("av", 0))
        if qk_mp is not None:
            m1 = getattr(attn, "matmul1", None)
            if m1 is not None and not isinstance(m1, SCMatMul):
                attn.matmul1 = SCMatMul(
                    sc_prec=sc_prec, mode="bipolar", op="qk",
                    mp_cfg=qk_mp,
                )
        if av_mp is not None:
            m2 = getattr(attn, "matmul2", None)
            if m2 is not None and not isinstance(m2, SCMatMul):
                attn.matmul2 = SCMatMul(
                    sc_prec=sc_prec, mode="bipolar", op="av",
                    mp_cfg=av_mp,
                )

        # qkv_proj, out_proj (linear)
        qkv_mp = ms.entry_to_mpconfig(block_spec.get("qkv_proj", 0))
        if qkv_mp is not None and isinstance(getattr(attn, "qkv", None), nn.Linear):
            attn.qkv = SCLinear(attn.qkv, sc_prec, mode="bipolar",
                                mp_cfg=qkv_mp, operator="qkv_proj")
        outp_mp = ms.entry_to_mpconfig(block_spec.get("out_proj", 0))
        if outp_mp is not None and isinstance(getattr(attn, "proj", None), nn.Linear):
            attn.proj = SCLinear(attn.proj, sc_prec, mode="bipolar",
                                 mp_cfg=outp_mp, operator="out_proj")

    # mlp.fc1 / mlp.fc2 (linear)
    if mlp is not None:
        fc1_mp = ms.entry_to_mpconfig(block_spec.get("mlp_fc1", 0))
        if fc1_mp is not None and isinstance(getattr(mlp, "fc1", None), nn.Linear):
            mlp.fc1 = SCLinear(mlp.fc1, sc_prec, mode="bipolar",
                               mp_cfg=fc1_mp, operator="mlp_fc1")
        fc2_mp = ms.entry_to_mpconfig(block_spec.get("mlp_fc2", 0))
        if fc2_mp is not None and isinstance(getattr(mlp, "fc2", None), nn.Linear):
            mlp.fc2 = SCLinear(mlp.fc2, sc_prec, mode="bipolar",
                               mp_cfg=fc2_mp, operator="mlp_fc2")

    return blk


def _eva_blocks(model: nn.Module) -> nn.ModuleList:
    """Resolve the EVA ViT's block ModuleList. Accepts either the full
    Detectron2 model or just the EVA ViT (``backbone.net`` view)."""
    if hasattr(model, "backbone") and hasattr(model.backbone, "net"):
        return model.backbone.net.blocks
    return model.blocks


def _patch_model_with_sl_map(model: nn.Module, sc_prec: int,
                             sl_map: dict) -> dict:
    """Patch every block of ``model`` according to ``sl_map``. ``model`` may
    be either the full Detectron2 model or the EVA ViT view; both expose
    a ``blocks`` ModuleList we can mutate in place."""
    blocks = _eva_blocks(model)
    nb = len(blocks)
    stats: dict[str, int] = {op: 0 for op in SC_OP_NAMES}
    for bi in range(nb):
        spec = ms.build_current_spec(sl_map, SC_OP_NAMES, bi)
        new_blk = _patch_block_with_spec(blocks[bi], sc_prec, spec)
        blocks[bi] = new_blk
        for op in SC_OP_NAMES:
            entry = spec.get(op, 0)
            mp = ms.entry_to_mpconfig(entry)
            if mp is not None:
                stats[op] += 1
    return stats


# --------------------------------------------------------------------------
# DetBackend — concrete adapter implementing MPSearchBackend
# --------------------------------------------------------------------------


class DetBackend(ms.MPSearchBackend):
    op_names = SC_OP_NAMES
    n_blocks = 40

    def __init__(self, *, sc_prec: int, n_search: int, batch_size: int,
                 workers: int, sc_config: str,
                 d2_datasets: str = "", ckpt: str = "", size: int = 0):
        self._sc_prec = sc_prec
        self._n_search = n_search
        self._batch_size = batch_size
        self._workers = workers
        self._sc_config = sc_config
        self._d2_kw = dict(d2_datasets=d2_datasets, ckpt=ckpt, size=size)

    def op_macs(self, op: str, block_idx: int) -> float:
        if op in _EVA_LINEAR_MACS:
            return float(_EVA_LINEAR_MACS[op])
        if op == "qk" or op == "av":
            return float(_EVA_QK_AV_GLOBAL if block_idx in _EVA_GLOBAL_BLOCKS
                         else _EVA_QK_AV_WINDOW)
        raise KeyError(f"unknown det op {op!r}")

    def comp_macs(self) -> float:
        return float(_EVA_COMP_MACS)

    def load_sensitivity(self) -> dict[tuple[str, int], float]:
        return _load_sensitivity_grid(self._sc_prec)

    def build_active_ops(self, sc_config: str) -> set[tuple[str, int]]:
        if sc_config not in _FINE_KS:
            raise ValueError(f"unknown sc_config: {sc_config}")
        k = _FINE_KS[sc_config]
        if k == 0:
            return {(op, bi) for op in SC_OP_NAMES for bi in range(self.n_blocks)}
        sched = _load_skip_schedule(k, self._sc_prec, self.n_blocks)
        return {
            (op, bi)
            for op in SC_OP_NAMES
            for bi in range(self.n_blocks)
            if sched[op][bi]
        }

    def load_model(self, device: torch.device) -> nn.Module:
        """Return the EVA ViT (``model.backbone.net``) so that the search's
        ``model_sc(imgs)`` accepts a (B, 3, H, W) tensor. Calling the full
        Detectron2 cascade_mask_rcnn with a tensor breaks because it
        expects list[dict]. The same trick is used by
        qwt_det_compensate.py's calibrate_qwt invocation."""
        model, _, _, _ = load_model_and_loader(self._n_search, **self._d2_kw)
        model.eval()
        if device.type == "cpu":
            model = model.cpu()
        else:
            model = model.to(device)
        return model.backbone.net

    def get_blocks(self, model: nn.Module) -> list[nn.Module]:
        return list(_eva_blocks(model))

    def build_loader(self, n_images: int, seed: int,
                     batch_size: int, workers: int) -> DataLoader:
        # The d2 test_loader yields list[dict]; we wrap it so each iter
        # yields (image_tensor,) — what the algorithm expects (it does
        # `imgs = batch[0] if isinstance(batch, (tuple, list)) else batch`).
        # ``preprocess_image`` lives on the full Detectron2 cascade_mask_rcnn,
        # so we build the model+loader pair once and use the model just for
        # preprocessing.
        model_for_pp, loader, _, _ = load_model_and_loader(
            n_images, **self._d2_kw)
        model_for_pp.eval()
        device = torch.device("cuda")

        class _DetCalibLoader:
            def __init__(self, _loader, _model, _max):
                self._loader = _loader
                self._model = _model
                self._max = _max
                # Algorithm does `loader.dataset.__len__()`; expose dataset.
                self.dataset = type("D", (), {"__len__": lambda s: _max})()

            def __iter__(self):
                n = 0
                for batched_inputs in self._loader:
                    images = (self._model.preprocess_image(batched_inputs)
                              .tensor.to(device))
                    yield (images,)
                    n += images.size(0)
                    if n >= self._max:
                        return

        return _DetCalibLoader(loader, model_for_pp, n_images)

    def patch_model_with_sl_map(self, model: nn.Module, sc_prec: int,
                                sl_map: dict) -> dict:
        return _patch_model_with_sl_map(model, sc_prec, sl_map)

    def patch_block_with_spec(self, block: nn.Module, sc_prec: int,
                              block_spec: dict) -> nn.Module:
        return _patch_block_with_spec(block, sc_prec, block_spec)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_int_map(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    if not text:
        return out
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"expected key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def _parse_levels(text: str) -> list[int]:
    return sorted({int(x.strip()) for x in text.split(",") if x.strip()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc_config", default="skip_worst30",
                    choices=list(_FINE_KS))
    ap.add_argument("--target_main_sl", type=int, default=192,
                    help="Target average main-path stoc_len under the budget. "
                         "192 ≈ avg192 / log192. 128 = p7-equivalent.")
    ap.add_argument("--levels", default="64,96,128,192,256",
                    help="Allowed per-unit stoc_len levels.")
    ap.add_argument("--search_ops", default="mlp_fc1,mlp_fc2",
                    help="Searchable ops. Equal-MAC DP requires all selected "
                         "ops to share the same per-block MAC; mlp_fc1+mlp_fc2 "
                         "satisfies this trivially on EVA. qkv_proj/out_proj "
                         "have different MACs (and from mlp), so default "
                         "to fixed.")
    ap.add_argument("--fixed_ops",
                    default="qk=256,av=256,qkv_proj=192,out_proj=192",
                    help="Pinned stoc_len per op (excluded from search).")
    ap.add_argument("--op_min_levels", default="",
                    help="Per-op minimum stoc_len (e.g. 'mlp_fc2=192').")
    ap.add_argument("--op_max_levels", default="",
                    help="Per-op maximum stoc_len.")
    ap.add_argument("--proxy", default="comp_residual",
                    choices=["raw_mse", "comp_residual"])
    ap.add_argument("--init_mode", default="uniform_repair",
                    choices=["sensitivity_dp", "uniform_repair", "map_repair"])
    ap.add_argument("--init_level", type=int, default=192)
    ap.add_argument("--init_sl_map_json", default="")
    ap.add_argument("--n_search", type=int, default=8,
                    help="Number of calibration images for the search proxy.")
    ap.add_argument("--search_seed", type=int, default=1)
    ap.add_argument("--max_iters", type=int, default=4)
    ap.add_argument("--pair_eval_topk", type=int, default=0)
    ap.add_argument("--lookahead_blocks", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--fwd_chunk", type=int, default=2,
                    help="Det images are 1280²; use tiny chunks.")
    ap.add_argument("--cache_dtype", default="float16",
                    choices=["float16", "float32"])
    ap.add_argument("--sc_prec", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--d2_datasets", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--size", type=int, default=0,
                    help="Override input square size (multiple of 256). "
                         "0=default 1280. Use 1024 for ~36% speedup; "
                         "sensitivity grid was generated at 1280 so this "
                         "is a slight prior mismatch but the swap phase "
                         "overrides it via real forward MSE.")
    ap.add_argument("--out_json",
                    default="sensitivity/sl_maps/avg192_mp.json")
    args = ap.parse_args()

    os.environ.setdefault("DETECTRON2_DATASETS",
                          args.d2_datasets or
                          "/scratch/nbleier_owned_root/nbleier_owned1/shared_data")
    random.seed(args.search_seed)
    torch.manual_seed(args.search_seed)
    device = torch.device("cuda")

    backend = DetBackend(
        sc_prec=args.sc_prec, n_search=args.n_search,
        batch_size=args.batch_size, workers=args.workers,
        sc_config=args.sc_config,
        d2_datasets=args.d2_datasets, ckpt=args.ckpt, size=args.size,
    )

    active_ops = backend.build_active_ops(args.sc_config)
    levels = _parse_levels(args.levels)
    search_ops = {x.strip() for x in args.search_ops.split(",") if x.strip()}
    fixed_ops = _parse_int_map(args.fixed_ops)
    op_min_levels = _parse_int_map(args.op_min_levels)
    op_max_levels = _parse_int_map(args.op_max_levels)
    sens_map = backend.load_sensitivity()
    init_sl_map = None
    if args.init_sl_map_json:
        with open(args.init_sl_map_json) as f:
            d = json.load(f)
        init_sl_map = d.get("sl_map", d) if isinstance(d, dict) else d

    n_blocks = backend.n_blocks
    search_units = ms.build_search_units(
        active_ops=active_ops, search_ops=search_ops,
        n_bins=1, bin_ops=set(),
    )
    spaces = ms.build_level_spaces(
        search_units=search_units, global_levels=levels,
        op_min_levels=op_min_levels, op_max_levels=op_max_levels,
    )

    print(f"[swap_search] active_ops={len(active_ops)}  "
          f"search_units={len(search_units)}  "
          f"levels={levels}  fixed_ops={fixed_ops}", flush=True)

    sl_map, init_meta = ms.build_initial_sl_map(
        op_names=backend.op_names,
        active_ops=active_ops, search_ops=search_ops,
        search_units=search_units, fixed_ops=fixed_ops,
        target_main_sl=args.target_main_sl,
        level_spaces=spaces, sens_map=sens_map,
        op_macs_fn=backend.op_macs, n_blocks=n_blocks,
        init_mode=args.init_mode, init_level=args.init_level,
        init_sl_map=init_sl_map, n_bins=1,
    )

    cache_dtype = (torch.float16 if args.cache_dtype == "float16"
                   else torch.float32)
    loader = backend.build_loader(args.n_search, args.search_seed,
                                  args.batch_size, args.workers)

    init_main = ms.compute_main_sl(sl_map, active_ops, backend.op_macs)
    print(f"[swap_search] initial main_sl={init_main:.2f} "
          f"alloc={ms.summarize_allocation(sl_map, active_ops, backend.op_macs)}",
          flush=True)
    print(f"[swap_search] initial per_op="
          f"{ms.summarize_per_op(sl_map, active_ops)}", flush=True)

    final_sl_map, history, last_iter = ms.iterative_swap_search(
        backend=backend,
        active_ops=active_ops, search_ops=search_ops,
        search_units=search_units, sl_map=sl_map, spaces=spaces,
        loader=loader, device=device, sc_prec=args.sc_prec,
        proxy=args.proxy, ridge=args.ridge, max_iters=args.max_iters,
        fwd_chunk=args.fwd_chunk, cache_dtype=cache_dtype, n_bins=1,
        pair_eval_topk=args.pair_eval_topk,
        lookahead_blocks=args.lookahead_blocks,
    )

    final_main = ms.compute_main_sl(final_sl_map, active_ops, backend.op_macs)
    final_eff = ms.compute_eff_sl(
        final_main, active_ops, backend.op_macs,
        comp_macs=backend.comp_macs(), n_blocks=n_blocks,
    )
    out = {
        "sc_config": args.sc_config,
        "target_main_sl": args.target_main_sl,
        "proxy": args.proxy,
        "levels": levels,
        "search_ops": sorted(search_ops),
        "fixed_ops": fixed_ops,
        "op_min_levels": op_min_levels,
        "op_max_levels": op_max_levels,
        "n_search": args.n_search,
        "search_seed": args.search_seed,
        "max_iters": args.max_iters,
        "pair_eval_topk": args.pair_eval_topk,
        "lookahead_blocks": args.lookahead_blocks,
        "init_meta": init_meta,
        "history": history,
        "last_iter": last_iter,
        "final_main_sl": final_main,
        "final_eff_sl": final_eff,
        "allocation": ms.summarize_allocation(final_sl_map, active_ops,
                                              backend.op_macs),
        "per_op_allocation": ms.summarize_per_op(final_sl_map, active_ops),
        "sl_map": {op: final_sl_map[op] for op in SC_OP_NAMES},
    }

    out_path = Path(args.out_json)
    if not out_path.is_absolute():
        out_path = _DET / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[swap_search] final main_sl={final_main:.2f}  eff_sl={final_eff:.2f}",
          flush=True)
    print(f"[swap_search] alloc={out['allocation']}", flush=True)
    print(f"[swap_search] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

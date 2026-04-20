"""Zero-hyperparameter automatic MP calibration.

Finds per-(block, op) free boundaries that minimize the post-QwT-compensation
residual ``||Y_fp - Y_sc - (X W + b)||^2`` on a small calibration set, via
coordinate descent in a Gauss-Seidel-over-blocks schedule.

Public entry points:

* ``oracle_search_op``   — search k-1 boundaries for one (block, op).
* ``oracle_search_block`` — outer loop over all ops in one block.
* ``auto_calibrate_mp``   — block-by-block calibrator that also installs QwT
                            ``CompensationBlock`` wrappers, mirroring the
                            structure of ``third_party/QwT-SC/QwT-vit-sc/
                            qwt_sc/compensation.py:calibrate_qwt``.

Nothing here trains with gradients. Oracle search = forward-only grid-plus-
coord-descent; comp fit = closed-form ridge. Both are borrowed straight
from QwT's philosophy ("no back-prop, calibration in minutes").
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn as nn

from mp_config import (
    FreeBoundaryMPConfig,
    set_current_block_idx,
)


# =====================================================================
# Ridge fitter with precomputed factorization
# =====================================================================

class RidgeFitter:
    """Cache the factorization of ``(X_aug^T X_aug + ridge I)`` so that
    residual norms can be scored in O(N D D_out) per call without re-solving.

    Inputs are expected **on GPU** and in float32 / float64. For large
    calibration sets keep the batch in-memory but chunk if it doesn't fit.
    """

    def __init__(self, X: torch.Tensor, ridge: float = 1e-2):
        N, D = X.shape
        ones = torch.ones(N, 1, device=X.device, dtype=X.dtype)
        self.X_aug = torch.cat([X, ones], dim=-1)  # [N, D+1]
        G = self.X_aug.t() @ self.X_aug            # [D+1, D+1]
        reg = torch.zeros_like(G)
        reg[:D, :D] = ridge * torch.eye(D, device=G.device, dtype=G.dtype)
        self.G_inv = torch.linalg.inv(G + reg)     # [D+1, D+1]

    def residual_norm_sq(self, R: torch.Tensor) -> float:
        """Return ``||R - X_aug @ sol||^2`` where sol = argmin_w ||R - X_aug w||^2."""
        rhs = self.X_aug.t() @ R                   # [D+1, D_out]
        sol = self.G_inv @ rhs
        resid = R - self.X_aug @ sol
        return float((resid.pow(2).sum()).item())


# =====================================================================
# Coordinate-descent oracle search
# =====================================================================

def _candidate_values(
    boundaries: torch.Tensor,
    j: int,
    n_candidates: int,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Linear grid of candidate values for boundaries[j], strictly between
    its neighbors (or the [eps, 1-eps] envelope at the ends).
    """
    k_minus_1 = boundaries.numel()
    lo = float(boundaries[j + 1]) + eps if j + 1 < k_minus_1 else eps
    hi = float(boundaries[j - 1]) - eps if j > 0 else 1.0 - eps
    if hi <= lo:
        # Degenerate; return current value so coord descent is a no-op here.
        return boundaries[j:j + 1].clone()
    return torch.linspace(lo, hi, n_candidates)


def oracle_search_op(
    *,
    run_block_sc: Callable[[], torch.Tensor],
    Y_fp: torch.Tensor,
    fitter: RidgeFitter,
    cfg: FreeBoundaryMPConfig,
    block_idx: int,
    op: str,
    init_boundaries: Optional[torch.Tensor] = None,
    n_candidates: int = 10,
    n_outer: int = 2,
    eps: float = 1e-3,
    log_fn: Callable[[str], None] = lambda _m: None,
) -> tuple[torch.Tensor, float]:
    """Coord-descent search for the k-1 free boundaries of one (block, op).

    Parameters
    ----------
    run_block_sc : callable with no args
        Runs the SC block on the pre-staged calibration input and returns
        Y_sc of shape ``[N, D_out]``. The caller is responsible for making
        sure the block's config reads ``cfg`` (by passing the same ``cfg``
        instance into ``patch_model``) and that ``set_current_block_idx``
        has been called for ``block_idx``.
    Y_fp : [N, D_out]
        FP target on the same input.
    fitter : RidgeFitter
        Pre-fit on the block's input X.
    cfg : FreeBoundaryMPConfig
        Mutated in place — boundaries for ``(block_idx, op)`` end up set
        to the best discovered values on return.

    Returns
    -------
    (best_boundaries, best_score)
    """
    k = len(cfg.stoc_len_levels)
    if init_boundaries is None:
        init_boundaries = cfg.default_boundaries()
    b = init_boundaries.detach().clone().float()

    cfg.set_boundaries(op, block_idx, b)
    Y_sc = run_block_sc()
    best_score = fitter.residual_norm_sq(Y_fp - Y_sc)
    log_fn(f"    [op={op}] init score={best_score:.4e}")

    for outer in range(n_outer):
        improved_this_round = False
        for j in range(k - 1):
            cands = _candidate_values(b, j, n_candidates, eps=eps)
            best_cand = float(b[j])
            best_cand_score = best_score
            for c in cands.tolist():
                b_trial = b.clone()
                b_trial[j] = c
                cfg.set_boundaries(op, block_idx, b_trial)
                Y_sc_try = run_block_sc()
                score = fitter.residual_norm_sq(Y_fp - Y_sc_try)
                if score < best_cand_score:
                    best_cand_score = score
                    best_cand = c
            if best_cand_score < best_score - 1e-9:
                b[j] = best_cand
                best_score = best_cand_score
                cfg.set_boundaries(op, block_idx, b)
                improved_this_round = True
            else:
                # restore best boundaries in cfg after sweeping candidates
                cfg.set_boundaries(op, block_idx, b)
        log_fn(f"    [op={op}] outer {outer}: score={best_score:.4e} "
               f"b={b.tolist()}")
        if not improved_this_round:
            break

    return b, best_score


def oracle_search_block(
    *,
    run_block_sc_for_op: Callable[[str], Callable[[], torch.Tensor]],
    Y_fp: torch.Tensor,
    fitter: RidgeFitter,
    cfg: FreeBoundaryMPConfig,
    block_idx: int,
    ops: Sequence[str],
    n_candidates: int = 10,
    n_outer_per_op: int = 2,
    n_outer_block: int = 2,
    log_fn: Callable[[str], None] = lambda _m: None,
) -> dict[str, dict[str, Any]]:
    """Search boundaries for every op in a block. Wraps multiple ops in an
    outer coord-descent (``n_outer_block``) since ops within a block couple
    through the block's output.

    Parameters
    ----------
    run_block_sc_for_op : callable
        ``op -> () -> Y_sc`` — closure producing a zero-arg forward that the
        op's boundaries affect. In practice the driver uses a single forward
        of the patched block (all ops feed into it), but different ops may
        want to swap the input feed — this signature leaves that flexible.

    Returns
    -------
    dict mapping op -> {"boundaries": Tensor[k-1], "score": float}
    """
    results: dict[str, dict[str, Any]] = {}
    # Initial pass: equal spacing for everyone.
    for op in ops:
        cfg.set_boundaries(op, block_idx, cfg.default_boundaries())

    for outer in range(n_outer_block):
        log_fn(f"  [block {block_idx}] outer block round {outer}")
        for op in ops:
            run_block_sc = run_block_sc_for_op(op)
            b, score = oracle_search_op(
                run_block_sc=run_block_sc,
                Y_fp=Y_fp,
                fitter=fitter,
                cfg=cfg,
                block_idx=block_idx,
                op=op,
                init_boundaries=cfg.get_boundaries(op, block_idx),
                n_candidates=n_candidates,
                n_outer=n_outer_per_op,
                log_fn=log_fn,
            )
            results[op] = {"boundaries": b, "score": score}

    return results


# =====================================================================
# Gauss-Seidel block calibrator (mirrors QwT calibrate_qwt)
# =====================================================================

@torch.no_grad()
def auto_calibrate_mp(
    *,
    model_fp: nn.Module,
    model_sc: nn.Module,
    blocks_fp: Sequence[nn.Module],
    blocks_sc_container,                     # list-like: __getitem__/__setitem__
    calib_loader,
    device: torch.device,
    n_calib: int,
    cfg: FreeBoundaryMPConfig,
    ops_per_block: Sequence[str],
    ridge: float = 1e-2,
    start_block: int = 0,
    fwd_chunk: int = 32,
    avg_sc_draws: int = 1,
    n_candidates: int = 10,
    n_outer_per_op: int = 2,
    n_outer_block: int = 2,
    comp_factory: Optional[Callable[[torch.Tensor, torch.Tensor], nn.Module]] = None,
    install_compensation: bool = True,
    log_fn: Callable[[str], None] = print,
) -> list[dict]:
    """Gauss-Seidel sequential calibration: per block i, (1) search oracle
    boundaries over each op, (2) fit QwT compensation W, b, (3) install the
    compensated block, (4) propagate X to i+1.

    This is the direct analog of ``calibrate_qwt`` with an oracle-search step
    interleaved before the compensation fit.

    Parameters
    ----------
    cfg : FreeBoundaryMPConfig
        **Mutated in place** — at return, ``cfg.boundaries`` is populated
        for every (block_idx, op) in ``0..n_blocks × ops_per_block``.
    ops_per_block : list[str]
        The ops whose boundaries should be searched on every block (same list
        reused for all blocks; detection of which ops are actually SC-patched
        is the caller's responsibility).

    Returns
    -------
    per-block report dicts: {block, r2, enabled, rmse_before, rmse_after,
                             boundaries_per_op, dt_s}
    """
    from qwt_sc.compensation import (  # type: ignore
        CompensationBlock, closed_form_ridge,
    )

    assert len(blocks_fp) == len(blocks_sc_container), (
        f"fp blocks ({len(blocks_fp)}) vs sc blocks "
        f"({len(blocks_sc_container)}) length mismatch")
    n_blocks = len(blocks_fp)

    # Install block-index pre-hooks so classifiers see the right (block,op).
    # We stash handles to keep them alive; they remain installed post-calib.
    hook_handles = []
    for i, blk in enumerate(blocks_sc_container):
        def _make_hook(idx):
            def _hook(_m, _args):
                set_current_block_idx(idx)
            return _hook
        hook_handles.append(blk.register_forward_pre_hook(_make_hook(i)))

    # Collect block-0 inputs via a pre-hook on model_sc's first block.
    first_block = blocks_sc_container[0]
    captured: list[torch.Tensor] = []

    def _cap_hook(_m, args):
        captured.append(args[0].detach().float().cpu())

    log_fn(f"[auto_calib] collecting first-block inputs from model_sc")
    handle = first_block.register_forward_pre_hook(_cap_hook)
    seen = 0
    try:
        for batch in calib_loader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(device, non_blocking=True)
            model_sc(imgs)
            seen += imgs.size(0)
            if seen >= n_calib:
                break
    finally:
        handle.remove()
    X_cur = torch.cat(captured, dim=0)[:n_calib]
    log_fn(f"[auto_calib] X_0 shape={tuple(X_cur.shape)}")

    report: list[dict] = []

    for i in range(n_blocks):
        t0 = time.time()
        blk_fp = blocks_fp[i]
        blk_sc = blocks_sc_container[i]
        set_current_block_idx(i)

        # ---- Y_fp target ----
        Y_fp_cpu = _forward_batched(blk_fp, X_cur, device, fwd_chunk)

        # ---- Pre-stage X and Y_fp on device for fitter + scoring ----
        X_flat_cpu = X_cur.reshape(-1, X_cur.size(-1))
        Y_fp_flat_cpu = Y_fp_cpu.reshape(-1, Y_fp_cpu.size(-1))
        X_flat = X_flat_cpu.to(device)
        Y_fp_flat = Y_fp_flat_cpu.to(device)
        fitter = RidgeFitter(X_flat, ridge=ridge)

        # ---- Baseline Y_sc (current boundaries — default for block i) ----
        def _run_block_sc():
            out = _forward_batched(blk_sc, X_cur, device, fwd_chunk)
            return out.reshape(-1, out.size(-1)).to(device)

        # optional SC averaging to reduce stochastic variance
        def _run_block_sc_avg():
            if avg_sc_draws <= 1:
                return _run_block_sc()
            acc = None
            for _ in range(avg_sc_draws):
                y = _run_block_sc()
                acc = y if acc is None else acc + y
            return acc / avg_sc_draws

        raw_Y_sc = _run_block_sc_avg()
        raw_rmse = float((Y_fp_flat - raw_Y_sc).pow(2).mean().sqrt().item())
        baseline_score = fitter.residual_norm_sq(Y_fp_flat - raw_Y_sc)

        log_fn(f"[auto_calib] block {i:2d}: baseline raw_rmse={raw_rmse:.4e} "
               f"score={baseline_score:.4e}")

        # ---- Oracle search over ops in this block ----
        per_op: dict[str, dict[str, Any]] = {}
        for outer in range(n_outer_block):
            for op in ops_per_block:
                b_init = cfg.get_boundaries(op, i)
                b_new, score = oracle_search_op(
                    run_block_sc=_run_block_sc_avg,
                    Y_fp=Y_fp_flat,
                    fitter=fitter,
                    cfg=cfg,
                    block_idx=i,
                    op=op,
                    init_boundaries=b_init,
                    n_candidates=n_candidates,
                    n_outer=n_outer_per_op,
                    log_fn=log_fn,
                )
                per_op[op] = {"boundaries": b_new.tolist(), "score": score}

        # ---- Refit QwT compensation W, b with final boundaries ----
        final_Y_sc = _run_block_sc_avg()
        R_flat = Y_fp_flat - final_Y_sc
        W, b_vec, r2 = closed_form_ridge(X_flat, R_flat, ridge=ridge)
        after_rmse = float((R_flat - (X_flat @ W + b_vec)).pow(2).mean().sqrt().item())

        enabled = (i >= start_block) and (after_rmse < raw_rmse)
        W_cpu = W.detach().cpu()
        b_cpu = b_vec.detach().cpu()

        if install_compensation:
            comp_module = (comp_factory(W_cpu, b_cpu)
                           if comp_factory is not None else None)
            new_block = CompensationBlock(
                block=blk_sc, W=W_cpu, b=b_cpu,
                r2=r2, enabled=enabled, comp_module=comp_module,
            ).to(device)
            blocks_sc_container[i] = new_block
            # pre-hook was attached to blk_sc; re-attach to new_block
            # (old hook still fires on blk_sc if it's still referenced, but
            # forwarding now goes through new_block; add a fresh hook).
            def _make_hook(idx):
                def _hook(_m, _args):
                    set_current_block_idx(idx)
                return _hook
            new_block.register_forward_pre_hook(_make_hook(i))
            X_next = _forward_batched(new_block, X_cur, device, fwd_chunk)
        else:
            X_next = _forward_batched(blk_sc, X_cur, device, fwd_chunk)

        dt = time.time() - t0
        log_fn(f"[auto_calib] block {i:2d}  r2={r2:+.4f}  "
               f"rmse {raw_rmse:.4e} -> {after_rmse:.4e}  "
               f"enabled={enabled}  ({dt:.1f}s)")

        report.append({
            "block": i,
            "r2": r2,
            "enabled": enabled,
            "rmse_before": raw_rmse,
            "rmse_after": after_rmse,
            "boundaries_per_op": per_op,
            "dt_s": dt,
        })

        X_cur = X_next
        del X_flat, Y_fp_flat, fitter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return report


def _forward_batched(block: nn.Module, X: torch.Tensor, device: torch.device,
                     chunk: int) -> torch.Tensor:
    """Run `block` on CPU-resident X batch-by-batch, return CPU output."""
    outs = []
    for s in range(0, X.size(0), chunk):
        xb = X[s:s + chunk].to(device, non_blocking=True)
        y = block(xb)
        outs.append(y.detach().float().cpu())
    return torch.cat(outs, dim=0)

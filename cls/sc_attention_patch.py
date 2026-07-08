"""Monkey-patch DINOv2 attention + MLP to use stochastic computing matmuls.

- Q @ K^T: batched bipolar enable-signal SC via sc_matmul(granularity="per_head").
- Attn @ V: per-row enable-signal SC via sc_matmul(granularity="per_row").
- MLP fc1 / fc2: SC linear via sc_matmul(granularity="per_row").

Softmax, layernorms, activations, and qkv/proj linears stay FP.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# cls/ lives under vit_sc/; SC primitives (sc/) and integration wrappers
# (sc_integration/) sit at the vit_sc/ root and are shared with det/.
HERE = Path(__file__).resolve().parent          # vit_sc/cls/
VIT_SC = HERE.parent                             # vit_sc/
sys.path.insert(0, str(VIT_SC / "sc"))
sys.path.insert(0, str(VIT_SC))
sys.path.insert(0, str(HERE))

from scmp_kernels.sc import sc_matmul  # noqa: E402  unified dispatcher (enable path)
from sc_integration import sc_linear as _scl  # noqa: E402
from sc_integration.noise_matmul import _noisy_matmul_core  # noqa: E402
from sc_integration.mp_linear import (  # noqa: E402
    MPConfig, RangeMPConfig, AdaptiveMPConfig, sc_prec_for_stoc_len,
)
from scmp_kernels.mp import (  # noqa: E402
    classify_rows_by_metric,
    adaptive_classify_rows,
    AutoMPBudgetLogger,
    get_current_block_idx,
)

# Re-export the moved primitives so existing
# ``from sc_attention_patch import SCLinear / _sc_linear / set_noise_model``
# call sites keep working. Read ``USE_NOISE_MODEL`` via ``_scl.USE_NOISE_MODEL``
# (qualified) so toggles via ``set_noise_model`` are seen at call time.
SCLinear = _scl.SCLinear
_sc_linear = _scl._sc_linear
_sc_linear_mp = _scl._sc_linear_mp
_mm_one_bucket = _scl._mm_one_bucket
_get_config = _scl._get_config
_CFG_CACHE = _scl._CFG_CACHE
set_noise_model = _scl.set_noise_model
set_vit_timestep = _scl.set_vit_timestep
get_vit_timestep = _scl.get_vit_timestep


def _classify_per_head(q: torch.Tensor,
                       cfg: "MPConfig | AdaptiveMPConfig") -> list[int]:
    """Per-head ``stoc_len`` assignment. metric = per-head ``|q|.amax((0,2,3))``.

    * ``MPConfig`` → fixed-quantile classifier.
    * ``AdaptiveMPConfig`` → timestep-aware classifier, reads (t, T) from
      ``sc_linear.get_vit_timestep()`` so ViT callers drive it externally.
    """
    H = q.shape[1]
    metric = q.float().abs().amax(dim=(0, 2, 3))  # (H,)
    if isinstance(cfg, AdaptiveMPConfig):
        t, T = _scl.get_vit_timestep()
        assignment = adaptive_classify_rows(metric, t, T, cfg, operator="qk")
    else:
        assignment = classify_rows_by_metric(
            metric, cfg.stoc_len_levels, cfg.level_fractions,
        )
    head_sl = [0] * H
    for sl, heads in assignment.level_row_indices.items():
        for h in heads:
            head_sl[h.item()] = int(sl)
    return head_sl


AV_METRIC_REVERSE = False
AV_METRIC_KIND = "entropy"  # "amax" | "entropy" | "ipr"


def set_av_metric_reverse(flag: bool):
    """When True, invert the AV row metric so LOW-amax rows get sl=256.

    Rationale: softmax rows with high max are near one-hot (easy to quant);
    rows with low max are near-uniform (needs precise sum of ~N terms).
    Reversing aligns precision allocation with difficulty.
    """
    global AV_METRIC_REVERSE
    AV_METRIC_REVERSE = bool(flag)


def set_av_metric_kind(kind: str):
    """Select the AV row peakiness metric:

    * ``amax``   — row.max() (legacy option; pair with AV_METRIC_REVERSE
                   to get "flat → high sl" direction)
    * ``entropy`` — Shannon entropy of the row; HIGHER entropy ⇒ flatter ⇒
                    more precision needed (passed straight as metric, no
                    reverse flag needed — already aligned; default)
    * ``ipr``    — inverse participation ratio 1/Σp²; effective support;
                    HIGHER IPR ⇒ flatter (same sign as entropy)
    """
    global AV_METRIC_KIND
    assert kind in ("amax", "entropy", "ipr"), f"unknown AV metric: {kind}"
    AV_METRIC_KIND = kind


def _av_row_score(attn_row: torch.Tensor) -> torch.Tensor:
    """Compute per-row peakiness score for AV. Higher = 'need more precision'.

    For amax we return attn_row.amax(-1) and rely on AV_METRIC_REVERSE for
    direction. For entropy / ipr, we return the raw measure where higher
    already means flatter (already the "reversed" direction).

    Input attn_row can be 1-D (amax already computed) or 2-D ((M, N)).
    """
    if AV_METRIC_KIND == "amax":
        metric = attn_row.amax(-1) if attn_row.dim() > 1 else attn_row
        return -metric if AV_METRIC_REVERSE else metric
    if attn_row.dim() == 1:
        raise ValueError(f"AV_METRIC_KIND={AV_METRIC_KIND} needs full "
                         "attn row, got 1-D input")
    if AV_METRIC_KIND == "entropy":
        safe = attn_row.clamp(min=1e-9)
        return -(attn_row * safe.log()).sum(-1)
    if AV_METRIC_KIND == "ipr":
        return 1.0 / (attn_row.pow(2).sum(-1) + 1e-9)
    raise ValueError(f"unknown AV_METRIC_KIND: {AV_METRIC_KIND}")


def _classify_attn_rows(attn_row: torch.Tensor,
                        cfg: "MPConfig | AdaptiveMPConfig") -> "RowAssignment":
    """Per-attn-row ``stoc_len`` assignment. See ``set_av_metric_kind`` for
    the choice of peakiness metric. Accepts 1-D (amax only) or 2-D (full row).
    """
    metric = _av_row_score(attn_row)
    if isinstance(cfg, AdaptiveMPConfig):
        t, T = _scl.get_vit_timestep()
        return adaptive_classify_rows(metric, t, T, cfg, operator="av")
    return classify_rows_by_metric(
        metric, cfg.stoc_len_levels, cfg.level_fractions,
    )


def _sc_qk_mp(q: torch.Tensor, k: torch.Tensor, sc_prec: int,
              mp_cfg: "MPConfig | AdaptiveMPConfig") -> torch.Tensor:
    """Per-head mixed-precision Q@K^T. q, k: (B, H, N, D). Returns (B, H, N, N)
    unscaled (caller applies self.scale). Accepts static or adaptive MP.
    """
    B, H, N, D = q.shape
    head_sl = _classify_per_head(q, mp_cfg)

    # Bucket heads by stoc_len, one matmul per bucket.
    stoc_len_to_heads: dict[int, list[int]] = {}
    for h, sl in enumerate(head_sl):
        stoc_len_to_heads.setdefault(sl, []).append(h)

    max_stoc_len = max(int(sl) for sl in mp_cfg.stoc_len_levels)
    compute_baseline = B * H * N * N * max_stoc_len
    compute_actual = 0.0
    for sl, heads in stoc_len_to_heads.items():
        if sl <= 0:
            continue
        compute_actual += B * len(heads) * N * N * int(sl)
    AutoMPBudgetLogger.record(
        get_current_block_idx(), "qk", compute_baseline, compute_actual,
    )

    output = torch.zeros(B, H, N, N, dtype=torch.float32, device=q.device)
    for sl, heads in stoc_len_to_heads.items():
        if sl == 0 or not heads:
            continue
        idx = torch.as_tensor(heads, dtype=torch.long, device=q.device)
        sp = sc_prec_for_stoc_len(sl)
        q_sub = q.index_select(1, idx).contiguous()  # (B, h', N, D)
        k_sub = k.index_select(1, idx).contiguous()

        if _scl.USE_NOISE_MODEL:
            out_sub = _noisy_matmul_core(
                q_sub.float(), k_sub.float(), L=sl,
                mode="bipolar", per_row_scale=False,
            )
            output.index_copy_(1, idx, out_sub.float())
        else:
            # Use the batched enable kernel (scmp_llm's QK MP path) — takes
            # floats + per-head min/max, accepts stoc_len natively.
            Bq, Hh, Nq, Dq = q_sub.shape
            qf = q_sub.float().reshape(Bq * Hh, Nq, Dq).contiguous()
            kf = k_sub.float().reshape(Bq * Hh, Nq, Dq).contiguous()
            q_maxs = qf.amax(dim=(1, 2))
            q_mins = qf.amin(dim=(1, 2))
            k_maxs = kf.amax(dim=(1, 2))
            k_mins = kf.amin(dim=(1, 2))
            cfg = _get_config(Dq, sp)
            attn_sub = sc_matmul(
                qf, kf, granularity="per_head",
                sc_prec=sp, config=cfg, stoc_len=int(sl),
            ).reshape(Bq, Hh, Nq, Nq)
            output.index_copy_(1, idx, attn_sub.float())

    return output


def _sc_av_mp(attn: torch.Tensor, v: torch.Tensor, sc_prec: int,
              mp_cfg: "MPConfig | AdaptiveMPConfig") -> torch.Tensor:
    """Per-attn-row mixed-precision Attn@V. attn: (B, H, N, N), v: (B, H, N, D).
    Per scmp_llm: for each (b, h), classify the N rows by row-max and bucket.
    Static or adaptive depending on ``mp_cfg`` type.
    """
    B, H, N, D = v.shape
    BH = B * H
    attn_flat = attn.float().reshape(BH, N, N)
    b_flat = v.float().transpose(-1, -2).reshape(BH, D, N)  # (BH, D, N)
    out = torch.zeros(BH, N, D, dtype=torch.float32, device=v.device)
    max_stoc_len = max(int(sl) for sl in mp_cfg.stoc_len_levels)
    compute_baseline = 0.0
    compute_actual = 0.0

    for i in range(BH):
        # Pass full attn rows (N, N) so entropy / ipr metrics can be computed.
        assignment = _classify_attn_rows(attn_flat[i], mp_cfg)
        for sl, rows in assignment.level_row_indices.items():
            if sl == 0 or rows.numel() == 0:
                continue
            compute_baseline += rows.numel() * D * max_stoc_len
            compute_actual += rows.numel() * D * int(sl)
            sp = sc_prec_for_stoc_len(int(sl))
            attn_sub = attn_flat[i].index_select(0, rows).contiguous()  # (n_sel, N)

            if _scl.USE_NOISE_MODEL:
                sub = _noisy_matmul_core(
                    attn_sub, b_flat[i], L=int(sl),
                    mode="bipolar", per_row_scale=True,
                )
            else:
                cfg = _get_config(N, sp)
                sub = sc_matmul(
                    attn_sub, b_flat[i], granularity="per_row",
                    group_a=1, group_b=1,
                    mode="bipolar", sc_prec=sp, config=cfg, stoc_len=int(sl),
                )
            out[i, rows] = sub

    AutoMPBudgetLogger.record(
        get_current_block_idx(), "av", compute_baseline, compute_actual,
    )
    return out.reshape(B, H, N, D)


def make_sc_attention_forward(sc_prec: int, sc_av: bool, sc_qk: bool = True,
                              qk_mp_cfg: MPConfig | None = None,
                              av_mp_cfg: MPConfig | None = None):
    """Return a forward(self, x) with optional SC Q@K^T and SC Attn@V.

    When ``qk_mp_cfg`` / ``av_mp_cfg`` are provided, per-head / per-attn-row
    fixed mixed precision is applied (scmp_llm-compatible). When they are
    ``None`` the original uniform path is taken — unchanged numerics.
    """

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        H = self.num_heads
        D = C // H
        qkv = self.qkv(x).reshape(B, N, 3, H, D)
        q, k, v = torch.unbind(qkv, 2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if sc_qk:
            if qk_mp_cfg is not None:
                attn = _sc_qk_mp(q, k, sc_prec, qk_mp_cfg).to(x.dtype) * self.scale
            elif _scl.USE_NOISE_MODEL:
                # (B,H,N,D) @ (B,H,N,D)^T via surrogate (per-head amax scale).
                attn = _noisy_matmul_core(
                    q.float(), k.float(), L=2 ** sc_prec,
                    mode="bipolar", per_row_scale=False,
                ).to(x.dtype) * self.scale
            else:
                # Aligned with scmp_llm / MP path: enable-signal batched kernel,
                # takes floats + per-head (min, max), faster and more numerically
                # stable than the packed-XNOR sc_matmul_qk_multihead path.
                qf = q.float().reshape(B * H, N, D).contiguous()
                kf = k.float().reshape(B * H, N, D).contiguous()
                q_maxs = qf.amax(dim=(1, 2))
                q_mins = qf.amin(dim=(1, 2))
                k_maxs = kf.amax(dim=(1, 2))
                k_mins = kf.amin(dim=(1, 2))
                cfg_qk = _get_config(D, sc_prec)
                attn_sc = sc_matmul(
                    qf, kf, granularity="per_head",
                    sc_prec=sc_prec, config=cfg_qk, stoc_len=2 ** sc_prec,
                ).reshape(B, H, N, N)
                attn = attn_sc.to(x.dtype) * self.scale
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if sc_av:
            if av_mp_cfg is not None:
                out = _sc_av_mp(attn, v, sc_prec, av_mp_cfg).to(x.dtype)
            else:
                # Use sc_matmul_grouped (unipolar + signs) with per-row quant to
                # handle the very wide dynamic range of softmax'd attn.
                a_flat = attn.float().reshape(B * H, N, N)
                b_flat = v.float().transpose(-1, -2).reshape(B * H, D, N)
                if _scl.USE_NOISE_MODEL:
                    out = _noisy_matmul_core(
                        a_flat, b_flat, L=2 ** sc_prec,
                        mode="bipolar", per_row_scale=True,
                    ).reshape(B, H, N, D).to(x.dtype)
                else:
                    cfg_av = _get_config(N, sc_prec)
                    out = torch.empty(B * H, N, D, dtype=torch.float32, device=x.device)
                    for i in range(B * H):
                        out[i] = sc_matmul(
                            a_flat[i], b_flat[i], granularity="per_row",
                            group_a=1, group_b=1,
                            mode="bipolar", sc_prec=sc_prec, config=cfg_av,
                        )
                    out = out.to(x.dtype).reshape(B, H, N, D)
        else:
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    return forward


# Canonical op names for the sc_ops vector API.
SC_OP_NAMES = ("mlp_fc1", "mlp_fc2", "qk", "av", "proj")


def _discover_blocks(model):
    """Return the list of transformer blocks in a DINOv2-style model."""
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        return list(model.backbone.blocks)
    if hasattr(model, "blocks"):
        return list(model.blocks)
    raise RuntimeError("Could not locate transformer blocks on model.")


def normalize_sc_ops_per_block(spec, n_blocks):
    """Normalize a per-block-per-op spec into a complete dict.

    Accepts ``spec`` as ``dict[op_name, list[int]]`` where each list has
    length ``n_blocks`` (or length 1 — broadcast — or a scalar 0/1). Missing
    op keys default to all-zero. Returns a dict with every key in
    ``SC_OP_NAMES`` mapped to a length-``n_blocks`` tuple of 0/1 ints.
    """
    if spec is None:
        return {k: tuple([0] * n_blocks) for k in SC_OP_NAMES}
    if not isinstance(spec, dict):
        raise TypeError("sc_ops_per_block must be a dict[op_name, list]")
    out = {}
    for k in SC_OP_NAMES:
        v = spec.get(k, 0)
        if isinstance(v, (int, bool)):
            out[k] = tuple([int(bool(v))] * n_blocks)
            continue
        v = list(v)
        if len(v) == 1:
            out[k] = tuple([int(bool(v[0]))] * n_blocks)
        elif len(v) == n_blocks:
            out[k] = tuple(int(bool(x)) for x in v)
        else:
            raise ValueError(
                f"sc_ops_per_block[{k!r}] length {len(v)} != n_blocks {n_blocks}")
    extra = set(spec) - set(SC_OP_NAMES)
    if extra:
        raise ValueError(f"unknown op names in sc_ops_per_block: {sorted(extra)}")
    return out


def parse_sc_ops(spec):
    """Normalize an ``sc_ops`` spec into a set of canonical op names.

    Accepts any of:
      * a set/list/tuple of names drawn from SC_OP_NAMES
      * a 5-element 0/1 vector in SC_OP_NAMES order
      * a comma-separated string of names (e.g. "mlp_fc1,av")
    Returns a set[str].
    """
    if spec is None:
        return set()
    if isinstance(spec, str):
        items = [s.strip() for s in spec.split(",") if s.strip()]
        return parse_sc_ops(items)
    items = list(spec)
    if len(items) == len(SC_OP_NAMES) and all(isinstance(x, (int, bool)) for x in items):
        return {n for n, v in zip(SC_OP_NAMES, items) if v}
    out = set()
    for x in items:
        if x not in SC_OP_NAMES:
            raise ValueError(f"unknown sc op: {x!r} (valid: {SC_OP_NAMES})")
        out.add(x)
    return out


LINEAR_OP_NAMES = ("mlp_fc1", "mlp_fc2", "qkv_proj", "out_proj")


def _normalize_linear_mp_spec(spec):
    """Normalize a linear_mp_spec dict into {op: (mp_cfg, adaptive_cfg, range_cfg, gsize)}.

    Each key must be one of ``LINEAR_OP_NAMES`` or the alias ``"proj"``
    (which expands to both ``qkv_proj`` and ``out_proj``). Each value is
    one of ``MPConfig``, ``AdaptiveMPConfig``, ``RangeMPConfig``, or a dict
    with any of::

        {"fixed": MPConfig, "adaptive": AdaptiveMPConfig,
         "range": RangeMPConfig, "range_group_size": int}
    """
    out: dict[str, tuple] = {}
    if not spec:
        return out
    for k, v in spec.items():
        if k == "proj":
            targets = ("qkv_proj", "out_proj")
        elif k in LINEAR_OP_NAMES:
            targets = (k,)
        else:
            raise ValueError(
                f"unknown linear_mp_spec key: {k!r}; "
                f"expected one of {LINEAR_OP_NAMES + ('proj',)}")
        if isinstance(v, MPConfig):
            entry = (v, None, None, 0)
        elif isinstance(v, AdaptiveMPConfig):
            entry = (None, v, None, 0)
        elif isinstance(v, RangeMPConfig):
            entry = (None, None, v, 0)
        elif isinstance(v, dict):
            entry = (
                v.get("fixed"),
                v.get("adaptive"),
                v.get("range"),
                int(v.get("range_group_size", 0)),
            )
        else:
            raise TypeError(
                f"linear_mp_spec[{k!r}] must be MPConfig, AdaptiveMPConfig, "
                f"RangeMPConfig, or dict; got {type(v).__name__}")
        for op in targets:
            out[op] = entry
    return out


def _mp_kwargs(linear_mp, op_name: str) -> dict:
    """Build the MP kwargs for SCLinear from a normalized spec dict."""
    if not linear_mp or op_name not in linear_mp:
        return {}
    mp_cfg, adaptive_cfg, range_cfg, gsize = linear_mp[op_name]
    return dict(mp_cfg=mp_cfg, adaptive_mp_cfg=adaptive_cfg,
                range_mp_cfg=range_cfg, range_mp_group_size=gsize,
                operator=op_name)


def patch_model(model, sc_prec: int = 8,
                sc_qk: bool = True, sc_av: bool = False, sc_mlp: bool = False,
                sc_qkv_proj: bool = False, sc_out_proj: bool = False,
                mlp_skip_last_k: int = 0,
                mlp_skip_first_k: int = 0,
                sc_mlp_mode: str = "bipolar",
                sc_proj_mode: str = "bipolar",
                mlp_fc_mask: str = "both",  # "both" | "fc1" | "fc2"
                mlp_sc_spec=None,           # list[tuple[int, str]] | None
                sc_ops=None,                # set/vec/str over SC_OP_NAMES
                sc_ops_per_block=None,      # dict[op_name, list[0/1]] length n_blocks
                linear_mp_spec=None,        # dict[op_name, MPConfig|RangeMPConfig|dict]
                attn_mp_spec=None):         # dict[{"qk","av"}, MPConfig] per-head / per-row MP
    """Apply SC patches to a DINOv2 ViT model in-place.

    Priority: ``sc_ops_per_block`` > ``sc_ops`` > legacy per-op bools.

    ``sc_ops`` (when given) takes priority over per-op bool flags. It accepts
    any form understood by ``parse_sc_ops`` — a set of names, a 5-vector
    ``[mlp_fc1, mlp_fc2, qk, av, proj]`` of 0/1, or a comma-separated string.
    The selected ops are turned on uniformly across all 24 blocks (no block-
    level granularity); "proj" enables both qkv_proj and out_proj together.

    ``mlp_sc_spec`` (if given) is an explicit list of
    ``(block_idx, "fc1"|"fc2")`` pairs naming exactly which MLP linears go SC.
    When set, it overrides ``mlp_skip_first_k`` / ``mlp_skip_last_k`` /
    ``mlp_fc_mask`` / ``sc_ops`` for MLP selection.

    Returns a dict counting patched modules.
    """
    # sc_ops dispatches to per-op bools (unless mlp_sc_spec is set for MLP).
    if sc_ops is not None:
        ops = parse_sc_ops(sc_ops)
        sc_qk = "qk" in ops
        sc_av = "av" in ops
        sc_qkv_proj = "proj" in ops
        sc_out_proj = "proj" in ops
        if mlp_sc_spec is None:
            want_fc1 = "mlp_fc1" in ops
            want_fc2 = "mlp_fc2" in ops
            sc_mlp = want_fc1 or want_fc2
            if want_fc1 and want_fc2:
                mlp_fc_mask = "both"
            elif want_fc1:
                mlp_fc_mask = "fc1"
            elif want_fc2:
                mlp_fc_mask = "fc2"
    stats = {"attn": 0, "mlp_fc1": 0, "mlp_fc2": 0, "qkv_proj": 0, "out_proj": 0}
    linear_mp = _normalize_linear_mp_spec(linear_mp_spec)
    qk_mp_cfg = (attn_mp_spec or {}).get("qk")
    av_mp_cfg = (attn_mp_spec or {}).get("av")
    _attn_ok = (MPConfig, AdaptiveMPConfig)
    if qk_mp_cfg is not None and not isinstance(qk_mp_cfg, _attn_ok):
        raise TypeError(
            f"attn_mp_spec['qk'] must be MPConfig or AdaptiveMPConfig, "
            f"got {type(qk_mp_cfg).__name__}")
    if av_mp_cfg is not None and not isinstance(av_mp_cfg, _attn_ok):
        raise TypeError(
            f"attn_mp_spec['av'] must be MPConfig or AdaptiveMPConfig, "
            f"got {type(av_mp_cfg).__name__}")

    # ---- highest-priority path: per-block per-op vectors ----
    if sc_ops_per_block is not None:
        blocks = _discover_blocks(model)
        pbm = normalize_sc_ops_per_block(sc_ops_per_block, len(blocks))
        for i, blk in enumerate(blocks):
            # Locate attention submodule (the one with qkv/proj/num_heads)
            attn_mod = None
            for m in blk.modules():
                if (hasattr(m, "qkv") and hasattr(m, "proj")
                        and hasattr(m, "num_heads")):
                    attn_mod = m
                    break
            q_on, a_on, p_on = pbm["qk"][i], pbm["av"][i], pbm["proj"][i]
            if attn_mod is not None and (q_on or a_on):
                fwd = make_sc_attention_forward(
                    sc_prec, sc_av=bool(a_on), sc_qk=bool(q_on),
                    qk_mp_cfg=qk_mp_cfg if q_on else None,
                    av_mp_cfg=av_mp_cfg if a_on else None)
                attn_mod.forward = fwd.__get__(attn_mod, type(attn_mod))
                stats["attn"] += 1
            if attn_mod is not None and p_on:
                if isinstance(attn_mod.qkv, nn.Linear):
                    attn_mod.qkv = SCLinear(attn_mod.qkv, sc_prec, mode=sc_proj_mode,
                                            **_mp_kwargs(linear_mp, "qkv_proj"))
                    stats["qkv_proj"] += 1
                if isinstance(attn_mod.proj, nn.Linear):
                    attn_mod.proj = SCLinear(attn_mod.proj, sc_prec, mode=sc_proj_mode,
                                             **_mp_kwargs(linear_mp, "out_proj"))
                    stats["out_proj"] += 1
            mlp = getattr(blk, "mlp", None)
            if mlp is not None:
                if pbm["mlp_fc1"][i] and isinstance(getattr(mlp, "fc1", None), nn.Linear):
                    mlp.fc1 = SCLinear(mlp.fc1, sc_prec, mode=sc_mlp_mode,
                                       **_mp_kwargs(linear_mp, "mlp_fc1"))
                    stats["mlp_fc1"] += 1
                if pbm["mlp_fc2"][i] and isinstance(getattr(mlp, "fc2", None), nn.Linear):
                    mlp.fc2 = SCLinear(mlp.fc2, sc_prec, mode=sc_mlp_mode,
                                       **_mp_kwargs(linear_mp, "mlp_fc2"))
                    stats["mlp_fc2"] += 1
        return stats

    if sc_qk or sc_av:
        fwd = make_sc_attention_forward(
            sc_prec, sc_av=sc_av, sc_qk=sc_qk,
            qk_mp_cfg=qk_mp_cfg if sc_qk else None,
            av_mp_cfg=av_mp_cfg if sc_av else None)
        for m in model.modules():
            if hasattr(m, "qkv") and hasattr(m, "proj") and hasattr(m, "num_heads"):
                m.forward = fwd.__get__(m, type(m))
                stats["attn"] += 1

    if sc_qkv_proj or sc_out_proj:
        for m in model.modules():
            if hasattr(m, "qkv") and hasattr(m, "proj") and hasattr(m, "num_heads"):
                if sc_qkv_proj and isinstance(m.qkv, nn.Linear):
                    m.qkv = SCLinear(m.qkv, sc_prec, mode=sc_proj_mode,
                                     **_mp_kwargs(linear_mp, "qkv_proj"))
                    stats["qkv_proj"] += 1
                if sc_out_proj and isinstance(m.proj, nn.Linear):
                    m.proj = SCLinear(m.proj, sc_prec, mode=sc_proj_mode,
                                      **_mp_kwargs(linear_mp, "out_proj"))
                    stats["out_proj"] += 1

    if sc_mlp:
        blocks = None
        if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
            blocks = list(model.backbone.blocks)
        elif hasattr(model, "blocks"):
            blocks = list(model.blocks)

        if mlp_sc_spec is not None:
            # Explicit list mode. Build per-block set of linears to patch.
            selected = {}  # block_idx -> set of "fc1"/"fc2"
            for item in mlp_sc_spec:
                bi, fc = item
                assert fc in ("fc1", "fc2"), f"bad linear name: {fc}"
                selected.setdefault(int(bi), set()).add(fc)
            if blocks is None:
                raise RuntimeError("mlp_sc_spec requires discoverable blocks")
            for bi, blk in enumerate(blocks):
                if not hasattr(blk, "mlp"):
                    continue
                mlp = blk.mlp
                want = selected.get(bi, set())
                if "fc1" in want and isinstance(getattr(mlp, "fc1", None), nn.Linear):
                    mlp.fc1 = SCLinear(mlp.fc1, sc_prec, mode=sc_mlp_mode,
                                       **_mp_kwargs(linear_mp, "mlp_fc1"))
                    stats["mlp_fc1"] += 1
                if "fc2" in want and isinstance(getattr(mlp, "fc2", None), nn.Linear):
                    mlp.fc2 = SCLinear(mlp.fc2, sc_prec, mode=sc_mlp_mode,
                                       **_mp_kwargs(linear_mp, "mlp_fc2"))
                    stats["mlp_fc2"] += 1
        else:
            # Legacy first-k / last-k block skipping path.
            skip_mlps = set()
            if blocks is not None and mlp_skip_last_k > 0:
                for blk in blocks[-mlp_skip_last_k:]:
                    if hasattr(blk, "mlp"):
                        skip_mlps.add(id(blk.mlp))
            if blocks is not None and mlp_skip_first_k > 0:
                for blk in blocks[:mlp_skip_first_k]:
                    if hasattr(blk, "mlp"):
                        skip_mlps.add(id(blk.mlp))
            for m in model.modules():
                if m.__class__.__name__ == "Mlp" and hasattr(m, "fc1") and hasattr(m, "fc2"):
                    if id(m) in skip_mlps:
                        continue
                    if mlp_fc_mask in ("both", "fc1") and isinstance(m.fc1, nn.Linear):
                        m.fc1 = SCLinear(m.fc1, sc_prec, mode=sc_mlp_mode,
                                         **_mp_kwargs(linear_mp, "mlp_fc1"))
                        stats["mlp_fc1"] += 1
                    if mlp_fc_mask in ("both", "fc2") and isinstance(m.fc2, nn.Linear):
                        m.fc2 = SCLinear(m.fc2, sc_prec, mode=sc_mlp_mode,
                                         **_mp_kwargs(linear_mp, "mlp_fc2"))
                        stats["mlp_fc2"] += 1

    return stats


# Back-compat alias
def patch_model_attention(model, sc_prec: int = 8):
    return patch_model(model, sc_prec=sc_prec, sc_qk=True)["attn"]

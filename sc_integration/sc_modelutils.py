"""
SC Model Utilities for Q-DiT.

This module provides utility functions for converting a DiT model to use
stochastic computing for specified operations.
"""

import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial

from models.models import DiTBlock
from ..qBlock import QuantDiTBlock
from ..quant import (
    quantize_activation_wrapper,
    quantize_attn_v_wrapper,
    quantize_attn_k_wrapper,
    quantize_attn_q_wrapper,
)
from .sc_controller import SCController
from .sc_precision_map import SCPrecisionMap
from .sc_block import SCDiTBlock
from .mp_config import classify_groups_by_range


def add_sc_wrapper(
    model,
    device,
    args,
    scales,
    sc_controller: SCController,
):
    """
    Convert model blocks to SC-enabled blocks.

    This function replaces DiTBlock or QuantDiTBlock instances with SCDiTBlock
    instances that support stochastic computing.

    Args:
        model: The DiT model
        device: Device to use for conversion
        args: Quantization arguments
        scales: Pre-computed activation scales (from calibration)
        sc_controller: SCController instance

    Returns:
        Model with SC-enabled blocks
    """
    blocks = model.blocks

    for i in tqdm(range(len(blocks)), desc="Adding SC wrappers"):
        args_i = copy.deepcopy(args)
        args_i.weight_group_size = args.weight_group_size[i]
        args_i.act_group_size = args.act_group_size[i]

        m = None
        if isinstance(blocks[i], DiTBlock):
            # Create SC-enabled block from original DiTBlock
            m = SCDiTBlock(
                dit_block=blocks[i],
                args=args_i,
                block_idx=i,
                sc_controller=sc_controller,
            )
        elif isinstance(blocks[i], QuantDiTBlock):
            raise NotImplementedError(
                "Converting QuantDiTBlock to SCDiTBlock is not yet supported. "
                "Please use add_sc_wrapper before quantization."
            )

        if m is None:
            continue

        m = m.to(device)

        # Configure activation quantizers
        nameTemplate = 'blocks.{}.{}.{}'
        m.attn.input_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'attn', 'qkv')]
        )
        m.attn.act_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'attn', 'proj')]
        )

        if args.quantize_bmm_input:
            m.attn.q_quant.configure(
                partial(quantize_attn_q_wrapper, args=args_i),
                None
            )
            m.attn.k_quant.configure(
                partial(quantize_attn_k_wrapper, args=args_i),
                None
            )
            m.attn.v_quant.configure(
                partial(quantize_attn_v_wrapper, args=args_i),
                None
            )

        m.mlp.input_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'mlp', 'fc1')]
        )
        m.mlp.act_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'mlp', 'fc2')]
        )

        blocks[i] = m.cpu()
        torch.cuda.empty_cache()

    return model


def quantize_sc_model(model, device, args, sc_controller: SCController = None):
    """
    Quantize weights in an SC-enabled model and build dispatch tables.

    After weight quantization, builds dispatch tables for any blocks
    that have per-group mixed precision configured.  If range-based MP
    is enabled (via sc_controller.range_mp_config), per-group stoc_lens
    are computed from weight (max-min) ranges before building dispatch tables.

    Args:
        model: Model with SCDiTBlock instances
        device: Device to use for quantization
        args: Quantization arguments
        sc_controller: Optional SCController; needed for range-based MP.

    Returns:
        Model with quantized weights and precomputed dispatch tables
    """
    blocks = model.blocks
    range_mp_config = (sc_controller.range_mp_config
                       if sc_controller is not None else None)

    for i in tqdm(range(len(blocks)), desc="Quantizing SC model"):
        args_i = copy.deepcopy(args)
        args_i.weight_group_size = args.weight_group_size[i]
        args_i.act_group_size = args.act_group_size[i]

        if not isinstance(blocks[i], SCDiTBlock):
            continue

        m = blocks[i].to(device)

        # Quantize weights
        m.mlp.fc1.quant()
        m.mlp.fc2.quant()
        m.attn.qkv.quant()
        m.attn.proj.quant()

        # Compute range-based per-group stoc_lens (after quantization)
        if range_mp_config is not None and sc_controller is not None:
            group_size = args_i.weight_channel_group
            for operator, layer in [("proj", m.attn.proj),
                                     ("mlp_fc1", m.mlp.fc1),
                                     ("mlp_fc2", m.mlp.fc2)]:
                group_stoc_lens = classify_groups_by_range(
                    layer.weight, group_size=group_size,
                    config=range_mp_config, operator=operator)
                cfg = sc_controller.precision_map.get(i, operator)
                cfg.group_stoc_lens = group_stoc_lens

            # input_proj: forward() splits QKV [3*C, C] into Q/K/V [C, C]
            # each, so compute range-based groups separately per sub-weight
            # and concatenate (Q first, then K, then V).
            qkv_w = m.attn.qkv.weight  # [3*C, C]
            C = qkv_w.shape[0] // 3
            gs_q = classify_groups_by_range(
                qkv_w[:C], group_size=group_size,
                config=range_mp_config, operator="input_proj")
            gs_k = classify_groups_by_range(
                qkv_w[C:2*C], group_size=group_size,
                config=range_mp_config, operator="input_proj")
            gs_v = classify_groups_by_range(
                qkv_w[2*C:], group_size=group_size,
                config=range_mp_config, operator="input_proj")
            cfg = sc_controller.precision_map.get(i, "input_proj")
            cfg.group_stoc_lens = gs_q + gs_k + gs_v

        # Build dispatch tables for mixed precision (after weights are quantized)
        m.attn._build_dispatch_tables()
        m.mlp._build_dispatch_tables()

        blocks[i] = m.cpu()
        torch.cuda.empty_cache()

    return model


def create_sc_controller_from_args(args, model) -> SCController:
    """
    Create an SCController from command-line arguments.

    If --sc_config is provided, loads a SCPrecisionMap from JSON.
    Otherwise, builds from legacy layerwise args.

    Args:
        args: Parsed arguments with SC parameters
        model: The model (to get total_blocks)

    Returns:
        Configured SCController instance
    """
    total_blocks = len(model.blocks)
    total_timesteps = args.num_sampling_steps

    skip_str = getattr(args, 'sc_skip_blocks', '')
    sc_skip_blocks = set()
    if skip_str:
        sc_skip_blocks = {int(x.strip()) for x in skip_str.split(',') if x.strip()}

    # Check for JSON config
    sc_config_path = getattr(args, 'sc_config', None)
    precision_map = None
    if sc_config_path:
        precision_map = SCPrecisionMap.from_json(sc_config_path)
        print(f"Loaded SC precision map from {sc_config_path}")
        print(precision_map.summary())

    controller = SCController(
        timewise=args.timewise,
        qklayerwise=args.qklayerwise,
        total_timesteps=total_timesteps,
        total_blocks=total_blocks,
        sc_prec=args.sc_prec,
        avlayerwise=getattr(args, 'avlayerwise', 0.0),
        projlayerwise=getattr(args, 'projlayerwise', 0.0),
        mlplayerwise=getattr(args, 'mlplayerwise', 0.0),
        inputprojlayerwise=getattr(args, 'inputprojlayerwise', 0.0),
        reverse_layerwise=getattr(args, 'reverse_layerwise', False),
        sc_skip_blocks=sc_skip_blocks,
        sc_enable=getattr(args, 'sc_enable', False),
        noise_model=getattr(args, 'sc_noise_model', False),
        noise_local_correction=getattr(args, 'sc_noise_local_correction', 0.15),
        noise_global_correction=getattr(args, 'sc_noise_global_correction', 0.60),
        precision_map=precision_map,
    )

    # Save effective config if requested
    save_path = getattr(args, 'save_sc_config', None)
    if save_path:
        controller.precision_map.to_json(save_path)
        print(f"Saved SC precision config to {save_path}")

    return controller

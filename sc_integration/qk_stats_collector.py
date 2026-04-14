"""
Q/K Statistics Collector for analyzing attention distributions in DiT.

This module provides tools to collect and analyze the distribution of
query (Q) and key (K) tensors across heads, layers, and timesteps.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class HeadStats:
    """Statistics for a single attention head."""
    abs_max: float = 0.0
    abs_mean: float = 0.0
    abs_min: float = 0.0
    std: float = 0.0
    sparsity: float = 0.0  # Fraction of values with |x| < threshold

    def to_dict(self) -> dict:
        return {
            'abs_max': self.abs_max,
            'abs_mean': self.abs_mean,
            'abs_min': self.abs_min,
            'std': self.std,
            'sparsity': self.sparsity,
        }


@dataclass
class LayerTimestepStats:
    """Statistics for one layer at one timestep."""
    timestep: int
    layer_idx: int
    q_heads: List[HeadStats] = field(default_factory=list)
    k_heads: List[HeadStats] = field(default_factory=list)

    # Global stats for this layer/timestep
    q_global_max: float = 0.0
    k_global_max: float = 0.0

    def to_dict(self) -> dict:
        return {
            'timestep': self.timestep,
            'layer_idx': self.layer_idx,
            'q_heads': [h.to_dict() for h in self.q_heads],
            'k_heads': [h.to_dict() for h in self.k_heads],
            'q_global_max': self.q_global_max,
            'k_global_max': self.k_global_max,
        }


class QKStatsCollector:
    """
    Collects Q/K statistics from attention layers during inference.

    Usage:
        collector = QKStatsCollector()
        collector.register_hooks(model)

        # Run inference
        for timestep in timesteps:
            collector.set_timestep(timestep)
            model(x, t, y)

        collector.remove_hooks()
        stats = collector.get_all_stats()
    """

    def __init__(self, sparsity_threshold: float = 0.01):
        """
        Args:
            sparsity_threshold: Values with |x| < threshold are considered "sparse"
        """
        self.sparsity_threshold = sparsity_threshold
        self.current_timestep: Optional[int] = None
        self.stats: Dict[Tuple[int, int], LayerTimestepStats] = {}  # (timestep, layer) -> stats
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.layer_indices: Dict[nn.Module, int] = {}

    def set_timestep(self, timestep: int):
        """Set current timestep (call before each diffusion step)."""
        self.current_timestep = timestep

    def register_hooks(self, model: nn.Module):
        """Register forward hooks on all attention layers."""
        layer_idx = 0

        for name, module in model.named_modules():
            # Look for attention modules (QuantAttention or SCAttention)
            if hasattr(module, 'qkv') and hasattr(module, 'num_heads'):
                self.layer_indices[module] = layer_idx
                hook = module.register_forward_hook(self._create_hook(layer_idx))
                self.hooks.append(hook)
                layer_idx += 1

        print(f"Registered hooks on {layer_idx} attention layers")

    def _create_hook(self, layer_idx: int):
        """Create a forward hook for a specific layer."""
        def hook(module, input, output):
            if self.current_timestep is None:
                return

            x = input[0]  # Input to attention
            B, N, C = x.shape

            # Get qkv projection
            qkv = module.qkv(x)
            qkv = qkv.reshape(B, N, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # Each: (B, H, N, D)

            # Apply normalization if present
            if hasattr(module, 'q_norm') and module.q_norm is not None:
                q = module.q_norm(q)
                k = module.k_norm(k)

            # Collect statistics
            self._collect_stats(q, k, layer_idx)

        return hook

    def _collect_stats(self, q: torch.Tensor, k: torch.Tensor, layer_idx: int):
        """Collect statistics from q and k tensors."""
        B, H, N, D = q.shape

        stats = LayerTimestepStats(
            timestep=self.current_timestep,
            layer_idx=layer_idx,
        )

        # Global max
        stats.q_global_max = q.abs().max().item()
        stats.k_global_max = k.abs().max().item()

        # Per-head statistics
        for h in range(H):
            q_h = q[:, h, :, :]  # (B, N, D)
            k_h = k[:, h, :, :]

            stats.q_heads.append(self._compute_head_stats(q_h))
            stats.k_heads.append(self._compute_head_stats(k_h))

        # Store
        key = (self.current_timestep, layer_idx)
        self.stats[key] = stats

    def _compute_head_stats(self, x: torch.Tensor) -> HeadStats:
        """Compute statistics for a single head's tensor."""
        x_flat = x.flatten()
        abs_x = x_flat.abs()

        return HeadStats(
            abs_max=abs_x.max().item(),
            abs_mean=abs_x.mean().item(),
            abs_min=abs_x.min().item(),
            std=x_flat.std().item(),
            sparsity=(abs_x < self.sparsity_threshold).float().mean().item(),
        )

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print("Removed all hooks")

    def get_all_stats(self) -> Dict[Tuple[int, int], LayerTimestepStats]:
        """Get all collected statistics."""
        return self.stats

    def analyze_head_magnitude_variation(self) -> dict:
        """
        Analyze how much head magnitudes vary within each layer.

        Returns dict with analysis results.
        """
        results = {
            'per_layer': {},
            'summary': {},
        }

        # Group by layer
        layer_stats = defaultdict(list)
        for (timestep, layer_idx), stats in self.stats.items():
            layer_stats[layer_idx].append(stats)

        for layer_idx, stats_list in sorted(layer_stats.items()):
            # Aggregate across timesteps
            q_head_maxes = defaultdict(list)  # head_idx -> list of max values
            k_head_maxes = defaultdict(list)

            for stats in stats_list:
                for h, head_stats in enumerate(stats.q_heads):
                    q_head_maxes[h].append(head_stats.abs_max)
                for h, head_stats in enumerate(stats.k_heads):
                    k_head_maxes[h].append(head_stats.abs_max)

            # Compute average max per head
            q_avg_max = {h: sum(v)/len(v) for h, v in q_head_maxes.items()}
            k_avg_max = {h: sum(v)/len(v) for h, v in k_head_maxes.items()}

            # Variation ratio: max_head / min_head
            q_values = list(q_avg_max.values())
            k_values = list(k_avg_max.values())

            q_ratio = max(q_values) / min(q_values) if min(q_values) > 0 else float('inf')
            k_ratio = max(k_values) / min(k_values) if min(k_values) > 0 else float('inf')

            results['per_layer'][layer_idx] = {
                'q_head_avg_max': q_avg_max,
                'k_head_avg_max': k_avg_max,
                'q_max_min_ratio': q_ratio,
                'k_max_min_ratio': k_ratio,
                'q_global_max': max(q_values),
                'q_global_min': min(q_values),
            }

        # Summary across all layers
        all_q_ratios = [v['q_max_min_ratio'] for v in results['per_layer'].values()]
        all_k_ratios = [v['k_max_min_ratio'] for v in results['per_layer'].values()]

        results['summary'] = {
            'avg_q_head_ratio': sum(all_q_ratios) / len(all_q_ratios),
            'max_q_head_ratio': max(all_q_ratios),
            'avg_k_head_ratio': sum(all_k_ratios) / len(all_k_ratios),
            'max_k_head_ratio': max(all_k_ratios),
        }

        return results

    def analyze_quantization_impact(self, sc_prec: int = 8) -> dict:
        """
        Analyze the impact of per-tensor vs per-head quantization on real data.

        Returns dict with quantization analysis.
        """
        quant_max = 2 ** (sc_prec - 1) - 1  # 127 for sc_prec=8

        results = {
            'per_layer': {},
            'summary': {},
        }

        total_wasted_levels_per_tensor = 0
        total_heads = 0

        # Group by layer
        layer_stats = defaultdict(list)
        for (timestep, layer_idx), stats in self.stats.items():
            layer_stats[layer_idx].append(stats)

        for layer_idx, stats_list in sorted(layer_stats.items()):
            layer_results = {
                'q_per_head_utilization': {},
                'k_per_head_utilization': {},
            }

            # Get global max across all timesteps for this layer
            q_global_max = max(s.q_global_max for s in stats_list)
            k_global_max = max(s.k_global_max for s in stats_list)

            # Per-head max (averaged across timesteps)
            num_heads = len(stats_list[0].q_heads)

            for h in range(num_heads):
                q_head_max = max(s.q_heads[h].abs_max for s in stats_list)
                k_head_max = max(s.k_heads[h].abs_max for s in stats_list)

                # With per-tensor quantization, what range does this head use?
                q_util = (q_head_max / q_global_max) * 100 if q_global_max > 0 else 0
                k_util = (k_head_max / k_global_max) * 100 if k_global_max > 0 else 0

                layer_results['q_per_head_utilization'][h] = q_util
                layer_results['k_per_head_utilization'][h] = k_util

                # Wasted levels
                wasted = quant_max - int(q_util / 100 * quant_max)
                total_wasted_levels_per_tensor += wasted
                total_heads += 1

            layer_results['q_global_max'] = q_global_max
            layer_results['k_global_max'] = k_global_max
            layer_results['num_heads'] = num_heads

            results['per_layer'][layer_idx] = layer_results

        results['summary'] = {
            'avg_wasted_levels_per_tensor': total_wasted_levels_per_tensor / total_heads if total_heads > 0 else 0,
            'total_heads_analyzed': total_heads,
        }

        return results

    def print_summary(self):
        """Print a summary of collected statistics."""
        if not self.stats:
            print("No statistics collected yet.")
            return

        # Get unique timesteps and layers
        timesteps = sorted(set(t for t, l in self.stats.keys()))
        layers = sorted(set(l for t, l in self.stats.keys()))

        print(f"\n{'='*70}")
        print("Q/K STATISTICS SUMMARY")
        print(f"{'='*70}")
        print(f"Timesteps collected: {len(timesteps)} ({min(timesteps)} to {max(timesteps)})")
        print(f"Layers collected: {len(layers)}")

        # Analyze head magnitude variation
        print(f"\n{'='*70}")
        print("HEAD MAGNITUDE VARIATION (why per-head quantization helps)")
        print(f"{'='*70}")

        analysis = self.analyze_head_magnitude_variation()

        print(f"\nSummary across all layers:")
        print(f"  Average Q head max/min ratio: {analysis['summary']['avg_q_head_ratio']:.2f}x")
        print(f"  Maximum Q head max/min ratio: {analysis['summary']['max_q_head_ratio']:.2f}x")
        print(f"  Average K head max/min ratio: {analysis['summary']['avg_k_head_ratio']:.2f}x")
        print(f"  Maximum K head max/min ratio: {analysis['summary']['max_k_head_ratio']:.2f}x")

        print(f"\nPer-layer details (Q head max/min ratio):")
        for layer_idx, layer_data in sorted(analysis['per_layer'].items()):
            ratio = layer_data['q_max_min_ratio']
            print(f"  Layer {layer_idx:2d}: {ratio:6.2f}x  (max={layer_data['q_global_max']:.4f}, min={layer_data['q_global_min']:.4f})")

        # Quantization impact
        print(f"\n{'='*70}")
        print("QUANTIZATION IMPACT ANALYSIS")
        print(f"{'='*70}")

        quant_analysis = self.analyze_quantization_impact()

        print(f"\nWith per-tensor quantization (sc_prec=8, quant_max=127):")
        print(f"  Average wasted quantization levels per head: {quant_analysis['summary']['avg_wasted_levels_per_tensor']:.1f} / 127")

        # Show worst cases
        print(f"\nWorst utilization per layer (Q):")
        for layer_idx, layer_data in sorted(quant_analysis['per_layer'].items())[:5]:
            utils = layer_data['q_per_head_utilization']
            min_util = min(utils.values())
            min_head = min(utils.keys(), key=lambda h: utils[h])
            print(f"  Layer {layer_idx:2d}: Head {min_head} uses only {min_util:.1f}% of range")

        print(f"\n{'='*70}")

    def save_stats(self, filepath: str):
        """Save statistics to JSON file."""
        data = {
            str(k): v.to_dict() for k, v in self.stats.items()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved statistics to {filepath}")

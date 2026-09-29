"""Stochastic Number Generator (SNG) module.

This module provides:
- RNGPool: Manages a pool of RNG instances with caching
- SNG: Wraps RNG + scrambling for stochastic number generation
- Scrambling utilities
"""
from __future__ import annotations

import numpy as np
from typing import Optional
from rng import RNG, create_rng


def apply_scramble(sequence: np.ndarray, scramble: list[int]) -> np.ndarray:
    """
    Apply bit permutation to sequence values.

    Args:
        sequence: Array of integer values
        scramble: Permutation list where scramble[i] = source bit for output bit i

    Returns:
        Scrambled sequence
    """
    result = np.zeros_like(sequence)
    for i, src in enumerate(scramble):
        result |= ((sequence >> src) & 1) << i
    return result


def apply_scramble_batch(sequences: np.ndarray, scrambles: list[list[int]], sc_prec: int) -> np.ndarray:
    """
    Apply different scrambles to multiple sequences efficiently.

    Args:
        sequences: Array of shape (n_elements, stoc_len)
        scrambles: List of n_elements permutations (or None for no scramble)
        sc_prec: Bit precision

    Returns:
        Scrambled sequences of same shape
    """
    n_elements, stoc_len = sequences.shape
    result = np.zeros_like(sequences)

    for bit_pos in range(sc_prec):
        # Gather source bit positions for this output bit across all elements
        src_bits = np.array([s[bit_pos] if s is not None else bit_pos for s in scrambles])
        # Extract source bits and place at output position
        result |= ((sequences >> src_bits[:, None]) & 1) << bit_pos

    return result


def reverse_permutation(n_bits: int) -> list[int]:
    """Generate reverse bit permutation (achieves minimum SCC per paper)."""
    return list(range(n_bits - 1, -1, -1))


def generate_random_permutation(n_bits: int) -> list[int]:
    """Generate a random bit permutation."""
    import random
    perm = list(range(n_bits))
    random.shuffle(perm)
    return perm


class RNGPool:
    """
    Manages a pool of RNG instances with sequence caching.

    The pool creates RNGs from config and caches their sequences
    to avoid regenerating the same sequence multiple times.
    """

    def __init__(self, rng_configs: list[dict], sc_prec: int):
        """
        Initialize the RNG pool.

        Args:
            rng_configs: List of RNG configurations, each containing:
                - type: "lfsr" or "true_random"
                - seed: RNG seed
                - taps: (for LFSR) tap positions
            sc_prec: Bit precision (LFSR length)
        """
        self.sc_prec = sc_prec
        self.rngs: list[RNG] = []
        self._sequence_cache: dict[int, np.ndarray] = {}

        for cfg in rng_configs:
            rng = create_rng(
                rng_type=cfg.get("type", "lfsr"),
                length=sc_prec,
                **{k: v for k, v in cfg.items() if k != "type"}
            )
            self.rngs.append(rng)

    def __len__(self):
        return len(self.rngs)

    def get_sequence(self, rng_id: int, stoc_len: int) -> np.ndarray:
        """
        Get sequence from an RNG (cached).

        Args:
            rng_id: Index into the RNG pool
            stoc_len: Length of sequence to generate

        Returns:
            NumPy array of random values
        """
        if rng_id not in self._sequence_cache:
            self._sequence_cache[rng_id] = self.rngs[rng_id].simulate(stoc_len)
        return self._sequence_cache[rng_id]

    def clear_cache(self):
        """Clear the sequence cache."""
        self._sequence_cache.clear()


class SNG:
    """
    Stochastic Number Generator.

    Wraps an RNG reference + optional scrambling to produce
    stochastic bitstreams for SC computation.
    """

    def __init__(self, rng_pool: RNGPool, rng_id: int,
                 scramble: Optional[list[int]] = None,
                 complement: bool = False):
        """
        Initialize the SNG.

        Args:
            rng_pool: The RNG pool to draw sequences from
            rng_id: Index of the RNG to use from the pool
            scramble: Optional bit permutation to apply
            complement: If True, apply value-complement (max-1) - seq after
                scramble. Same uniform distribution → same expected SC value;
                different realization → ANTITHETIC SC noise. Used for the
                comp's K-side to give η_antithetic ≈ -η_default exactly.
        """
        self.rng_pool = rng_pool
        self.rng_id = rng_id
        self.scramble = scramble
        self.complement = bool(complement)

    def get_sequence(self, stoc_len: int) -> np.ndarray:
        """
        Get the random sequence for this SNG.

        Args:
            stoc_len: Length of sequence

        Returns:
            NumPy array of (possibly scrambled / complemented) random values
        """
        seq = self.rng_pool.get_sequence(self.rng_id, stoc_len)
        if self.scramble is not None:
            seq = apply_scramble(seq, self.scramble)
        else:
            seq = seq.copy()
        if self.complement:
            max_val = 1 << self.rng_pool.sc_prec
            seq = (max_val - 1) - seq
        return seq


class SNGBank:
    """
    Manages SNGs for a matrix (Q or K).

    Efficiently generates all sequences for a matrix's elements.
    """

    def __init__(self, rng_pool: RNGPool, sng_configs: list[dict]):
        """
        Initialize the SNG bank.

        Args:
            rng_pool: The RNG pool
            sng_configs: List of SNG configs, one per element:
                - rng_id: Index into RNG pool
                - scramble: Optional permutation list
        """
        self.rng_pool = rng_pool
        self.sng_configs = sng_configs
        self.sngs = [
            SNG(rng_pool, cfg["rng_id"], cfg.get("scramble"),
                complement=cfg.get("complement", False))
            for cfg in sng_configs
        ]

    def __len__(self):
        return len(self.sngs)

    def get_all_sequences(self, stoc_len: int) -> np.ndarray:
        """
        Get sequences for all SNGs efficiently.

        Args:
            stoc_len: Length of each sequence

        Returns:
            Array of shape (n_elements, stoc_len)
        """
        n_elements = len(self.sngs)

        # Group SNGs by rng_id for efficient batch processing
        # First, get unique rng_ids and their sequences
        rng_id_to_seq = {}
        for sng in self.sngs:
            if sng.rng_id not in rng_id_to_seq:
                rng_id_to_seq[sng.rng_id] = self.rng_pool.get_sequence(sng.rng_id, stoc_len)

        # Check if we can use batch scrambling (all use same base RNG)
        unique_rng_ids = set(sng.rng_id for sng in self.sngs)

        any_complement = any(sng.complement for sng in self.sngs)

        if len(unique_rng_ids) == 1 and not any_complement:
            # All SNGs share same RNG - use efficient batch scrambling
            base_rng_id = list(unique_rng_ids)[0]
            base_seq = rng_id_to_seq[base_rng_id]

            # Collect scrambles
            scrambles = [sng.scramble for sng in self.sngs]

            # Check if any scrambling needed
            if all(s is None for s in scrambles):
                # No scrambling - broadcast single sequence
                return np.broadcast_to(base_seq, (n_elements, stoc_len)).copy()
            else:
                # Batch scramble
                base_seqs = np.broadcast_to(base_seq, (n_elements, stoc_len)).copy()
                return apply_scramble_batch(base_seqs, scrambles, self.rng_pool.sc_prec)
        else:
            # Mixed RNGs OR any complement flag set — process individually so
            # the per-SNG complement handling in SNG.get_sequence runs.
            sequences = np.zeros((n_elements, stoc_len), dtype=np.int32)
            for i, sng in enumerate(self.sngs):
                sequences[i] = sng.get_sequence(stoc_len)
            return sequences

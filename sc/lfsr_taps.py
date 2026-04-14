"""LFSR Taps Search and Management.

This module provides functions to:
1. Search for all valid maximal-length LFSR tap configurations
2. Save/load tap configurations to/from file
3. Provide tap configurations for use in stochastic computing

A maximal-length LFSR of N bits produces a sequence of period 2^N - 1,
visiting all non-zero states exactly once before repeating.

Usage:
    # First time: search and save all valid taps (do this once)
    from lfsr_taps import search_and_save_all_taps
    search_and_save_all_taps(max_bits=16)

    # Later: load and use
    from lfsr_taps import get_taps
    taps_8bit = get_taps(8)  # Returns list of all valid 8-bit tap configs
"""
from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path
from typing import Optional

from rng import LFSR


# Default file path for storing tap configurations
TAPS_FILE = Path(__file__).parent / "lfsr_taps_data.json"


def search_taps(n_bits: int, verbose: bool = True) -> list[list[int]]:
    """
    Search for all valid maximal-length LFSR tap configurations.

    Performs exhaustive enumeration of all possible tap combinations
    and keeps only those that produce maximum-length sequences.

    For an N-bit LFSR, a maximal-length sequence has period 2^N - 1.

    Args:
        n_bits: Number of bits in the LFSR
        verbose: Print progress information

    Returns:
        List of valid tap configurations, each as a list of bit positions
        (0-indexed from LSB, sorted in descending order)

    Note:
        This is computationally expensive for large n_bits.
        - 8 bits: ~few seconds
        - 12 bits: ~minutes
        - 16 bits: ~hours
    """
    max_period = 2 ** n_bits - 1
    valid_taps = []

    # We need at least 2 taps for a valid LFSR
    # The highest tap must be at position n_bits - 1 (MSB)
    # Try all combinations of 2, 4, 6, ... taps (even number required for maximal length)

    bit_positions = list(range(n_bits))
    total_combinations = 0

    for n_taps in range(2, n_bits + 1, 2):  # Even number of taps only
        # MSB (n_bits - 1) must always be a tap
        # Choose remaining n_taps - 1 positions from bits 0 to n_bits - 2
        remaining_positions = list(range(n_bits - 1))
        for combo in combinations(remaining_positions, n_taps - 1):
            total_combinations += 1

    if verbose:
        print(f"Searching {n_bits}-bit LFSR taps ({total_combinations} combinations)...")

    checked = 0
    for n_taps in range(2, n_bits + 1, 2):
        remaining_positions = list(range(n_bits - 1))
        for combo in combinations(remaining_positions, n_taps - 1):
            # Build tap list: MSB + selected positions, sorted descending
            taps = sorted([n_bits - 1] + list(combo), reverse=True)

            # Test if this produces maximal-length sequence
            if _is_maximal_length(taps, n_bits, max_period):
                valid_taps.append(taps)

            checked += 1
            if verbose and checked % 1000 == 0:
                print(f"  Checked {checked}/{total_combinations}, found {len(valid_taps)} valid")

    if verbose:
        print(f"  Found {len(valid_taps)} valid tap configurations for {n_bits} bits")

    return valid_taps


def _is_maximal_length(taps: list[int], n_bits: int, expected_period: int) -> bool:
    """Check if LFSR taps produce a maximal-length sequence."""
    try:
        lfsr = LFSR(seed=1, length=n_bits, taps=taps)
        seen = set()
        for _ in range(expected_period + 1):
            val = lfsr.step()
            if val in seen:
                return len(seen) == expected_period
            seen.add(val)
        return len(seen) == expected_period
    except Exception:
        return False


def save_taps(taps_dict: dict[int, list[list[int]]], filepath: Optional[Path] = None):
    """
    Save tap configurations to JSON file.

    Args:
        taps_dict: Dict mapping bit width to list of valid tap configs
        filepath: Output file path (default: lfsr_taps_data.json)
    """
    if filepath is None:
        filepath = TAPS_FILE

    # Convert keys to strings for JSON
    json_dict = {str(k): v for k, v in taps_dict.items()}

    with open(filepath, 'w') as f:
        json.dump(json_dict, f, indent=2)

    print(f"Saved tap configurations to {filepath}")


def load_taps(filepath: Optional[Path] = None) -> dict[int, list[list[int]]]:
    """
    Load tap configurations from JSON file.

    Args:
        filepath: Input file path (default: lfsr_taps_data.json)

    Returns:
        Dict mapping bit width to list of valid tap configs
    """
    if filepath is None:
        filepath = TAPS_FILE

    if not os.path.exists(filepath):
        return {}

    with open(filepath, 'r') as f:
        json_dict = json.load(f)

    # Convert keys back to integers
    return {int(k): v for k, v in json_dict.items()}


def get_taps(n_bits: int, filepath: Optional[Path] = None) -> list[list[int]]:
    """
    Get all valid tap configurations for given bit width.

    Loads from file if available, otherwise returns empty list.
    Use search_and_save_all_taps() first to populate the file.

    Args:
        n_bits: Number of bits
        filepath: Taps file path (default: lfsr_taps_data.json)

    Returns:
        List of valid tap configurations
    """
    taps_dict = load_taps(filepath)
    return taps_dict.get(n_bits, [])


def search_and_save_all_taps(bit_widths: list[int] = None,
                              filepath: Optional[Path] = None,
                              verbose: bool = True):
    """
    Search for all valid taps for multiple bit widths and save to file.

    This is the main preprocessing function. Run it once to generate
    all tap configurations, then use get_taps() to load them.

    Args:
        bit_widths: List of bit widths to search (default: [4, 5, 6, 7, 8, 10, 12])
        filepath: Output file path (default: lfsr_taps_data.json)
        verbose: Print progress information

    Example:
        >>> search_and_save_all_taps([8, 10, 12, 16])
        >>> taps_8 = get_taps(8)  # Now available
    """
    if bit_widths is None:
        bit_widths = [4, 5, 6, 7, 8, 10, 12]

    if filepath is None:
        filepath = TAPS_FILE

    # Load existing taps to avoid re-searching
    taps_dict = load_taps(filepath)

    for n_bits in bit_widths:
        if n_bits in taps_dict:
            if verbose:
                print(f"Skipping {n_bits} bits (already have {len(taps_dict[n_bits])} configs)")
            continue

        taps_dict[n_bits] = search_taps(n_bits, verbose=verbose)

    save_taps(taps_dict, filepath)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        bit_widths = [int(x) for x in sys.argv[1:]]
    else:
        bit_widths = [4, 5, 6, 7, 8, 9]

    print(f"Searching for valid LFSR taps for bit widths: {bit_widths}")
    search_and_save_all_taps(bit_widths)

    # Print summary
    print("\nSummary:")
    taps_dict = load_taps()
    for n_bits in sorted(taps_dict.keys()):
        print(f"  {n_bits} bits: {len(taps_dict[n_bits])} valid tap configurations")

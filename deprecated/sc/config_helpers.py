"""Helper functions for creating common RNG/SNG configurations."""
from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from sng import reverse_permutation, generate_random_permutation
from lfsr_taps import get_taps, load_taps


# Default directory for saved configs
CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_REGISTRY_FILE = CONFIG_DIR / "config_registry.json"


def get_taps_choices(sc_prec: int) -> list[list[int]]:
    """
    Get all valid tap configurations for given precision.

    Loads from pre-computed file (lfsr_taps_data.json).
    Run `python lfsr_taps.py <bits>` to generate the data first.

    Args:
        sc_prec: Bit precision

    Returns:
        List of valid tap configurations

    Raises:
        ValueError: If no taps available for given precision
    """
    taps = get_taps(sc_prec)
    if not taps:
        raise ValueError(
            f"No pre-computed taps for sc_prec={sc_prec}. "
            f"Run `python lfsr_taps.py {sc_prec}` to generate."
        )
    return taps


def get_default_taps(sc_prec: int) -> list[int]:
    """Get default LFSR taps for a given precision (first from pre-computed list)."""
    taps_list = get_taps_choices(sc_prec)
    return taps_list[0]


def make_default_config(q_e: int, k_e: int, sc_prec: int,
                        taps: Optional[list[int]] = None,
                        seed: int = 125) -> dict:
    """
    Create default config: shared LFSR, K uses reverse scrambling.

    This matches the paper's optimal configuration (67% SCC reduction).

    Args:
        q_e: Number of Q embedding elements
        k_e: Number of K embedding elements
        sc_prec: Bit precision
        taps: LFSR taps (uses default if None)
        seed: RNG seed

    Returns:
        Config dict for matmul_sc
    """
    if taps is None:
        taps = get_default_taps(sc_prec)

    k_scramble = reverse_permutation(sc_prec)

    return {
        "rng_pool": [
            {"type": "lfsr", "seed": seed, "taps": taps},
        ],
        "sng": {
            "q": [{"rng_id": 0, "scramble": None} for _ in range(q_e)],
            "k": [{"rng_id": 0, "scramble": k_scramble} for _ in range(k_e)],
        }
    }


def make_independent_config(q_e: int, k_e: int, sc_prec: int,
                            q_taps: list[int], k_taps: list[int],
                            q_seed: int, k_seed: int,
                            q_scramble: Optional[list[int]] = None,
                            k_scramble: Optional[list[int]] = None) -> dict:
    """
    Create config with independent LFSRs for Q and K.

    Args:
        q_e: Number of Q embedding elements
        k_e: Number of K embedding elements
        sc_prec: Bit precision
        q_taps: LFSR taps for Q
        k_taps: LFSR taps for K
        q_seed: Seed for Q's LFSR
        k_seed: Seed for K's LFSR
        q_scramble: Optional scramble for Q (applied to all elements)
        k_scramble: Optional scramble for K (applied to all elements)

    Returns:
        Config dict for matmul_sc
    """
    return {
        "rng_pool": [
            {"type": "lfsr", "seed": q_seed, "taps": q_taps},
            {"type": "lfsr", "seed": k_seed, "taps": k_taps},
        ],
        "sng": {
            "q": [{"rng_id": 0, "scramble": q_scramble} for _ in range(q_e)],
            "k": [{"rng_id": 1, "scramble": k_scramble} for _ in range(k_e)],
        }
    }


def make_per_element_scramble_config(q_e: int, k_e: int, sc_prec: int,
                                     taps: Optional[list[int]] = None,
                                     seed: int = 125,
                                     k_reverse_of_q: bool = True) -> dict:
    """
    Create config where each element has its own scrambling.

    Args:
        q_e: Number of Q embedding elements
        k_e: Number of K embedding elements
        sc_prec: Bit precision
        taps: LFSR taps (uses default if None)
        seed: RNG seed
        k_reverse_of_q: If True, K's scramble is reverse of Q's scramble

    Returns:
        Config dict for matmul_sc
    """
    if taps is None:
        taps = get_default_taps(sc_prec)

    assert q_e == k_e, "Q and K must have same embedding size for paired scrambling"

    sng_q = []
    sng_k = []

    for _ in range(q_e):
        scramble_q = generate_random_permutation(sc_prec)
        if k_reverse_of_q:
            scramble_k = list(reversed(scramble_q))
        else:
            scramble_k = generate_random_permutation(sc_prec)

        sng_q.append({"rng_id": 0, "scramble": scramble_q})
        sng_k.append({"rng_id": 0, "scramble": scramble_k})

    return {
        "rng_pool": [
            {"type": "lfsr", "seed": seed, "taps": taps},
        ],
        "sng": {
            "q": sng_q,
            "k": sng_k,
        }
    }


def make_fully_independent_config(q_e: int, k_e: int, sc_prec: int,
                                  taps_choices: Optional[list[list[int]]] = None) -> dict:
    """
    Create config where each element has its own independent LFSR.

    Each RNG gets randomly selected taps and seed for true independence.

    Args:
        q_e: Number of Q embedding elements
        k_e: Number of K embedding elements
        sc_prec: Bit precision
        taps_choices: List of valid tap configurations to choose from

    Returns:
        Config dict for matmul_sc
    """
    if taps_choices is None:
        taps_choices = get_taps_choices(sc_prec)

    rng_pool = []

    # Create LFSRs for Q elements - each with random taps and seed
    q_rng_ids = []
    for i in range(q_e):
        taps = random.choice(taps_choices)
        seed = random.randint(1, 2 ** sc_prec - 1)
        rng_pool.append({"type": "lfsr", "seed": seed, "taps": taps})
        q_rng_ids.append(len(rng_pool) - 1)

    # Create LFSRs for K elements - each with random taps and seed
    k_rng_ids = []
    for i in range(k_e):
        taps = random.choice(taps_choices)
        seed = random.randint(1, 2 ** sc_prec - 1)
        rng_pool.append({"type": "lfsr", "seed": seed, "taps": taps})
        k_rng_ids.append(len(rng_pool) - 1)

    return {
        "rng_pool": rng_pool,
        "sng": {
            "q": [{"rng_id": rng_id, "scramble": None} for rng_id in q_rng_ids],
            "k": [{"rng_id": rng_id, "scramble": None} for rng_id in k_rng_ids],
        }
    }


def make_separated_random_config(q_e: int, k_e: int, sc_prec: int,
                                  taps_choices: Optional[list[list[int]]] = None) -> dict:
    """
    Create random config with strict Q/K separation (all Q use RNG_0, all K use RNG_1).

    This is the optimal architecture - searches over taps and seeds only.

    Args:
        q_e: Number of Q embedding elements
        k_e: Number of K embedding elements
        sc_prec: Bit precision
        taps_choices: List of valid tap configurations to choose from

    Returns:
        Random config dict for matmul_sc
    """
    if taps_choices is None:
        taps_choices = get_taps_choices(sc_prec)

    q_taps = random.choice(taps_choices)
    k_taps = random.choice(taps_choices)
    q_seed = random.randint(1, 2 ** sc_prec - 1)
    k_seed = random.randint(1, 2 ** sc_prec - 1)

    return {
        "rng_pool": [
            {"type": "lfsr", "seed": q_seed, "taps": q_taps},
            {"type": "lfsr", "seed": k_seed, "taps": k_taps},
        ],
        "sng": {
            "q": [{"rng_id": 0, "scramble": None} for _ in range(q_e)],
            "k": [{"rng_id": 1, "scramble": None} for _ in range(k_e)],
        }
    }


def make_random_config(q_e: int, k_e: int, sc_prec: int,
                       taps_choices: Optional[list[list[int]]] = None,
                       use_scrambling: bool = True) -> dict:
    """
    Create a random configuration for DSE with 2 RNGs (Q→RNG_0, K→RNG_1).

    All Q elements use RNG_0, all K elements use RNG_1.
    Each element can have its own random scrambling.

    Args:
        q_e: Number of Q embedding elements
        k_e: Number of K embedding elements
        sc_prec: Bit precision
        taps_choices: List of valid tap configurations to choose from
        use_scrambling: Whether to use random per-element scrambling

    Returns:
        Random config dict for matmul_sc
    """
    if taps_choices is None:
        taps_choices = get_taps_choices(sc_prec)

    # Create 2 RNGs: RNG_0 for Q, RNG_1 for K
    q_taps = random.choice(taps_choices)
    k_taps = random.choice(taps_choices)
    q_seed = random.randint(1, 2 ** sc_prec - 1)
    k_seed = random.randint(1, 2 ** sc_prec - 1)

    rng_pool = [
        {"type": "lfsr", "seed": q_seed, "taps": q_taps},
        {"type": "lfsr", "seed": k_seed, "taps": k_taps},
    ]

    # Create SNG configs - all Q use RNG_0, all K use RNG_1
    sng_q = []
    sng_k = []

    for _ in range(q_e):
        scramble = generate_random_permutation(sc_prec) if use_scrambling and random.random() > 0.5 else None
        sng_q.append({"rng_id": 0, "scramble": scramble})

    for _ in range(k_e):
        scramble = generate_random_permutation(sc_prec) if use_scrambling and random.random() > 0.5 else None
        sng_k.append({"rng_id": 1, "scramble": scramble})

    return {
        "rng_pool": rng_pool,
        "sng": {
            "q": sng_q,
            "k": sng_k,
        }
    }


def make_single_rng_reverse_config(q_e: int, k_e: int, sc_prec: int,
                                    taps_choices: Optional[list[list[int]]] = None,
                                    use_per_element_scrambling: bool = True) -> dict:
    """
    Create a random configuration for DSE with single RNG where K = reverse(Q).

    Single RNG shared by Q and K. Each Q element can have random scrambling,
    and the corresponding K element uses the reverse of Q's scrambling.

    Args:
        q_e: Number of Q embedding elements
        k_e: Number of K embedding elements
        sc_prec: Bit precision
        taps_choices: List of valid tap configurations to choose from
        use_per_element_scrambling: If True, each Q gets random scramble and K gets reverse.
                                    If False, Q has no scramble and K uses global reverse.

    Returns:
        Random config dict for matmul_sc
    """
    if taps_choices is None:
        taps_choices = get_taps_choices(sc_prec)

    assert q_e == k_e, "Q and K must have same size for paired reverse scrambling"

    taps = random.choice(taps_choices)
    seed = random.randint(1, 2 ** sc_prec - 1)

    rng_pool = [{"type": "lfsr", "seed": seed, "taps": taps}]

    sng_q = []
    sng_k = []

    if use_per_element_scrambling:
        # Each Q element gets random scramble, K gets reverse
        for _ in range(q_e):
            q_scramble = generate_random_permutation(sc_prec)
            k_scramble = list(reversed(q_scramble))
            sng_q.append({"rng_id": 0, "scramble": q_scramble})
            sng_k.append({"rng_id": 0, "scramble": k_scramble})
    else:
        # Q has no scramble, K uses global reverse
        k_scramble = reverse_permutation(sc_prec)
        for _ in range(q_e):
            sng_q.append({"rng_id": 0, "scramble": None})
            sng_k.append({"rng_id": 0, "scramble": k_scramble})

    return {
        "rng_pool": rng_pool,
        "sng": {
            "q": sng_q,
            "k": sng_k,
        }
    }


# =============================================================================
# Config Save/Load Functions
# =============================================================================

def save_config(
    config: dict,
    filepath: Optional[Path] = None,
    operation: str = "matmul",
    sc_prec: int = 8,
    binary_prec: str = "fp8_e4m3",
    operand_sizes: Optional[dict] = None,
    error: Optional[float] = None,
    input_seed: Optional[int] = None,
    dse_params: Optional[dict] = None,
    notes: str = "",
) -> Path:
    """
    Save a config to file with metadata.

    The saved format is general and supports:
    - Any RNG types (lfsr, true_random, etc.)
    - Shared RNGs between operands (via rng_id references)
    - Any scrambling permutations
    - Input seed for exact reproducibility

    Args:
        config: The config dict with "rng_pool" and "sng" keys
        filepath: Where to save (default: auto-generated in configs/)
        operation: Operation type ("matmul", "add", "mul", etc.)
        sc_prec: Stochastic computing precision
        binary_prec: Input binary precision ("fp8_e4m3", "int8", etc.)
        operand_sizes: Dict of operand sizes, e.g. {"q": 64, "k": 64}
        error: Error achieved with this config
        input_seed: Numpy random seed for exact reproducibility
        dse_params: DSE parameters used to find this config
        notes: Additional notes

    Returns:
        Path where config was saved
    """
    # Create config directory if needed
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Generate filepath if not provided
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{operation}_{sc_prec}bit_{timestamp}.json"
        filepath = CONFIG_DIR / filename

    # Build metadata
    metadata = {
        "operation": operation,
        "sc_prec": sc_prec,
        "binary_prec": binary_prec,
        "operand_sizes": operand_sizes or {},
        "error": error,
        "input_seed": input_seed,
        "created_at": datetime.now().isoformat(),
        "dse_params": dse_params,
        "notes": notes,
    }

    # Build full document
    doc = {
        "metadata": metadata,
        "config": config,
    }

    # Save to file
    with open(filepath, 'w') as f:
        json.dump(doc, f, indent=2)

    return Path(filepath)


def load_config(filepath: Path) -> tuple[dict, dict]:
    """
    Load a config from file.

    Args:
        filepath: Path to the config file

    Returns:
        Tuple of (config, metadata)
    """
    with open(filepath, 'r') as f:
        doc = json.load(f)

    return doc["config"], doc["metadata"]


def save_best_config(
    config: dict,
    error: float,
    operation: str = "matmul",
    sc_prec: int = 8,
    binary_prec: str = "fp8_e4m3",
    operand_sizes: Optional[dict] = None,
    input_seed: Optional[int] = None,
    dse_params: Optional[dict] = None,
    notes: str = "",
) -> Path:
    """
    Save the best config and update the registry.

    The registry tracks the best config for each (operation, sc_prec, binary_prec) combo.

    Args:
        config: The best config dict
        error: Error achieved
        operation: Operation type
        sc_prec: Stochastic computing precision
        binary_prec: Input binary precision
        operand_sizes: Dict of operand sizes
        input_seed: Numpy random seed for reproducibility
        dse_params: DSE parameters used
        notes: Additional notes

    Returns:
        Path where config was saved
    """
    # Save the config file
    filepath = save_config(
        config=config,
        operation=operation,
        sc_prec=sc_prec,
        binary_prec=binary_prec,
        operand_sizes=operand_sizes,
        error=error,
        input_seed=input_seed,
        dse_params=dse_params,
        notes=notes,
    )

    # Update registry
    registry = _load_registry()

    key = f"{operation}_{sc_prec}bit_{binary_prec}"

    # Check if this is better than existing
    if key in registry:
        existing_error = registry[key].get("error")
        if existing_error is not None and error >= existing_error:
            # Existing is better or equal, don't update
            return filepath

    registry[key] = {
        "filepath": str(filepath),
        "error": error,
        "operation": operation,
        "sc_prec": sc_prec,
        "binary_prec": binary_prec,
        "operand_sizes": operand_sizes,
        "input_seed": input_seed,
        "updated_at": datetime.now().isoformat(),
    }

    _save_registry(registry)

    return filepath


def load_best_config(
    operation: str = "matmul",
    sc_prec: int = 8,
    binary_prec: str = "fp8_e4m3",
) -> tuple[Optional[dict], Optional[dict]]:
    """
    Load the best config for a given operation/precision.

    Args:
        operation: Operation type
        sc_prec: Stochastic computing precision
        binary_prec: Input binary precision

    Returns:
        Tuple of (config, metadata) or (None, None) if not found
    """
    registry = _load_registry()
    key = f"{operation}_{sc_prec}bit_{binary_prec}"

    if key not in registry:
        return None, None

    filepath = registry[key]["filepath"]
    if not os.path.exists(filepath):
        return None, None

    return load_config(filepath)


def list_saved_configs() -> list[dict]:
    """
    List all configs in the registry.

    Returns:
        List of registry entries with metadata
    """
    registry = _load_registry()
    return list(registry.values())


def _load_registry() -> dict:
    """Load the config registry."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not CONFIG_REGISTRY_FILE.exists():
        return {}

    with open(CONFIG_REGISTRY_FILE, 'r') as f:
        return json.load(f)


def _save_registry(registry: dict):
    """Save the config registry."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)


# =============================================================================
# Sobol Config Functions (SCGen-compatible)
# =============================================================================

def make_sobol_simple_config(q_e: int, k_e: int, sc_prec: int = 8) -> dict:
    """
    Create simple Sobol config with decorrelated Q and K (no DSE needed).

    Q uses default seed [1,1,1,...], K uses seed [1,3,1,1,...].
    These seeds produce sequences with SCC ≈ 0 for accurate SC multiplication.

    Args:
        q_e: Number of Q embedding elements
        k_e: Number of K embedding elements
        sc_prec: Bit precision (default 8)

    Returns:
        Config dict for matmul_sc
    """
    return {
        "rng_pool": [
            {"type": "sobol", "seed_type": "q"},  # Q: seed [1,1,1,...]
            {"type": "sobol", "seed_type": "k"},  # K: seed [1,3,1,1,...] (decorrelated)
        ],
        "sng": {
            "q": [{"rng_id": 0, "scramble": None} for _ in range(q_e)],
            "k": [{"rng_id": 1, "scramble": None} for _ in range(k_e)],
        }
    }


def make_sobol_antithetic_config(q_e: int, k_e: int, sc_prec: int = 8) -> dict:
    """Sobol config with VALUE-COMPLEMENT antithetic on K-side (true antithetic).

    Same Sobol RNGs as ``make_sobol_simple_config`` for both Q and K, but the
    K-side SNGs apply ``τ' = (max−1) − τ`` after their RNG draw. Same uniform
    distribution → preserved low-discrepancy and preserved expected SC matmul
    value. Different bit realization → SC noise η_antithetic = −η_default to
    leading order.

    For variant selection: pairing this with ``make_sobol_simple_config``
    realizes the polarity-flip primitive `pol ∈ {+1, −1}` from the math.
    Per-block pick of pol then provably reduces ‖e_FP + pol·η‖² by
    `2|⟨e_FP, η⟩|` (positive iff e_FP correlates with the comp's noise).
    """
    return {
        "rng_pool": [
            {"type": "sobol", "seed_type": "q"},
            {"type": "sobol", "seed_type": "k"},
        ],
        "sng": {
            "q": [{"rng_id": 0, "scramble": None} for _ in range(q_e)],
            "k": [{"rng_id": 1, "scramble": None, "complement": True}
                  for _ in range(k_e)],
        }
    }


def make_sobol_altseed_config(q_e: int, k_e: int, sc_prec: int = 8,
                                k_seed: Optional[list[int]] = None) -> dict:
    """Sobol config with an *alternative* K-side direction-vector seed.

    Same Sobol RNG family as ``make_sobol_simple_config`` for Q (low-
    discrepancy preserved); K uses a different valid Sobol seed → an
    independent low-discrepancy sequence with low SCC vs Q. Different bit
    pattern, *same* uniformity properties, *different* SC noise realization.

    For the QwT-SC compensator, this gives a hardware-friendly knob: pick
    one of a small set of K-side Sobol seeds per block at calibration so the
    comp's SC noise correlates with — and partially cancels — the block's
    residual SC noise.

    Args:
        q_e, k_e, sc_prec: as in ``make_sobol_simple_config``
        k_seed: Length-``sc_prec`` direction vector. Default for sc_prec=8 is
            ``[1, 3, 7, 15, 31, 63, 127, 255]`` — the canonical Niederreiter-
            style seed (different family than k=[1,1,1,1,9,1,41,255]).
    """
    if k_seed is None:
        if sc_prec == 8:
            k_seed = [1, 3, 7, 15, 31, 63, 127, 255]
        else:
            # Generic valid seed: seed[i] = 2^(i+1) - 1 (odd, in range).
            k_seed = [(1 << (i + 1)) - 1 for i in range(sc_prec)]
    return {
        "rng_pool": [
            {"type": "sobol", "seed_type": "q"},
            {"type": "sobol", "seed": list(k_seed)},
        ],
        "sng": {
            "q": [{"rng_id": 0, "scramble": None} for _ in range(q_e)],
            "k": [{"rng_id": 1, "scramble": None} for _ in range(k_e)],
        }
    }


def make_sobol_dse_config(q_e: int, k_e: int, sc_prec: int,
                          use_scrambling: bool = True) -> dict:
    """
    Create Sobol config with random seeds for DSE.

    Uses random valid Sobol seeds for Q and K, with optional scrambling.
    DSE searches over seed combinations.

    Args:
        q_e: Number of Q embedding elements
        k_e: Number of K embedding elements
        sc_prec: Bit precision (for scrambling)
        use_scrambling: Whether to apply random scrambling

    Returns:
        Random config dict for matmul_sc (for DSE)
    """
    from rng import Sobol

    # Generate random seeds
    q_seed = Sobol.random_seed(sc_prec)
    k_seed = Sobol.random_seed(sc_prec)

    rng_pool = [
        {"type": "sobol", "seed": q_seed},
        {"type": "sobol", "seed": k_seed},
    ]

    # Create SNG configs with optional scrambling
    sng_q = []
    sng_k = []

    for _ in range(q_e):
        scramble = generate_random_permutation(sc_prec) if use_scrambling and random.random() > 0.5 else None
        sng_q.append({"rng_id": 0, "scramble": scramble})

    for _ in range(k_e):
        scramble = generate_random_permutation(sc_prec) if use_scrambling and random.random() > 0.5 else None
        sng_k.append({"rng_id": 1, "scramble": scramble})

    return {
        "rng_pool": rng_pool,
        "sng": {
            "q": sng_q,
            "k": sng_k,
        }
    }


def make_sobol_custom_seed_config(q_e: int, k_e: int, sc_prec: int,
                                   q_seed: list = None, k_seed: list = None,
                                   use_scrambling: bool = False) -> dict:
    """
    Create Sobol config with custom seeds.

    Args:
        q_e: Number of Q embedding elements
        k_e: Number of K embedding elements
        sc_prec: Bit precision (for scrambling)
        q_seed: Custom seed for Q (default: [1,1,1,...])
        k_seed: Custom seed for K (default: [1,3,1,1,...])
        use_scrambling: Whether to apply scrambling

    Returns:
        Config dict for matmul_sc
    """
    rng_pool = [
        {"type": "sobol", "seed": q_seed, "seed_type": "q"},
        {"type": "sobol", "seed": k_seed, "seed_type": "k"},
    ]

    sng_q = []
    sng_k = []

    for _ in range(q_e):
        scramble = generate_random_permutation(sc_prec) if use_scrambling else None
        sng_q.append({"rng_id": 0, "scramble": scramble})

    for _ in range(k_e):
        scramble = generate_random_permutation(sc_prec) if use_scrambling else None
        sng_k.append({"rng_id": 1, "scramble": scramble})

    return {
        "rng_pool": rng_pool,
        "sng": {
            "q": sng_q,
            "k": sng_k,
        }
    }

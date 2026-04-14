"""Random Number Generator implementations for Stochastic Computing."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import random


class RNG(ABC):
    """Base class for Random Number Generators."""

    @abstractmethod
    def simulate(self, num_cycles: int) -> np.ndarray:
        """
        Generate a sequence of random integers.

        Args:
            num_cycles: Number of random values to generate

        Returns:
            NumPy array of random integers.
        """
        ...

    @abstractmethod
    def reset(self):
        """Reset the RNG to its initial state."""
        ...


class LFSR(RNG):
    """Linear Feedback Shift Register RNG implementation."""

    def __init__(self, seed: int, length: int, taps: list[int]):
        """
        Initialize the LFSR.

        Args:
            seed: Initial seed value (integer, non-zero)
            length: Number of bits/flip-flops in the register
            taps: List of bit positions for feedback taps (0-indexed from LSB)
        """
        if seed == 0:
            raise ValueError("LFSR seed must be non-zero")
        self.seed = seed
        self.length = length
        self.taps = taps
        self.reg = seed  # Current register state

    def reset(self):
        """Reset the register to initial seed."""
        self.reg = self.seed

    def step(self) -> int:
        """Perform one LFSR cycle and return the output."""
        output = self.reg
        xor_bit = 0
        for bit in self.taps:
            xor_bit ^= (self.reg >> bit) & 1
        self.reg = (self.reg >> 1) | (xor_bit << (self.length - 1))
        return output

    def simulate(self, num_cycles: int = None) -> np.ndarray:
        """
        Generate a sequence of random integers.

        Args:
            num_cycles: Number of values to generate. Defaults to 2^length.

        Returns:
            NumPy array of random integers (register values per cycle).
        """
        if num_cycles is None:
            num_cycles = 2 ** self.length
        self.reset()
        return np.array([self.step() for _ in range(num_cycles)], dtype=np.int32)


class TrueRandom(RNG):
    """True random number generator using Python's random module."""

    def __init__(self, seed: int, length: int):
        """
        Initialize the TrueRandom generator.

        Args:
            seed: Seed for the random number generator
            length: Number of bits (determines max value: 2^length - 1)
        """
        self.seed = seed
        self.length = length
        self.max_val = 2 ** length - 1

    def reset(self):
        """Reset the random generator to initial seed."""
        random.seed(self.seed)

    def simulate(self, num_cycles: int = None) -> np.ndarray:
        """
        Generate a sequence of random integers.

        Args:
            num_cycles: Number of values to generate. Defaults to 2^length.

        Returns:
            NumPy array of random integers in range [0, 2^length - 1].
        """
        if num_cycles is None:
            num_cycles = 2 ** self.length
        self.reset()
        # Generate random values in [0, max_val] (matching UnarySim convention)
        return np.array([random.randint(0, self.max_val) for _ in range(num_cycles)], dtype=np.int32)


class Sobol(RNG):
    """
    Sobol low-discrepancy sequence generator (SCGen-compatible).

    Uses Gray-code XOR algorithm with direction vectors derived from seed.
    Different seeds produce independent sequences with low SCC.

    For stochastic computing:
    - Use seed_type="q" for Q operand (default seed [1,1,1,...])
    - Use seed_type="k" for K operand (default seed [1,3,1,1,...])
    - These produce sequences with SCC ≈ 0 for accurate multiplication
    """

    def __init__(self, length: int, seed: list = None, seed_type: str = "q"):
        """
        Initialize the Sobol sequence generator.

        Args:
            length: Number of bits (precision)
            seed: Direction vector seed [s0, s1, ..., s_{n-1}].
                  s0 must be 1, s_i must be odd in [1, 2^(i+1)).
                  If None, uses default seed based on seed_type.
            seed_type: "q" or "k" - selects default seed for Q or K operand.
        """
        self.length = length
        self.max_val = 2 ** length

        # Set seed
        if seed is not None:
            self.seed = list(seed)
        else:
            self.seed = self._default_seed(seed_type)

        # Compute direction vectors from seed (SCGen formula)
        self._direction_vectors = self._compute_direction_vectors()

        # State
        self._index = 0
        self._value = 0

    def _default_seed(self, seed_type: str) -> list:
        """Generate default seed for given length and type."""
        if seed_type == "k":
            # Optimal K seed with SCC ≈ 0 relative to Q seed
            # Found via search: gives SCC = 0.000046 with default Q seed
            if self.length == 8:
                return [1, 1, 1, 1, 9, 1, 41, 255]
            # Fallback for other lengths: use [1,1,1,...] with different later elements
            seed = [1] * self.length
            if self.length >= 5:
                seed[4] = 9
            if self.length >= 7:
                seed[6] = 41
            return seed
        # Q seed: simple [1,1,1,...]
        return [1] * self.length

    def _compute_direction_vectors(self) -> list:
        """
        Compute direction vectors from seed (SCGen algorithm).

        Vs[i] = seed[i] / 2^(i+1) * 2^length
        """
        Vs = []
        for i in range(self.length):
            v = int(self.seed[i] / (1 << (i + 1)) * (1 << self.length))
            Vs.append(v)
        return Vs

    def _lsz(self, n: int) -> int:
        """Least Significant Zero position (0-indexed)."""
        if n == 0:
            return 0
        pos = 0
        while (n >> pos) & 1:
            pos += 1
        return pos

    def reset(self):
        """Reset to initial state."""
        self._index = 0
        self._value = 0

    def _step(self) -> int:
        """Generate next Sobol value using Gray-code XOR."""
        val = self._value  # Return current value before XOR
        k = self._lsz(self._index)
        self._index += 1
        if k < self.length:
            self._value ^= self._direction_vectors[k]
        return val

    def step(self) -> int:
        """Generate next value in [0, 2^length-1] (matching UnarySim convention)."""
        return self._step()

    def simulate(self, num_cycles: int = None) -> np.ndarray:
        """
        Generate a sequence of Sobol numbers.

        Args:
            num_cycles: Number of values. Defaults to 2^length.

        Returns:
            NumPy array of quasi-random integers in [0, 2^length-1].
        """
        if num_cycles is None:
            num_cycles = 2 ** self.length
        self.reset()
        return np.array([self.step() for _ in range(num_cycles)], dtype=np.int32)

    @staticmethod
    def random_seed(length: int) -> list:
        """
        Generate a random valid Sobol seed.

        seed[0] must be 1, seed[i] must be odd in [1, 2^(i+1)).
        """
        seed = [1]  # seed[0] is always 1
        for i in range(1, length):
            max_val = 1 << (i + 1)
            val = random.randint(0, max_val // 2 - 1) * 2 + 1
            seed.append(val)
        return seed


def create_rng(rng_type: str, length: int, **kwargs) -> RNG:
    """
    Factory function to create RNG instances.

    Args:
        rng_type: Type of RNG ("lfsr", "true_random", "sobol")
        length: Bit width (precision)
        **kwargs: Additional arguments for specific RNG types
            - For LFSR: seed, taps
            - For TrueRandom: seed
            - For Sobol: seed (list) or seed_type ("q" or "k")

    Returns:
        RNG instance
    """
    if rng_type == "lfsr":
        return LFSR(seed=kwargs["seed"], length=length, taps=kwargs["taps"])
    elif rng_type == "true_random":
        return TrueRandom(seed=kwargs["seed"], length=length)
    elif rng_type == "sobol":
        return Sobol(
            length=length,
            seed=kwargs.get("seed"),
            seed_type=kwargs.get("seed_type", "q")
        )
    else:
        raise ValueError(f"Unknown RNG type: {rng_type}")

"""Per-episode records and their summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    success: bool
    max_reward: float
    max_coverage: float
    steps: int
    final_block_pose: tuple[float, float, float]
    watch_max_coverage: tuple[float, ...] = ()
    displacement_overlapped_agent: bool = False


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must be in [0, {trials}], got {successes}")

    rate = successes / trials
    z_squared = z * z
    denominator = 1 + z_squared / trials
    center = (rate + z_squared / (2 * trials)) / denominator
    half_width = (
        z * math.sqrt(rate * (1 - rate) / trials + z_squared / (4 * trials * trials)) / denominator
    )
    low = 0.0 if successes == 0 else center - half_width
    high = 1.0 if successes == trials else center + half_width
    return low, high


def spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Spearman rank correlation, with tied values given their average rank."""
    if len(first) != len(second):
        raise ValueError(f"lengths differ: {len(first)} and {len(second)}")
    first_centered = _ranks(first) - (len(first) + 1) / 2
    second_centered = _ranks(second) - (len(second) + 1) / 2
    denominator = math.sqrt(
        float(first_centered @ first_centered * (second_centered @ second_centered))
    )
    return float(first_centered @ second_centered) / denominator


def _ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    _, inverse, counts = np.unique(array[order], return_inverse=True, return_counts=True)
    average_by_value = np.bincount(inverse, weights=np.arange(1, len(array) + 1)) / counts
    ranks = np.empty(len(array))
    ranks[order] = average_by_value[inverse]
    return ranks


def summarize(results: Sequence[EpisodeResult]) -> dict:
    if not results:
        raise ValueError("cannot summarize zero episodes")

    trials = len(results)
    successes = sum(result.success for result in results)
    return {
        "trials": trials,
        "successes": successes,
        "success_rate": successes / trials,
        "wilson_95": wilson_interval(successes, trials),
        "mean_max_reward": sum(result.max_reward for result in results) / trials,
        "mean_max_coverage": sum(result.max_coverage for result in results) / trials,
        "mean_steps": sum(result.steps for result in results) / trials,
    }

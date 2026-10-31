"""Turn the pilot rates into prediction ranges that include chance and rate uncertainty.

Each simulated replication draws the rates from Beta(successes + 1, failures + 1), then draws
episode outcomes. Ranges are the 2.5th and 97.5th percentiles. Uses resets only.

Usage: uv run python scripts/prediction_intervals.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from compass.design import MOVED_GOAL, PILOT_IN_SUPPORT, PILOT_OUT_OF_SUPPORT, SEEDS
from compass.env_ops import make_env
from compass.metrics import spearman
from compass.policy_loader import load_pretrained
from compass.se2 import goal_frame
from compass.support import start_in_support

GATE1 = (40, 50)
REPLICATIONS = 20_000
RNG_SEED = 20261031
GRID_PREDICTION = Path("../results/grid_prediction.json")
OUTPUT = Path("../results/prediction_intervals.json")


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    trials = len(SEEDS)
    default_rate = _beta(rng, GATE1)
    in_rate, out_rate = _beta(rng, PILOT_IN_SUPPORT), _beta(rng, PILOT_OUT_OF_SUPPORT)
    moved_in_support = _starts_in_support(MOVED_GOAL)

    normal = rng.binomial(trials, default_rate)
    moved = rng.binomial(moved_in_support, in_rate) + rng.binomial(
        trials - moved_in_support, out_rate
    )
    report = {
        "replications": REPLICATIONS,
        "rng_seed": RNG_SEED,
        "rates": {
            "gate1": GATE1,
            "in_support": PILOT_IN_SUPPORT,
            "out_of_support": PILOT_OUT_OF_SUPPORT,
        },
        "condition_normal": _interval(normal),
        "moved_goal_starts_in_support": moved_in_support,
        "condition_moved_with_compass": _interval(moved),
        "grid": _grid(rng, in_rate, out_rate),
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rates"}, indent=2))


def _beta(rng: np.random.Generator, counts: tuple[int, int]) -> np.ndarray:
    successes, total = counts
    return rng.beta(successes + 1, total - successes + 1, size=REPLICATIONS)


def _interval(counts: np.ndarray) -> list[int]:
    low, high = np.percentile(counts, [2.5, 97.5])
    return [int(np.floor(low)), int(np.ceil(high))]


def _starts_in_support(goal) -> int:
    normalizer = load_pretrained(device="cpu").normalizer
    frame = goal_frame(goal)
    env = make_env()
    try:
        flags = []
        for seed in SEEDS:
            observation, _ = env.reset(seed=seed)
            flags.append(
                start_in_support(
                    frame, observation["agent_pos"], observation["environment_state"], normalizer
                )
            )
        return sum(flags)
    finally:
        env.close()


def _grid(rng: np.random.Generator, in_rate: np.ndarray, out_rate: np.ndarray) -> dict:
    cells = [
        cell
        for cell in json.loads(GRID_PREDICTION.read_text(encoding="utf-8"))["cells"]
        if not cell["pilot_goal"]
    ]
    in_support = np.array([cell["starts_in_support"] for cell in cells])
    predicted = np.array([cell["predicted_successes"] for cell in cells])
    edge = np.array([cell["edge"] for cell in cells])
    trials = len(SEEDS)

    counts = rng.binomial(in_support, in_rate[:, None]) + rng.binomial(
        trials - in_support, out_rate[:, None]
    )
    rho_if_h1 = np.array([spearman(in_support, row) for row in counts])
    pooled_rate = (in_support * in_rate[:, None] + (trials - in_support) * out_rate[:, None]).sum(
        axis=1
    ) / (trials * len(cells))
    no_relation = rng.binomial(trials, np.repeat(pooled_rate[:, None], len(cells), axis=1))
    rho_if_no_relation = np.array([spearman(in_support, row) for row in no_relation])
    residual = counts - predicted
    gap_if_no_walls = residual[:, edge].mean(axis=1) - residual[:, ~edge].mean(axis=1)
    return {
        "cells_judged": len(cells),
        "edge_cells": int(edge.sum()),
        "inner_cells": int((~edge).sum()),
        "rho_if_h1": [float(x) for x in np.percentile(rho_if_h1, [2.5, 50, 97.5])],
        "rho_95th_if_no_relation": float(np.percentile(rho_if_no_relation, 95)),
        "edge_minus_inner_residual_if_no_walls": [
            float(x) for x in np.percentile(gap_if_no_walls, [2.5, 5, 50, 95, 97.5])
        ],
    }


if __name__ == "__main__":
    main()

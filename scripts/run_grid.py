"""Run the pre-registered goal grid with the compass (predictions.md).

Each cell is run exactly like condition 3: the 20 seeds as one batch, torch seed 1.
Results are written after every cell so an interrupted run keeps what it finished.

Usage: uv run python scripts/run_grid.py --device mps
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from compass.agent import CompassAgent, PolicyAgent
from compass.design import SEEDS, grid_goals, is_edge, is_pilot_goal
from compass.policy_loader import load_pretrained
from compass.rollout import EpisodeSpec, run_batch

TORCH_SEED = 1
PREDICTION = Path("../results/grid_prediction.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--output", type=Path, default=Path("../results/grid.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plain = PolicyAgent(load_pretrained(device=args.device))
    predicted = {
        tuple(cell["goal"]): cell
        for cell in json.loads(PREDICTION.read_text(encoding="utf-8"))["cells"]
    }
    started = time.perf_counter()
    cells = []
    for index, goal in enumerate(grid_goals(), start=1):
        specs = [EpisodeSpec(seed=seed, goal=goal) for seed in SEEDS]
        results = run_batch(CompassAgent(plain, [goal] * len(specs)), specs, torch_seed=TORCH_SEED)
        successes = sum(item.success for item in results)
        cells = [
            *cells,
            {
                "goal": goal,
                "successes": successes,
                "starts_in_support": predicted[goal]["starts_in_support"],
                "predicted_successes": predicted[goal]["predicted_successes"],
                "edge": is_edge(goal),
                "pilot_goal": is_pilot_goal(goal),
                "episodes": [dataclasses.asdict(item) for item in results],
            },
        ]
        _write(args.output, cells, time.perf_counter() - started)
        print(f"{index:2d}/50 goal {goal} -> {successes}/20", flush=True)


def _write(path: Path, cells: list[dict], elapsed: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "seeds": [SEEDS[0], SEEDS[-1]],
        "torch_seed": TORCH_SEED,
        "elapsed_seconds": elapsed,
        "cells": cells,
    }
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

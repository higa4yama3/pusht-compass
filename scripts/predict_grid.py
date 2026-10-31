"""Predict, before running the policy, how often the compass succeeds at each grid goal.

Each start is checked against the training range in the goal's frame, then given the pilot's
success rate for that kind of start. Uses resets only; the policy never moves.

Usage: uv run python scripts/predict_grid.py
"""

from __future__ import annotations

import json
from pathlib import Path

from compass.design import (
    GRID_ANGLES,
    GRID_POSITIONS,
    PILOT_IN_SUPPORT,
    PILOT_OUT_OF_SUPPORT,
    SEEDS,
    grid_goals,
    is_edge,
    is_pilot_goal,
)
from compass.env_ops import make_env
from compass.policy_loader import load_pretrained
from compass.se2 import goal_frame
from compass.support import start_in_support

OUTPUT = Path("../results/grid_prediction.json")


def main() -> None:
    normalizer = load_pretrained(device="cpu").normalizer
    in_rate = PILOT_IN_SUPPORT[0] / PILOT_IN_SUPPORT[1]
    out_rate = PILOT_OUT_OF_SUPPORT[0] / PILOT_OUT_OF_SUPPORT[1]
    starts = _starts()

    cells = []
    for goal in grid_goals():
        frame = goal_frame(goal)
        flags = [start_in_support(frame, *starts[seed], normalizer) for seed in SEEDS]
        predicted = sum(in_rate if flag else out_rate for flag in flags)
        cells.append(
            {
                "goal": goal,
                "starts_in_support": sum(flags),
                "predicted_successes": round(predicted, 1),
                "edge": is_edge(goal),
                "pilot_goal": is_pilot_goal(goal),
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "seeds": [SEEDS[0], SEEDS[-1]],
                "rate_in_support": in_rate,
                "rate_out_of_support": out_rate,
                "cells": cells,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _print_tables(cells)


def _starts() -> dict[int, tuple]:
    env = make_env()
    try:
        starts = {}
        for seed in SEEDS:
            observation, _ = env.reset(seed=seed)
            starts[seed] = (observation["agent_pos"], observation["environment_state"])
        return starts
    finally:
        env.close()


def _print_tables(cells: list[dict]) -> None:
    by_goal = {tuple(cell["goal"]): cell for cell in cells}
    for angle in GRID_ANGLES:
        print(f"\nangle {angle:.3f}  (predicted successes / 20, starts in support)")
        for y in GRID_POSITIONS:
            row = [by_goal[(x, y, angle)] for x in GRID_POSITIONS]
            print(
                f"y={y:5.0f} "
                + " ".join(
                    f"{c['predicted_successes']:5.1f}({c['starts_in_support']:2d})" for c in row
                )
            )


if __name__ == "__main__":
    main()

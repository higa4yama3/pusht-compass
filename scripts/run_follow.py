"""Post-hoc, not pre-registered: hop the goal four times within one episode.

Runs the hop chain on seeds 101-120 and records how many goals each seed reached.

Usage: uv run python scripts/run_follow.py --device mps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compass.agent import CompassAgent, PolicyAgent
from compass.design import SEEDS
from compass.follow import follow_goals, hop_goals
from compass.policy_loader import load_pretrained

MAX_STEPS_PER_GOAL = 260
OUTPUT = Path("../results/follow.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _measure(PolicyAgent(load_pretrained(device=args.device)), hop_goals())


def _measure(plain: PolicyAgent, goals) -> dict:
    outcomes = [
        follow_goals(plain, CompassAgent, seed=seed, goals=goals, max_steps=MAX_STEPS_PER_GOAL)
        for seed in SEEDS
    ]
    report = {
        "note": "post-hoc, not in predictions.md",
        "goals": goals,
        "max_steps_per_goal": MAX_STEPS_PER_GOAL,
        "seeds": [
            {"seed": item.seed, "reached": item.reached, "steps": item.steps} for item in outcomes
        ],
        "reached_all": sum(len(item.reached) == len(goals) for item in outcomes),
        "reached_counts": [len(item.reached) for item in outcomes],
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"reached all {len(goals)} goals: {report['reached_all']}/{len(SEEDS)}")
    print("goals reached per seed:", report["reached_counts"])
    return report


if __name__ == "__main__":
    main()

"""Gate 2, second half: with the compass, does the T reach goals the policy never saw?

Usage: uv run python scripts/check_goals.py --device mps
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

from compass.agent import CompassAgent, PolicyAgent
from compass.env_ops import DEFAULT_GOAL
from compass.metrics import summarize
from compass.policy_loader import load_pretrained
from compass.rollout import EpisodeSpec, run_batch

CENTRAL_GOALS = {
    "left_up_rot60": (200.0, 300.0, DEFAULT_GOAL[2] + math.pi / 3),
    "right_down_rot-90": (320.0, 200.0, DEFAULT_GOAL[2] - math.pi / 2),
    "center_rot180": (256.0, 256.0, DEFAULT_GOAL[2] + math.pi),
}
SEEDS = tuple(range(101, 121))
# Gate 2 passes when the compass reaches every central goal at least this often.
MIN_SUCCESSES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--output", type=Path, default=Path("../results/gate2_central_goals.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_agent = PolicyAgent(load_pretrained(device=args.device))
    report = {"seeds": [SEEDS[0], SEEDS[-1]], "min_successes": MIN_SUCCESSES, "goals": {}}

    for name, goal in CENTRAL_GOALS.items():
        specs = [EpisodeSpec(seed=seed, goal=goal) for seed in SEEDS]
        compass = CompassAgent(policy_agent, [goal] * len(specs))
        with_compass = run_batch(compass, specs, torch_seed=1)
        without_compass = run_batch(policy_agent, specs, torch_seed=1)
        report["goals"][name] = {
            "goal": goal,
            "with_compass": summarize(with_compass),
            "without_compass": summarize(without_compass),
            "episodes_with_compass": [dataclasses.asdict(item) for item in with_compass],
            "episodes_without_compass": [dataclasses.asdict(item) for item in without_compass],
        }
        print(
            f"{name:18s} compass {summarize(with_compass)['successes']:2d}/20  "
            f"plain {summarize(without_compass)['successes']:2d}/20"
        )

    passed = all(
        entry["with_compass"]["successes"] >= MIN_SUCCESSES for entry in report["goals"].values()
    )
    report["gate"] = "PASS" if passed else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"GATE 2: {report['gate']}")


if __name__ == "__main__":
    main()

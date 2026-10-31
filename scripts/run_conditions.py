"""Run the pre-registered condition table (predictions.md).

Usage: uv run python scripts/run_conditions.py --device mps
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from compass.agent import CompassAgent, PolicyAgent
from compass.design import MOVED_GOAL, SEEDS
from compass.env_ops import DEFAULT_GOAL, Pose
from compass.metrics import EpisodeResult, summarize
from compass.policy_loader import load_pretrained
from compass.rollout import EpisodeSpec, run_batch

TORCH_SEED = 1
TOLD_GOAL: Pose = (MOVED_GOAL[0], MOVED_GOAL[1], MOVED_GOAL[2] + math.pi / 2)
DISPLACE_AFTER = 44
DISPLACEMENT = (60.0, 0.0)
NOISE_LEVELS = (0.0, 2.0, 5.0, 10.0, 20.0)
NEAR_PX = 30.0
REACHED = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--output", type=Path, default=Path("../results/conditions.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plain = PolicyAgent(load_pretrained(device=args.device))
    started = time.perf_counter()
    report = {"seeds": [SEEDS[0], SEEDS[-1]], "torch_seed": TORCH_SEED, "conditions": {}}

    def run(name: str, agent, make_spec: Callable[[int], EpisodeSpec], derive=None) -> None:
        results = run_batch(agent, [make_spec(seed) for seed in SEEDS], torch_seed=TORCH_SEED)
        entry = {"summary": summarize(results), **(derive(results) if derive else {})}
        entry["episodes"] = [dataclasses.asdict(item) for item in results]
        report["conditions"][name] = entry
        extra = {key: value for key, value in entry.items() if key not in ("summary", "episodes")}
        print(f"{name:28s} {entry['summary']['successes']:2d}/20  {extra}")

    run("1_normal", plain, lambda seed: EpisodeSpec(seed=seed))
    run(
        "2_moved_without_compass",
        plain,
        lambda seed: EpisodeSpec(seed=seed, goal=MOVED_GOAL, watch_goals=(DEFAULT_GOAL,)),
        _near_training_goal,
    )
    run(
        "3_moved_with_compass",
        CompassAgent(plain, [MOVED_GOAL] * len(SEEDS)),
        lambda seed: EpisodeSpec(seed=seed, goal=MOVED_GOAL),
    )
    run(
        "4_told_rotated_90",
        CompassAgent(plain, [TOLD_GOAL] * len(SEEDS)),
        lambda seed: EpisodeSpec(seed=seed, goal=MOVED_GOAL, watch_goals=(TOLD_GOAL,)),
        _rotated_landing,
    )
    run(
        "5_displaced_after_44",
        plain,
        lambda seed: EpisodeSpec(
            seed=seed, displace_after=DISPLACE_AFTER, displacement=DISPLACEMENT
        ),
        _overlaps,
    )
    for sigma in NOISE_LEVELS:
        run(
            f"6_noise_{sigma:g}px",
            plain,
            lambda seed, sigma=sigma: EpisodeSpec(seed=seed, keypoint_noise=sigma),
        )

    report["elapsed_seconds"] = time.perf_counter() - started
    report["told_goal"] = TOLD_GOAL
    report["moved_goal"] = MOVED_GOAL
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['elapsed_seconds']:.0f}s")


def _near_training_goal(results: Sequence[EpisodeResult]) -> dict:
    distances = [math.dist(item.final_block_pose[:2], DEFAULT_GOAL[:2]) for item in results]
    return {
        "stopped_within_30px_of_training_goal": sum(d <= NEAR_PX for d in distances),
        "median_distance_to_training_goal": statistics.median(distances),
        "median_distance_to_current_goal": statistics.median(
            math.dist(item.final_block_pose[:2], MOVED_GOAL[:2]) for item in results
        ),
    }


def _rotated_landing(results: Sequence[EpisodeResult]) -> dict:
    offsets = [_angle_offset_degrees(item.final_block_pose[2], MOVED_GOAL[2]) for item in results]
    return {
        "reached_told_goal": sum(item.watch_max_coverage[0] > REACHED for item in results),
        "median_angle_offset_from_true_goal_deg": statistics.median(offsets),
        "median_distance_to_true_goal": statistics.median(
            math.dist(item.final_block_pose[:2], MOVED_GOAL[:2]) for item in results
        ),
    }


def _overlaps(results: Sequence[EpisodeResult]) -> dict:
    return {"overlapped_agent": sum(item.displacement_overlapped_agent for item in results)}


def _angle_offset_degrees(angle: float, reference: float) -> float:
    return math.degrees(math.atan2(math.sin(angle - reference), math.cos(angle - reference)))


if __name__ == "__main__":
    main()

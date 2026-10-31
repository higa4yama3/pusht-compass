"""Gate 1: does the converted checkpoint reproduce the published 71% success rate?

Usage: uv run python scripts/check_baseline.py --device mps
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import time
from pathlib import Path

from compass.agent import PolicyAgent
from compass.metrics import summarize
from compass.policy_loader import REPO_ID, REVISION, load_pretrained
from compass.rollout import EpisodeSpec, run_batch

PUBLISHED_SUCCESS_RATE = 0.71
PUBLISHED_MEAN_MAX_REWARD = 0.97
# 95% range of successes out of 50 when the true rate is 71%.
GATE_SUCCESS_RANGE = (29, 42)
GATE_MIN_MEAN_MAX_REWARD = 0.93


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("../results/baseline.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    agent = PolicyAgent(load_pretrained(device=args.device))

    started = time.perf_counter()
    results = []
    for offset in range(0, len(seeds), args.batch_size):
        batch = [EpisodeSpec(seed=seed) for seed in seeds[offset : offset + args.batch_size]]
        results.extend(run_batch(agent, batch, torch_seed=offset))
    elapsed = time.perf_counter() - started

    summary = summarize(results)
    passed = (
        GATE_SUCCESS_RANGE[0] <= summary["successes"] <= GATE_SUCCESS_RANGE[1]
        and summary["mean_max_reward"] >= GATE_MIN_MEAN_MAX_REWARD
    )
    report = {
        "gate": "PASS" if passed else "FAIL",
        "policy": {"repo_id": REPO_ID, "revision": REVISION},
        "versions": {
            name: importlib.metadata.version(name) for name in ("lerobot", "gym-pusht", "torch")
        },
        "device": args.device,
        "seeds": [seeds[0], seeds[-1]],
        "elapsed_seconds": elapsed,
        "seconds_per_episode": elapsed / len(results),
        "published": {
            "success_rate": PUBLISHED_SUCCESS_RATE,
            "mean_max_reward": PUBLISHED_MEAN_MAX_REWARD,
        },
        "gate_criteria": {
            "success_range": GATE_SUCCESS_RANGE,
            "min_mean_max_reward": GATE_MIN_MEAN_MAX_REWARD,
        },
        "summary": summary,
        "episodes": [dataclasses.asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"successes {summary['successes']}/{summary['trials']}  wilson95 {summary['wilson_95']}")
    print(f"mean max reward {summary['mean_max_reward']:.3f}")
    print(f"{elapsed:.1f}s total, {elapsed / len(results):.2f}s per episode")
    print(f"GATE 1: {report['gate']}")


if __name__ == "__main__":
    main()

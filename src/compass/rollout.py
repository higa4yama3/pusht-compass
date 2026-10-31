"""Run several PushT episodes in lockstep so the policy sees them as one batch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch

from compass.agent import Agent
from compass.env_ops import (
    DEFAULT_GOAL,
    MAX_STEPS,
    Pose,
    block_pose,
    coverage,
    coverage_against,
    displace_block,
    make_env,
    observe,
    set_goal,
)
from compass.metrics import EpisodeResult

SUCCESS_THRESHOLD = 0.95
# Keeps the observation noise apart from the env's and torch's random streams.
NOISE_STREAM = 7919


@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    goal: Pose = DEFAULT_GOAL
    displace_after: int | None = None
    displacement: tuple[float, float] = (0.0, 0.0)
    keypoint_noise: float = 0.0
    watch_goals: tuple[Pose, ...] = ()


@dataclass(frozen=True)
class _Progress:
    done: bool = False
    success: bool = False
    max_coverage: float = 0.0
    steps: int = 0
    watch_max: tuple[float, ...] = ()
    overlapped: bool = False


def run_batch(
    agent: Agent,
    specs: Sequence[EpisodeSpec],
    max_steps: int = MAX_STEPS,
    torch_seed: int = 0,
) -> list[EpisodeResult]:
    """Roll out one episode per spec. Finished episodes stop stepping but stay in the batch."""
    envs = [make_env(max_steps) for _ in specs]
    try:
        observations = [_reset(env, spec) for env, spec in zip(envs, specs, strict=True)]
        progress = [_start(env, spec) for env, spec in zip(envs, specs, strict=True)]
        noise = [_noise_rng(spec) for spec in specs]
        torch.manual_seed(torch_seed)
        agent.reset()
        for step in range(max_steps):
            if all(item.done for item in progress):
                break
            displaced = [
                _maybe_displace(env, spec, item, observation, step)
                for env, spec, item, observation in zip(
                    envs, specs, progress, observations, strict=True
                )
            ]
            observations = [observation for observation, _ in displaced]
            progress = [item for _, item in displaced]
            actions = agent.act(*_stack(observations, specs, noise))
            stepped = [
                _step(env, spec, action, item) if not item.done else (observation, item)
                for env, spec, action, item, observation in zip(
                    envs, specs, actions, progress, observations, strict=True
                )
            ]
            observations = [observation for observation, _ in stepped]
            progress = [item for _, item in stepped]
        return [
            _result(spec, item, env) for spec, item, env in zip(specs, progress, envs, strict=True)
        ]
    finally:
        for env in envs:
            env.close()


def _reset(env, spec: EpisodeSpec) -> dict:
    observation, _ = env.reset(seed=spec.seed)
    set_goal(env, spec.goal)
    return observation


def _start(env, spec: EpisodeSpec) -> _Progress:
    return _Progress(
        max_coverage=coverage(env),
        watch_max=tuple(coverage_against(env, goal) for goal in spec.watch_goals),
    )


def _noise_rng(spec: EpisodeSpec) -> np.random.Generator | None:
    if spec.keypoint_noise <= 0:
        return None
    return np.random.default_rng([NOISE_STREAM, spec.seed])


def _maybe_displace(
    env, spec: EpisodeSpec, item: _Progress, observation: dict, step: int
) -> tuple[dict, _Progress]:
    if item.done or spec.displace_after != step:
        return observation, item
    overlapped = displace_block(env, spec.displacement)
    return observe(env), replace(item, overlapped=overlapped)


def _stack(
    observations: Sequence[dict],
    specs: Sequence[EpisodeSpec],
    noise: Sequence[np.random.Generator | None],
) -> tuple[np.ndarray, np.ndarray]:
    agent_pos = np.stack([np.asarray(item["agent_pos"]) for item in observations])
    keypoints = np.stack(
        [
            _seen_keypoints(np.asarray(item["environment_state"]), spec, rng)
            for item, spec, rng in zip(observations, specs, noise, strict=True)
        ]
    )
    return agent_pos, keypoints


def _seen_keypoints(
    keypoints: np.ndarray, spec: EpisodeSpec, rng: np.random.Generator | None
) -> np.ndarray:
    if rng is None:
        return keypoints
    return keypoints + rng.normal(0.0, spec.keypoint_noise, size=keypoints.shape)


def _step(env, spec: EpisodeSpec, action: np.ndarray, item: _Progress) -> tuple[dict, _Progress]:
    observation, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
    success = item.success or bool(info["is_success"])
    watch_max = tuple(
        max(best, coverage_against(env, goal))
        for best, goal in zip(item.watch_max, spec.watch_goals, strict=True)
    )
    return observation, replace(
        item,
        done=terminated or truncated or success,
        success=success,
        max_coverage=max(item.max_coverage, float(info["coverage"])),
        steps=item.steps + 1,
        watch_max=watch_max,
    )


def _result(spec: EpisodeSpec, item: _Progress, env) -> EpisodeResult:
    return EpisodeResult(
        seed=spec.seed,
        success=item.success,
        max_reward=min(item.max_coverage / SUCCESS_THRESHOLD, 1.0),
        max_coverage=item.max_coverage,
        steps=item.steps,
        final_block_pose=block_pose(env),
        watch_max_coverage=item.watch_max,
        displacement_overlapped_agent=item.overlapped,
    )

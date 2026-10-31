"""One episode in which the goal hops several times. The compass is handed over again at each hop."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from compass.env_ops import DEFAULT_GOAL, Pose, coverage, make_env, set_goal

# Each hop moves the goal 80 px and turns it, from where the previous goal was.
HOPS: tuple[Pose, ...] = (
    (80.0, 0.0, math.pi / 4),
    (0.0, 80.0, -math.pi / 3),
    (-80.0, 0.0, math.pi / 2),
    (0.0, -80.0, -math.pi / 4),
)
REACHED_COVERAGE = 0.95
HOLD_STEPS = 8


@dataclass(frozen=True)
class FollowOutcome:
    seed: int
    reached: tuple[int, ...]
    steps: int


def hop_goals(start: Pose = DEFAULT_GOAL, hops: Sequence[Pose] = HOPS) -> tuple[Pose, ...]:
    goals: tuple[Pose, ...] = (start,)
    for dx, dy, turn in hops:
        x, y, angle = goals[-1]
        goals = (*goals, (x + dx, y + dy, angle + turn))
    return goals


def follow_goals(
    inner,
    compass_factory: Callable,
    seed: int,
    goals: Sequence[Pose],
    max_steps: int,
    hold: int = HOLD_STEPS,
    torch_seed: int = 1,
) -> FollowOutcome:
    """Give each goal up to `max_steps`; a goal counts as reached after `hold` covered steps."""
    env = make_env(max_steps=len(goals) * max_steps + 1)
    try:
        observation, _ = env.reset(seed=seed)
        torch.manual_seed(torch_seed)
        agent = compass_factory(inner, [goals[0]])
        agent.reset()
        reached: tuple[int, ...] = ()
        steps = 0
        for index, goal in enumerate(goals):
            set_goal(env, goal)
            if index > 0:
                agent = agent.retarget([goal])
            streak = 0
            for _ in range(max_steps):
                action = agent.act(
                    np.asarray(observation["agent_pos"])[None],
                    np.asarray(observation["environment_state"])[None],
                )[0]
                observation, *_ = env.step(np.asarray(action, dtype=np.float32))
                steps += 1
                streak = streak + 1 if coverage(env) > REACHED_COVERAGE else 0
                if streak >= hold:
                    reached = (*reached, index)
                    break
        return FollowOutcome(seed=seed, reached=reached, steps=steps)
    finally:
        env.close()

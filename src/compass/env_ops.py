"""Small operations on gym-pusht environments.

gym-pusht hard-codes the goal at (256, 256, pi/4) inside `_setup`, which runs on every reset, so the
goal has to be set again after each reset.
"""

from __future__ import annotations

import math
import warnings

import gymnasium as gym
import numpy as np
from shapely.geometry import Point

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
    import gym_pusht  # noqa: F401  registers gym_pusht/PushT-v0
    from gym_pusht.envs.pusht import pymunk_to_shapely

ENV_ID = "gym_pusht/PushT-v0"
DEFAULT_GOAL: tuple[float, float, float] = (256.0, 256.0, math.pi / 4)
MAX_STEPS = 300
VISUALIZATION_SIZE = 680
AGENT_RADIUS = 15.0

Pose = tuple[float, float, float]


def make_env(max_steps: int = MAX_STEPS) -> gym.Env:
    return gym.make(
        ENV_ID,
        obs_type="environment_state_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=max_steps,
        visualization_width=VISUALIZATION_SIZE,
        visualization_height=VISUALIZATION_SIZE,
    )


def set_goal(env: gym.Env, goal: Pose) -> None:
    env.unwrapped.goal_pose = np.array(goal, dtype=np.float64)


def block_pose(env: gym.Env) -> Pose:
    block = env.unwrapped.block
    return (float(block.position[0]), float(block.position[1]), float(block.angle))


def coverage(env: gym.Env) -> float:
    return float(env.unwrapped._get_coverage())


def coverage_against(env: gym.Env, goal: Pose) -> float:
    """Share of `goal` covered by the block, the same measure gym-pusht uses for its own goal."""
    inner = env.unwrapped
    goal_geometry = pymunk_to_shapely(
        inner.get_goal_pose_body(np.array(goal, dtype=np.float64)), inner.block.shapes
    )
    block_geometry = pymunk_to_shapely(inner.block, inner.block.shapes)
    return float(goal_geometry.intersection(block_geometry).area / goal_geometry.area)


def displace_block(env: gym.Env, offset: tuple[float, float]) -> bool:
    """Teleport the block by `offset` and report whether it now overlaps the agent."""
    inner = env.unwrapped
    x, y = inner.block.position
    inner.block.position = (x + offset[0], y + offset[1])
    inner.space.reindex_shapes_for_body(inner.block)
    block_geometry = pymunk_to_shapely(inner.block, inner.block.shapes)
    return bool(block_geometry.distance(Point(*inner.agent.position)) < AGENT_RADIUS)


def observe(env: gym.Env) -> dict:
    return env.unwrapped.get_obs()

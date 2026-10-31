"""Agents that turn PushT observations into target positions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch

from compass.env_ops import Pose
from compass.policy_loader import ENV_STATE_KEY, STATE_KEY, LoadedPolicy
from compass.se2 import Frame, goal_frame, to_canonical, to_world

ARENA_LIMITS = (0.0, 512.0)


class Agent(Protocol):
    def reset(self) -> None: ...

    def act(self, agent_pos: np.ndarray, keypoints: np.ndarray) -> np.ndarray: ...


class PolicyAgent:
    """The pretrained policy as is, in world coordinates."""

    def __init__(self, loaded: LoadedPolicy) -> None:
        self._loaded = loaded

    def reset(self) -> None:
        self._loaded.policy.reset()

    def act(self, agent_pos: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        normalizer = self._loaded.normalizer
        batch = {
            STATE_KEY: self._to_tensor(normalizer.normalize(STATE_KEY, agent_pos)),
            ENV_STATE_KEY: self._to_tensor(normalizer.normalize(ENV_STATE_KEY, keypoints)),
        }
        with torch.inference_mode():
            normalized_action = self._loaded.policy.select_action(batch)
        return normalizer.unnormalize_action(normalized_action.detach().cpu().numpy())

    def _to_tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(
            self._loaded.device
        )


class CompassAgent:
    """Shows the inner agent the world as if each episode's goal were the training goal."""

    def __init__(self, inner: Agent, goals: Sequence[Pose]) -> None:
        self._inner = inner
        self.goals: tuple[Pose, ...] = tuple(goals)
        self._frames: tuple[Frame, ...] = tuple(goal_frame(goal) for goal in self.goals)

    def reset(self) -> None:
        self._inner.reset()

    def retarget(self, goals: Sequence[Pose]) -> CompassAgent:
        """A new compass for new goals. The policy's history was seen in the old frame, so it goes."""
        self._inner.reset()
        return CompassAgent(self._inner, goals)

    def act(self, agent_pos: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        if len(agent_pos) != len(self._frames):
            raise ValueError(f"got {len(agent_pos)} episodes for {len(self._frames)} goals")
        canonical_pos = np.stack(
            [to_canonical(frame, row) for frame, row in zip(self._frames, agent_pos, strict=True)]
        )
        canonical_keypoints = np.stack(
            [
                to_canonical(frame, row.reshape(-1, 2)).reshape(-1)
                for frame, row in zip(self._frames, keypoints, strict=True)
            ]
        )
        canonical_action = self._inner.act(canonical_pos, canonical_keypoints)
        world_action = np.stack(
            [
                to_world(frame, row)
                for frame, row in zip(self._frames, np.asarray(canonical_action), strict=True)
            ]
        )
        return np.clip(world_action, *ARENA_LIMITS)

"""Rigid transforms that move a goal onto the policy's training goal.

`to_canonical` maps world coordinates into the frame where `goal` sits at the default goal
(256, 256, pi/4). The policy only ever saw that goal, so it is shown the world from there.
Written as `R p + t` so that the default goal is an exact identity, not just a near one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from compass.env_ops import DEFAULT_GOAL, Pose


@dataclass(frozen=True)
class Frame:
    rotation: float
    translation: tuple[float, float]

    def matrix(self) -> np.ndarray:
        cosine, sine = math.cos(self.rotation), math.sin(self.rotation)
        return np.array([[cosine, -sine], [sine, cosine]])


def goal_frame(goal: Pose, canonical: Pose = DEFAULT_GOAL) -> Frame:
    rotation = canonical[2] - goal[2]
    cosine, sine = math.cos(rotation), math.sin(rotation)
    rotated_x = cosine * goal[0] - sine * goal[1]
    rotated_y = sine * goal[0] + cosine * goal[1]
    return Frame(
        rotation=rotation, translation=(canonical[0] - rotated_x, canonical[1] - rotated_y)
    )


def to_canonical(frame: Frame, points: np.ndarray) -> np.ndarray:
    """Points shaped (..., 2) from world to the canonical frame."""
    values = np.asarray(points, dtype=np.float64)
    return values @ frame.matrix().T + np.asarray(frame.translation)


def to_world(frame: Frame, points: np.ndarray) -> np.ndarray:
    """Points shaped (..., 2) from the canonical frame back to the world."""
    values = np.asarray(points, dtype=np.float64)
    return (values - np.asarray(frame.translation)) @ frame.matrix()


def pose_to_canonical(frame: Frame, pose: Pose) -> Pose:
    (x, y), angle = to_canonical(frame, np.array([pose[0], pose[1]])), pose[2] + frame.rotation
    return (float(x), float(y), float(angle))

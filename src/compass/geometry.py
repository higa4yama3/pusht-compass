"""The T's shape in gym-pusht, without a simulator.

gym-pusht builds the T from two boxes with scale 30 and length 4: a 120 x 30 bar and a 30 x 90
stem below it. The walls are segments at 5 and 506 with radius 2.
"""

from __future__ import annotations

import math

import numpy as np

from compass.env_ops import Pose

# In the order pymunk stores them, which is the order of gym-pusht's keypoints.
T_LOCAL_CORNERS = np.array(
    [
        [-60.0, 0.0],
        [60.0, 0.0],
        [60.0, 30.0],
        [-60.0, 30.0],
        [-15.0, 30.0],
        [15.0, 30.0],
        [15.0, 120.0],
        [-15.0, 120.0],
    ]
)
# The T's outline, walking around the whole shape once.
T_LOCAL_OUTLINE = np.array(
    [
        [-60.0, 0.0],
        [60.0, 0.0],
        [60.0, 30.0],
        [15.0, 30.0],
        [15.0, 120.0],
        [-15.0, 120.0],
        [-15.0, 30.0],
        [-60.0, 30.0],
    ]
)
WALL_INNER = (7.0, 504.0)


def t_corners(pose: Pose) -> np.ndarray:
    """The 8 corners of a T at `pose`, in the same order as gym-pusht's keypoints."""
    return _place(T_LOCAL_CORNERS, pose)


def t_outline(pose: Pose) -> np.ndarray:
    """The T's outline at `pose` as a closed polygon (first point repeated at the end)."""
    placed = _place(T_LOCAL_OUTLINE, pose)
    return np.vstack([placed, placed[:1]])


def _place(local: np.ndarray, pose: Pose) -> np.ndarray:
    cosine, sine = math.cos(pose[2]), math.sin(pose[2])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + np.array(pose[:2])


def fits_inside_walls(pose: Pose) -> bool:
    corners = t_corners(pose)
    return bool(np.all(corners >= WALL_INNER[0]) and np.all(corners <= WALL_INNER[1]))

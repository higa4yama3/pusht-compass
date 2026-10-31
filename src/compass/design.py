"""The experiment as fixed before running it. Changing anything here after the predictions are
committed has to be reported."""

from __future__ import annotations

import math

from compass.env_ops import DEFAULT_GOAL, Pose

SEEDS: tuple[int, ...] = tuple(range(101, 121))

GRID_POSITIONS: tuple[float, ...] = (96.0, 176.0, 256.0, 336.0, 416.0)
GRID_ANGLES: tuple[float, ...] = (DEFAULT_GOAL[2], DEFAULT_GOAL[2] + math.pi)
EDGE_POSITIONS = frozenset({GRID_POSITIONS[0], GRID_POSITIONS[-1]})

# Pilot (gate 2, three central goals): successes when the start, seen through the compass,
# was inside the training range versus outside it.
PILOT_IN_SUPPORT = (23, 32)
PILOT_OUT_OF_SUPPORT = (11, 28)
PILOT_GOALS: tuple[Pose, ...] = (
    (200.0, 300.0, DEFAULT_GOAL[2] + math.pi / 3),
    (320.0, 200.0, DEFAULT_GOAL[2] - math.pi / 2),
    (256.0, 256.0, DEFAULT_GOAL[2] + math.pi),
)

# Goal used by the condition table. Not one of the pilot goals.
MOVED_GOAL: Pose = (300.0, 320.0, DEFAULT_GOAL[2] - math.pi / 3)


def grid_goals() -> tuple[Pose, ...]:
    return tuple(
        (x, y, angle) for angle in GRID_ANGLES for y in GRID_POSITIONS for x in GRID_POSITIONS
    )


def is_edge(goal: Pose) -> bool:
    return goal[0] in EDGE_POSITIONS or goal[1] in EDGE_POSITIONS


def is_pilot_goal(goal: Pose) -> bool:
    return any(_same_pose(goal, pilot) for pilot in PILOT_GOALS)


def _same_pose(first: Pose, second: Pose) -> bool:
    angle_gap = math.atan2(math.sin(first[2] - second[2]), math.cos(first[2] - second[2]))
    return (
        math.isclose(first[0], second[0])
        and math.isclose(first[1], second[1])
        and abs(angle_gap) < 1e-9
    )

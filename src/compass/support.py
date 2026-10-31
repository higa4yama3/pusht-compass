"""Is a start, seen through the compass, inside what the policy was trained on?

The training range is the min/max that came with the checkpoint, per coordinate.
"""

from __future__ import annotations

import numpy as np

from compass.policy_loader import ENV_STATE_KEY, STATE_KEY, MinMaxNormalizer
from compass.se2 import Frame, to_canonical


def start_in_support(
    frame: Frame,
    agent_pos: np.ndarray,
    keypoints: np.ndarray,
    normalizer: MinMaxNormalizer,
) -> bool:
    canonical_pos = to_canonical(frame, np.asarray(agent_pos).reshape(2))
    canonical_keypoints = to_canonical(frame, np.asarray(keypoints).reshape(8, 2)).reshape(16)
    return _inside(canonical_pos, normalizer, STATE_KEY) and _inside(
        canonical_keypoints, normalizer, ENV_STATE_KEY
    )


def _inside(value: np.ndarray, normalizer: MinMaxNormalizer, key: str) -> bool:
    return bool(
        np.all(value >= normalizer.minimum[key]) and np.all(value <= normalizer.maximum[key])
    )

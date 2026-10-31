"""Load the 2024 checkpoint of lerobot/diffusion_pusht_keypoints into LeRobot 0.6.1.

The checkpoint predates LeRobot's processor pipeline: its config uses `input_shapes`, and the
min/max statistics live inside the weights. Current `DiffusionPolicy` does not normalize, so the
statistics are pulled out into `MinMaxNormalizer` and applied around the policy.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from safetensors.torch import load_file

REPO_ID = "lerobot/diffusion_pusht_keypoints"
REVISION = "58570fc39828d28efa5457aa297a52be27ac3a10"

STATE_KEY = "observation.state"
ENV_STATE_KEY = "observation.environment_state"
ACTION_KEY = "action"

# The legacy normalizer divides by (max - min + 1e-8) and unnormalizes without it.
LEGACY_MIN_MAX_EPS = 1e-8

COPIED_FIELDS = (
    "beta_end",
    "beta_schedule",
    "beta_start",
    "clip_sample",
    "clip_sample_range",
    "crop_is_random",
    "crop_shape",
    "diffusion_step_embed_dim",
    "do_mask_loss_for_padding",
    "down_dims",
    "horizon",
    "kernel_size",
    "n_action_steps",
    "n_groups",
    "n_obs_steps",
    "noise_scheduler_type",
    "num_inference_steps",
    "num_train_timesteps",
    "prediction_type",
    "pretrained_backbone_weights",
    "spatial_softmax_num_keypoints",
    "use_film_scale_modulation",
    "use_group_norm",
    "vision_backbone",
)
TUPLE_FIELDS = frozenset({"crop_shape", "down_dims"})
INPUT_TYPES = MappingProxyType({STATE_KEY: FeatureType.STATE, ENV_STATE_KEY: FeatureType.ENV})
BUFFER_GROUPS = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
FEATURE_BY_BUFFER = MappingProxyType(
    {key.replace(".", "_"): key for key in (STATE_KEY, ENV_STATE_KEY, ACTION_KEY)}
)


@dataclass(frozen=True)
class MinMaxNormalizer:
    minimum: Mapping[str, np.ndarray]
    maximum: Mapping[str, np.ndarray]

    def normalize(self, key: str, value: np.ndarray) -> np.ndarray:
        low, high = self.minimum[key], self.maximum[key]
        scaled = (np.asarray(value, dtype=np.float32) - low) / (high - low + LEGACY_MIN_MAX_EPS)
        return scaled * 2.0 - 1.0

    def unnormalize_action(self, value: np.ndarray) -> np.ndarray:
        low, high = self.minimum[ACTION_KEY], self.maximum[ACTION_KEY]
        return (np.asarray(value, dtype=np.float32) + 1.0) / 2.0 * (high - low) + low


@dataclass(frozen=True)
class LoadedPolicy:
    policy: DiffusionPolicy
    normalizer: MinMaxNormalizer
    device: str


def build_config(legacy: Mapping[str, Any], device: str = "cpu") -> DiffusionConfig:
    _require_min_max(legacy)
    copied = {name: _config_value(name, legacy[name]) for name in COPIED_FIELDS}
    input_features = {
        key: PolicyFeature(type=INPUT_TYPES[key], shape=tuple(shape))
        for key, shape in legacy["input_shapes"].items()
    }
    output_features = {
        key: PolicyFeature(type=FeatureType.ACTION, shape=tuple(shape))
        for key, shape in legacy["output_shapes"].items()
    }
    return DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        device=device,
        **copied,
    )


def split_state_dict(
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], MinMaxNormalizer]:
    """Separate model weights from the legacy normalization buffers."""
    model_state = {key: value for key, value in state.items() if not _is_buffer(key)}
    stats = {group: _collect_group(state, group) for group in BUFFER_GROUPS}
    if not _same_stats(stats["normalize_targets"], stats["unnormalize_outputs"]):
        raise ValueError(
            "action statistics differ between normalize_targets and unnormalize_outputs"
        )

    merged = {**stats["normalize_inputs"], **stats["unnormalize_outputs"]}
    return model_state, MinMaxNormalizer(
        minimum=MappingProxyType({key: _frozen(value["min"]) for key, value in merged.items()}),
        maximum=MappingProxyType({key: _frozen(value["max"]) for key, value in merged.items()}),
    )


def load_pretrained(device: str = "cpu", revision: str = REVISION) -> LoadedPolicy:
    root = Path(snapshot_download(REPO_ID, revision=revision))
    legacy = json.loads((root / "config.json").read_text(encoding="utf-8"))
    policy = DiffusionPolicy(build_config(legacy, device=device))
    model_state, normalizer = split_state_dict(load_file(root / "model.safetensors"))
    policy.load_state_dict(model_state, strict=True)
    policy.to(device)
    policy.eval()
    return LoadedPolicy(policy=policy, normalizer=normalizer, device=device)


def _require_min_max(legacy: Mapping[str, Any]) -> None:
    modes = {**legacy["input_normalization_modes"], **legacy["output_normalization_modes"]}
    unsupported = {key: mode for key, mode in modes.items() if mode != "min_max"}
    if unsupported:
        raise ValueError(f"only min_max normalization is supported, got {unsupported}")


def _config_value(name: str, value: Any) -> Any:
    if name in TUPLE_FIELDS and value is not None:
        return tuple(value)
    return value


def _is_buffer(key: str) -> bool:
    return key.split(".", 1)[0] in BUFFER_GROUPS


def _collect_group(
    state: Mapping[str, torch.Tensor], group: str
) -> dict[str, dict[str, np.ndarray]]:
    collected: dict[str, dict[str, np.ndarray]] = {}
    for key, value in state.items():
        prefix, _, rest = key.partition(".")
        if prefix != group:
            continue
        buffer_name, stat = rest.rsplit(".", 1)
        feature = FEATURE_BY_BUFFER[buffer_name.removeprefix("buffer_")]
        collected = {
            **collected,
            feature: {**collected.get(feature, {}), stat: value.detach().cpu().numpy()},
        }
    return collected


def _same_stats(
    first: Mapping[str, Mapping[str, np.ndarray]], second: Mapping[str, Mapping[str, np.ndarray]]
) -> bool:
    if first.keys() != second.keys():
        return False
    return all(
        np.array_equal(first[feature][stat], second[feature][stat])
        for feature in first
        for stat in ("min", "max")
    )


def _frozen(array: np.ndarray) -> np.ndarray:
    copy = np.array(array, dtype=np.float32)
    copy.setflags(write=False)
    return copy

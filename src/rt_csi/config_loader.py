"""Merge configs/default.yaml + configs/<scene>.yaml + CLI overrides."""

from __future__ import annotations

import os
from typing import Any

import yaml

from .paths import repository_root
from .scenes import SceneConfig


HERE = str(repository_root())
DEFAULT_CONFIG = os.path.join(HERE, "configs", "default.yaml")


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ValueError(f"{path} did not parse as a mapping")
    return d


def load_scene_config(scene_name: str, configs_dir: str | None = None,
                      overrides: dict[str, Any] | None = None) -> SceneConfig:
    cfg_dict: dict[str, Any] = {}
    cfg_dict.update(_load_yaml(DEFAULT_CONFIG))

    cfg_dir = configs_dir or os.path.join(HERE, "configs")
    scene_yaml = os.path.join(cfg_dir, f"{scene_name}.yaml")
    if os.path.isfile(scene_yaml):
        cfg_dict.update(_load_yaml(scene_yaml))

    if overrides:
        cfg_dict.update({k: v for k, v in overrides.items() if v is not None})

    cfg_dict["name"] = scene_name

    # Coerce list -> tuple for tx_position so the dataclass is hashable-friendly.
    if cfg_dict.get("tx_position") is not None:
        cfg_dict["tx_position"] = tuple(cfg_dict["tx_position"])

    valid_keys = set(SceneConfig.__dataclass_fields__.keys())
    unknown = set(cfg_dict.keys()) - valid_keys
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    return SceneConfig(**cfg_dict)

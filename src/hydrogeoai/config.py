"""Configuration loading with `inherits:` deep-merge support."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve(path: str | os.PathLike) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load a YAML config. A key `inherits: <path>` is merged underneath the file's own keys."""
    p = resolve(path)
    with open(p) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherits", None)
    if parent:
        cfg = _deep_merge(load_config(parent), cfg)
    return cfg


def data_dir() -> Path:
    return resolve(os.environ.get("HYDROGEOAI_DATA_DIR", "data"))


def results_dir() -> Path:
    return resolve(os.environ.get("HYDROGEOAI_RESULTS_DIR", "results"))


def registry_url() -> str:
    default = f"sqlite:///{resolve('experiments_registry/registry.db')}"
    return os.environ.get("HYDROGEOAI_REGISTRY_URL", default)

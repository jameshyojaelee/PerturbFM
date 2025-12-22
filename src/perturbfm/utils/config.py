"""Lightweight config resolver."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

from perturbfm.utils.hashing import sha256_json


def load_yaml(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def resolve_config(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    return merge_dicts(defaults, overrides)


def config_hash(cfg: Dict[str, Any]) -> str:
    return sha256_json(cfg)[:8]

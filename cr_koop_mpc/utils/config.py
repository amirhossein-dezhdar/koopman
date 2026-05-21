"""YAML config loading with attribute access and simple inheritance.

A config may contain

    extends: default.yaml

in which case the parent YAML is loaded first and the child values are deep-
merged on top.  This keeps diagnostic overrides such as configs/event_pure.yaml
small and prevents copy/paste drift between experiment configs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dict subclass with attribute access for nested keys."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        super().__init__()
        if data is None:
            return
        for k, v in data.items():
            self[k] = Config(v) if isinstance(v, dict) else v

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    out = dict(parent)
    for k, v in child.items():
        if k == "extends":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Circular config inheritance detected at {path}")
    seen.add(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    parent_name = raw.get("extends")
    if parent_name:
        parent_path = Path(parent_name)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        parent = _load_raw(parent_path, seen)
        raw = _deep_merge(parent, raw)
    return raw


def load_config(path: str | Path) -> Config:
    """Load a YAML config and return a ``Config`` with nested attribute access."""
    return Config(_load_raw(Path(path)))

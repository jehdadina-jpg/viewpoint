"""
Central configuration loader.

All tunables live in config.yaml so that swapping data providers, changing the
universe, or tweaking Black-Litterman parameters never requires a code change.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

BACKEND_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("SBL_CONFIG", BACKEND_DIR / "config.yaml"))


class Config(dict):
    """dict with attribute access + nested `get_path('a.b.c')`."""

    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        return Config(v) if isinstance(v, dict) else v

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


@lru_cache(maxsize=1)
def load_config() -> Config:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw)


def resolve_path(rel: str) -> Path:
    """Paths in config.yaml are relative to the backend/ directory."""
    p = Path(rel)
    return p if p.is_absolute() else BACKEND_DIR / p


def env_secret(env_name: str) -> str | None:
    """Secrets are only ever read from the environment (never committed)."""
    v = os.environ.get(env_name)
    return v.strip() if v else None

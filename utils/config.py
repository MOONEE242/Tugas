"""Central config loader for the trading ecosystem.

Usage:
    from utils.config import load_config
    cfg = load_config()
    cfg["risk"]["risk_per_trade_pct"]
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@functools.lru_cache(maxsize=8)
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_secret(env_var_name: str, required: bool = True) -> str | None:
    """Fetch a credential from the environment (never from config.yaml)."""
    value = os.environ.get(env_var_name)
    if required and not value:
        raise RuntimeError(
            f"Environment variable '{env_var_name}' is not set. "
            "Set MT5 credentials via env vars, see README.md."
        )
    return value

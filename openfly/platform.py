"""Resolve OpenFly-Platform install and import sim bridges from upstream eval."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def openfly_root() -> Path:
    root = Path(os.environ.get("OPENFLY_ROOT", Path.home() / "OpenFly-Platform"))
    if not root.is_dir():
        raise FileNotFoundError(
            f"OPENFLY_ROOT not found: {root}\n"
            "Run: bash ~/drone_project/openfly/setup.sh"
        )
    return root


def load_eval_module() -> ModuleType:
    """Import OpenFly train/eval.py (AirsimBridge, UEBridge, GSBridge, helpers)."""
    root = openfly_root()
    eval_path = root / "train" / "eval.py"
    if not eval_path.is_file():
        raise FileNotFoundError(f"Missing {eval_path}")

    train_dir = str(root / "train")
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)

    spec = importlib.util.spec_from_file_location("openfly_upstream_eval", eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {eval_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pos_ratio_for_env(env_name: str) -> float:
    if "gs" in env_name:
        return 5.15
    return 1.0


def make_bridge(env_name: str, eval_mod: ModuleType | None = None):
    eval_mod = eval_mod or load_eval_module()
    if "airsim" in env_name:
        return eval_mod.AirsimBridge(env_name), 1.0
    if "ue" in env_name:
        return eval_mod.UEBridge(ue_ip="127.0.0.1", ue_port="9000", env_name=env_name), 1.0
    if "gs" in env_name:
        return eval_mod.GSBridge(env_name), 5.15
    raise ValueError(f"Unknown OpenFly env type in {env_name!r}")

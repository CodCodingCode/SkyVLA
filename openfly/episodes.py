"""Load OpenFly VLN episodes from HuggingFace Annotation JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_ANNOTATION_ROOT = Path(
    os.environ.get(
        "OPENFLY_ANNOTATION_DIR",
        str(Path.home() / "assets" / "OpenFly" / "Annotation"),
    )
)

SPLIT_FILES = {
    "eval_test": "eval_test.json",  # shipped with OpenFly-Platform configs/
    "seen": "seen.json",
    "unseen": "unseen.json",
    "train": "train.json",
}


def annotation_path(split: str, root: Path | None = None) -> Path:
    root = root or DEFAULT_ANNOTATION_ROOT
    if split == "eval_test":
        platform = os.environ.get("OPENFLY_ROOT", str(Path.home() / "OpenFly-Platform"))
        p = Path(platform) / "configs" / "eval_test.json"
        if p.is_file():
            return p
    name = SPLIT_FILES.get(split)
    if name is None:
        raise KeyError(f"Unknown split {split!r}. Choices: {list(SPLIT_FILES)}")
    return root / name


def load_episodes(
    split: str = "unseen",
    *,
    root: Path | None = None,
    max_episodes: int = 0,
    env_filter: str | None = None,
    per_env_max_episodes: int = 0,
) -> list[dict[str, Any]]:
    """Load OpenFly episodes for a split.

    Filtering precedence (applied in this order):
      1. ``env_filter``: substring match on ``image_path[0]`` (env name).
      2. ``per_env_max_episodes``: cap episodes-per-env in the natural
         json ordering. Use this to balance training across all envs
         when total data is otherwise dominated by one env (e.g.
         env_ue_bigcity is ~3x the next-largest env in the local image
         dump). Same envs will still appear in the same relative order
         as the source json — we just stop after N episodes of each.
      3. ``max_episodes``: global cap, applied last.
    """
    path = annotation_path(split, root)
    if not path.is_file():
        raise FileNotFoundError(
            f"OpenFly annotations not found: {path}\n"
            f"Run: bash ~/drone_project/openfly/setup.sh"
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if env_filter:
        data = [ep for ep in data if env_filter in ep.get("image_path", "")]
    if per_env_max_episodes > 0:
        counts: dict[str, int] = {}
        filtered: list[dict[str, Any]] = []
        for ep in data:
            env = ep.get("image_path", "").split("/")[0]
            if counts.get(env, 0) < per_env_max_episodes:
                counts[env] = counts.get(env, 0) + 1
                filtered.append(ep)
        data = filtered
    if max_episodes > 0:
        data = data[:max_episodes]
    return data


def group_by_env(episodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for ep in episodes:
        env_name = ep["image_path"].split("/")[0]
        groups.setdefault(env_name, []).append(ep)
    return groups


def episode_env_name(episode: dict[str, Any]) -> str:
    return episode["image_path"].split("/")[0]

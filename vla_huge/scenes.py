"""HUGE-Bench scene registry.

The HUGE-Bench paper ships 4 outdoor 3DGS-Mesh digital-twin scenes; the
Isaac Sim toolchain isn't public yet, so the USD assets below are
placeholders. Each entry resolves to ``warehouse_full`` so the rest of
the training pipeline can be smoke-tested end-to-end. When HUGE drops
its sim, edit ``usd_path`` to the real asset and add scene-specific
POIs / spawn poses.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HugeScene:
    """One HUGE-Bench digital-twin scene.

    Attributes:
        scene_id: HUGE env_id used by ``huge_bench/dataset.py`` (e.g. ``1_office``).
        usd_path: USD relative to ``ISAAC_NUCLEUS_DIR`` (warehouse placeholder
            until HUGE sim drops; replace with the real outdoor asset).
        env_spacing: per-env tile spacing (large outdoor scenes need wide tiles).
        spawn_altitude: drone start height in scene-local frame (m).
        prompts: instruction bank — one of these is randomly chosen per episode.
            Currently mirrors HUGE-Bench task0 instructions for that scene.
    """

    scene_id: str
    usd_path: str
    env_spacing: float = 80.0
    spawn_altitude: float = 5.0
    prompts: list[str] = field(default_factory=list)


# Placeholder USD: the same warehouse_full asset every other vla_warehouse run
# uses. Replace with real HUGE 3DGS-Mesh paths when available.
_PLACEHOLDER_USD = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"


SCENES: dict[str, HugeScene] = {
    "1_office": HugeScene(
        scene_id="1_office",
        usd_path=_PLACEHOLDER_USD,
        env_spacing=60.0,
        spawn_altitude=3.0,
        prompts=[
            "Fly to 70 meters above the twin curved office building complex.",
            "Reach the rooftop garden between the two office towers.",
            "Hover above the main entrance plaza of the office complex.",
        ],
    ),
    "2_park": HugeScene(
        scene_id="2_park",
        usd_path=_PLACEHOLDER_USD,
        env_spacing=80.0,
        spawn_altitude=4.0,
        prompts=[
            "Fly to the wooden gazebo in the centre of the park.",
            "Reach the pond at the south end of the park.",
            "Hover above the playground near the park entrance.",
        ],
    ),
    "3_campus": HugeScene(
        scene_id="3_campus",
        usd_path=_PLACEHOLDER_USD,
        env_spacing=70.0,
        spawn_altitude=4.0,
        prompts=[
            "Fly to the central courtyard between the campus buildings.",
            "Reach the parking lot south of the main academic building.",
            "Hover above the tallest tower on the east side of campus.",
        ],
    ),
    "4_lake": HugeScene(
        scene_id="4_lake",
        usd_path=_PLACEHOLDER_USD,
        env_spacing=100.0,
        spawn_altitude=6.0,
        prompts=[
            "Fly to 70 meters above the large lake surface.",
            "Reach the boathouse on the east shore of the lake.",
            "Hover above the small island near the centre of the lake.",
        ],
    ),
}


def get_scene(scene_id: str) -> HugeScene:
    if scene_id not in SCENES:
        raise KeyError(
            f"Unknown HUGE scene '{scene_id}'. Available: {sorted(SCENES.keys())}"
        )
    return SCENES[scene_id]


def all_scene_ids() -> list[str]:
    return list(SCENES.keys())

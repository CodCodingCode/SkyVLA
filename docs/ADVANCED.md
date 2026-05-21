# Advanced and experimental modules

The four-stage curriculum (`hover` -> `waypoint_nav` -> `lang_nav` -> `vla`) is the main path through this repo. Everything below is optional: alternative encoders, alternative backbones, domain fine-tunes for richer scenes, and offline-RL baselines. Each module ships its own README with the full details; this page is just the index.

## Stage 3 alternatives

- [lang_nav_siglip/](../lang_nav_siglip) — drop-in replacement for `lang_nav/` that swaps frozen CLIP for SigLIP and exposes an extra cosine-similarity feature, giving a 1546-dim observation. Use [scripts/transfer_waypoint_to_vla_siglip.py](../scripts/transfer_waypoint_to_vla_siglip.py) to bootstrap from a Stage 2 waypoint checkpoint.

## Stage 4 alternatives

- [pi/](../pi) — Pi0's PaliGemma backbone is fully frozen and only a `256 x 256` MLP action head is trained on top, with PPO at `lr=3e-4`. Bootstrap with [scripts/transfer_waypoint_to_pi0.py](../scripts/transfer_waypoint_to_pi0.py). Cheaper than full VLA fine-tuning and sometimes more stable.

## Stage 4 domain fine-tunes

These start from an existing Stage 4 VLA checkpoint and continue training in richer environments.

- [vla_warehouse/](../vla_warehouse) — fine-tune in NVIDIA's built-in warehouse, hospital, and office USD scenes. Targets are real objects (forklifts, pallets, beds) instead of the three primitive shapes from the main arena. Works on both x86_64 and aarch64. See [vla_warehouse/README.md](../vla_warehouse/README.md).
- [vla_cesium/](../vla_cesium) — fine-tune in real-world city tiles streamed from Cesium for Omniverse. The Cesium extension is x86_64-only and targets specific Isaac Sim versions. See [vla_cesium/README.md](../vla_cesium/README.md).

## Inference-only and baselines

- [vla_universal/](../vla_universal) — scan-then-navigate inference loop with no training: the drone first builds a semantic map of its surroundings, then routes to the language target. See [vla_universal/README.md](../vla_universal/README.md).
- [huge_bench/](../huge_bench) — offline behaviour-cloning baseline trained on the HUGE-Bench dataset. Lives outside the PPO curriculum entirely. See [huge_bench/README.md](../huge_bench/README.md).

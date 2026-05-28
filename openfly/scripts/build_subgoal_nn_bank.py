#!/usr/bin/env python3
"""Build a SigLIP-feature bank over OpenFly frames for nearest-neighbor
retrieval visualization of SubgoalDiT predictions.

For each unique frame in the chosen split, encode it through PaliGemma
once, mean-pool the 256 SigLIP tokens to a single (2048,) descriptor,
and save the resulting ``(N, 2048)`` matrix alongside the frame's RGB
path. The bank is the search pool that :file:`eval_subgoal_nn.py` later
uses to map predicted subgoal *features* back to a real RGB frame
("show me the dataset image that most resembles what the DiT predicted").

Design choices:

* **Mean-pool over tokens, not flatten.** Flattening to ~524k-d would
  blow the bank to ~tens of GB and isn't needed — for *scene-level*
  retrieval ("which real frame looks most like this prediction") the
  global descriptor wins. Per-patch matching would over-weight precise
  spatial composition, which isn't what we care about for subgoal QA.
* **Cosine-normalize at save time.** Retrieval is L2 over unit vectors
  = cosine similarity, and pre-normalizing means the eval script can
  use plain matmul instead of pairwise cos. ~2× faster, no quality
  cost.
* **Index by RGB path, dedup.** The same frame can appear as
  ``rgb`` for one step and ``subgoal_rgb`` for another step; we encode
  it only once.
* **Sidecar JSON for metadata.** Per-frame instruction / pose / env
  goes in a parallel ``.json`` so the viewer can show context without
  loading the dataset.

Output layout (under ``--out_dir``):
  bank.pt    — {"features": (N, 2048) fp16 normalized, "paths": list[str]}
  meta.json  — [{"path": str, "env": str, "instruction": str, ...}, ...]

Usage:
  python -m openfly.scripts.build_subgoal_nn_bank \\
    --split unseen --batch_size 16 \\
    --out_dir ~/drone_project/logs/openfly/subgoal_nn/bank_unseen
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_DRONE_ROOT = Path(__file__).resolve().parents[2]
if str(_DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DRONE_ROOT))

from openfly.dataset import OpenFlyDataset, collate
from openfly.train_subgoal_dit import _build_processor, _tokenise_batch
from vla.vla_policy import PaliGemmaFeatureExtractor


@torch.no_grad()
def _encode_paths_unique(
    paligemma: PaliGemmaFeatureExtractor,
    processor,
    ds: OpenFlyDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], list[dict]]:
    """Walk the dataset, encode each unique RGB path exactly once."""
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    seen_paths: dict[str, int] = {}      # path -> index in features
    feats: list[torch.Tensor] = []        # each (2048,) fp16 normalized
    meta: list[dict] = []                 # parallel to feats

    t0 = time.time()
    n_encoded = 0
    for step, batch in enumerate(loader):
        rgb = batch["rgb"].to(device, non_blocking=True)
        subgoal_rgb = batch["subgoal_rgb"].to(device, non_blocking=True)
        # Tokenize once (we don't actually need text_embed downstream;
        # the dummy ids just satisfy PaliGemma's forward signature).
        input_ids, attention_mask = _tokenise_batch(
            processor, batch["instruction"], batch["sub_instruction"],
            rgb_dummy=rgb[0], device=device,
        )

        for which, frames, paths_attr, pose_attr in (
            ("curr",    rgb,         "rgb_path",         "pose"),
            ("subgoal", subgoal_rgb, "subgoal_rgb_path", "subgoal_pose"),
        ):
            paths = batch.get(paths_attr)
            if paths is None:
                continue
            # Filter to frames whose paths we haven't already encoded.
            to_encode: list[int] = []
            for i, p in enumerate(paths):
                if p in seen_paths:
                    continue
                seen_paths[p] = len(feats) + len(to_encode)  # tentative
                to_encode.append(i)
            if not to_encode:
                continue
            sel_frames = frames[to_encode]
            sel_ids = input_ids[to_encode]
            sel_mask = attention_mask[to_encode]
            pv = paligemma.preprocess_images(sel_frames)
            _, siglip = paligemma.forward_tokens(pv, sel_ids, sel_mask)
            paligemma.clear_cache()
            # Mean-pool over 256 tokens, L2-normalize.
            pooled = siglip.mean(dim=1).float()
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            feats.extend(pooled.cpu().to(torch.float16))

            for local_i, batch_i in enumerate(to_encode):
                pose = batch[pose_attr][batch_i].tolist()
                meta.append({
                    "path": paths[batch_i],
                    "which": which,
                    "instruction": batch["instruction"][batch_i],
                    "sub_instruction": batch["sub_instruction"][batch_i],
                    "pose": pose,
                })
            n_encoded += len(to_encode)

        if step % 20 == 0:
            dt = time.time() - t0
            print(
                f"[build_nn_bank] step={step:04d} unique_encoded={n_encoded} "
                f"elapsed={dt:.1f}s",
                flush=True,
            )

    features = torch.stack(feats, dim=0) if feats else torch.empty(0, 2048)
    paths_out = [m["path"] for m in meta]
    return features, paths_out, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="unseen",
                        help="OpenFly split to encode (typically 'unseen' — the "
                        "same one used for val_cos so retrievals are over the "
                        "same population).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--env_filter", type=str, default=None)
    parser.add_argument("--paligemma_model", default="google/paligemma-3b-pt-224")
    parser.add_argument("--paligemma_dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--out_dir",
        default=str(Path(os.environ.get("DRONE_PROJECT", Path.home() / "drone_project"))
                    / "logs" / "openfly" / "subgoal_nn" / "bank"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[build_nn_bank] split={args.split} out_dir={out_dir}")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    ds = OpenFlyDataset(
        split=args.split,
        history_frames=0,
        env_filter=args.env_filter,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
        require_images=True,
        oversample_stop=1.0,
        subgoal_pairing="semantic_only",  # deterministic; we only care about frames
    )
    if len(ds) == 0:
        raise RuntimeError("empty dataset")
    print(f"[build_nn_bank] dataset: {len(ds)} steps")

    paligemma = PaliGemmaFeatureExtractor(
        model_name=args.paligemma_model, lora_rank=8, lora_alpha=16.0,
        dtype=dtype_map[args.paligemma_dtype],
    ).to(device)
    paligemma.eval()
    for p in paligemma.parameters():
        p.requires_grad = False

    processor = _build_processor(args.paligemma_model)
    features, paths, meta = _encode_paths_unique(
        paligemma, processor, ds,
        batch_size=args.batch_size, num_workers=args.num_workers, device=device,
    )

    bank_path = out_dir / "bank.pt"
    meta_path = out_dir / "meta.json"
    torch.save({"features": features, "paths": paths}, bank_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(
        f"[build_nn_bank] saved features={tuple(features.shape)} "
        f"({features.element_size() * features.nelement() / 1e6:.1f} MB) "
        f"to {bank_path}"
    )
    print(f"[build_nn_bank] saved metadata for {len(meta)} frames to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

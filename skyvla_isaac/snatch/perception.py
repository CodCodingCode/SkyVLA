"""SNATCH perception (Agent A2) — dual-camera depth sensing + encoders.

Provides:
  * `top_camera_cfg()` / `bottom_camera_cfg()` — isaaclab `CameraCfg`s for the two
    onboard depth cameras (top forward RealSense-like, bottom downward Pi-cam-like),
    matching the resolutions / FoV / mounts in snatch/DESIGN.md.
  * `DepthEncoder` — a ResNet-18 (torchvision) that maps a single-channel metric
    depth image to a 512-dim feature.
  * `build_obs_latents(...)` — concatenate top+bottom latents -> (N, 1024).
  * `freeze_backbone(...)` — freeze the conv trunk, leave the final projection trainable.

The camera cfgs are intentionally importable WITHOUT a running Isaac app: we defer
the `isaaclab` imports into the cfg builder functions, so the torch-only encoder
parts of this module load fine for unit tests / offline use.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# --- depth pre-processing constants (DESIGN.md) ---
DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 5.0
FEATURE_DIM = 512
OBS_LATENT_DIM = 2 * FEATURE_DIM  # 1024

# default pinhole horizontal aperture (mm) used by isaaclab PinholeCameraCfg
_DEFAULT_H_APERTURE_MM = 20.955


def _focal_from_fov(fov_deg: float, aperture_mm: float = _DEFAULT_H_APERTURE_MM) -> float:
    """Convert a HORIZONTAL field-of-view (degrees) to a pinhole focal length (mm).

    isaaclab's PinholeCameraCfg defines the intrinsics via
    ``fov = 2 * atan(aperture / (2 * focal))`` so
    ``focal = aperture / (2 * tan(fov / 2))``.
    """
    half = math.radians(fov_deg) / 2.0
    return aperture_mm / (2.0 * math.tan(half))


# =====================================================================
# Camera configs
# =====================================================================
def top_camera_cfg(prim_path: str = "/World/envs/env_.*/Drone/top_cam"):
    """Top forward-facing depth camera (RealSense-like): 848x480, FoV 87 deg.

    Mounted on the drone body looking forward (+x). Depth via
    ``distance_to_image_plane`` (planar metric depth).
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    # TiledCamera: one batched render pass for ALL envs (the plain Camera sensor
    # exhausts the RTX descriptor pool past a handful of envs). Resolution is reduced
    # to a tiled-friendly tile (real cam is 848x480, but the depth encoder is
    # resolution-agnostic) so visuomotor RL scales to many parallel envs.
    return TiledCameraCfg(
        prim_path=prim_path,
        height=120,
        width=212,                                  # ~16:9, matches 848x480 aspect
        update_period=0.0,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_focal_from_fov(87.0),
            horizontal_aperture=_DEFAULT_H_APERTURE_MM,
            clipping_range=(0.05, 20.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),  # optical frame looking forward (+x)
            convention="ros",
        ),
    )


def bottom_camera_cfg(prim_path: str = "/World/envs/env_.*/Drone/bottom_cam"):
    """Bottom downward-facing depth camera (Pi-cam-like): 640x480, FoV 120 deg.

    Mounted under the drone body looking straight down (-z). Depth via
    ``distance_to_image_plane``.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    return TiledCameraCfg(
        prim_path=prim_path,
        height=120,
        width=160,                                  # 4:3, matches 640x480 aspect
        update_period=0.0,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_focal_from_fov(120.0),
            horizontal_aperture=_DEFAULT_H_APERTURE_MM,
            clipping_range=(0.05, 20.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.05),
            rot=(0.0, 1.0, 0.0, 0.0),  # 180deg about x -> looking straight down
            convention="ros",
        ),
    )


# =====================================================================
# Depth encoder
# =====================================================================
def _build_resnet18():
    """Build a torchvision ResNet-18, using ImageNet weights if reachable,
    otherwise random init (offline-graceful)."""
    from torchvision.models import resnet18

    try:  # new-style weights enum (torchvision >= 0.13)
        from torchvision.models import ResNet18_Weights
        try:
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            # download failed / offline -> random init
            net = resnet18(weights=None)
    except Exception:  # very old torchvision
        try:
            net = resnet18(pretrained=True)
        except Exception:
            net = resnet18(pretrained=False)
    return net


def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    """Clip metric depth to [DEPTH_MIN_M, DEPTH_MAX_M] and normalize to [0, 1].

    Accepts (N, H, W) or (N, 1, H, W); returns (N, 1, H, W). Non-finite values
    (e.g. background inf from the renderer) are mapped to the far clip.
    """
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)  # (N,1,H,W)
    elif depth.dim() == 4 and depth.shape[1] != 1:
        # tolerate trailing-channel (N,H,W,1)
        if depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2)
    depth = torch.nan_to_num(depth, nan=DEPTH_MAX_M, posinf=DEPTH_MAX_M, neginf=DEPTH_MIN_M)
    depth = depth.clamp(DEPTH_MIN_M, DEPTH_MAX_M)
    return (depth - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)


class DepthEncoder(nn.Module):
    """ResNet-18 depth encoder: (N,H,W) metric depth -> (N, 512) feature.

    The single-channel depth is normalized to [0,1] then repeated to 3 channels
    so the standard 3-channel ResNet stem can be used directly. The classifier
    head is replaced by Identity, giving a 512-dim feature, and a learnable
    linear projection (512->512) sits on top as the trainable "head" used by
    `freeze_backbone`.
    """

    def __init__(self, out_dim: int = FEATURE_DIM):
        super().__init__()
        net = _build_resnet18()
        net.fc = nn.Identity()  # ResNet-18 trunk outputs 512-d after avgpool
        self.backbone = net
        self.proj = nn.Linear(512, out_dim)
        self.out_dim = out_dim

    def preprocess(self, depth: torch.Tensor) -> torch.Tensor:
        x = _normalize_depth(depth)            # (N,1,H,W) in [0,1]
        x = x.repeat(1, 3, 1, 1)               # -> 3 channels
        return x

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(depth)
        feat = self.backbone(x)                # (N, 512)
        return self.proj(feat)                 # (N, out_dim)


def build_obs_latents(
    top_depth: torch.Tensor,
    bottom_depth: torch.Tensor,
    enc_top: DepthEncoder,
    enc_bottom: DepthEncoder,
) -> torch.Tensor:
    """Encode both depth streams with their (separate) encoders and concat.

    Returns (N, 1024). `enc_top` and `enc_bottom` MUST be distinct instances
    (no shared weights) — the two cameras see very different distributions.
    """
    f_top = enc_top(top_depth)        # (N, 512)
    f_bot = enc_bottom(bottom_depth)  # (N, 512)
    return torch.cat([f_top, f_bot], dim=-1)  # (N, 1024)


def freeze_backbone(encoder: DepthEncoder) -> DepthEncoder:
    """Freeze the ResNet conv trunk; keep the final projection trainable.

    Sets ``requires_grad=False`` on every backbone parameter and puts the
    backbone in eval mode (so BatchNorm running stats stay fixed). The `proj`
    head remains trainable. Returns the encoder for chaining.
    """
    for p in encoder.backbone.parameters():
        p.requires_grad = False
    encoder.backbone.eval()
    for p in encoder.proj.parameters():
        p.requires_grad = True
    return encoder

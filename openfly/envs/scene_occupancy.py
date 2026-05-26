"""Per-scene voxel occupancy grid for OpenFly collision checking.

The OpenFly paper (Section 6.2) defines task success as the UAV stopping
within 20 m of the target *and* completing the trajectory without
collision — checked externally against the scene's point cloud, not
through AirSim's physics:

  > "Each environment provides corresponding point clouds that enable
  >  collision checking. If a collision occurs, the task is counted as
  >  a failure."

This module loads those point clouds (one ``.pcd`` per scene from
``IPEC-COMMUNITY/OpenFly_DataGen/pcd_map`` on HuggingFace), voxelizes
them, and exposes a fast ``is_occupied(world_xyz)`` query. Used by
``AirSimVLNEnv`` to flag steps where the kinematic teleport would
place the drone inside an obstacle.

Scenes without a published PCD (the 3DGS scenes — `env_gs_*` — and
GTA — `env_game_gtav`) return ``is_occupied=False`` always and are
recorded as ``available=False``. The env can still run on them; it
just won't penalise collisions for those scenes.

Memory layout
-------------
City-scale scenes (env_ue_bigcity ≈ 2.3 GB raw PCD) would blow up
Python sets if we stored voxel coords as tuples. Instead we hash each
voxel ``(vx, vy, vz)`` to a single int64 key with generous range
(±2^20 in each axis ⇒ 1M-voxel range per axis, i.e. ±3 km at 3 m
voxel size — well past every OpenFly scene), then store the unique
sorted keys in a 1D ``np.int64`` array. Lookup is
``np.searchsorted``, O(log N), and memory is 8 bytes per occupied
voxel.

For env_ue_bigcity at 3 m voxel size we get a few million voxels and
~50 MB of memory — easy.

Cache
-----
Building the voxel grid takes 30 s–3 min depending on PCD size. We
cache the int64-key arrays to ``.voxel_cache/<env>_v<voxel>_e<expand>.npz``
under the PCD directory so subsequent loads are instant.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np


_DEFAULT_PCD_DIR = Path(
    os.environ.get(
        "OPENFLY_PCD_DIR",
        str(Path.home() / "assets" / "OpenFly" / "openfly_datagen" / "pcd_map"),
    )
)

# Hash range: ±2^20 voxels = ±1,048,576 per axis. At 3 m voxel size this
# covers ±3 km in each direction — comfortably bigger than every OpenFly
# scene's bounding box. If a future scene exceeds this, the offset will
# need to grow (the int64 will hold up to ~±2^21 per axis safely).
_AXIS_OFFSET = 1 << 20
_AXIS_BITS = 21  # 2^21 > 2 * AXIS_OFFSET, so the three axes pack into int64 cleanly


def _hash_voxel(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
    """Pack int voxel coords into a single int64 hash key."""
    a = vx.astype(np.int64) + _AXIS_OFFSET
    b = vy.astype(np.int64) + _AXIS_OFFSET
    c = vz.astype(np.int64) + _AXIS_OFFSET
    return (a << (2 * _AXIS_BITS)) | (b << _AXIS_BITS) | c


def _hash_voxel_scalar(vx: int, vy: int, vz: int) -> int:
    a = int(vx) + _AXIS_OFFSET
    b = int(vy) + _AXIS_OFFSET
    c = int(vz) + _AXIS_OFFSET
    return (a << (2 * _AXIS_BITS)) | (b << _AXIS_BITS) | c


class SceneOccupancy:
    """Voxelized obstacle map for one OpenFly scene.

    Parameters
    ----------
    env_name:
        Canonical OpenFly env name, e.g. ``env_ue_bigcity``. Used to
        find ``<pcd_root>/<env_name>.pcd``.
    voxel_size:
        Edge length of one occupancy voxel in metres. Defaults to 3 m
        to match OpenFly's smallest forward macro. Smaller values are
        more precise but use more memory.
    expand:
        Dilate the occupancy set by ``expand`` voxels in each direction
        (Chebyshev neighbourhood). Useful as a safety margin if you
        want to treat the drone as having non-zero volume. Default 0.
    pcd_root:
        Directory containing ``<env_name>.pcd`` files. Defaults to
        ``$OPENFLY_PCD_DIR`` or ``~/assets/OpenFly/openfly_datagen/pcd_map``.
    cache_dir:
        Where to write the int64-key cache. Defaults to a
        ``.voxel_cache`` subdir next to ``pcd_root``.
    """

    def __init__(
        self,
        env_name: str,
        *,
        voxel_size: float = 3.0,
        expand: int = 0,
        pcd_root: Path | str | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        self.env_name = env_name
        self.voxel_size = float(voxel_size)
        self.expand = int(expand)

        pcd_root = Path(pcd_root or _DEFAULT_PCD_DIR)
        pcd_path = pcd_root / f"{env_name}.pcd"

        # Unavailable scenes (env_gs_*, env_game_gtav as of the upstream
        # release). Env will quietly skip collision checking for these.
        if not pcd_path.is_file():
            self.available = False
            self._keys = None
            print(
                f"[scene_occupancy] no PCD for {env_name} "
                f"({pcd_path}); collision check disabled for this scene"
            )
            return
        self.available = True

        cache_dir = Path(cache_dir or pcd_root / ".voxel_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{env_name}_v{voxel_size}_e{expand}.npz"

        if cache_path.is_file():
            try:
                data = np.load(cache_path)
                self._keys = np.asarray(data["keys"], dtype=np.int64)
                print(
                    f"[scene_occupancy] {env_name}: loaded {len(self._keys):,} "
                    f"voxel keys from cache"
                )
                return
            except Exception as e:
                print(f"[scene_occupancy] cache load failed for {cache_path}: {e}")

        self._keys = self._build_keys_from_pcd(pcd_path)
        try:
            np.savez_compressed(cache_path, keys=self._keys)
            print(f"[scene_occupancy] cached → {cache_path}")
        except Exception as e:
            print(f"[scene_occupancy] cache save failed: {e}")

    def _build_keys_from_pcd(self, pcd_path: Path) -> np.ndarray:
        import open3d as o3d
        t0 = time.time()
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        pts = np.asarray(pcd.points, dtype=np.float64)
        print(
            f"[scene_occupancy] {pcd_path.name}: {len(pts):,} points "
            f"loaded in {time.time() - t0:.1f}s"
        )

        # Voxelize: floor(pts / voxel) → integer voxel indices
        vox = np.floor(pts / self.voxel_size).astype(np.int64)
        keys = _hash_voxel(vox[:, 0], vox[:, 1], vox[:, 2])

        # Optional dilation: expand each voxel by ±N in each axis.
        if self.expand > 0:
            r = self.expand
            offs = np.array(
                [(dx, dy, dz)
                 for dx in range(-r, r + 1)
                 for dy in range(-r, r + 1)
                 for dz in range(-r, r + 1)],
                dtype=np.int64,
            )
            # Broadcasted dilation: (N_pts, K_offsets, 3) → (N_pts * K, 3)
            dilated = vox[:, None, :] + offs[None, :, :]
            dilated = dilated.reshape(-1, 3)
            keys = _hash_voxel(dilated[:, 0], dilated[:, 1], dilated[:, 2])

        keys = np.unique(keys)  # also sorts
        print(
            f"[scene_occupancy] {pcd_path.name}: {len(keys):,} occupied voxels "
            f"at {self.voxel_size}m (expand={self.expand}) "
            f"[total build time {time.time() - t0:.1f}s]"
        )
        return keys

    # ---- query API -----------------------------------------------

    def is_occupied(self, world_xyz: Sequence[float]) -> bool:
        """O(log N) occupancy query for a single world-frame point.

        Returns ``False`` when the scene has no PCD (i.e. unsupported
        scene like env_game_gtav). Callers that want stricter behaviour
        can check ``available`` first.
        """
        if not self.available or self._keys is None:
            return False
        vx = int(world_xyz[0] // self.voxel_size)
        vy = int(world_xyz[1] // self.voxel_size)
        vz = int(world_xyz[2] // self.voxel_size)
        key = _hash_voxel_scalar(vx, vy, vz)
        idx = np.searchsorted(self._keys, key)
        return bool(idx < len(self._keys) and self._keys[idx] == key)

    def num_voxels(self) -> int:
        return 0 if self._keys is None else int(len(self._keys))


# Process-global cache so each scene's voxel grid is built at most once
# per Python process. The env loads scenes lazily on episode reset, but
# subsequent resets to the same scene must not re-read the PCD.
_SCENE_CACHE: dict[tuple[str, float, int], SceneOccupancy] = {}


def get_scene_occupancy(
    env_name: str,
    *,
    voxel_size: float = 3.0,
    expand: int = 0,
) -> SceneOccupancy:
    """Process-cached factory. Same args → same instance."""
    key = (env_name, float(voxel_size), int(expand))
    if key not in _SCENE_CACHE:
        _SCENE_CACHE[key] = SceneOccupancy(
            env_name, voxel_size=voxel_size, expand=expand
        )
    return _SCENE_CACHE[key]

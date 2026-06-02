"""Object detectors that return a pixel bounding box for a text query.

Two implementations with the same interface ``detect(rgb, query, pose, K) -> box``
(box = (x0,y0,x1,y1) in pixels, or ``None`` if not found):

* ``OpenAIDetector`` — a real VLM (GPT-4o vision) via a stdlib HTTPS call (no
  ``openai`` package needed). The API key is read from ``OPENAI_API_KEY`` or a
  gitignored ``/home/ubuntu/SkyVLA/.openai_key`` — NEVER hard-code it.
* ``SyntheticDetector`` — a keyless oracle that projects a known WORLD target
  into the current camera to synthesise a box. Lets the full box->lift->fly path
  run without any API key (swap to OpenAIDetector by changing one flag).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.request

import numpy as np

_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_KEY_PATHS = ("/home/ubuntu/SkyVLA/.openai_key", os.path.expanduser("~/.openai_key"))


def _load_key() -> str:
    k = os.environ.get("OPENAI_API_KEY")
    if k:
        return k.strip()
    for p in _KEY_PATHS:
        if os.path.exists(p):
            return open(p).read().strip()
    raise RuntimeError(
        "No OpenAI key. Set OPENAI_API_KEY or write your key to "
        "/home/ubuntu/SkyVLA/.openai_key (gitignored). Do NOT paste keys in chat.")


def _png_b64(rgb: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _parse_box(txt: str, W: int, H: int):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except Exception:
        return None
    if not d.get("found") or "box" not in d:
        return None
    try:
        b = [float(x) for x in d["box"]]
    except Exception:
        return None
    if max(b) <= 1.0:                                 # normalized 0..1 -> pixels
        b = [b[0] * W, b[1] * H, b[2] * W, b[3] * H]
    b = [max(0, min(W, b[0])), max(0, min(H, b[1])),
         max(0, min(W, b[2])), max(0, min(H, b[3]))]
    if b[2] <= b[0] or b[3] <= b[1]:
        return None
    return tuple(b)


class OpenAIDetector:
    def __init__(self, model: str = "gpt-4o", key: str | None = None):
        self.model = model
        self.key = key or _load_key()

    def detect(self, rgb, query, pose=None, K=None):
        rgb = np.asarray(rgb)
        H, W = rgb.shape[:2]
        prompt = (
            f"Find the {query} in this image. Respond with ONLY compact JSON: "
            f'{{"found": true, "box": [x0, y0, x1, y1]}} or {{"found": false}}. '
            f"The box is in PIXEL coordinates of a {W}x{H} image, origin top-left, "
            f"x right, y down. No prose, no code fences.")
        body = {
            "model": self.model, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url":
                    {"url": f"data:image/png;base64,{_png_b64(rgb)}"}},
            ]}],
        }
        req = urllib.request.Request(
            _OPENAI_URL, data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=45) as r:
            out = json.loads(r.read())
        txt = out["choices"][0]["message"]["content"]
        return _parse_box(txt, W, H)


class SyntheticDetector:
    """Keyless oracle: projects a fixed world target into the camera to make a box.
    ``pose`` must be the OpenCV camera-to-world used everywhere else (det = +1)."""

    def __init__(self, world_target, box_px: int = 140):
        self.t = np.asarray(world_target, np.float32)
        self.box_px = box_px

    def detect(self, rgb, query, pose, K):
        rgb = np.asarray(rgb)
        H, W = rgb.shape[:2]
        pose = np.asarray(pose, np.float32)
        pc = np.linalg.inv(pose) @ np.array([*self.t, 1.0], np.float32)
        if pc[2] <= 0.1:                              # behind the camera
            return None
        u = float(K[0, 0]) * pc[0] / pc[2] + float(K[0, 2])
        v = float(K[1, 1]) * pc[1] / pc[2] + float(K[1, 2])
        if not (0 <= u < W and 0 <= v < H):          # outside the frame
            return None
        # box shrinks with distance, so it looks like a real detection
        s = max(40.0, self.box_px * 4.0 / max(1.0, float(pc[2])))
        return (u - s / 2, v - s / 2, u + s / 2, v + s / 2)

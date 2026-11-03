"""Draw detections on a frame: mask tint, box, prompt + track id label."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .sam_parse import Detection

COLORS = {"person": (255, 64, 64), "forklift": (255, 196, 0), "hard hat": (64, 160, 255),
          "high-visibility vest": (64, 255, 96), "safety vest": (64, 255, 96)}


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw(image: str | Path | Image.Image, dets: list[Detection], title: str | None = None,
         max_side: int = 1280) -> Image.Image:
    im = (Image.open(image) if not isinstance(image, Image.Image) else image).convert("RGB")
    W, H = im.size
    arr = np.asarray(im).astype(np.float32)
    for d in dets:
        if d.mask is None:
            continue
        c = np.array(COLORS.get(d.prompt, (255, 0, 255)), dtype=np.float32)
        m = d.full_mask(W, H).astype(bool)
        arr[m] = arr[m] * 0.55 + c * 0.45
    im = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    s = min(1.0, max_side / max(W, H))
    if s < 1.0:
        im = im.resize((round(W * s), round(H * s)), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    lw = max(2, round(min(im.size) / 300))
    f = _font(max(12, round(min(im.size) / 40)))
    for d in dets:
        c = COLORS.get(d.prompt, (255, 0, 255))
        x1, y1, x2, y2 = (v * s for v in d.box)
        dr.rectangle([x1, y1, x2, y2], outline=c, width=lw)
        lab = f"{d.prompt} #{d.track_id}" + (f" {d.confidence:.2f}" if d.confidence is not None else "")
        tb = dr.textbbox((0, 0), lab, font=f)
        ty = y1 - (tb[3] - tb[1]) - 6 if y1 > (tb[3] - tb[1]) + 6 else y1 + 2
        dr.rectangle([x1, ty, x1 + tb[2] - tb[0] + 6, ty + tb[3] - tb[1] + 6], fill=c)
        dr.text((x1 + 3, ty + 1), lab, fill=(0, 0, 0), font=f)
    if title:
        tb = dr.textbbox((0, 0), title, font=f)
        dr.rectangle([0, 0, tb[2] + 10, tb[3] + 10], fill=(0, 0, 0))
        dr.text((5, 3), title, fill=(255, 255, 255), font=f)
    return im


def save_jpeg(im: Image.Image, path: str | Path, max_bytes: int = 200_000) -> int:
    """Save as JPEG under max_bytes (lower quality, then shrink). Returns the size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = im
    while True:
        for q in (85, 75, 65, 55):
            buf = io.BytesIO()
            cur.save(buf, "JPEG", quality=q, optimize=True)
            if buf.tell() <= max_bytes:
                path.write_bytes(buf.getvalue())
                return buf.tell()
        cur = cur.resize((round(cur.width * 0.85), round(cur.height * 0.85)), Image.LANCZOS)

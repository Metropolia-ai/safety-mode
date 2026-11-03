"""Parse SAM 3.1 output_text from the Meta Model API into Detection objects.

Wire format (dev.meta.ai/docs/sam/reading-segmentation, checked 27.10.2026):
one line per frame, ``<Nf>`` then comma-separated records
``id<|box;x1=..;y1=..;x2=..;y2=..;w=W;h=H[;c=conf]|><|mask;x=0;y=0[;c=conf];data=H,W,<enc>payload|>``.
Box corners are inclusive source pixels of the media that was sent. The mask
raster is box-local: its H x W equals the box size (y2-y1+1, x2-x1+1), checked
on live responses. Mask payloads are decoded with Meta's own codec
(``meta_sam_parser.decode_mask_to_raster``, ``pip install meta-sam-parser``),
which the docs point to; the codec is a custom base85 plus range coder.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_LINE = re.compile(r"^<(\d+)f(?:;[^>]*)?>(.*)$")
_RECORD = re.compile(r",?(\d+)((?:<\|[A-Za-z][A-Za-z0-9_.-]*(?:;[^|]*)?\|>)+)")
_TOKEN = re.compile(r"<\|([A-Za-z][A-Za-z0-9_.-]*)(?:;([^|]*))?\|>")
_MASK_DATA = re.compile(r"^(\d+),(\d+),([!~].+)$")


class SamParseError(ValueError):
    pass


@dataclass
class Detection:
    prompt: str
    frame_index: int
    track_id: int
    box: list[float]            # [x1, y1, x2, y2] in source pixels, x2/y2 exclusive
    mask: np.ndarray | None     # uint8 H x W, 0/1, box-local raster (None if not decoded)
    mask_box: list[float]       # source-pixel rectangle the mask raster covers
    frame_size: tuple[int, int] = (0, 0)   # (w, h) of the media SAM saw
    confidence: float | None = None
    mask_shape: tuple[int, int] = (0, 0)   # (H, W) declared on the wire
    mask_payload: str = field(default="", repr=False)

    def full_mask(self, width: int, height: int) -> np.ndarray:
        """Mask pasted into a width x height canvas (source pixels)."""
        from PIL import Image
        out = np.zeros((height, width), dtype=np.uint8)
        if self.mask is None:
            return out
        x1, y1, x2, y2 = (int(round(v)) for v in self.mask_box)
        x1c, y1c, x2c, y2c = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0 or x2c <= x1c or y2c <= y1c:
            return out
        m = self.mask
        if m.shape != (bh, bw):
            m = np.asarray(Image.fromarray(m * 255).resize((bw, bh), Image.NEAREST)) > 127
            m = m.astype(np.uint8)
        out[y1c:y2c, x1c:x2c] = m[y1c - y1:y2c - y1, x1c - x1:x2c - x1]
        return out


def _fields(body: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not body:
        return out
    # data=... is always last in a mask token and its payload never contains ';'
    for part in body.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def decode_mask(payload: str, height: int, width: int) -> np.ndarray:
    from meta_sam_parser import SegmentationMask, decode_mask_to_raster
    enc = "lossless" if payload.startswith("~") else "one_bit"
    raster = decode_mask_to_raster(SegmentationMask(enc, payload, width, height))
    return np.frombuffer(raster, dtype=np.uint8).reshape(height, width).copy()


def parse_output(text: str, prompt: str, *, scale: tuple[float, float] = (1.0, 1.0),
                 decode_masks: bool = True) -> list[Detection]:
    """Parse a whole output_text. ``scale`` = (sx, sy) from sent pixels to source pixels."""
    sx, sy = scale
    dets: list[Detection] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE.match(line)
        if not m:
            raise SamParseError(f"line without <Nf> header: {line[:60]!r}")
        frame = int(m.group(1))
        body = m.group(2)
        pos = 0
        while pos < len(body):
            r = _RECORD.match(body, pos)
            if not r:
                raise SamParseError(f"bad record at {pos} in frame {frame}: {body[pos:pos+60]!r}")
            pos = r.end()
            tokens = {name: _fields(b) for name, b in _TOKEN.findall(r.group(2))}
            if "box" not in tokens:
                raise SamParseError(f"record without box in frame {frame}")
            b = tokens["box"]
            x1, y1, x2, y2 = (int(b[k]) for k in ("x1", "y1", "x2", "y2"))
            fw, fh = int(b["w"]), int(b["h"])
            conf = float(b["c"]) if "c" in b else None
            box_sent = [x1, y1, x2 + 1, y2 + 1]       # inclusive -> exclusive
            box = [box_sent[0] * sx, box_sent[1] * sy, box_sent[2] * sx, box_sent[3] * sy]
            mask = None
            shape = (0, 0)
            payload = ""
            if "mask" in tokens:
                md = _MASK_DATA.match(tokens["mask"].get("data", ""))
                if not md:
                    raise SamParseError(f"bad mask data in frame {frame}")
                shape = (int(md.group(1)), int(md.group(2)))
                payload = md.group(3)
                if decode_masks:
                    mask = decode_mask(payload, *shape)
            dets.append(Detection(prompt=prompt, frame_index=frame, track_id=int(r.group(1)),
                                  box=box, mask=mask, mask_box=list(box), frame_size=(fw, fh),
                                  confidence=conf, mask_shape=shape, mask_payload=payload))
    return dets

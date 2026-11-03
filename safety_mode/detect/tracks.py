"""Bridge SAM 3.1 track ids across short breaks (1 fps records of video mode).

SAM 3.1 gives an object a new id when it is hidden for a moment (a person behind a rack
frame, out of frame for a second). Rule: when an id stops and a new id of the same prompt
starts within max_gap_s, and the new box overlaps the last known box of the old id
(IoU >= iou_min) or its foot point is within foot_m metres of the old one on the floor
(homography of the camera, when there is one), the new id takes the old id. Several new
ids competing for old ids are matched greedily, best overlap first, one to one. Every
bridge is returned (and logged) so the caller can write it next to the data.

Records keep the SAM id in "sam_track_id" when it was changed.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

IOU_MIN = 0.4
FOOT_M = 0.3


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _foot_m(a, b, size, H) -> Optional[float]:
    if H is None or not size:
        return None
    w, h = size
    try:
        pa = H.to_floor(((a[0] + a[2]) / 2 / w, a[3] / h))
        pb = H.to_floor(((b[0] + b[2]) / 2 / w, b[3] / h))
    except ValueError:
        return None
    return ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5


def bridge_prompt(seconds: dict, prompt: str, *, max_gap_s: float = 1.0, iou_min: float = IOU_MIN,
                  foot_m: float = FOOT_M, frame_size=None, H=None) -> tuple[dict, list]:
    """seconds: {stem: [{"box", "score", "track_id"}]} for one prompt (stem NNN = second + 1).
    Returns (new records per stem, bridges [{prompt, second, from, to, last_seen_s, iou, foot_m}])."""
    mapping: dict = {}          # SAM id -> bridged id
    last: dict = {}             # bridged id -> (second, box)
    out: dict = {}
    bridges: list = []
    for stem in sorted(seconds):
        s = int(stem) - 1
        recs = seconds[stem]
        present = {mapping[d["track_id"]] for d in recs if d["track_id"] in mapping}
        new = [d for d in recs if d["track_id"] not in mapping]
        cands = []
        for d in new:
            for b, (ls, lbox) in last.items():
                if b in present or not (1 <= s - ls <= max_gap_s + 1 + 1e-6):
                    continue
                o = iou(d["box"], lbox)
                fm = _foot_m(d["box"], lbox, frame_size, H)
                if o >= iou_min:
                    cands.append(((0, -o), d, b, o, fm))
                elif fm is not None and fm <= foot_m:
                    cands.append(((1, fm), d, b, o, fm))
        used_new, used_old = set(), set()
        for _, d, b, o, fm in sorted(cands, key=lambda c: c[0]):
            if d["track_id"] in used_new or b in used_old:
                continue
            used_new.add(d["track_id"])
            used_old.add(b)
            mapping[d["track_id"]] = b
            br = {"prompt": prompt, "second": s, "from": d["track_id"], "to": b, "last_seen_s": last[b][0],
                  "iou": round(o, 2), "foot_m": None if fm is None else round(fm, 2)}
            bridges.append(br)
            log.info("bridge %s #%s -> #%s at s%d (last seen s%d, IoU %.2f, foot %s m)", prompt, br["from"],
                     br["to"], s, br["last_seen_s"], o, br["foot_m"])
        for d in new:
            mapping.setdefault(d["track_id"], d["track_id"])
        out[stem] = []
        for d in recs:
            b = mapping[d["track_id"]]
            r = dict(d, track_id=b)
            if b != d["track_id"]:
                r["sam_track_id"] = d["track_id"]
            out[stem].append(r)
            last[b] = (s, d["box"])
        out[stem].sort(key=lambda r: r["track_id"])
    return out, bridges

"""Shared helpers for explain.py and report.py: structured facts only, no images.

Positions in words come from boxes (and zone polygons when a zones file is given), so the model
never has to guess where something is. Guards check model text against the facts it was given.
"""
import json
import re

PERSON_WORDS = ("person", "worker", "pedestrian", "operator", "people", "human")
DEFAULT_FRAME = (1280, 720)

_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")


def person_like(name):
    n = name.lower()
    return any(k in n for k in PERSON_WORDS)


def parse_json(raw):
    """json.loads, tolerating markdown fences or prose around one object."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        i, j = raw.find("{"), raw.rfind("}")
        if i < 0 or j <= i:
            raise
        return json.loads(raw[i:j + 1])


def _point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def norm_box(box, frame_size=None):
    """Box -> normalised 0..1. Boxes with all values <= 1 are taken as already normalised."""
    if all(0 <= v <= 1.0 for v in box):
        return [float(v) for v in box]
    w, h = frame_size or DEFAULT_FRAME
    return [box[0] / w, box[1] / h, box[2] / w, box[3] / h]


def position_words(box, frame_size=None, zones=None):
    """'on the right side of the frame, mid-distance from the camera, inside Fire exit'.

    Horizontal: thirds by the box centre. Distance: by the bottom edge (ground contact point);
    low in the frame = near the camera. Zones: the ground contact point (bottom middle) in a polygon.
    """
    x1, y1, x2, y2 = norm_box(box, frame_size)
    cx, by = (x1 + x2) / 2, y2
    side = "on the left side" if cx < 1 / 3 else "on the right side" if cx > 2 / 3 else "in the centre"
    dist = "near the camera" if by > 0.8 else "far from the camera" if by < 0.55 else "mid-distance from the camera"
    words = [f"{side} of the frame", dist]
    inside = [z.get("title") or z["name"] for z in (zones or []) if z.get("polygon") and _point_in_poly(cx, by, z["polygon"])]
    if inside:
        words.append("inside " + " and ".join(inside))
    return ", ".join(words)


def numbers_in(text):
    """Set of numbers in a text, as normalised strings ('1,3' -> '1.3', '2.0' -> '2')."""
    out = set()
    for m in _NUM.findall(text or ""):
        v = float(m.replace(",", "."))
        out.add(f"{v:g}")
    return out


def duration_words(s):
    s = int(round(s))
    if s < 60:
        return f"{s} s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} min {s} s" if s else f"{m} min"
    h, m = divmod(m, 60)
    return f"{h} h {m} min"


def add_seconds(hhmmss, seconds):
    """'16:20:40' + 840 -> '16:34:40'. Returns None for other time formats."""
    m = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})", hhmmss or "")
    if not m:
        return None
    t = (int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(round(seconds))) % 86400
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


LANG_NAMES = {"en": "English", "fi": "Finnish", "ru": "Russian", "sv": "Swedish"}


def lang_name(lang):
    return LANG_NAMES.get(lang, lang)

"""Plane geometry in normalised image coordinates (0..1) and the floor homography."""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

Point = Sequence[float]
Box = Sequence[float]

WEAR_MIN_OVERLAP = 0.3

_UPPER = ("hat", "helmet", "goggle", "glasses", "visor", "face", "mask", "ear")
_MIDDLE = ("vest", "jacket", "harness", "coat", "shirt", "apron")
_LOWER = ("boot", "shoe", "trouser", "pant")


def point_in_polygon(pt: Point, polygon: Sequence[Point]) -> bool:
    """Ray casting; a point exactly on an edge counts as inside."""
    x, y = pt
    n = len(polygon)
    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # on-edge check
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) < 1e-12 and min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12 \
                and min(y1, y2) - 1e-12 <= y <= max(y1, y2) + 1e-12:
            return True
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xi:
                inside = not inside
    return inside


def foot_point(box: Box) -> tuple[float, float]:
    """Bottom-centre of the box: feet of a person, wheels of a machine."""
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def overlap_ratio(a: Box, b: Box) -> float:
    """Share of box a that lies inside box b (0..1)."""
    area = box_area(a)
    return intersection(a, b) / area if area > 0 else 0.0


class Homography:
    """Image (normalised) -> floor metres, from 4 point pairs of floor_calibration."""

    def __init__(self, matrix: np.ndarray):
        self.H = np.asarray(matrix, dtype=float)
        self.H_inv = np.linalg.inv(self.H)

    @classmethod
    def from_points(cls, src: Sequence[Point], dst: Sequence[Point]) -> "Homography":
        if len(src) != 4 or len(dst) != 4:
            raise ValueError("floor calibration needs exactly 4 point pairs")
        rows, rhs = [], []
        for (x, y), (u, v) in zip(src, dst):
            rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
            rhs.append(u)
            rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
            rhs.append(v)
        h = np.linalg.solve(np.array(rows, float), np.array(rhs, float))
        return cls(np.append(h, 1.0).reshape(3, 3))

    @classmethod
    def from_calibration(cls, calibration: Optional[dict]) -> Optional["Homography"]:
        if not calibration:
            return None
        return cls.from_points(calibration["points_px"], calibration["points_m"])

    @staticmethod
    def _apply(M: np.ndarray, pt: Point) -> tuple[float, float]:
        x, y, w = M @ np.array([pt[0], pt[1], 1.0])
        if abs(w) < 1e-12:
            raise ValueError("point maps to infinity (on the horizon)")
        return (float(x / w), float(y / w))

    def to_floor(self, pt: Point) -> tuple[float, float]:
        return self._apply(self.H, pt)

    def to_image(self, pt: Point) -> tuple[float, float]:
        return self._apply(self.H_inv, pt)


def _seg_point_dist(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def _segments_intersect(a1, a2, b1, b2) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    d1, d2 = orient(b1, b2, a1), orient(b1, b2, a2)
    d3, d4 = orient(a1, a2, b1), orient(a1, a2, b2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def bottom_edge_floor(box: Box, H: Homography) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bottom edge of the box projected to the floor (a segment in metres)."""
    x1, _, x2, y2 = box
    return H.to_floor((x1, y2)), H.to_floor((x2, y2))


def distance_m(a: Box, b: Box, H: Homography) -> float:
    """Floor distance in metres between the nearest points of the two bottom edges."""
    a1, a2 = bottom_edge_floor(a, H)
    b1, b2 = bottom_edge_floor(b, H)
    if _segments_intersect(a1, a2, b1, b2):
        return 0.0
    return min(_seg_point_dist(a1, b1, b2), _seg_point_dist(a2, b1, b2),
               _seg_point_dist(b1, a1, a2), _seg_point_dist(b2, a1, a2))


def _seg_closest(p, a, b) -> tuple[float, float]:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return (ax + t * dx, ay + t * dy)


def nearest_floor_points(a: Box, b: Box, H: Homography):
    """(distance_m(a, b, H), point on a's bottom edge, point on b's bottom edge), points in floor metres.
    The two points are the ones the distance is measured between (for drawing the line)."""
    a1, a2 = bottom_edge_floor(a, H)
    b1, b2 = bottom_edge_floor(b, H)
    if _segments_intersect(a1, a2, b1, b2):
        # crossing edges: distance 0 at the crossing point
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = a1, a2, b1, b2
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        q = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        return 0.0, q, q
    cands = [(p, _seg_closest(p, b1, b2), False) for p in (a1, a2)] + \
            [(p, _seg_closest(p, a1, a2), True) for p in (b1, b2)]
    best = min(cands, key=lambda c: math.hypot(c[0][0] - c[1][0], c[0][1] - c[1][1]))
    p, q, swapped = best
    d = math.hypot(p[0] - q[0], p[1] - q[1])
    return (d, q, p) if swapped else (d, p, q)


def ppe_region(ppe_name: str) -> str:
    """Which part of the person a PPE item sits on, by its name."""
    n = ppe_name.lower()
    if any(k in n for k in _UPPER):
        return "upper"
    if any(k in n for k in _MIDDLE):
        return "middle"
    if any(k in n for k in _LOWER):
        return "lower"
    return "whole"


def region_box(person_box: Box, region: str) -> list[float]:
    x1, y1, x2, y2 = person_box
    h = (y2 - y1) / 3.0
    if region == "upper":
        return [x1, y1, x2, y1 + h]
    if region == "middle":
        return [x1, y1 + h, x2, y1 + 2 * h]
    if region == "lower":
        return [x1, y1 + 2 * h, x2, y2]
    return [x1, y1, x2, y2]


def mask_share_in_box(mask, box: Box) -> float:
    """Share of mask pixels inside a normalised box; mask covers the whole frame."""
    m = np.asarray(mask, dtype=bool)
    total = int(m.sum())
    if total == 0:
        return 0.0
    H, W = m.shape[:2]
    c0, r0 = int(math.floor(box[0] * W)), int(math.floor(box[1] * H))
    c1, r1 = int(math.ceil(box[2] * W)), int(math.ceil(box[3] * H))
    c0, r0 = max(0, c0), max(0, r0)
    c1, r1 = min(W, c1), min(H, r1)
    if c1 <= c0 or r1 <= r0:
        return 0.0
    return float(m[r0:r1, c0:c1].sum()) / total


def mask_overlap(person_box: Box, ppe_name: str, ppe_box: Box, ppe_mask=None,
                 threshold: float = WEAR_MIN_OVERLAP) -> tuple[bool, float]:
    """Is this PPE item on this person?

    With a mask: share of the PPE mask inside the person-box third for that PPE
    (upper for hard_hat-like names, middle for vest-like). Without a mask: share
    of the PPE box inside that third. Both against `threshold` (0.3).
    """
    region = region_box(person_box, ppe_region(ppe_name))
    if ppe_mask is not None:
        share = mask_share_in_box(ppe_mask, region)
    else:
        share = overlap_ratio(ppe_box, region)
    return share >= threshold, share

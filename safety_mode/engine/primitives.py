"""Engine primitives (schemas/README.md), evaluated on one frame.

Primitives that bind an object (in_zone, distance_lt, wearing, pose,
height_above_floor) get an assignment `sigma` {object name: instance index or
None}; None means no instance of that object is in the frame, and the result
is then "absent" (value None), which evaluate.py treats as false.
count_in_zone and machine_state look at the whole frame and bind nothing.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Optional

from .geometry import Homography, distance_m as _distance_m, foot_point, mask_overlap, \
    overlap_ratio, point_in_polygon
from .types import NotEvaluable

OPERATOR_INSIDE_SHARE = 0.5  # person box >= 50 % inside the machine box ...
OPERATOR_LIFT_SHARE = 0.1    # ... and its bottom edge >= 10 % of the machine box height above the machine's


def is_operator(p_box, m_box) -> bool:
    """A person riding the machine (in the cab, on the platform), not a pedestrian.

    Box containment alone is not enough: a reach truck's box runs from the floor to the
    top of the mast, so a pedestrian beside the mast is inside it too (warehouse-team
    s18, s22). The rider's feet are above the machine's floor line; a pedestrian stands
    on the floor at or in front of the machine's bottom edge.
    """
    lift = p_box[3] <= m_box[3] - OPERATOR_LIFT_SHARE * (m_box[3] - m_box[1])
    return lift and overlap_ratio(p_box, m_box) >= OPERATOR_INSIDE_SHARE

_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}

# argument keys that bind an object instance, per primitive
BINDING_ARGS = {
    "in_zone": ("object",),
    "distance_lt": ("a", "b"),
    "wearing": ("person",),
    "pose": ("person",),
    "height_above_floor": ("object",),
}


@dataclass
class NDet:
    """Detection with a normalised box."""
    name: str
    index: int
    box: list
    mask: object = None
    score: Optional[float] = None
    track_id: Optional[int] = None


@dataclass
class Context:
    dets: dict                       # name -> list[NDet]
    zones: dict                      # zone name -> polygon (normalised)
    H: Optional[Homography]
    machine_states: dict             # machine name -> "running" | "stopped"

    def get(self, name: str, idx: Optional[int]) -> Optional[NDet]:
        if idx is None:
            return None
        return self.dets.get(name, [])[idx]


@dataclass
class PrimResult:
    value: Optional[bool]            # None = absent (no instance to talk about)
    objs: list = field(default_factory=list)   # [(name, index)]
    measurements: dict = field(default_factory=dict)


def _zone(ctx: Context, zone: str):
    poly = ctx.zones.get(zone)
    if poly is None:
        return NotEvaluable(f"zone '{zone}' is not defined for this camera")
    return poly


def in_zone(ctx: Context, sigma: dict, object: str, zone: str):
    poly = _zone(ctx, zone)
    if isinstance(poly, NotEvaluable):
        return poly
    d = ctx.get(object, sigma.get(object))
    if d is None:
        return PrimResult(None)
    return PrimResult(point_in_polygon(foot_point(d.box), poly), [(object, d.index)],
                      {"object": object})


def count_in_zone(ctx: Context, sigma: dict, object: str, zone: str, op: str, n: int):
    poly = _zone(ctx, zone)
    if isinstance(poly, NotEvaluable):
        return poly
    if op not in _OPS:
        return NotEvaluable(f"unknown operator '{op}'")
    inside = [d for d in ctx.dets.get(object, []) if point_in_polygon(foot_point(d.box), poly)]
    return PrimResult(_OPS[op](len(inside), n), [(object, d.index) for d in inside],
                      {"count": len(inside), "object": object})


def distance_lt(ctx: Context, sigma: dict, a: str, b: str, m: float):
    if ctx.H is None:
        return NotEvaluable("no floor calibration for this camera: distance_lt cannot be computed")
    da = ctx.get(a, sigma.get(a))
    if da is None:
        return PrimResult(None)
    if a == b:
        others = [d for d in ctx.dets.get(b, []) if d.index != da.index]
    else:
        db = ctx.get(b, sigma.get(b))
        if db is None:
            return PrimResult(None)
        others = [db]
    best = None
    for db in others:
        # the operator sitting in (or standing on) the machine is not a pedestrian
        if is_operator(db.box, da.box) or is_operator(da.box, db.box):
            continue
        dist = _distance_m(da.box, db.box, ctx.H)
        if best is None or dist < best[0]:
            best = (dist, db)
    if best is None:
        return PrimResult(False, [(a, da.index)])
    dist, db = best
    return PrimResult(dist < m, [(a, da.index), (b, db.index)], {"distance_m": round(dist, 2)})


def wearing(ctx: Context, sigma: dict, person: str, ppe: list):
    p = ctx.get(person, sigma.get(person))
    if p is None:
        return PrimResult(None)
    missing = []
    for item in ppe:
        if not any(mask_overlap(p.box, item, d.box, d.mask)[0] for d in ctx.dets.get(item, [])):
            missing.append(item)
    return PrimResult(not missing, [(person, p.index)],
                      {"missing_ppe": missing, "worn_ppe": [i for i in ppe if i not in missing]})


def pose(ctx: Context, sigma: dict, person: str, pose: str):
    return NotEvaluable("pose is not implemented in v1 (no keypoints from SAM 3.1)")


def height_above_floor(ctx: Context, sigma: dict, object: str, m: float):
    return NotEvaluable("height_above_floor is not implemented in v1: with one camera "
                        "a lifted object has no floor point to measure from")


def machine_state(ctx: Context, sigma: dict, machine: str, state: str):
    actual = ctx.machine_states.get(machine)
    if actual is None:
        return NotEvaluable(f"machine state of '{machine}' is not set")
    objs = [(machine, d.index) for d in ctx.dets.get(machine, [])]
    return PrimResult(actual == state, objs, {"machine_state": actual})


PRIMITIVES = {
    "in_zone": in_zone,
    "count_in_zone": count_in_zone,
    "distance_lt": distance_lt,
    "wearing": wearing,
    "pose": pose,
    "height_above_floor": height_above_floor,
    "machine_state": machine_state,
}

# primitives that can never be computed in v1 (whole rule is "not checkable")
NOT_IMPLEMENTED = {"pose", "height_above_floor"}

REQUIRED_ARGS = {
    "in_zone": ("object", "zone"),
    "count_in_zone": ("object", "zone", "op", "n"),
    "distance_lt": ("a", "b", "m"),
    "wearing": ("person", "ppe"),
    "pose": ("person", "pose"),
    "height_above_floor": ("object", "m"),
    "machine_state": ("machine", "state"),
}

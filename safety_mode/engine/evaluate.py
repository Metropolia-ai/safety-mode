"""Walk the condition tree of rule.json on one frame.

Semantics: the rule is true on a frame if there is an assignment of detected
instances to the object names the condition binds (in_zone.object,
distance_lt.a/b, wearing.person, pose.person, height_above_floor.object) that
makes the tree true. So in r02 `all[in_zone(person), not wearing(person)]`
both nodes talk about the same person.

Inside the tree four values are used: T, F, NE (not evaluable) and ABS (the
bound object is not in the frame). ABS behaves as false in all/any, `not`
keeps it ABS (so `not wearing(person)` is not true when there is no person),
and at the root ABS becomes false. NE follows Kleene logic: all = F if any
child is F, any = T if any child is T, otherwise NE wins over the rest.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .geometry import Homography
from .primitives import BINDING_ARGS, PRIMITIVES, REQUIRED_ARGS, Context, NDet, PrimResult
from .types import ConditionResult, Detection, Frame, MatchedObject, NotEvaluable, Truth

T, F, NE, ABS = "T", "F", "NE", "ABS"
COMBINATORS = ("all", "any", "not")
MAX_ASSIGNMENTS = 50_000


@dataclass
class _Node:
    v: str
    objs: set = field(default_factory=set)
    meas: dict = field(default_factory=dict)
    reason: Optional[str] = None


def merge_measurements(into: dict, new: dict) -> dict:
    for k, val in new.items():
        if k not in into:
            into[k] = list(val) if isinstance(val, list) else val
        elif isinstance(val, list):
            into[k] = into[k] + [x for x in val if x not in into[k]]
        elif k == "distance_m":
            into[k] = min(into[k], val)
        elif k == "count":
            into[k] = max(into[k], val)
        else:
            into[k] = val
    return into


# ---------------------------------------------------------------- validation

def node_key(node) -> str:
    if not isinstance(node, dict) or len(node) != 1:
        raise ValueError(f"condition node must be an object with exactly one key: {node!r}")
    return next(iter(node))


def validate_condition(node, objects: Iterable[str], zones: Iterable[str], depth: int = 1) -> list[str]:
    """Structural check of a condition tree against rule.schema.json + names."""
    objects, zones = set(objects), set(zones)
    errors: list[str] = []
    try:
        key = node_key(node)
    except ValueError as e:
        return [str(e)]
    arg = node[key]
    if key in ("all", "any"):
        if depth >= 3:
            errors.append(f"'{key}' at level {depth}: schema allows combinators only on levels 1-2")
        if not isinstance(arg, list):
            return errors + [f"'{key}' must be an array"]
        for child in arg:
            errors += validate_condition(child, objects, zones, depth + 1)
    elif key == "not":
        if depth >= 3:
            errors.append(f"'not' at level {depth}: schema allows combinators only on levels 1-2")
        errors += validate_condition(arg, objects, zones, depth + 1)
    elif key in PRIMITIVES:
        if not isinstance(arg, dict):
            return [f"'{key}' arguments must be an object"]
        for req in REQUIRED_ARGS[key]:
            if req not in arg:
                errors.append(f"'{key}' is missing argument '{req}'")
        extra = set(arg) - set(REQUIRED_ARGS[key])
        if extra:
            errors.append(f"'{key}' has unknown arguments {sorted(extra)}")
        for k in ("object", "a", "b", "person", "machine"):
            if k in arg and arg[k] not in objects:
                errors.append(f"'{key}.{k}' = '{arg[k]}' is not in rule.objects")
        if key == "wearing":
            for p in arg.get("ppe", []):
                if p not in objects:
                    errors.append(f"'wearing.ppe' item '{p}' is not in rule.objects")
        if "zone" in arg and arg["zone"] not in zones:
            errors.append(f"zone '{arg['zone']}' is not defined for this camera")
    else:
        errors.append(f"unknown condition node '{key}'")
    return errors


def primitives_used(node) -> set[str]:
    key = node_key(node)
    if key in ("all", "any"):
        return set().union(*[primitives_used(c) for c in node[key]]) if node[key] else set()
    if key == "not":
        return primitives_used(node[key])
    return {key}


def bound_names(node) -> list[str]:
    key = node_key(node)
    if key in ("all", "any"):
        out: list[str] = []
        for c in node[key]:
            out += [n for n in bound_names(c) if n not in out]
        return out
    if key == "not":
        return bound_names(node[key])
    return [node[key][a] for a in BINDING_ARGS.get(key, ()) if a in node[key]]


# ---------------------------------------------------------------- evaluation

def _eval(node, ctx: Context, sigma: dict) -> _Node:
    key = node_key(node)
    arg = node[key]
    if key == "all":
        kids = [_eval(c, ctx, sigma) for c in arg]
        vs = [k.v for k in kids]
        if F in vs:
            return _Node(F)
        if ABS in vs:
            return _Node(ABS)
        if NE in vs:
            return _Node(NE, reason=next(k.reason for k in kids if k.v == NE))
        out = _Node(T)
        for k in kids:
            out.objs |= k.objs
            merge_measurements(out.meas, k.meas)
        return out
    if key == "any":
        kids = [_eval(c, ctx, sigma) for c in arg]
        trues = [k for k in kids if k.v == T]
        if trues:
            out = _Node(T)
            for k in trues:
                out.objs |= k.objs
                merge_measurements(out.meas, k.meas)
            return out
        nes = [k for k in kids if k.v == NE]
        if nes:
            return _Node(NE, reason=nes[0].reason)
        return _Node(F) if any(k.v == F for k in kids) else _Node(ABS)
    if key == "not":
        k = _eval(arg, ctx, sigma)
        flip = {T: F, F: T}.get(k.v, k.v)
        return _Node(flip, k.objs, k.meas, k.reason)
    res = PRIMITIVES[key](ctx, sigma, **arg)
    if isinstance(res, NotEvaluable):
        return _Node(NE, reason=res.reason)
    assert isinstance(res, PrimResult)
    if res.value is None:
        return _Node(ABS)
    return _Node(T if res.value else F, set(res.objs), dict(res.measurements))


def _true_result(objs: set, meas: dict, ctx: Context) -> ConditionResult:
    matched = []
    for name, idx in sorted(objs):
        d: NDet = ctx.dets[name][idx]
        matched.append(MatchedObject(name, idx, list(d.box), d.track_id, d.score))
    counts: dict = {}
    for m in matched:
        counts[m.name] = counts.get(m.name, 0) + 1
    meas["object_counts"] = counts
    return ConditionResult(Truth.TRUE, matched, meas)


def tracked(names: list[str], ctx: Context) -> bool:
    """Episodes per object: the rule binds objects and every detection of them has a track_id."""
    return bool(names) and all(d.track_id is not None for n in names for d in ctx.dets.get(n, []))


def evaluate_context(condition: dict, ctx: Context) -> ConditionResult:
    """True if some assignment of instances makes the tree true. With track ids (see `tracked`)
    the result also carries by_track: the true assignments grouped by the tracked bound
    objects that made them true, {(("person", 2),): ConditionResult}."""
    names = bound_names(condition)
    domains = [list(range(len(ctx.dets.get(n, [])))) or [None] for n in names]
    total = 1
    for d in domains:
        total *= len(d)
    if total > MAX_ASSIGNMENTS:
        return ConditionResult(Truth.NOT_EVALUABLE, reason=f"too many object combinations ({total})")
    use_ids = tracked(names, ctx)
    objs: set = set()
    meas: dict = {}
    groups: dict = {}            # track key -> (objs, meas)
    any_true, ne_reason = False, None
    for combo in itertools.product(*domains):
        sigma = dict(zip(names, combo))
        r = _eval(condition, ctx, sigma)
        if r.v == T:
            any_true = True
            objs |= r.objs
            merge_measurements(meas, r.meas)
            if use_ids:
                key = tuple(sorted({(n, ctx.dets[n][i].track_id) for n, i in r.objs if n in names}))
                g_objs, g_meas = groups.setdefault(key, (set(), {}))
                g_objs |= r.objs
                merge_measurements(g_meas, r.meas)
        elif r.v == NE and ne_reason is None:
            ne_reason = r.reason
    if any_true:
        res = _true_result(objs, meas, ctx)
        if use_ids:
            res.by_track = {k: _true_result(o, m, ctx) for k, (o, m) in sorted(groups.items())}
        return res
    if ne_reason is not None:
        return ConditionResult(Truth.NOT_EVALUABLE, reason=ne_reason)
    return ConditionResult(Truth.FALSE)


def zones_to_polygons(zones) -> dict:
    """Accept {name: polygon}, a list of zone dicts, or a whole zones file dict."""
    if isinstance(zones, dict) and "zones" in zones:
        zones = zones["zones"]
    if isinstance(zones, dict):
        return {k: [tuple(p) for p in v] for k, v in zones.items()}
    return {z["name"]: [tuple(p) for p in z["polygon"]] for z in zones}


def normalise_detections(detections: Iterable[Detection], size: Optional[tuple], box_units: str) -> dict:
    out: dict = {}
    for d in detections:
        x1, y1, x2, y2 = (float(v) for v in d.box)
        if box_units == "px":
            if not size:
                raise ValueError("frame size unknown: pass Frame.size or a zones file with frame_size")
            w, h = size
            x1, x2, y1, y2 = x1 / w, x2 / w, y1 / h, y2 / h
        lst = out.setdefault(d.name, [])
        lst.append(NDet(d.name, len(lst), [x1, y1, x2, y2], d.mask, d.score, d.track_id))
    return out


def evaluate(rule: dict, frame: Frame, zones, calibration=None, machine_states: Optional[dict] = None,
             *, frame_size: Optional[tuple] = None, box_units: str = "px") -> ConditionResult:
    """Evaluate rule.condition on one frame.

    zones: {name: polygon} / list of zone dicts / zones file dict (polygons 0..1).
    calibration: floor_calibration dict, a Homography, or None.
    machine_states: {machine name: "running"|"stopped"}; frame.machine_states override it.
    """
    cond = rule.get("condition")
    if cond is None:
        return ConditionResult(Truth.NOT_EVALUABLE, reason=rule.get("unsupported") or "rule has no condition")
    H = calibration if isinstance(calibration, Homography) or calibration is None \
        else Homography.from_calibration(calibration)
    states = dict(machine_states or {})
    states.update(frame.machine_states or {})
    ctx = Context(normalise_detections(frame.detections, frame.size or frame_size, box_units),
                  zones_to_polygons(zones), H, states)
    return evaluate_context(cond, ctx)

"""Deterministic condition engine: detections + zones + rule.json -> events.

    from safety_mode.engine import Engine, Frame, Detection
    eng = Engine(["rules/r02.json"], "test-footage/zones/<clip>.json",
                 machine_states={"press": "running"})
    for frame in frames:            # Frame(t_s, [Detection(name, box_px, mask, score, track_id)])
        for ev in eng.process(frame):
            ...                     # Event opened / updated / closed on this frame
    eng.finish()                    # close what is still active at the end of the clip
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Union

from .evaluate import (bound_names, evaluate, evaluate_context, normalise_detections,
                       primitives_used, validate_condition, zones_to_polygons)
from .geometry import Homography
from .primitives import NOT_IMPLEMENTED, Context
from .state import RuleState
from .types import (ConditionResult, Detection, Event, Frame, MatchedObject, NotEvaluable,
                    Truth)

__all__ = ["Engine", "RuleRejected", "Detection", "Frame", "Event", "ConditionResult",
           "MatchedObject", "NotEvaluable", "Truth", "evaluate", "Homography", "check_rule"]

_SCHEMA = Path(__file__).resolve().parents[2] / "schemas" / "rule.schema.json"


class RuleRejected(ValueError):
    """The rule cannot run on this camera (unsupported, invalid, unknown zone)."""

    def __init__(self, reasons: dict):
        self.reasons = reasons
        super().__init__("; ".join(f"{k}: {v}" for k, v in reasons.items()))


def _load_json(x):
    if isinstance(x, (str, Path)):
        return json.loads(Path(x).read_text(encoding="utf-8"))
    return x


def _schema_errors(rule: dict) -> list[str]:
    try:
        import jsonschema
    except ImportError:
        return []
    if not _SCHEMA.exists():
        return []
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    return [f"schema: {e.message}" for e in v.iter_errors(rule)][:3]


def check_rule(rule: dict, zone_names: Iterable[str], has_calibration: bool) -> tuple[Optional[str], Optional[str]]:
    """Returns (reject_reason, not_evaluable_reason); both None = runs normally."""
    if rule.get("condition") is None:
        return ("unsupported: " + (rule.get("unsupported") or "rule has no condition")), None
    errs = _schema_errors(rule)
    objects = [o["name"] for o in rule.get("objects", [])]
    errs += validate_condition(rule["condition"], objects, zone_names)
    if errs:
        return "; ".join(errs), None
    used = primitives_used(rule["condition"])
    ni = sorted(used & NOT_IMPLEMENTED)
    if ni:
        return None, f"uses {', '.join(ni)}, not implemented in v1"
    if "distance_lt" in used and not has_calibration:
        return None, "no floor calibration for this camera: distance_lt cannot be computed"
    return None, None


class Engine:
    """Runs a set of rules on frames of one camera.

    rules: rule dicts or paths to rule.json.
    zones_file: path or dict in the test-footage/zones/*.json format
        (zones with normalised polygons, frame_size, floor_calibration).
    machine_states: {machine object name: "running"|"stopped"}, manual in v1;
        Frame.machine_states overrides it per frame.
    frame_period_s: how long one frame "lasts" (1.0 at 1 fps).
    max_gap_s: not-true time an episode tolerates before closing (default 1 s: one
        missing frame at 1 fps does not split an episode, test-footage/expected.md; 0 = none).
    box_units: "px" (frame pixels, default) or "norm" (0..1).
    episodes: "object" (default): when the detections of the objects a rule binds carry
        track_id, one episode per tracked object set ("person #2"), events get object_ids and
        subject; without ids one episode per rule. "rule": one episode per rule even with ids.
    skip_rejected: False = raise RuleRejected; True = keep them in self.rejected.
    """

    def __init__(self, rules: Iterable[Union[dict, str, Path]], zones_file, machine_states: Optional[dict] = None,
                 *, frame_period_s: float = 1.0, max_gap_s: float = 1.0, box_units: str = "px",
                 skip_rejected: bool = False, episodes: str = "object"):
        if episodes not in ("object", "rule"):
            raise ValueError("episodes must be 'object' or 'rule'")
        zf = _load_json(zones_file)
        self.zones = zones_to_polygons(zf)
        self.frame_size = tuple(zf["frame_size"]) if isinstance(zf, dict) and zf.get("frame_size") else None
        cal = zf.get("floor_calibration") if isinstance(zf, dict) else None
        self.H = Homography.from_calibration(cal)
        self.machine_states = dict(machine_states or {})
        self.frame_period_s = frame_period_s
        self.box_units = box_units
        self.rejected: dict[str, str] = {}
        self.not_evaluable: dict[str, str] = {}
        self.states: dict[str, RuleState] = {}
        self._next_id = 0
        for r in rules:
            rule = _load_json(r)
            rid = rule.get("rule_id", "?")
            reject, ne = check_rule(rule, self.zones, self.H is not None)
            if reject:
                self.rejected[rid] = reject
                continue
            if ne:
                self.not_evaluable[rid] = ne
            self.states[rid] = RuleState(rule, frame_period_s, max_gap_s, per_object=episodes == "object")
        if self.rejected and not skip_rejected:
            raise RuleRejected(self.rejected)

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    @property
    def events(self) -> list[Event]:
        out = []
        for st in self.states.values():
            out += st.closed + st.active
        return sorted(out, key=lambda e: e.event_id)

    def evaluate(self, frame: Frame) -> dict[str, ConditionResult]:
        states = dict(self.machine_states)
        states.update(frame.machine_states or {})
        dets = normalise_detections(frame.detections, frame.size or self.frame_size, self.box_units)
        ctx = Context(dets, self.zones, self.H, states)
        return {rid: evaluate_context(st.rule["condition"], ctx) for rid, st in self.states.items()}

    def process(self, frame: Frame) -> list[Event]:
        """Feed one frame; returns snapshots (copies) of events opened, updated or closed on it."""
        out: list[Event] = []
        for rid, res in self.evaluate(frame).items():
            out += self.states[rid].step(frame.t_s, res, self._new_id)
        return out

    def finish(self) -> list[Event]:
        out: list[Event] = []
        for st in self.states.values():
            out += st.finish()
        return out

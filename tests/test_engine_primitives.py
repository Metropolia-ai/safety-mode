"""Each engine primitive and the condition walker on one frame."""
import json
from pathlib import Path

import pytest

from safety_mode.engine import Detection, Frame, Truth, evaluate
from safety_mode.engine.geometry import Homography

ZONES = {"z": [(0.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 1.0)]}   # bottom-left quarter
CAL = json.loads((Path(__file__).resolve().parent.parent / "test-footage" / "zones" /
                  "pexels-6079421-forklift-busy-warehouse.json").read_text())["floor_calibration"]


def rule(cond, objects):
    return {"rule_id": "t", "objects": [{"name": o, "prompt": o} for o in objects], "zones": ["z"],
            "condition": cond, "min_duration_s": 1, "severity": "low"}


def ev(cond, objects, dets, **kw):
    return evaluate(rule(cond, objects), Frame(0, dets), ZONES, box_units="norm", **kw)


def P(box, name="person"):
    return Detection(name, box)


def test_in_zone_uses_foot_point():
    c = {"in_zone": {"object": "person", "zone": "z"}}
    assert ev(c, ["person"], [P([0.1, 0.6, 0.2, 0.9])]).truth is Truth.TRUE
    # box overlaps the zone but the feet are right of it
    assert ev(c, ["person"], [P([0.45, 0.6, 0.65, 0.9])]).truth is Truth.FALSE
    assert ev(c, ["person"], []).truth is Truth.FALSE


def test_count_in_zone_ops():
    dets = [P([0.1, 0.6, 0.2, 0.9]), P([0.3, 0.6, 0.4, 0.9]), P([0.7, 0.6, 0.8, 0.9])]
    def c(op, n):
        return {"count_in_zone": {"object": "person", "zone": "z", "op": op, "n": n}}
    r = ev(c(">", 1), ["person"], dets)
    assert r.truth is Truth.TRUE and r.measurements["count"] == 2 and len(r.objects) == 2
    assert ev(c("==", 2), ["person"], dets).truth is Truth.TRUE
    assert ev(c(">=", 3), ["person"], dets).truth is Truth.FALSE
    assert ev(c("<", 1), ["person"], []).truth is Truth.TRUE


def test_distance_lt_needs_calibration():
    c = {"distance_lt": {"a": "forklift", "b": "person", "m": 2}}
    dets = [Detection("forklift", [0.3, 0.6, 0.5, 0.8]), P([0.55, 0.65, 0.6, 0.8])]
    r = ev(c, ["forklift", "person"], dets)
    assert r.truth is Truth.NOT_EVALUABLE and "calibration" in r.reason
    r = ev(c, ["forklift", "person"], dets, calibration=CAL)
    assert r.truth is Truth.TRUE and r.measurements["distance_m"] < 2


def test_distance_lt_excludes_operator_inside_machine_box():
    c = {"distance_lt": {"a": "forklift", "b": "person", "m": 2}}
    dets = [Detection("forklift", [0.3, 0.5, 0.5, 0.8]), P([0.35, 0.52, 0.42, 0.7])]
    assert ev(c, ["forklift", "person"], dets, calibration=CAL).truth is Truth.FALSE


def test_wearing_and_missing_ppe():
    c = {"wearing": {"person": "person", "ppe": ["hard_hat", "vest"]}}
    person = P([0.4, 0.2, 0.6, 0.8])
    hat = Detection("hard_hat", [0.45, 0.18, 0.55, 0.26])
    vest = Detection("vest", [0.42, 0.36, 0.58, 0.55])
    objs = ["person", "hard_hat", "vest"]
    assert ev(c, objs, [person, hat, vest]).truth is Truth.TRUE
    r = ev({"not": c}, objs, [person, hat])
    assert r.truth is Truth.TRUE and r.measurements["missing_ppe"] == ["vest"]


def test_not_wearing_without_person_is_false():
    c = {"not": {"wearing": {"person": "person", "ppe": ["vest"]}}}
    assert ev(c, ["person", "vest"], []).truth is Truth.FALSE


def test_machine_state_manual():
    c = {"machine_state": {"machine": "press", "state": "running"}}
    assert ev(c, ["press"], [], machine_states={"press": "running"}).truth is Truth.TRUE
    assert ev(c, ["press"], [], machine_states={"press": "stopped"}).truth is Truth.FALSE
    r = ev(c, ["press"], [])
    assert r.truth is Truth.NOT_EVALUABLE and "press" in r.reason


def test_pose_and_height_are_not_evaluable():
    r = ev({"pose": {"person": "person", "pose": "lying"}}, ["person"], [P([0.1, 0.6, 0.2, 0.9])])
    assert r.truth is Truth.NOT_EVALUABLE
    r = ev({"height_above_floor": {"object": "person", "m": 1.2}}, ["person"], [P([0.1, 0.6, 0.2, 0.9])])
    assert r.truth is Truth.NOT_EVALUABLE


def test_kleene_logic_with_not_evaluable():
    inz = {"in_zone": {"object": "person", "zone": "z"}}
    pose = {"pose": {"person": "person", "pose": "lying"}}
    outside = [P([0.7, 0.6, 0.8, 0.9])]
    inside = [P([0.1, 0.6, 0.2, 0.9])]
    assert ev({"all": [inz, pose]}, ["person"], outside).truth is Truth.FALSE
    assert ev({"all": [inz, pose]}, ["person"], inside).truth is Truth.NOT_EVALUABLE
    assert ev({"any": [inz, pose]}, ["person"], inside).truth is Truth.TRUE
    assert ev({"not": pose}, ["person"], inside).truth is Truth.NOT_EVALUABLE


def test_same_person_binding_in_all():
    # one person in the zone with full PPE, another without PPE outside the zone -> false
    c = {"all": [{"in_zone": {"object": "person", "zone": "z"}},
                 {"not": {"wearing": {"person": "person", "ppe": ["vest"]}}}]}
    p_in = P([0.1, 0.3, 0.3, 0.9])
    vest = Detection("vest", [0.12, 0.5, 0.28, 0.65])
    p_out = P([0.7, 0.3, 0.9, 0.9])
    r = ev(c, ["person", "vest"], [p_in, vest, p_out])
    assert r.truth is Truth.FALSE
    # without the vest the person in the zone matches, the one outside does not
    r = ev(c, ["person", "vest"], [p_in, p_out])
    assert r.truth is Truth.TRUE and [o.index for o in r.objects] == [0]

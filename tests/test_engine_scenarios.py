"""Scenario tests: rules/r*.json on the real zone files with hand-made detections."""
import json
from pathlib import Path

import pytest

from safety_mode.engine import Detection, Engine, Frame, RuleRejected, Truth
from safety_mode.engine.geometry import Homography, distance_m, overlap_ratio

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "rules"
ZDIR = ROOT / "test-footage" / "zones"
BUSY = ZDIR / "pexels-6079421-forklift-busy-warehouse.json"
TEAM = ZDIR / "pexels-15170997-warehouse-team.json"
OPER = ZDIR / "pexels-32243653-forklift-operator.json"


def R(n):
    return RULES / f"r{n:02d}.json"


def px(zfile, nbox):
    w, h = json.loads(Path(zfile).read_text())["frame_size"]
    x1, y1, x2, y2 = nbox
    return [x1 * w, y1 * h, x2 * w, y2 * h]


def D(name, zfile, nbox):
    return Detection(name, px(zfile, nbox))


def play(eng, frames):
    """frames: list of detection lists, one per second. Returns events after finish()."""
    for t, dets in enumerate(frames):
        eng.process(Frame(t, dets))
    eng.finish()
    return eng.events


# ---------------------------------------------------------------- r01
def r01_person_in_yellow():
    return D("person", TEAM, [0.66, 0.40, 0.70, 0.55])


def test_r01_person_in_yellow_zone_while_press_running():
    eng = Engine([R(1)], TEAM, machine_states={"press": "running"})
    evs = play(eng, [[r01_person_in_yellow()]] * 2)
    (ev,) = evs
    assert ev.start_s == 0 and ev.end_s == 1 and ev.severity == "high"
    assert ev.explanation_inputs["person_count"] == 1


def test_r01_press_stopped_or_person_outside():
    eng = Engine([R(1)], TEAM, machine_states={"press": "stopped"})
    assert play(eng, [[r01_person_in_yellow()]] * 3) == []
    eng = Engine([R(1)], TEAM, machine_states={"press": "running"})
    assert play(eng, [[D("person", TEAM, [0.2, 0.6, 0.25, 0.8])]] * 3) == []


def test_r01_per_frame_machine_state_and_missing_state():
    eng = Engine([R(1)], TEAM)
    res = eng.evaluate(Frame(0, [r01_person_in_yellow()]))["r1"]
    assert res.truth is Truth.NOT_EVALUABLE
    eng.process(Frame(0, [r01_person_in_yellow()], machine_states={"press": "running"}))
    eng.process(Frame(1, [r01_person_in_yellow()], machine_states={"press": "stopped"}))
    eng.process(Frame(2, [r01_person_in_yellow()], machine_states={"press": "stopped"}))
    (ev,) = eng.events                  # default max_gap_s=1: closes on the 2nd not-true frame
    assert ev.status == "closed" and ev.end_s == 0


# ---------------------------------------------------------------- r02
P_IN = [0.27, 0.5, 0.33, 0.8]          # feet on the shop floor (past_gate)
HAT = [0.28, 0.49, 0.32, 0.53]         # upper third 0.50-0.60
VEST = [0.275, 0.61, 0.325, 0.68]      # middle third 0.60-0.70


def test_r02_no_ppe_three_seconds_event_at_s2():
    eng = Engine([R(2)], BUSY)
    evs = play(eng, [[D("person", BUSY, P_IN)]] * 5)
    (ev,) = evs
    assert ev.start_s == 2 and ev.end_s == 4 and ev.condition_since_s == 0
    assert ev.explanation_inputs["missing_ppe"] == "hard hat, vest"


def test_r02_full_ppe_no_event_and_missing_vest_only():
    eng = Engine([R(2)], BUSY)
    full = [D("person", BUSY, P_IN), D("hard_hat", BUSY, HAT), D("vest", BUSY, VEST)]
    assert play(eng, [full] * 5) == []
    eng = Engine([R(2)], BUSY)
    (ev,) = play(eng, [[D("person", BUSY, P_IN), D("hard_hat", BUSY, HAT)]] * 3)
    assert ev.measurements["missing_ppe"] == ["vest"]


def test_r02_pedestrian_behind_the_gate_is_outside_zone():
    eng = Engine([R(2)], BUSY)
    assert play(eng, [[D("person", BUSY, [0.41, 0.45, 0.43, 0.56])]] * 5) == []


# ---------------------------------------------------------------- r03
H_BUSY = Homography.from_calibration(json.loads(BUSY.read_text())["floor_calibration"])


def floor_box(x0_m, x1_m, y_m, height_norm):
    (u0, v), (u1, _) = H_BUSY.to_image((x0_m, y_m)), H_BUSY.to_image((x1_m, y_m))
    return [u0, v - height_norm, u1, v]


FORK = floor_box(0.0, 1.5, 2.0, 0.2)


@pytest.mark.parametrize("gap_m,expect_event", [(1.5, True), (3.0, False)])
def test_r03_forklift_pedestrian_distance(gap_m, expect_event):
    person = floor_box(1.5 + gap_m, 1.9 + gap_m, 2.0, 0.15)
    assert distance_m(FORK, person, H_BUSY) == pytest.approx(gap_m, abs=1e-6)
    assert overlap_ratio(person, FORK) < 0.5
    eng = Engine([R(3)], BUSY)
    evs = play(eng, [[D("forklift", BUSY, FORK), D("person", BUSY, person)]] * 2)
    if expect_event:
        (ev,) = evs
        assert ev.start_s == 0 and ev.explanation_inputs["distance_m"] == pytest.approx(1.5, abs=0.01)
        assert {o["name"] for o in ev.objects} == {"forklift", "person"}
    else:
        assert evs == []


def test_r03_operator_inside_forklift_box_excluded():
    x1, y1, x2, y2 = FORK
    operator = [x1 + 0.3 * (x2 - x1), y1 + 0.02, x1 + 0.5 * (x2 - x1), y2 - 0.05]
    assert overlap_ratio(operator, FORK) >= 0.5
    eng = Engine([R(3)], BUSY)
    assert play(eng, [[D("forklift", BUSY, FORK), D("person", BUSY, operator)]] * 3) == []


def test_r03_pedestrian_beside_reach_truck_mast_is_not_the_operator():
    """Regression, warehouse-team s22 (SAM 3.1 boxes, px): the reach-truck box spans the
    mast, the pedestrian is 91 % inside it but stands on the floor in front of it."""
    truck = [363, 514, 759, 1302]
    pedestrian = [597, 980, 745, 1332]
    rider = [839, 973, 912, 1069]                    # s0: operator on the platform
    rider_truck = [814, 0, 1074, 1262]
    eng = Engine([R(3)], TEAM)
    (ev,) = play(eng, [[Detection("forklift", truck), Detection("person", pedestrian)]])
    assert ev.explanation_inputs["distance_m"] < 1
    eng = Engine([R(3)], TEAM)
    assert play(eng, [[Detection("forklift", rider_truck), Detection("person", rider)]] * 2) == []


def test_r03_without_calibration_is_not_evaluable():
    eng = Engine([R(3)], OPER)
    assert "calibration" in eng.not_evaluable["r3"]
    res = eng.evaluate(Frame(0, [D("forklift", OPER, [0.1, 0.2, 0.5, 0.9]),
                                 D("person", OPER, [0.6, 0.2, 0.7, 0.9])]))["r3"]
    assert res.truth is Truth.NOT_EVALUABLE


# ---------------------------------------------------------------- r05
PALLET_IN_EXIT = [0.52, 0.55, 0.58, 0.61]


def test_r05_fire_exit_blocked_for_60_s():
    eng = Engine([R(5)], BUSY)
    (ev,) = play(eng, [[D("pallet", BUSY, PALLET_IN_EXIT)]] * 61)
    assert ev.start_s == 59 and ev.end_s == 60
    assert ev.explanation_inputs["object"] == "pallet"


def test_r05_clip_shorter_than_window_no_event():
    eng = Engine([R(5)], BUSY)
    assert play(eng, [[D("pallet", BUSY, PALLET_IN_EXIT), D("box", BUSY, [0.1, 0.8, 0.2, 0.9])]] * 25) == []


# ---------------------------------------------------------------- r07
CHARGING = {"frame_size": [1280, 720], "floor_calibration": None,
            "zones": [{"name": "charging_zone", "polygon": [[0.6, 0.6], [0.9, 0.6], [0.9, 0.95], [0.6, 0.95]]}]}


def test_r07_two_people_in_charging_zone_5_s():
    two = [D("person", BUSY, [0.62, 0.5, 0.66, 0.8]),
           D("person", BUSY, [0.75, 0.5, 0.79, 0.85])]
    eng = Engine([R(7)], CHARGING)
    (ev,) = play(eng, [two] * 6)
    assert ev.start_s == 4 and ev.explanation_inputs["count"] == 2
    eng = Engine([R(7)], CHARGING)
    assert play(eng, [two[:1]] * 6) == []


# ---------------------------------------------------------------- r08
def test_r08_forklift_on_pedestrian_lane():
    eng = Engine([R(8)], BUSY)
    (ev,) = play(eng, [[D("forklift", BUSY, [0.8, 0.55, 1.0, 0.79])]] * 2)
    assert ev.start_s == 1 and ev.severity == "medium"


def test_r08_box_overlaps_lane_but_wheels_outside():
    eng = Engine([R(8)], BUSY)
    assert play(eng, [[D("forklift", BUSY, [0.55, 0.5, 0.85, 0.75])]] * 4) == []


def test_r08_single_frame_is_not_enough():
    eng = Engine([R(8)], BUSY)
    assert play(eng, [[D("forklift", BUSY, [0.8, 0.55, 1.0, 0.79])], []]) == []


# ---------------------------------------------------------------- load-time status
@pytest.mark.parametrize("n", [4, 9])
def test_unsupported_rules_rejected_at_load(n):
    with pytest.raises(RuleRejected) as e:
        Engine([R(n)], BUSY)
    assert f"r{n}" in e.value.reasons and "unsupported" in e.value.reasons[f"r{n}"]


def test_unknown_zone_rejected():
    with pytest.raises(RuleRejected) as e:
        Engine([R(8)], TEAM)
    assert "pedestrian_lane" in e.value.reasons["r8"]


def test_r06_r10_load_but_not_evaluable():
    eng = Engine([R(6), R(10)], TEAM)
    assert set(eng.not_evaluable) == {"r6", "r10"}
    person_in_rack = D("person", TEAM, [0.75, 0.40, 0.79, 0.53])
    res = eng.evaluate(Frame(0, [person_in_rack]))
    assert res["r6"].truth is Truth.NOT_EVALUABLE and res["r10"].truth is Truth.NOT_EVALUABLE
    assert play(eng, [[person_in_rack]] * 12) == []


def test_all_ten_rules_with_skip_rejected():
    eng = Engine([R(n) for n in range(1, 11)], BUSY, skip_rejected=True)
    assert set(eng.rejected) == {"r4", "r6", "r7", "r9"}   # r6/r7 zones not on this camera
    assert set(eng.states) == {"r1", "r2", "r3", "r5", "r8", "r10"}

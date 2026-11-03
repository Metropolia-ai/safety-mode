"""Pipeline on the committed SAM 3.1 video-mode cache (detections/<clip>.video.json), offline."""
import pytest

from safety_mode import pipeline
from safety_mode.pipeline import DetectionSource, intervals, run_clip


@pytest.fixture(autouse=True)
def no_frames(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "FRAMES_DIR", tmp_path)    # sizes and frame list from the JSON


def video(clip, rules, **kw):
    return run_clip(clip, rules, source=DetectionSource("offline", detector="video"), **kw)


def test_default_detector_is_video():
    assert pipeline.DEFAULT_DETECTOR == "video"
    assert DetectionSource("offline").compact_path("x").name == "x.video.json"


def test_forklift_operator_r02_one_person():
    run = video("forklift-operator", ["r02"])
    assert run.calls == 0 and run.detector == "video"
    (ev,) = run.events
    assert (ev.start_s, ev.end_s, ev.subject) == (2, 9, "person #0")
    assert ev.object_ids == [{"name": "person", "track_id": 0}]


def test_warehouse_team_vest_rule_is_about_one_person():
    run = video("warehouse-team", ["r02-vest"])
    assert [(int(e.start_s), int(e.end_s), e.subject) for e in run.events] == [(3, 12, "person #1")]
    assert run.events[0].explanation_inputs["subject"] == "person #1"


def test_warehouse_team_r03_finds_reach_truck_image_mode_missed():
    # image mode: forklift not found on s3, s4, s11 (results-2026-10-27.md); video mode keeps the track
    run = video("warehouse-team", ["r03"])
    assert all(run.timeline["r3"][s] == "T" for s in (3, 4, 11))
    assert {e.object_ids[0]["name"] for e in run.events} == {"forklift"}


def test_busy_warehouse_r11_forklift_track():
    run = video("busy-warehouse", ["r11"])
    assert [(int(e.start_s), int(e.end_s), e.subject) for e in run.events] == [(1, 3, "forklift #0")]


def test_image_detector_has_no_ids_and_rule_episodes():
    run = run_clip("warehouse-team", ["r02"], source=DetectionSource("offline", detector="image"))
    assert intervals(run, "r2") == [(3, 22)]
    assert run.events[0].subject == "" and run.events[0].object_ids == []


def test_video_with_rule_episodes_matches_rule_level_timing():
    run = video("warehouse-team", ["r02"], episodes="rule")
    assert intervals(run, "r2") == [(3, 22)]


def test_warehouse_team_r02_per_worker_after_bridging():
    # person #10 (SAM id after the worker in the coverall was hidden at s13) is bridged to #5
    run = video("warehouse-team", ["r02"])
    assert [(int(e.start_s), int(e.end_s), e.subject) for e in run.events] == [
        (3, 12, "person #1"), (10, 16, "person #5")]


def test_explanation_facts_carry_the_subject():
    import dataclasses
    from safety_mode.explain import build_facts
    run = video("forklift-operator", ["r02"])
    facts = build_facts(dataclasses.asdict(run.events[0]), run.rules["r2"])
    assert facts["event"]["subject"] == "person #0" and facts["event"]["object_ids"] == ["person #0"]
    assert facts["objects"][0]["label"] == "person #0"


# ---- sample clips added 30.10 (test-footage/candidates/REPORT-2026-10-30.md): violation starts and ends in frame

@pytest.mark.parametrize("rule", ["r02", "r02-vest"])
def test_worker_crosses_floor_alert_s7_s21(rule):
    run = video("man-crosses-floor", [rule], episodes="rule")
    rid = next(iter(run.timeline))
    assert run.timeline[rid] == "....." + "T" * 17 + ".."     # in the zone from s5, gone after s21
    assert [(int(e.start_s), int(e.end_s), e.subject) for e in run.events] == [(7, 21, "person #0")]


def test_forklift_crosses_walkway_alert_s6_s7():
    run = video("forklift-crosses-aisle", ["r08"], episodes="rule")
    assert run.timeline["r8"] == ".....TTT...."                 # s8: only the tail, foot point outside
    assert [(int(e.start_s), int(e.end_s), e.subject) for e in run.events] == [(6, 7, "forklift #0")]


@pytest.mark.parametrize("rule", ["r02", "r02-vest"])
def test_site_vest_check_alert_s2_s5(rule):
    run = video("site-worker-no-vest", [rule], episodes="rule")
    rid = next(iter(run.timeline))
    # s3: the vest of #0 overlaps #1's box (no masks), the 1 s gap keeps the alert; s10: #0 behind scaffolding
    assert run.timeline[rid] == "TTT.TT....T"
    assert [(int(e.start_s), int(e.end_s), e.subject) for e in run.events] == [(2, 5, "person #1")]

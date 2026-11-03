"""State machine timing: idle -> candidate -> active -> closed at frame timestamps."""
import pytest

from safety_mode.engine import Detection, Engine, Frame

ZONES = {"frame_size": [100, 100], "floor_calibration": None,
         "zones": [{"name": "z", "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]}]}
IN = Detection("person", [0.4, 0.6, 0.5, 0.9])
OUT = Detection("person", [0.4, 0.1, 0.5, 0.3])


def rule(min_s):
    return {"rule_id": "rz", "title": "t", "objects": [{"name": "person", "prompt": "person"}],
            "zones": ["z"], "condition": {"in_zone": {"object": "person", "zone": "z"}},
            "min_duration_s": min_s, "severity": "medium",
            "explanation_template": "{person_count} for {duration_s} s at {time}",
            "unsupported": None, "suggested_rewrite": None}


def run(min_s, pattern, **kw):
    """pattern: string of 1/0 per second; returns (engine, list of (t, events))."""
    eng = Engine([rule(min_s)], ZONES, box_units="norm", **kw)
    period = kw.get("frame_period_s", 1.0)
    log = []
    for i, c in enumerate(pattern):
        log.append((i * period, eng.process(Frame(i * period, [IN if c == "1" else OUT]))))
    return eng, log


def test_window_not_reached_no_event():
    eng, log = run(3, "1100")
    assert all(not evs for _, evs in log) and eng.events == []


def test_event_starts_when_window_reached_and_closes_on_false():
    eng, log = run(3, "1110", max_gap_s=0)
    assert log[0][1] == [] and log[1][1] == []
    (ev,) = log[2][1]
    assert ev.start_s == 2 and ev.condition_since_s == 0 and ev.status == "active"
    (closed,) = log[3][1]
    assert closed.event_id == ev.event_id and closed.status == "closed" and closed.end_s == 2
    assert closed.explanation_inputs["duration_s"] == 3
    assert closed.explanation_inputs["time"] == "00:00"


def test_one_event_per_episode_with_updates():
    eng, log = run(1, "111101", max_gap_s=0)
    assert len(eng.events) == 2
    first, second = eng.events
    assert (first.start_s, first.end_s, first.status) == (0, 3, "closed")
    assert second.start_s == 5 and second.status == "active"
    assert [len(e) for _, e in log] == [1, 1, 1, 1, 1, 1]   # new, upd, upd, upd, closed, new


def test_default_gap_is_one_frame():
    eng, _ = run(1, "11011")            # default max_gap_s=1: one missing frame keeps the episode
    assert len(eng.events) == 1 and eng.events[0].end_s == 4
    eng, _ = run(1, "110011")
    assert len(eng.events) == 2


def test_gap_does_not_bridge_a_candidate():
    """Regression, warehouse-team r2-vest s19-s22: C on s19-s20, nobody s21, another person
    s22 must not add up to a 3 s window; the gap only keeps a raised event open."""
    eng, _ = run(3, "11011")
    assert eng.events == []
    eng, _ = run(3, "111011")
    (ev,) = eng.events
    assert (ev.condition_since_s, ev.start_s, ev.end_s) == (0, 2, 5)


def test_gap_tolerance_keeps_episode_open():
    eng, log = run(1, "11011", max_gap_s=1)
    assert len(eng.events) == 1 and eng.events[0].end_s == 4
    eng, log = run(1, "110011", max_gap_s=1)
    assert len(eng.events) == 2 and eng.events[0].end_s == 1


def test_finish_closes_active_event():
    eng, _ = run(2, "0111")
    (ev,) = eng.finish()
    assert ev.status == "closed" and ev.start_s == 2 and ev.end_s == 3 and ev.condition_since_s == 1


def test_two_fps_period():
    eng, log = run(1, "0110", frame_period_s=0.5)
    (ev,) = log[2][1]
    assert ev.start_s == pytest.approx(1.0) and ev.condition_since_s == pytest.approx(0.5)

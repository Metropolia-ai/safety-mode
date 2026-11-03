"""Offline tests of explain.py and report.py: fake client with canned JSON, no network."""
import dataclasses
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest

from safety_mode.explain import (ExplainError, build_facts, check_explanation, explain_event,
                                 explain_event_result, explain_event_template)
from safety_mode.facts import position_words
from safety_mode.llm import LLMConfig
from safety_mode.report import ReportError, ReportInputError, build_digest, check_report, daily_report, daily_report_result

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "safety_mode" / "samples"


def rule(n):
    return json.loads((ROOT / "rules" / f"r{n:02d}.json").read_text(encoding="utf-8"))


def report_input():
    return json.loads((SAMPLES / "report-input-2026-11-02.json").read_text(encoding="utf-8"))


def event_r3():
    return json.loads((SAMPLES / "event-r3.json").read_text(encoding="utf-8"))


def api_error(cls=openai.APIConnectionError, msg="connection refused"):
    err = cls.__new__(cls)  # built without an HTTP request object
    Exception.__init__(err, msg)
    return err


class FakeClient:
    """Returns canned replies in order (an Exception instance is raised); records every request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        content = self.replies.pop(0)
        if isinstance(content, Exception):
            raise content
        if not isinstance(content, str):
            content = json.dumps(content)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def cfg(**kw):
    return LLMConfig(api_key="fake", base_url="https://fake.example/v1", model="fake-model", **kw)


GOOD_EXPL = {"explanation": "A forklift came within 1.3 m of a person, the limit is 2 m, near the centre of the frame. "
                            "It lasted 2 s from 10:42:17.", "severity_check": "high"}

GOOD_MD = """**Safety report — Warehouse A, camera cam-3 — Mon 2026-11-02**

**Summary.** 7 events, 2 high, 5 medium. No critical events. Four of the five medium events are Rule 2 between 06:55 and 07:10. Camera uptime 23 h 48 min.

**High**
1. 10:42:17 — Rule 3 *Forklift within 2 m of a pedestrian*. A forklift came within 1.3 m of a person. Duration 2 s. Frame: `evt-0412.jpg`.
2. 14:05:03 — Rule 1 *Nobody in the yellow zone while the press runs*. One person in the yellow zone while the press was running. Duration 6 s. Frame: `evt-0418.jpg`.

**Medium**
3–6. 06:55:12–07:10:22 — Rule 2 *PPE past the gate*. Four people past the gate without a vest. Frames: `evt-0401.jpg`, `evt-0402.jpg`, `evt-0403.jpg`, `evt-0404.jpg`.
7. 16:20:40 — Rule 5 *Fire exit kept clear*. A pallet in the fire-exit zone for 14 min, until 16:35. Frame: `evt-0423.jpg`.

**Patterns and suggestions**
- Vest events cluster between 06:55 and 07:10. Consider a vest rack at the gate.

**Rule health.** Rule 6 had 0 events. Rule 9 was disabled at 11:00 after 3 false positives (stack partly outside the frame).
"""
GOOD_REPORT = {"markdown": GOOD_MD, "patterns": ["Vest events cluster between 06:55 and 07:10."],
               "suggestions": ["Consider a vest rack at the gate."]}


# ---------- explain ----------

def test_position_words_thirds_depth_and_zone():
    assert position_words([1100, 400, 1200, 700], (1280, 720)) == "on the right side of the frame, near the camera"
    assert position_words([0.0, 0.1, 0.1, 0.3]) == "on the left side of the frame, far from the camera"
    zone = {"name": "z", "title": "Fire exit", "polygon": [[0.4, 0.5], [0.6, 0.5], [0.6, 0.7], [0.4, 0.7]]}
    assert position_words([0.45, 0.3, 0.55, 0.6], zones=[zone]).endswith("inside Fire exit")


def test_facts_have_no_boxes_and_carry_limit():
    f = build_facts(event_r3(), rule(3))
    s = json.dumps(f)
    assert "412" not in s.replace("evt-0412", "")  # box pixels are turned into words
    assert f["measurements"] == {"min_floor_distance_m": 1.3}
    assert f["rule"]["condition"]["distance_lt"]["m"] == 2
    assert f["objects"][0]["position"]


def test_explain_valid_first_try():
    fc = FakeClient(GOOD_EXPL)
    res = explain_event_result(event_r3(), rule(3), client_config=cfg(), client=fc)
    assert res["source"] == "model" and res["explanation"] == GOOD_EXPL["explanation"]
    assert len(fc.calls) == 1
    kw = fc.calls[0]
    assert kw["response_format"]["json_schema"]["name"] == "alert_explanation"
    assert kw["response_format"]["json_schema"]["strict"] is True
    assert "Write in English" in kw["messages"][0]["content"]
    assert "Forklift within 2 m of a pedestrian" in kw["messages"][1]["content"]
    assert "image_url" not in json.dumps(kw["messages"])


def test_explain_language_goes_into_prompt():
    fc = FakeClient(GOOD_EXPL)
    explain_event(event_r3(), rule(3), "fi", client_config=cfg(), client=fc)
    assert "Write in Finnish" in fc.calls[0]["messages"][0]["content"]


def test_explain_schema_error_then_retry():
    fc = FakeClient({"explanation": "x", "severity_check": "severe"}, GOOD_EXPL)
    res = explain_event_result(event_r3(), rule(3), client_config=cfg(), client=fc)
    assert res["source"] == "model" and len(fc.calls) == 2
    assert "severity_check" in fc.calls[1]["messages"][-1]["content"]


def test_explain_invented_number_is_rejected():
    facts = build_facts(event_r3(), rule(3))
    assert check_explanation(GOOD_EXPL["explanation"], facts) == []
    errs = check_explanation("A forklift at 15 km/h came within 1.3 m of a person.", facts)
    assert errs and "15" in errs[0]
    assert check_explanation("Rule 3: forklift 1,3 m from a person at 10:42:17.", facts) == []


def test_explain_falls_back_to_template_after_two_bad_answers():
    bad = {"explanation": "Forklift 0.4 m from a worker carrying 3 boxes.", "severity_check": "high"}
    fc = FakeClient(bad, bad)
    res = explain_event_result(event_r3(), rule(3), client_config=cfg(), client=fc)
    assert res["source"] == "template" and len(fc.calls) == 2
    assert res["explanation"] == "Forklift came within 1.3 m of a pedestrian for 2 s, at 10:42:17."
    with pytest.raises(ExplainError):
        explain_event_result(event_r3(), rule(3), client_config=cfg(), client=FakeClient(bad, bad), fallback=False)


def test_explain_falls_back_when_api_is_down():
    fc = FakeClient(api_error())
    res = explain_event_result(event_r3(), rule(3), client_config=cfg(), client=fc)
    assert res["source"] == "template" and "APIConnectionError" in res["errors"][0]
    # no key at all: make_client raises, the template still answers
    res = explain_event_result(event_r3(), rule(3), client_config=LLMConfig(api_key=""))
    assert res["source"] == "template"


def test_template_engine_event_shape():
    ev = {"rule_id": "r2", "start_s": 4, "end_s": 6, "condition_since_s": 2, "severity": "medium",
          "objects": [{"name": "person", "index": 0, "box": [0.1, 0.4, 0.2, 0.9]}], "measurements": {},
          "explanation_inputs": {"time": "00:02", "duration_s": 5, "person_count": 1, "missing_ppe": "vest",
                                 "distance_m": None, "count": None, "height_m": None, "object": "person"}}
    assert explain_event_template(ev, rule(2)) == "1 person(s) on the warehouse floor without vest for 5 s, at 00:02."


def test_template_report_shape_without_inputs():
    ev = report_input()["events"][-1]  # r5 pallet, 840 s
    assert explain_event_template(ev, rule(5)) == "Pallet left in the fire-exit zone for 840 s, since 16:20:40."
    ev = copy.deepcopy(report_input()["events"][0])  # r2: no missing_ppe in report-input measurements
    assert explain_event_template(ev, rule(2)) == "1 person(s) on the warehouse floor without required PPE for 4 s, at 06:55:12."


# ---------- report ----------

def test_report_input_is_validated_before_any_call():
    bad = report_input()
    del bad["events"][0]["frame"]
    fc = FakeClient()
    with pytest.raises(ReportInputError):
        daily_report(bad, client_config=cfg(), client=fc)
    assert fc.calls == []


def test_digest_counts_groups_and_end_times():
    d = build_digest(report_input())
    assert d["totals"] == {"events": 7, "critical": 0, "high": 2, "medium": 5, "low": 0}
    assert d["high_and_critical_ids"] == ["evt-0412", "evt-0418"]
    g = {x["rule_id"]: x for x in d["groups"]}
    assert g["r2"]["ids"] == ["evt-0401", "evt-0402", "evt-0403", "evt-0404"]
    assert g["r5"]["last_end_time"] == "16:34:40"
    assert d["weekday"] == "Mon" and d["camera_uptime"] == "23 h 48 min"


def test_report_prompt_gets_every_event():
    fc = FakeClient(GOOD_REPORT)
    res = daily_report_result(report_input(), client_config=cfg(), client=fc)
    assert res["markdown"] == GOOD_MD and res["suggestions"] == GOOD_REPORT["suggestions"]
    kw = fc.calls[0]
    assert kw["response_format"]["json_schema"]["name"] == "daily_report"
    user = kw["messages"][1]["content"]
    for e in report_input()["events"]:
        assert e["id"] in user and e["frame"] in user and e["explanation"] in user
    system = kw["messages"][0]["content"]
    assert "must not" in system and "legal judgements" in system and "invent events" in system


@pytest.mark.parametrize("edit,needle", [
    (lambda m: m.replace("`evt-0418.jpg`", "`frame.jpg`"), "missing: ['evt-0418']"),
    (lambda m: m.replace("`evt-0423.jpg`", "`evt-0423.jpg`, `evt-0999.jpg`"), "evt-0999"),
    (lambda m: m.replace("Rule 6 had", "Rule 8 had"), "rule numbers [8]"),
    (lambda m: m.replace("until 16:35", "until 17:05"), "17:05"),
    (lambda m: m + "\nThis breaches safety regulations and the site is liable.", "liable"),
])
def test_report_guards(edit, needle):
    d = build_digest(report_input())
    assert check_report(GOOD_MD, d) == []
    errs = check_report(edit(GOOD_MD), d)
    assert any(needle in e for e in errs), errs


def test_report_retry_then_error():
    bad = dict(GOOD_REPORT, markdown=GOOD_MD.replace("evt-0412", "evt-0499"))
    fc = FakeClient(bad, GOOD_REPORT)
    assert daily_report(report_input(), client_config=cfg(), client=fc) == GOOD_MD
    assert len(fc.calls) == 2 and "evt-0499" in fc.calls[1]["messages"][-1]["content"]
    with pytest.raises(ReportError):
        daily_report(report_input(), client_config=cfg(), client=FakeClient(bad, {"markdown": 1}))


def test_facts_quote_closest_distance_of_the_episode():
    """Engine events keep the last second's distance_m and the episode minimum; the facts use the minimum."""
    from safety_mode.explain import build_facts
    ev = {"rule_id": "r3", "start_s": 1, "end_s": 6, "condition_since_s": 1, "severity": "high",
          "objects": [], "measurements": {"distance_m": 1.54, "min_distance_m": 0.26, "duration_s": 6},
          "explanation_inputs": {"time": "00:01", "duration_s": 6}}
    assert "0.26" in json.dumps(build_facts(ev, rule(3)))
    assert "1.54" not in json.dumps(build_facts(ev, rule(3)))


def test_positions_are_taken_at_alert_time_not_at_episode_end():
    # clip A: person #0 walks right to left; the alert goes up at s7 (right side), the episode ends at
    # s21 at the left edge. The explanation gives 00:07, so the position must be the one at s7.
    from safety_mode import pipeline
    from safety_mode.explain import build_facts
    from safety_mode.pipeline import DetectionSource, run_clip
    run = run_clip("man-crosses-floor", ["r02"], source=DetectionSource("offline", detector="video"), episodes="rule")
    (ev,) = run.events
    assert (ev.start_s, ev.end_s) == (7, 21)
    start_box, end_box = ev.objects_at_start[0]["box"], ev.objects[0]["box"]
    assert (start_box[0] + start_box[2]) / 2 > 2 / 3 and (end_box[0] + end_box[2]) / 2 < 1 / 3   # normalised boxes
    facts = build_facts(dataclasses.asdict(ev), run.rules["r2"], {"frame_size": [1280, 720], "zones": []})
    assert facts["objects"][0]["position"].startswith("on the right side")


def test_positions_fall_back_to_objects_without_start_snapshot():
    from safety_mode.explain import build_facts
    ev = {"rule_id": "r2", "start_s": 0, "end_s": 0, "objects": [{"name": "person", "box": [0, 0, 100, 700]}]}
    assert build_facts(ev, {"rule_id": "r2"}, {"frame_size": [1280, 720], "zones": []})["objects"][0]["position"].startswith("on the left side")

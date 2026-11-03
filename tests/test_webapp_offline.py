"""Demo page API on the committed caches (demo/cache/compile.json, detections/*.video.json), no network."""
import pytest
from fastapi.testclient import TestClient

from safety_mode import webapp
from safety_mode.webapp import EXAMPLES, JUDGE_CLIPS, app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("SM_LLM_API_KEY", raising=False)    # cached results only; a miss falls back to the template
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    return TestClient(app)


DEFAULT = {"man-crosses-floor": ("r2", [(7, 21, "person #0")]),
           "site-worker-no-vest": ("r2-vest", [(2, 5, "person #1")]),
           "forklift-crosses-aisle": ("r8", [(6, 7, "forklift #0")]),
           "warehouse-team": ("r3", [(1, 6, None), (10, 12, None), (15, 15, None), (18, 22, None)])}


def test_judge_clip_order():
    assert JUDGE_CLIPS == ["man-crosses-floor", "site-worker-no-vest", "forklift-crosses-aisle", "warehouse-team"]
    assert set(EXAMPLES) == {"r2", "r2-vest", "r3", "r8"}


@pytest.mark.parametrize("clip", JUDGE_CLIPS)
def test_default_example_runs_from_cache(client, clip):
    rid, want = DEFAULT[clip]
    j = client.post("/api/compile", json={"text": EXAMPLES[rid], "clip": clip, "rule_id": rid}).json()
    assert j["cached"], "example compile must be pre-cached (demo/precompute.py)"
    run = client.post("/api/run", json={"clip": clip, "rule": j["rule"], "episodes": "rule"}).json()
    assert run["detector"] == "video"
    got = [(int(e["start_s"]), int(e["end_s"]), e["subject"]) for e in run["events"]]
    assert [(a, b) for a, b, _ in got] == [(a, b) for a, b, _ in want]
    assert all(w is None or g == w for (_, _, g), (_, _, w) in zip(got, want))
    # the first alert's explanation comes from the committed Spark cache, not the template
    assert run["events"][0]["explanation_source"] == "model"


@pytest.mark.parametrize("clip", JUDGE_CLIPS + ["webcam"])
@pytest.mark.parametrize("rid", list(EXAMPLES))
def test_every_example_is_cached_with_this_cameras_zone_titles(client, clip, rid):
    j = client.post("/api/compile", json={"text": EXAMPLES[rid], "clip": clip, "rule_id": rid}).json()
    assert j["cached"]
    titles = {z["title"] for z in webapp.zones_for(clip)["zones"]}
    other = {"Shop floor", "Site floor", "Walkway crossing", "Warehouse floor"} - titles
    text = " ".join(str(j["rule"].get(k) or "") for k in ("title", "unsupported", "suggested_rewrite"))
    # refusals list the camera's zones as "name (Title)"; a result cached for another camera must not leak its titles
    assert not any(f"({t})" in text for t in other), text


def test_report_input_carries_the_engine_alert_start_and_end(client):
    # clip A, example 1: the engine raises the alert at s7 (true since s5) and it ends at s21
    rule = client.post("/api/compile", json={"text": EXAMPLES["r2"], "clip": "man-crosses-floor", "rule_id": "r2"}).json()["rule"]
    (ev,) = client.post("/api/run", json={"clip": "man-crosses-floor", "rule": rule, "episodes": "rule"}).json()["events"]
    assert (ev["condition_since_s"], ev["start_s"], ev["end_s"]) == (5, 7, 21)
    (rin,) = webapp.report_events([ev])
    assert (rin["time"], rin["end_time"]) == ("00:07", "00:21")
    from safety_mode.report import build_digest
    data = {"site": "s", "camera": "c", "date": "2026-10-30", "uptime_min": 1, "events": [rin],
            "rules": [{"id": "r2", "title": rule["title"], "events": 1, "false_positives_marked": 0, "enabled": True}],
            "manager_actions": []}
    from safety_mode.report import validate_input
    validate_input(data)
    (d,) = build_digest(data)["events"]
    assert (d["time"], d["end_time"]) == ("00:07", "00:21")


def test_zone_keys_never_reach_prose():
    from safety_mode.compiler import humanize_zone_names
    zones = [{"name": "past_gate", "title": "Shop floor"}, {"name": "yellow_zone", "title": "Yellow zone (parking spot)"}]
    r = humanize_zone_names({"title": "PPE on past_gate", "unsupported": "needs the crossing (marked: past_gate (Shop floor), "
                             "`yellow_zone` (Yellow zone (parking spot)))", "suggested_rewrite": None,
                             "zones": ["past_gate"]}, zones)
    assert r["title"] == "PPE on Shop floor" and r["zones"] == ["past_gate"]
    assert "past_gate" not in r["unsupported"] and "yellow_zone" not in r["unsupported"]
    assert "(marked: Shop floor," in r["unsupported"]

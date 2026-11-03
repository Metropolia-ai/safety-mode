"""Per-IP and daily limits, the operator override (SM_ACCESS_KEY) and live SAM 3.1 detections merged into
/api/run, no network."""
import pytest
from fastapi.testclient import TestClient

from safety_mode import webapp
from safety_mode.webapp import EXAMPLES, app

KEY = "test-operator-key-0123456789abcdef"
CLIP = "man-crosses-floor"     # 24 s, person #0 crosses the shop floor 5-21 s (video mode, track ids)

GLOVES = {"rule_id": "custom", "title": "Gloves on the shop floor", "severity": "medium", "min_duration_s": 3,
          "objects": [{"name": "person", "prompt": "person"}, {"name": "gloves", "prompt": "gloves"}],
          "zones": ["past_gate"],
          "condition": {"all": [{"in_zone": {"object": "person", "zone": "past_gate"}},
                                {"not": {"wearing": {"person": "person", "ppe": ["gloves"]}}}]},
          "explanation_template": "Person on the shop floor without {missing_ppe} for {duration_s} at {time}.",
          "unsupported": None, "suggested_rewrite": None}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("SM_LLM_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("SM_ACCESS_KEY", KEY)
    for v in ("SM_LIMIT_SAM_FRAMES_PER_HOUR", "SM_DAILY_SAM_FRAMES", "SM_LIMIT_SPARK_PER_HOUR", "SM_DAILY_SPARK_CALLS",
              "SM_LIMIT_SCOUT_PER_HOUR", "SM_DAILY_SCOUT_RUNS", "SM_LIMIT_SCOUT_STEPS_PER_HOUR", "SM_DAILY_SCOUT_STEPS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(webapp, "LIMITER", webapp.Limiter())
    return TestClient(app)


def person_box(sec):
    """The cached person #0 box at this second, normalised (where live gloves would sit)."""
    dj = webapp.detections_for(CLIP, "video")
    w, h = dj["frame_size"]
    recs = dj["frames"][f"{sec + 1:03d}"].get("person", [])
    return [[d["box"][0] / w, d["box"][1] / h, d["box"][2] / w, d["box"][3] / h] for d in recs]


def extra_gloves(worn_from=None):
    """Live 'gloves' for every second: none (empty list), or on person #0's hands from worn_from on."""
    out = {}
    for s in range(24):
        boxes = []
        if worn_from is not None and s >= worn_from:
            for b in person_box(s):
                hy = b[1] + (b[3] - b[1]) * 0.55
                boxes.append({"box": [b[0], hy, b[2], hy + (b[3] - b[1]) * 0.1], "score": 0.8})
        out[str(s)] = {"gloves": boxes}
    return out


JPEG = "/9j/AA=="      # starts with FF D8: passes the JPEG check, SAM is faked below


@pytest.fixture()
def fake_sam(monkeypatch):
    """/api/detect without the network: MODEL_API_KEY set, SamClient answers with no detections."""
    monkeypatch.setenv("MODEL_API_KEY", "x")

    class R:
        source_size = (640, 480)

        def detections(self, decode_masks=False):
            return []

    class Sam:
        calls = 1

        def __init__(self, **kw):
            pass

        def segment_image(self, data, prompt):
            return R()

    monkeypatch.setattr("safety_mode.detect.SamClient", Sam)


def detect(client, headers=None, ip="10.0.0.1"):
    return client.post("/api/detect", json={"jpeg": JPEG, "prompts": ["gloves"]},
                       headers={"X-Forwarded-For": ip, **(headers or {})})


def test_cached_examples_need_nothing(client):
    j = client.post("/api/compile", json={"text": EXAMPLES["r2"], "clip": CLIP, "rule_id": "r2"}).json()
    assert j["cached"] and j["status"]["live"] == []
    r = client.post("/api/run", json={"clip": CLIP, "rule": j["rule"], "episodes": "rule"})
    assert r.status_code == 200
    assert [(e["start_s"], e["end_s"]) for e in r.json()["events"]] == [(7, 21)]


def test_live_calls_are_open_without_a_key(client, fake_sam):
    r = detect(client)
    assert r.status_code == 200, r.text
    assert r.json()["detections"] == {"gloves": []}
    # no Spark configured in tests: past the limiter, 503, never 401
    for path, body in [("/api/compile", {"text": "Nobody may stand on the shop floor without gloves", "clip": CLIP}),
                       ("/api/report", {"events": [{"start_s": 1}], "rules": []}),
                       ("/api/scout/step", {"clip": CLIP})]:
        assert client.post(path, json=body).status_code == 503, path
    r = client.post("/api/run", json={"clip": CLIP, "rule": GLOVES, "episodes": "rule", "extra_detections": extra_gloves()})
    assert r.status_code == 200


def test_default_limits():
    L = webapp.Limiter.limits
    assert L("spark") == (30, 500) and L("sam") == (300, 2500)
    assert L("scout") == (2, 30) and L("scout_step")[0] == 40


def test_env_overrides_the_limits(monkeypatch):
    monkeypatch.setenv("SM_LIMIT_SAM_FRAMES_PER_HOUR", "7")
    monkeypatch.setenv("SM_DAILY_SPARK_CALLS", "9")
    assert webapp.Limiter.limits("sam")[0] == 7 and webapp.Limiter.limits("spark")[1] == 9


def test_hourly_limit_per_address_says_when_it_resets(client, fake_sam, monkeypatch):
    monkeypatch.setenv("SM_LIMIT_SAM_FRAMES_PER_HOUR", "3")
    for _ in range(3):
        assert detect(client).status_code == 200
    r = detect(client)
    assert r.status_code == 429
    msg = r.json()["detail"]
    assert msg.startswith("Hourly limit for this address reached (3 SAM 3.1 frames), try again in 60 minutes.")
    assert detect(client, ip="10.0.0.2").status_code == 200      # another address has its own hour


def test_reset_time_follows_the_oldest_hit(monkeypatch):
    monkeypatch.setenv("SM_LIMIT_SPARK_PER_HOUR", "2")
    lim = webapp.Limiter()
    now = [1_000_000.0]
    monkeypatch.setattr(webapp.time, "time", lambda: now[0])
    assert lim.take("a", "spark") is None
    now[0] += 20 * 60
    assert lim.take("a", "spark") is None
    now[0] += 5 * 60
    assert "try again in 35 minutes" in lim.take("a", "spark")   # the first hit expires 35 min from now
    now[0] += 35 * 60 + 1
    assert lim.take("a", "spark") is None


def test_daily_cap_applies_to_everyone(client, fake_sam, monkeypatch):
    monkeypatch.setenv("SM_DAILY_SAM_FRAMES", "2")
    assert detect(client, ip="10.0.0.1").status_code == 200
    assert detect(client, ip="10.0.0.2").status_code == 200
    for r in (detect(client, ip="10.0.0.3"), detect(client, headers={"X-Access-Key": KEY})):
        assert r.status_code == 429
        assert r.json()["detail"].startswith("The demo's daily limit of SAM 3.1 frames is used up, it resets in ")


def test_operator_override_skips_the_per_ip_limit(client, fake_sam, monkeypatch):
    monkeypatch.setenv("SM_LIMIT_SAM_FRAMES_PER_HOUR", "1")
    assert detect(client).status_code == 200
    assert detect(client).status_code == 429
    assert detect(client, headers={"X-Access-Key": "wrong"}).status_code == 429
    assert detect(client, headers={"X-Access-Key": KEY}).status_code == 200


def test_no_override_when_the_key_is_unset(client, fake_sam, monkeypatch):
    monkeypatch.delenv("SM_ACCESS_KEY")
    monkeypatch.setenv("SM_LIMIT_SAM_FRAMES_PER_HOUR", "1")
    assert detect(client).status_code == 200
    assert detect(client, headers={"X-Access-Key": ""}).status_code == 429
    assert detect(client, headers={"X-Access-Key": "anything"}).status_code == 429
    assert "exempt" not in client.get("/api/health?key=").json()


def test_health_shows_limits_to_everyone(client, fake_sam):
    j = client.get("/api/health", headers={"X-Forwarded-For": "10.0.0.1"}).json()
    lim = j["limits"]
    assert lim["sam_frames_per_hour_per_ip"] == 300 and lim["sam_frames_per_day"] == 2500
    assert lim["spark_calls_per_hour_per_ip"] == 30 and lim["spark_calls_per_day"] == 500
    assert lim["scout_runs_per_hour_per_ip"] == 2 and lim["scout_runs_per_day"] == 30
    assert lim["scout_steps_per_hour_per_ip"] == 40
    assert j["remaining"]["sam_frames"] == 300 and j["remaining_today"]["sam_frames"] == 2500
    assert "exempt" not in j and not any("key" in k and k not in ("spark_key", "sam_key") for k in j)
    detect(client)
    j = client.get("/api/health", headers={"X-Forwarded-For": "10.0.0.1"}).json()
    assert j["remaining"]["sam_frames"] == 299 and j["remaining_today"]["sam_frames"] == 2499
    assert client.get("/api/health", headers={"X-Access-Key": KEY}).json()["exempt"] is True
    assert client.get(f"/api/health?key={KEY}").json()["exempt"] is True
    assert "exempt" not in client.get("/api/health?key=wrong").json()
    assert KEY not in client.get(f"/api/health?key={KEY}").text


def test_only_safe_synonyms_map_to_cached_prompts():
    avail = ["forklift", "hard hat", "person", "safety vest"]
    assert webapp.resolve_prompt("helmet", avail) == "hard hat"
    assert webapp.resolve_prompt("vest", avail) == "safety vest"
    assert webapp.resolve_prompt("high-visibility vest", avail) == "safety vest"
    assert webapp.resolve_prompt("worker", avail) == "person"
    assert webapp.resolve_prompt("Workers", avail) == "person"
    for p in ("hat", "vehicle", "pallet truck", "man", "operator", "gloves", "pallet"):
        assert webapp.resolve_prompt(p, avail) is None, p


def test_status_lists_live_prompts_and_clip_seconds():
    st = webapp.choose_detector(GLOVES, CLIP)[1]
    assert st["live"] == ["gloves"] and st["prompts"]["person"] == "person"
    assert st["seconds"] == 24 and st["frame_size"] == [1280, 720] and st["detector"] == "video"


def test_run_needs_live_detections_for_every_second(client):
    extra = extra_gloves()
    del extra["5"]
    r = client.post("/api/run", json={"clip": CLIP, "rule": GLOVES, "episodes": "rule", "extra_detections": extra})
    assert r.status_code == 409 and "1 of 24" in r.json()["detail"]
    r = client.post("/api/run", json={"clip": CLIP, "rule": GLOVES, "episodes": "rule",
                                      "extra_detections": {"0": {"gloves": [{"box": ["a", 0, 1, 1]}]}}})
    assert r.status_code == 400


@pytest.mark.parametrize("episodes", ["rule", "object"])
def test_live_gloves_merge_with_cached_tracked_people(client, episodes):
    # no gloves anywhere: the person on the floor breaks the rule like in example 1 (alert 7-21 s)
    r = client.post("/api/run", json={"clip": CLIP, "rule": GLOVES, "episodes": episodes,
                                      "extra_detections": extra_gloves()})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["live_prompts"] == ["gloves"] and j["detector"] == "video"
    assert [(e["start_s"], e["end_s"]) for e in j["events"]] == [(7, 21)]
    # explanations: no Spark in tests -> template, the run still works
    assert j["events"][0]["explanation"]
    # gloves put on from 12 s: the alert ends at 11 s (gap of 1 s allowed)
    j2 = client.post("/api/run", json={"clip": CLIP, "rule": GLOVES, "episodes": episodes,
                                       "extra_detections": extra_gloves(worn_from=12)}).json()
    assert [(e["start_s"], e["end_s"]) for e in j2["events"]] == [(7, 11)]
    shown = [d for d in j2["seconds"][15]["dets"] if d["prompt"] == "gloves"]
    assert shown and all(d["live"] and d["id"] is None for d in shown)
    person = [d for d in j2["seconds"][15]["dets"] if d["prompt"] == "person"]
    assert person and person[0]["id"] == 0          # cached prompts keep their track ids


def test_explanations_over_the_spark_limit_fall_back_to_template(client, monkeypatch):
    monkeypatch.setenv("SM_LLM_API_KEY", "x")         # Spark "configured": the limit decides
    monkeypatch.setenv("SM_LIMIT_SPARK_PER_HOUR", "0")
    called = []
    monkeypatch.setattr("safety_mode.explain.explain_event_result", lambda *a, **k: called.append(1) or {})
    rule = dict(GLOVES, objects=[{"name": "person", "prompt": "person"}, {"name": "gloves", "prompt": "hard hat"}])
    r = client.post("/api/run", json={"clip": CLIP, "rule": rule, "episodes": "rule"})
    assert r.status_code == 200 and not called
    ev = r.json()["events"][0]
    assert ev["explanation_source"] in ("template", "model")
    if ev["explanation_source"] == "template":
        assert ev["explanation_note"].startswith("Hourly limit for this address reached")
    r = client.post("/api/explain", json={"event": {"start_s": 3, "rule_id": "x", "measurements": {"duration_s": 3}},
                                          "rule": rule, "clip": CLIP})
    assert r.status_code == 429 and "try again in" in r.json()["detail"]

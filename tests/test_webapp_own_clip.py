"""Own clip (decoded in the browser, zones drawn there) and zones drawn on the webcam: /api/compile, /api/run,
/api/step, /api/explain take camera {zones, frame_size, seconds}. No network: rules are written by hand."""
import pytest
from fastapi.testclient import TestClient

from safety_mode import webapp
from safety_mode.webapp import app

ZONE = {"name": "keyboard_zone", "title": "Keyboard zone", "polygon": [[0.2, 0.5], [0.8, 0.5], [0.8, 1.0], [0.2, 1.0]]}
CAM = {"zones": [ZONE], "frame_size": [1280, 720], "seconds": 6}
CUP_RULE = {
    "rule_id": "custom-1", "title": "No cup in Keyboard zone", "severity": "medium", "min_duration_s": 2,
    "objects": [{"name": "cup", "prompt": "cup"}], "zones": ["keyboard_zone"],
    "condition": {"in_zone": {"object": "cup", "zone": "keyboard_zone"}},
    "explanation_template": "A cup stood in the Keyboard zone for {duration_s} at {time}.",
    "unsupported": None, "suggested_rewrite": None,
}
CUP_IN = {"box": [0.4, 0.6, 0.5, 0.8], "score": 0.9}      # foot point (0.45, 0.8): inside the zone
CUP_OUT = {"box": [0.85, 0.3, 0.95, 0.45], "score": 0.9}  # foot point (0.9, 0.45): outside


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("SM_LLM_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("SM_ACCESS_KEY", raising=False)
    return TestClient(app)


def extra(per_second):
    return {str(s): {"cup": dets} for s, dets in enumerate(per_second)}


def run(client, cam=CAM, rule=CUP_RULE, ex=None):
    body = {"clip": "own", "rule": rule, "episodes": "rule", "camera": cam}
    if ex is not None:
        body["extra_detections"] = ex
    return client.post("/api/run", json=body)


def test_own_clip_runs_on_live_boxes_only(client):
    r = run(client, ex=extra([[], [CUP_IN], [CUP_IN], [CUP_IN], [CUP_OUT], []]))
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["title"] == "Your clip" and j["duration_s"] == 6 and j["detector"] == "image"
    assert j["status"]["live"] == ["cup"] and j["status"]["seconds"] == 6
    assert j["timeline"] == ".TTT.."
    (ev,) = j["events"]
    assert (ev["condition_since_s"], ev["start_s"], ev["end_s"]) == (1, 2, 3)
    # overlay boxes come back normalised, marked live, without track ids
    (d,) = j["seconds"][1]["dets"]
    assert d["live"] and d["id"] is None and d["box"] == pytest.approx(CUP_IN["box"], abs=1e-3)
    # no Spark on this server: the template, which names the user's zone title
    assert "Keyboard zone" in ev["explanation"] and "keyboard_zone" not in ev["explanation"]


def test_own_clip_needs_zones_and_every_second(client):
    assert run(client, ex=extra([[]] * 5)).status_code == 409             # second 5 missing
    r = run(client, cam={"frame_size": [1280, 720], "seconds": 6, "zones": []}, ex=extra([[]] * 6))
    assert r.status_code == 400 and "zone" in r.json()["detail"]
    assert run(client, cam={**CAM, "seconds": 31}, ex=extra([[]] * 31)).status_code == 400


@pytest.mark.parametrize("zone, why", [
    ({**ZONE, "name": "Keyboard Zone"}, "lowercase"),
    ({**ZONE, "title": "  "}, "name"),
    ({**ZONE, "polygon": [[0, 0], [1, 1]]}, "3 to"),
    ({**ZONE, "polygon": [[0, 0], [1, "x"], [1, 1]]}, "bad point"),
])
def test_drawn_zones_are_validated(client, zone, why):
    r = run(client, cam={**CAM, "zones": [zone]}, ex=extra([[]] * 6))
    assert r.status_code == 400 and why in r.json()["detail"]


def test_two_zones_with_one_name_are_refused(client):
    r = run(client, cam={**CAM, "zones": [ZONE, ZONE]}, ex=extra([[]] * 6))
    assert r.status_code == 400 and "same name" in r.json()["detail"]


def test_points_are_clamped_to_the_frame():
    cam = webapp.camera_from_body("own", {"camera": {**CAM, "zones": [{**ZONE, "polygon": [[-0.2, 0], [1.3, 0], [0.5, 1]]}]}})
    assert cam["zones"][0]["polygon"] == [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
    assert cam["floor_calibration"] is None


def test_own_clip_status_has_no_calibration_and_no_machine_state():
    cam = webapp.camera_from_body("own", {"camera": CAM})
    dist = {**CUP_RULE, "objects": [{"name": "cup", "prompt": "cup"}, {"name": "person", "prompt": "person"}], "zones": [],
            "condition": {"distance_lt": {"a": "cup", "b": "person", "m": 1}}}
    st = webapp.rule_status(dist, "own", cam=cam)
    assert "floor calibration" in st["not_evaluable"] and st["live"] == ["cup", "person"]
    mach = {**CUP_RULE, "zones": [], "objects": [{"name": "forklift", "prompt": "forklift"}],
            "condition": {"machine_state": {"machine": "forklift", "state": "running"}}}
    assert "machine state" in webapp.rule_status(mach, "own", cam=cam)["not_evaluable"]


def test_compile_key_depends_on_the_drawn_titles():
    a = webapp.camera_from_body("own", {"camera": CAM})
    b = webapp.camera_from_body("own", {"camera": {**CAM, "zones": [{**ZONE, "title": "Desk edge"}]}})
    assert webapp.compile_key("No cup in the keyboard zone.", "own", a) != webapp.compile_key("No cup in the keyboard zone.", "own", b)
    # the sample clips keep their old keys (committed cache)
    assert webapp.compile_key("x", "man-crosses-floor") == webapp.compile_key("x", "man-crosses-floor", None)


def test_compile_on_own_clip_without_zones_is_refused(client):
    r = client.post("/api/compile", json={"text": "No cup in the keyboard zone.", "clip": "own"})
    assert r.status_code == 400 and "zone" in r.json()["detail"]


def test_compile_on_own_clip_uses_the_drawn_zones(client, monkeypatch):
    seen = {}

    def fake_compile(text, zones, rule_id="r1", **kw):
        seen["zones"] = zones
        return {**CUP_RULE, "rule_id": rule_id, "title": "No cup in keyboard_zone (Keyboard zone)"}
    monkeypatch.setattr("safety_mode.compiler.compile_rule", fake_compile)
    monkeypatch.setattr(webapp, "spark_available", lambda: True)
    monkeypatch.setattr(webapp.COMPILE_CACHE, "put", lambda k, v: None)
    j = client.post("/api/compile", json={"text": "No cup in the keyboard zone, unique 1.", "clip": "own", "camera": CAM}).json()
    assert [z["name"] for z in seen["zones"]] == ["keyboard_zone"] and seen["zones"][0]["title"] == "Keyboard zone"
    assert j["status"]["live"] == ["cup"] and j["status"]["seconds"] == 6 and not j["status"]["rejected"]


def test_webcam_zone_is_added_to_the_whole_frame(client):
    cam = webapp.camera_from_body("webcam", {"camera": {"zones": [ZONE]}})
    assert [z["name"] for z in cam["zones"]] == ["past_gate", "keyboard_zone"]
    assert webapp.camera_from_body("webcam", {}) is None
    frames = [{"t": 2 * i, "detections": {"cup": [d] if d else []}} for i, d in enumerate([CUP_OUT, CUP_IN, CUP_IN, None])]
    j = client.post("/api/step", json={"rule": CUP_RULE, "frame_period_s": 2, "frames_so_far": frames,
                                       "camera": {"zones": [ZONE]}}).json()
    assert j["timeline"] == ".TT." and not j["status"]["rejected"]
    assert j["events"] and j["events"][0]["start_s"] == 2
    # without the drawn zone the webcam does not know it
    j = client.post("/api/step", json={"rule": CUP_RULE, "frame_period_s": 2, "frames_so_far": frames}).json()
    assert j["status"]["rejected"]


@pytest.mark.parametrize("k", list(webapp.DESK_RULES))
def test_desk_chips_are_cached_and_on_the_page(client, k):
    d = webapp.DESK_RULES[k]
    page = (webapp.REPO / "web" / "index.html").read_text(encoding="utf-8")
    assert d["text"] in page
    body = {"text": d["text"], "clip": "webcam", "rule_id": "custom-1"}
    if d.get("zone"):
        body["camera"] = {"zones": [{"name": webapp.DESK_ZONE_KEY[d["zone"]], "title": d["zone"],
                                     "polygon": [[0.3, 0.6], [0.8, 0.6], [0.8, 0.95], [0.3, 0.95]]}]}
    j = client.post("/api/compile", json=body).json()     # a cache hit
    assert j["cached"] and j["rule"]["min_duration_s"] == 2
    assert sorted(o["prompt"] for o in j["rule"]["objects"]) == sorted(d["prompts"])
    assert not j["status"]["rejected"] and not j["status"]["not_evaluable"]

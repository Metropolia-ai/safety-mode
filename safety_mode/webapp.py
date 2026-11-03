"""Live demo page backend: FastAPI, stateless, made for Vercel serverless.

The browser holds the session (detections so far, events, alerts); every endpoint recomputes
from what the request carries. Clips run on the committed SAM 3.1 detections
(test-footage/detections/<clip>.json, no API cost); a prompt with no cached detections (status.live)
is run live: the page posts the clip's 1 fps frames to /api/detect and sends the boxes back to
/api/run as extra_detections. The webcam sends one JPEG every 2 s.

    POST /api/compile {text, clip, rule_id?, camera?}   rule.json via Muse Spark (cached for the example rules)
    POST /api/run     {clip, rule, extra_detections?, camera?}   events + per-second detections (cached JSON + live)

clip "own" is the user's own video: it never leaves the browser; the page sends the zones drawn on it as
camera {zones: [{name, title, polygon}], frame_size, seconds} and SAM 3.1 boxes for every second
(extra_detections, all prompts live). The webcam takes the same camera.zones (added to its whole-frame zone).
    POST /api/detect  {jpeg, prompts[]}           SAM 3.1 on one frame (webcam, or a clip frame for a live prompt)
    POST /api/step    {rule, frames_so_far, frame_period_s?}   engine over the webcam frames so far
    POST /api/explain {event, rule, clip}         one paragraph (Muse Spark, template fallback)
    POST /api/report  {events, rules, clip}       session report, markdown (Muse Spark)
    POST /api/scout/step {clip} | {state, sig}    Rule Scout agent, one step per request (safety_mode/scout.py)
    GET  /api/health

Local: SM_STATIC_DIR=<build dir> uvicorn safety_mode.webapp:app  (serves the page too).
Spend guards: every live feature is open to everyone. Calls that spend Meta API money count against
per-IP hourly limits and a daily cap per process (best effort: Vercel may run several instances); cached
answers never count. Over a limit the API answers 429 with a sentence that says when it resets. A request
carrying SM_ACCESS_KEY (operator override) skips the per-IP limits, not the daily cap. Webcam session
length is enforced by the page and by the per-IP frame limit.
"""
from __future__ import annotations

import base64
import binascii
import dataclasses
import datetime as dt
import hashlib
import hmac
import json
import math
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .engine import Detection, Engine, Frame, Truth, check_rule
from .pipeline import DEFAULT_DETECTOR, DETS_DIR, ZONES_DIR, parse_machine_spec

log = logging.getLogger("safety_mode.webapp")

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "demo" / "cache"           # committed: compile and explanation results
TMP_CACHE = Path(os.environ.get("SM_TMP_CACHE", "/tmp/sm-cache"))
WRITE_REPO_CACHE = os.environ.get("SM_CACHE_WRITE") == "1"   # only when precomputing locally

if not os.environ.get("SM_LLM_API_KEY") and os.environ.get("MODEL_API_KEY"):
    os.environ["SM_LLM_API_KEY"] = os.environ["MODEL_API_KEY"]   # one Meta key for Spark and SAM

# ------------------------------------------------------------------ clips

CLIPS = {
    # sample clips (page selector, in this order); added 30.10, test-footage/candidates/REPORT-2026-10-30.md
    "man-crosses-floor": {"file": "pexels-38707991-man-crosses-floor", "title": "Worker crosses the floor",
                          "machine": "running", "machine_note": "running for the whole clip"},
    "site-worker-no-vest": {"file": "pexels-32245123-site-worker-no-vest", "title": "Site: vest check",
                            "machine": "running", "machine_note": "running for the whole clip"},
    "forklift-crosses-aisle": {"file": "pexels-31751344-forklift-crosses-aisle", "title": "Forklift crosses the walkway",
                               "machine": "running", "machine_note": "running for the whole clip"},
    "warehouse-team": {"file": "pexels-15170997-warehouse-team", "title": "Team at the racks",
                       "machine": "running", "machine_note": "running for the whole clip"},
    # out of the selector since 30.10, still reachable with ?clip=
    "forklift-operator": {"file": "pexels-32243653-forklift-operator", "title": "Operator close-up",
                          "machine": "running", "machine_note": "running for the whole clip"},
    "busy-warehouse": {"file": "pexels-6079421-forklift-busy-warehouse", "title": "Forklift aisle",
                       "machine": "running:0-8", "machine_note": "running from 0 to 8 s, stopped after"},
}
JUDGE_CLIPS = ["man-crosses-floor", "site-worker-no-vest", "forklift-crosses-aisle", "warehouse-team"]
OWN = "own"                      # the user's own clip: decoded in the browser, zones drawn there
OWN_TITLE = "Your clip"
OWN_MAX_SECONDS = 30             # the page samples the first 30 s at 1 fps
MAX_DRAWN_ZONES = 8
MAX_ZONE_POINTS = 64
ZONE_KEY = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
WEBCAM_ZONES = {
    "clip": "webcam", "frame_size": [1280, 720], "floor_calibration": None,
    "zones": [{"name": "past_gate", "title": "Warehouse floor", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
               "note": "The floor area the camera sees: the whole webcam frame."}],
}
EXAMPLES = {
    "r2": "Everyone on the warehouse floor must wear a hard hat and a high-visibility vest.",
    "r2-vest": "Everyone on the floor must wear a high-visibility vest.",
    "r3": "A forklift must not come closer than 2 m to a pedestrian.",
    "r8": "No forklift in the walkway crossing.",
}
# "Try at your desk" chips on the webcam (web/index.html DESK): rules a visitor can act out at a laptop.
# Compiled once into demo/cache/compile.json (demo/precompute.py --desk); the cup rule needs the zone the page
# asks the user to draw, titled "Keyboard zone". Checked on Pexels desk clips (test-footage/README.md).
DESK_RULES = {
    "glasses": {"text": "Everyone in view must wear glasses, alert after 2 seconds.", "prompts": ["person", "glasses"]},
    "headphones": {"text": "Everyone in view must wear headphones, alert after 2 seconds.", "prompts": ["person", "headphones"]},
    "cup": {"text": "No cup in the keyboard zone for more than 2 seconds.", "prompts": ["cup"], "zone": "Keyboard zone"},
}
DESK_ZONE_KEY = {"Keyboard zone": "keyboard_zone"}
# SAM prompts the clips have no cached detections for, mapped to a cached prompt that means the same thing.
# Only safe synonyms (the cached prompt finds exactly these objects); any other prompt runs SAM 3.1 live.
ALIASES = {
    "hard hat": ["helmet", "hardhat", "safety helmet", "hard hats", "construction helmet", "safety hard hat",
                 "protective helmet"],
    "safety vest": ["vest", "hi-vis vest", "hi vis vest", "high vis vest", "reflective vest", "safety vests",
                    "high visibility vest", "high-visibility vest", "hi-viz vest", "high-vis vest", "vests"],
    "person": ["worker", "workers", "pedestrian", "pedestrians", "people", "human", "humans", "persons",
               "employee", "employees"],
    "forklift": ["lift truck", "forklift truck", "fork lift", "forklifts"],
}

_json_cache: dict = {}


def _load_json(path: Path):
    if path not in _json_cache:
        _json_cache[path] = json.loads(path.read_text(encoding="utf-8"))
    return _json_cache[path]


def clip_info(clip: str) -> dict:
    if clip not in CLIPS:
        raise HTTPException(400, f"unknown clip {clip!r}; known: {', '.join(CLIPS)}, webcam")
    return CLIPS[clip]


def zones_for(clip: str, cam: dict | None = None) -> dict:
    """Zones file of a camera. cam = zones the user drew (camera_from_body) for the own clip or the webcam."""
    if cam is not None:
        return cam
    if clip == "webcam":
        return WEBCAM_ZONES
    if clip == OWN:
        raise HTTPException(400, "Draw at least one zone on your clip first, or press \u201cWhole frame\u201d.")
    return _load_json(ZONES_DIR / f"{clip_info(clip)['file']}.json")


def _drawn_zone(z, i: int) -> dict:
    if not isinstance(z, dict):
        raise HTTPException(400, f"zone {i + 1}: not an object")
    name, title, poly = z.get("name"), z.get("title"), z.get("polygon")
    if not isinstance(name, str) or not ZONE_KEY.match(name):
        raise HTTPException(400, f"zone {i + 1}: name must be lowercase letters, digits and _ (got {str(name)[:40]!r})")
    title = " ".join(str(title or "").split())[:60]
    if not title:
        raise HTTPException(400, f"zone {i + 1}: give the zone a name")
    if not isinstance(poly, list) or not 3 <= len(poly) <= MAX_ZONE_POINTS:
        raise HTTPException(400, f"zone \u201c{title}\u201d: a zone needs 3 to {MAX_ZONE_POINTS} points")
    pts = []
    for p in poly:
        if not (isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)):
            raise HTTPException(400, f"zone \u201c{title}\u201d: bad point {str(p)[:40]}")
        pts.append([round(min(1.0, max(0.0, float(v))), 4) for v in p])
    return {"name": name, "title": title, "polygon": pts, "note": "Drawn by the user on this page."}


def camera_from_body(clip: str, body: dict) -> dict | None:
    """Zones the user drew in the page -> a zones-file dict (no floor calibration: nothing to measure metres
    from). Own clip: required, with frame_size and seconds. Webcam: optional, added to the whole-frame zone."""
    if clip not in (OWN, "webcam"):
        return None
    cam = body.get("camera")
    drawn = cam.get("zones") if isinstance(cam, dict) else None
    if not drawn:
        if clip == OWN:
            raise HTTPException(400, "Draw at least one zone on your clip first, or press \u201cWhole frame\u201d.")
        return None
    if not isinstance(drawn, list) or len(drawn) > MAX_DRAWN_ZONES:
        raise HTTPException(400, f"Up to {MAX_DRAWN_ZONES} zones.")
    zones = [_drawn_zone(z, i) for i, z in enumerate(drawn)]
    base = list(WEBCAM_ZONES["zones"]) if clip == "webcam" else []
    names = [z["name"] for z in base + zones]
    if len(set(names)) != len(names):
        raise HTTPException(400, "Two zones have the same name. Rename one of them.")
    fs = cam.get("frame_size")
    if not (isinstance(fs, list) and len(fs) == 2 and all(isinstance(v, int) and 16 <= v <= 8192 for v in fs)):
        if clip == OWN:
            raise HTTPException(400, "camera.frame_size must be [width, height] in pixels")
        fs = list(WEBCAM_ZONES["frame_size"])
    out = {"clip": clip, "frame_size": list(fs), "floor_calibration": None, "zones": base + zones, "drawn": True}
    if clip == OWN:
        n = cam.get("seconds")
        if not (isinstance(n, int) and 1 <= n <= OWN_MAX_SECONDS):
            raise HTTPException(400, f"camera.seconds must be 1 to {OWN_MAX_SECONDS}")
        out["seconds"] = n
    return out


def detections_for(clip: str, detector: str = "image") -> dict:
    """Committed SAM 3.1 detections: <clip>.json (image mode, no ids) or <clip>.video.json
    (video mode, records carry track_id)."""
    f = clip_info(clip)["file"]
    video, image = DETS_DIR / f"{f}.video.json", DETS_DIR / f"{f}.json"
    if (detector == "video" or not image.exists()) and video.exists():   # clips added 30.10: video mode only
        return _load_json(video)
    return _load_json(image)


def resolve_prompt(prompt: str, available) -> str | None:
    p = " ".join(prompt.lower().replace("_", " ").split())
    avail = {a.lower(): a for a in available}
    if p in avail:
        return avail[p]
    if p.endswith("s") and p[:-1] in avail:
        return avail[p[:-1]]
    for target, words in ALIASES.items():
        if (p in words or p == target) and target in avail:
            return avail[target]
    return None


def machine_names(cond) -> set:
    out = set()
    if not cond:
        return out
    (k, v), = cond.items()
    if k in ("all", "any"):
        for c in v:
            out |= machine_names(c)
    elif k == "not":
        out |= machine_names(v)
    elif k == "machine_state":
        out.add(v["machine"])
    return out


def rule_status(rule: dict, clip: str, detector: str = "image", cam: dict | None = None) -> dict:
    """What this camera can do with the rule: rejected / not evaluable reasons, prompt mapping, machine note."""
    zf = zones_for(clip, cam)
    zone_names = [z["name"] for z in zf["zones"]]
    has_cal = bool(zf.get("floor_calibration"))
    try:
        rejected, ne = check_rule(rule, zone_names, has_cal)
    except Exception as e:  # malformed rule from the client
        rejected, ne = f"invalid rule: {e}", None
    st = {"rejected": rejected, "not_evaluable": ne, "prompts": {}, "machine": None, "live": []}
    if clip == OWN:
        # nothing is cached for the user's clip: every prompt runs live on its 1 fps frames
        st["prompts"] = {o["prompt"]: None for o in rule.get("objects", [])}
        st["live"] = list(st["prompts"])
        st["detector"] = "image"
        st["seconds"] = zf["seconds"]
        st["frame_size"] = list(zf["frame_size"])
        if machine_names(rule.get("condition")):
            st["not_evaluable"] = st["not_evaluable"] or "your clip has no machine state, so a machine_state " \
                                                         "condition cannot be checked"
    elif clip != "webcam":
        dj = detections_for(clip, detector)
        avail = dj["prompts"]
        st["prompts"] = {o["prompt"]: resolve_prompt(o["prompt"], avail) for o in rule.get("objects", [])}
        # no cached detections: SAM 3.1 runs live on the clip's 1 fps frames (the page does it)
        st["live"] = [p for p, v in st["prompts"].items() if v is None]
        st["detector"] = "video" if dj.get("mode") == "video" else "image"
        st["seconds"] = len(dj["frames"])
        st["frame_size"] = list(dj["frame_size"])
        ms = machine_names(rule.get("condition"))
        if ms:
            st["machine"] = f"Machine state is set by hand in this demo: {', '.join(sorted(ms))} " \
                            f"{clip_info(clip)['machine_note']}."
    elif machine_names(rule.get("condition")):
        st["not_evaluable"] = st["not_evaluable"] or "the webcam has no machine state, so a machine_state " \
                                                     "condition cannot be checked"
    return st


def choose_detector(rule: dict, clip: str, det: str | None = None, cam: dict | None = None) -> tuple[str, dict]:
    """Video mode (track ids) by default; image mode if the video cache lacks more of the rule's prompts."""
    if clip == OWN:
        return "image", rule_status(rule, clip, "image", cam)
    det = det if det in ("video", "image") else DEFAULT_DETECTOR
    status = rule_status(rule, clip, det)
    if clip != "webcam" and det == "video" and status["live"]:
        img = rule_status(rule, clip, "image")
        if len(img["live"]) < len(status["live"]):
            return "image", img
    return det, status


# ------------------------------------------------------------------ operator override

def operator_exempt(request: Request, allow_query: bool = False) -> bool:
    """True when SM_ACCESS_KEY is set and the request carries it: the per-IP limits do not apply."""
    want = os.environ.get("SM_ACCESS_KEY", "")
    if not want:
        return False
    got = request.headers.get("x-access-key") or ""
    if not got and allow_query:
        got = request.query_params.get("key") or ""
    return hmac.compare_digest(got.strip().encode(), want.encode())


# ------------------------------------------------------------------ spend guards

class Limiter:
    """Per-IP sliding hour + per-process daily cap. In memory, best effort."""

    def __init__(self):
        self.lock = threading.Lock()
        self.hits: dict = defaultdict(deque)
        self.daily: dict = defaultdict(int)

    @staticmethod
    def limits(kind: str) -> tuple[int, int]:
        """(per IP per hour, per instance per day). The only protection of the live features."""
        if kind == "sam":
            return (int(os.environ.get("SM_LIMIT_SAM_FRAMES_PER_HOUR", "300")),
                    int(os.environ.get("SM_DAILY_SAM_FRAMES", "2500")))
        if kind == "scout":      # live Rule Scout runs
            return (int(os.environ.get("SM_LIMIT_SCOUT_PER_HOUR", "2")),
                    int(os.environ.get("SM_DAILY_SCOUT_RUNS", "30")))
        if kind == "scout_step":  # requests inside those runs (signed state can be replayed; this caps it)
            return (int(os.environ.get("SM_LIMIT_SCOUT_STEPS_PER_HOUR", "40")),
                    int(os.environ.get("SM_DAILY_SCOUT_STEPS", "600")))
        return (int(os.environ.get("SM_LIMIT_SPARK_PER_HOUR", "30")),
                int(os.environ.get("SM_DAILY_SPARK_CALLS", "500")))

    def take(self, ip: str, kind: str, n: int = 1, exempt: bool = False) -> str | None:
        """Reserve n units. Returns the 429 sentence or None. exempt skips the per-IP limit."""
        per_hour, per_day = self.limits(kind)
        now = time.time()
        day = (dt.date.today().isoformat(), kind)
        with self.lock:
            q = self.hits[(ip, kind)]
            while q and q[0] < now - 3600:
                q.popleft()
            if self.daily[day] + n > per_day:
                return (f"The demo's daily limit of {kind_name(kind)} is used up, it resets in "
                        f"{_in(_to_midnight(now))}. The sample clips with cached results still work.")
            if not exempt and len(q) + n > per_hour:
                over = len(q) + n - per_hour      # this many of the oldest hits must expire first
                wait = q[min(over, len(q)) - 1] + 3600 - now if q else 3600
                return (f"Hourly limit for this address reached ({per_hour} {kind_name(kind)}), "
                        f"try again in {_in(wait)}. The sample clips with cached results still work.")
            q.extend([now] * n)
            self.daily[day] += n
        return None

    def remaining(self, ip: str, kind: str) -> int:
        per_hour, _ = self.limits(kind)
        with self.lock:
            q = self.hits[(ip, kind)]
            return max(0, per_hour - sum(1 for t in q if t >= time.time() - 3600))

    def remaining_today(self, kind: str) -> int:
        _, per_day = self.limits(kind)
        with self.lock:
            return max(0, per_day - self.daily[(dt.date.today().isoformat(), kind)])


def _to_midnight(now: float) -> float:
    t = dt.datetime.fromtimestamp(now)
    return (dt.datetime.combine(t.date() + dt.timedelta(days=1), dt.time()) - t).total_seconds()


def _in(seconds: float) -> str:
    """60 s -> "1 minute", 3000 s -> "50 minutes", 5 h -> "5 hours"."""
    m = max(1, -(-int(seconds) // 60))
    if m <= 90:
        return f"{m} minute" + ("s" if m != 1 else "")
    h = -(-m // 60)
    return f"{h} hours"


def kind_name(kind):
    return {"sam": "SAM 3.1 frames", "scout": "live Rule Scout runs",
            "scout_step": "Rule Scout steps"}.get(kind, "Muse Spark calls")


LIMITER = Limiter()


def client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "?")


# ------------------------------------------------------------------ caches

class JsonCache:
    """Committed file under demo/cache (read-only on Vercel) + a /tmp copy for new entries."""

    def __init__(self, name: str):
        self.repo = CACHE_DIR / f"{name}.json"
        self.tmp = TMP_CACHE / f"{name}.json"
        self.lock = threading.Lock()
        self.data: dict = {}
        for p in (self.repo, self.tmp):
            try:
                self.data.update(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            target = self.repo if WRITE_REPO_CACHE else self.tmp
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                old = {}
                if target.exists():
                    old = json.loads(target.read_text(encoding="utf-8"))
                old[key] = value
                target.write_text(json.dumps(old, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                  encoding="utf-8")
            except OSError as e:
                log.warning("cache write %s: %s", target, e)


COMPILE_CACHE = JsonCache("compile")
EXPLAIN_CACHE = JsonCache("explain")


def _h(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


def compile_key(text: str, clip: str, cam: dict | None = None) -> str:
    if cam is not None:
        # zones the user drew: their titles are in the compiled prose, so they are part of the key
        return _h(" ".join(text.lower().split()), sorted((z["name"], z["title"]) for z in cam["zones"]), "drawn")
    zone_names = sorted(z["name"] for z in zones_for(clip)["zones"])
    return _h(" ".join(text.lower().split()), zone_names)


def cached_rule(hit: dict, clip: str, cam: dict | None = None) -> dict:
    """A cached compile result for this camera. The key holds zone names only, so a result compiled for
    another camera with the same names may quote that camera's zone titles ("past_gate (Shop floor)");
    those are swapped for this camera's titles."""
    from .compiler import humanize_zone_names
    rule = json.loads(json.dumps(hit["rule"]))
    if cam is not None:
        return humanize_zone_names(rule, cam["zones"])
    src = hit.get("clip")
    if not src or src == clip:
        return humanize_zone_names(rule, zones_for(clip)["zones"])
    try:
        old = {z["name"]: z.get("title") for z in zones_for(src)["zones"]}
    except HTTPException:
        return rule
    new = {z["name"]: z.get("title") for z in zones_for(clip)["zones"]}
    for k in ("title", "unsupported", "suggested_rewrite", "explanation_template"):
        v = rule.get(k)
        if not isinstance(v, str):
            continue
        for name, t in old.items():
            if t and new.get(name) and t != new[name]:
                v = v.replace(f"{name} ({t})", f"{name} ({new[name]})").replace(t, new[name]) \
                     .replace(t.lower(), new[name].lower())
        rule[k] = v
    return humanize_zone_names(rule, zones_for(clip)["zones"])


# ------------------------------------------------------------------ engine helpers

def _event_dict(ev) -> dict:
    d = dataclasses.asdict(ev)
    d["duration_s"] = ev.measurements.get("duration_s")
    return d


def _hits(res) -> list:
    return [{"name": o.name, "box": [round(v, 4) for v in o.box], "id": o.track_id} for o in (res.objects or [])]


_TRUTH = {Truth.TRUE: "T", Truth.FALSE: ".", Truth.NOT_EVALUABLE: "?"}


def engine_state(st, t: float, period: float) -> dict:
    """State of the rule's episode machine after second t: idle, candidate (true for held_s of the
    min_s it needs) or alert (an event is open since since_s)."""
    tracks = list(st.tracks.values())
    act = [tr for tr in tracks if tr.phase == "active"]
    if act:
        return {"phase": "alert", "since_s": min(tr.event.condition_since_s for tr in act),
                "start_s": min(tr.event.start_s for tr in act), "min_s": act[0].min_duration_s}
    cand = [tr for tr in tracks if tr.phase == "candidate"]
    if cand:
        since = min(tr.since_s for tr in cand)
        return {"phase": "candidate", "since_s": since, "held_s": round(t - since + period, 2),
                "min_s": cand[0].min_duration_s}
    return {"phase": "idle", "min_s": float(st.rule.get("min_duration_s") or 0)}


def frame_diag(eng, rule: dict, dets, size, box_units: str) -> dict:
    """Foot points, zone membership, PPE overlap, distance lines of this second (engine/diagnose.py)."""
    from .engine.diagnose import frame_diagnostics
    from .engine.evaluate import normalise_detections
    from .engine.primitives import Context
    ctx = Context(normalise_detections(dets, size or eng.frame_size, box_units), eng.zones, eng.H, {})
    return frame_diagnostics(rule.get("condition"), ctx)


def run_engine(rule: dict, zones: dict, frames: list[tuple[float, list[Detection], tuple, dict | None]],
               box_units: str, frame_period_s: float = 1.0, episodes: str = "object") -> dict:
    eng = Engine([rule], zones, frame_period_s=frame_period_s, box_units=box_units, skip_rejected=True,
                 episodes=episodes)
    if eng.rejected:
        return {"rejected": next(iter(eng.rejected.values())), "events": [], "timeline": "", "seconds": []}
    rid = next(iter(eng.states))
    seconds, chars = [], []
    for t, dets, size, machines in frames:
        eng.process(Frame(t, dets, size=size, machine_states=machines))
        st = eng.states[rid]
        res = st.last_result
        chars.append(_TRUTH[res.truth])
        seconds.append({"t": t, "truth": _TRUTH[res.truth], "hits": _hits(res) if res.is_true else [],
                        "reason": res.reason, "state": engine_state(st, t, frame_period_s),
                        "diag": frame_diag(eng, rule, dets, size, box_units)})
    eng.finish()
    return {"rejected": None, "not_evaluable": eng.not_evaluable.get(rid), "events": [_event_dict(e) for e in eng.events],
            "timeline": "".join(chars), "seconds": seconds}


# ------------------------------------------------------------------ Spark wrappers

def spark_available() -> bool:
    return bool(os.environ.get("SM_LLM_API_KEY"))


def explain_cached(event: dict, rule: dict, clip: str, ip: str, allow_model: bool = True,
                   exempt: bool = False, strict: bool = False, cam: dict | None = None) -> dict:
    """Cached explanation, else Muse Spark (allow_model, within the limits), else the rule's template.
    Over the limit: strict (the /api/explain endpoint) answers 429, otherwise the template with a note."""
    from .explain import build_facts, explain_event_result, explain_event_template
    zf = zones_for(clip, cam)
    # Only the zones the rule names: "on the Warehouse floor", not every polygon the person stands in
    zones = {"frame_size": zf.get("frame_size"), "zones": [z for z in zf["zones"] if z["name"] in rule.get("zones", [])]}
    # The alert card shows the activation time and the episode length; the text must say the same:
    # time = start_s (when the condition had held min_duration_s), duration = whole episode
    event = dict(event)
    a = int(round(event.get("start_s", event.get("condition_since_s", 0)) or 0))
    event["time"] = f"{a // 60:02d}:{a % 60:02d}"
    event["duration_s"] = (event.get("measurements") or {}).get("duration_s", event.get("duration_s"))
    facts = build_facts(event, rule, zones)
    key = _h(facts)
    hit = EXPLAIN_CACHE.get(key)
    template = explain_event_template(event, rule)
    if hit:
        return {**hit, "template": template, "cached": True}
    if not allow_model or not spark_available():
        return {"explanation": template, "source": "template", "template": template, "cached": False,
                "note": None if allow_model else "not explained by the model (only the first 3 alerts are)"}
    err = LIMITER.take(ip, "spark", exempt=exempt)
    if err:
        if strict:
            raise HTTPException(429, err)
        return {"explanation": template, "source": "template", "template": template, "cached": False, "note": err}
    t0 = time.time()
    out = explain_event_result(event, rule, "en", zones=zones)
    rec = {"explanation": out["explanation"], "source": out["source"], "severity_check": out.get("severity_check")}
    if out["source"] == "model":
        EXPLAIN_CACHE.put(key, rec)
    return {**rec, "template": template, "cached": False, "ms": int((time.time() - t0) * 1000)}


# ------------------------------------------------------------------ app

app = FastAPI(title="Safety Mode demo API", docs_url=None, redoc_url=None, openapi_url=None)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"detail": f"Server error: {type(exc).__name__}"}, status_code=500)


@app.get("/api/health")
def health(request: Request):
    """Public: what is configured, the limits and what is left for this address this hour and today."""
    ip = client_ip(request)
    kinds = {"sam_frames": "sam", "spark_calls": "spark", "scout_runs": "scout", "scout_steps": "scout_step"}
    out = {"ok": True, "spark_key": spark_available(), "sam_key": bool(os.environ.get("MODEL_API_KEY")),
           "clips": {k: v["title"] for k, v in CLIPS.items()}, "examples": EXAMPLES,
           "webcam_max_frames": int(os.environ.get("SM_WEBCAM_MAX_FRAMES", "40")),
           "cached": {"compile": len(COMPILE_CACHE.data), "explain": len(EXPLAIN_CACHE.data)},
           "limits": {**{f"{k}_per_hour_per_ip": Limiter.limits(v)[0] for k, v in kinds.items()},
                      **{f"{k}_per_day": Limiter.limits(v)[1] for k, v in kinds.items()}},
           "remaining": {k: LIMITER.remaining(ip, v) for k, v in kinds.items()},
           "remaining_today": {k: LIMITER.remaining_today(v) for k, v in kinds.items()}}
    if operator_exempt(request, allow_query=True):
        out["exempt"] = True
    return out


@app.post("/api/compile")
def api_compile(request: Request, body: dict = Body(...)):
    text = (body.get("text") or "").strip()
    clip = body.get("clip") or "webcam"
    rule_id = body.get("rule_id") or "custom"
    if not text:
        raise HTTPException(400, "Write a rule first.")
    if len(text) > 400:
        raise HTTPException(400, "Keep the rule under 400 characters.")
    cam = camera_from_body(clip, body)
    zones = zones_for(clip, cam)
    key = compile_key(text, clip, cam)
    t0 = time.time()
    hit = COMPILE_CACHE.get(key)
    cached = bool(hit)
    if hit:
        rule = cached_rule(hit, clip, cam)
    else:
        if not spark_available():
            raise HTTPException(503, "Muse Spark is not configured on this server.")
        err = LIMITER.take(client_ip(request), "spark", exempt=operator_exempt(request))
        if err:
            raise HTTPException(429, err)
        from .compiler import CompileError, compile_rule
        try:
            rule = compile_rule(text, zones["zones"], rule_id=rule_id)
        except CompileError as e:
            raise HTTPException(422, "Muse Spark returned an invalid rule twice: " + "; ".join(e.errors)[:400])
        COMPILE_CACHE.put(key, {"text": text, "clip": clip, "rule": rule})
    rule["rule_id"] = rule_id
    status = rule_status(rule, clip, cam=cam) if clip == "webcam" else choose_detector(rule, clip, cam=cam)[1]
    return {"rule": rule, "status": status, "cached": cached,
            "ms": int((time.time() - t0) * 1000), "model": "muse-spark-1.3"}


@app.post("/api/run")
def api_run(request: Request, body: dict = Body(...)):
    clip = body.get("clip")
    rule = body.get("rule")
    if not isinstance(rule, dict):
        raise HTTPException(400, "rule must be the compiled rule object")
    t0 = time.time()
    cam = camera_from_body(clip, body) if clip == OWN else None
    if clip == OWN:
        # the user's clip: no cache, every prompt comes live from the page (extra_detections)
        info = {"title": OWN_TITLE, "machine": ""}
        det, status = choose_detector(rule, clip, cam=cam)
        dj = {"frame_size": cam["frame_size"], "frames": {f"{s + 1:03d}": {} for s in range(cam["seconds"])}}
    else:
        info = clip_info(clip)
        # video mode (track ids) by default; image mode if the video cache lacks a prompt the rule needs
        det, status = choose_detector(rule, clip, body.get("detector"))
        dj = detections_for(clip, det)
    zones = zones_for(clip, cam)
    size = tuple(dj["frame_size"])
    pmap = status["prompts"]
    live = {}
    if status["live"]:
        # prompts with no cached detections: SAM 3.1 ran live in the page (/api/detect)
        live = parse_extra_detections(body.get("extra_detections"), status["live"], len(dj["frames"]), size)
    mach = parse_machine_spec(info["machine"]) if info["machine"] else (lambda t: None)
    mnames = machine_names(rule.get("condition"))
    frames, overlay = [], []
    for stem in sorted(dj["frames"]):
        t = float(int(stem) - 1)
        rec = dj["frames"][stem]
        dets, shown = [], []
        for o in rule.get("objects", []):
            src = pmap.get(o["prompt"])
            for d in (rec.get(src, []) if src else []):
                dets.append(Detection(o["name"], d["box"], score=d.get("score"), track_id=d.get("track_id")))
                b = d["box"]
                shown.append({"name": o["name"], "prompt": o["prompt"], "score": d.get("score"), "id": d.get("track_id"),
                              "box": [round(b[0] / size[0], 4), round(b[1] / size[1], 4),
                                      round(b[2] / size[0], 4), round(b[3] / size[1], 4)]})
            if not src:
                # live image-mode boxes: no track ids (the engine then keeps one episode per rule for them)
                for b, score in live.get(int(t), {}).get(o["prompt"], []):
                    dets.append(Detection(o["name"], b, score=score))
                    shown.append({"name": o["name"], "prompt": o["prompt"], "score": score, "id": None, "live": True,
                                  "box": [round(b[0] / size[0], 4), round(b[1] / size[1], 4),
                                          round(b[2] / size[0], 4), round(b[3] / size[1], 4)]})
        frames.append((t, dets, size, {m: mach(t) for m in mnames} or None))
        overlay.append(shown)
    # per-object episodes when the boxes carry track ids; {"episodes": "rule"} = one episode per rule
    episodes = "rule" if body.get("episodes") == "rule" else "object"
    out = run_engine(rule, zones, frames, "px", episodes=episodes)
    for s, shown in zip(out["seconds"], overlay):
        s["dets"] = shown
    ip = client_ip(request)
    evs = out["events"]
    exempt = operator_exempt(request)
    with ThreadPoolExecutor(3) as ex:
        expl = list(ex.map(lambda ie: explain_cached(ie[1], rule, clip, ip, allow_model=ie[0] < 3, exempt=exempt, cam=cam),
                           enumerate(evs)))
    for e, x in zip(evs, expl):
        e["explanation"] = x["explanation"]
        e["explanation_source"] = x["source"]
        e["template"] = x["template"]
        e["explanation_note"] = x.get("note")
    return {"clip": clip, "title": info["title"], "frame_size": size, "duration_s": len(frames), "detector": det, "episodes": episodes,
            "status": status, "live_prompts": status["live"], **out, "ms": int((time.time() - t0) * 1000)}


MAX_LIVE_DETS = 60   # boxes per prompt per frame


def parse_extra_detections(extra, prompts: list, n_frames: int, size) -> dict:
    """{"<second>": {"<prompt>": [{"box": [x1, y1, x2, y2] in 0..1, "score"}]}} from the page's live SAM 3.1 run
    -> {second: {prompt: [(box_px, score)]}}. Every live prompt must be there for every second of the clip
    (an empty list = SAM found nothing), otherwise the run would miss objects without saying so."""
    if not isinstance(extra, dict):
        extra = {}
    w, h = size
    out: dict = {}
    missing = 0
    for s in range(n_frames):
        rec = extra.get(str(s))
        if not isinstance(rec, dict):
            rec = {}
        for p in prompts:
            lst = rec.get(p)
            if not isinstance(lst, list):
                missing += 1
                continue
            got = []
            for d in lst[:MAX_LIVE_DETS]:
                b = d.get("box") if isinstance(d, dict) else None
                if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in b)):
                    raise HTTPException(400, f"extra_detections: bad box at second {s} for {p!r}")
                x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in b)
                if x2 <= x1 or y2 <= y1:
                    continue
                sc = d.get("score")
                sc = float(sc) if isinstance(sc, (int, float)) and math.isfinite(sc) else None
                got.append(([x1 * w, y1 * h, x2 * w, y2 * h], sc))
            out.setdefault(s, {})[p] = got
    if missing:
        names = ", ".join(f"\u201c{p}\u201d" for p in prompts)
        raise HTTPException(409, f"There are no cached SAM 3.1 results for {names} on this clip, and live ones are "
                                 f"missing for {missing} of {n_frames * len(prompts)} frame-prompt pairs. "
                                 "Press Run again: the page runs SAM 3.1 live on the clip first.")
    return out


def _decode_jpeg(s: str) -> bytes:
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    try:
        data = base64.b64decode(s, validate=False)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "jpeg must be base64")
    if len(data) > 600_000:
        raise HTTPException(413, "Frame too large (limit 600 KB).")
    if data[:2] != b"\xff\xd8":
        raise HTTPException(400, "Not a JPEG.")
    return data


@app.post("/api/detect")
def api_detect(request: Request, body: dict = Body(...)):
    prompts = [p.strip() for p in (body.get("prompts") or []) if isinstance(p, str) and p.strip()][:4]
    if not prompts:
        raise HTTPException(400, "prompts[] is empty")
    if any(len(p) > 60 for p in prompts):
        raise HTTPException(400, "prompt too long")
    data = _decode_jpeg(body.get("jpeg") or "")
    if not os.environ.get("MODEL_API_KEY"):
        raise HTTPException(503, "SAM 3.1 is not configured on this server.")
    err = LIMITER.take(client_ip(request), "sam", exempt=operator_exempt(request))
    if err:
        raise HTTPException(429, err)
    from .detect import SamClient
    from .pipeline import SAM_OPTIONS
    sam = SamClient(cache_dir=Path(os.environ.get("SM_SAM_CACHE", str(TMP_CACHE / "sam31"))), timeout=25,
                    **SAM_OPTIONS)
    t0 = time.time()

    def one(p):
        r = sam.segment_image(data, p)
        w, h = r.source_size
        return p, [{"box": [round(d.box[0] / w, 4), round(d.box[1] / h, 4), round(d.box[2] / w, 4),
                            round(d.box[3] / h, 4)],
                    "score": round(d.confidence, 3) if d.confidence is not None else None, "id": d.track_id}
                   for d in r.detections(decode_masks=False)], (w, h)

    try:
        with ThreadPoolExecutor(len(prompts)) as ex:
            res = list(ex.map(one, prompts))
    except Exception as e:
        log.warning("SAM error: %s", e)
        raise HTTPException(502, f"SAM 3.1 request failed: {type(e).__name__}")
    return {"detections": {p: d for p, d, _ in res}, "frame_size": res[0][2], "calls": sam.calls,
            "ms": int((time.time() - t0) * 1000)}


@app.post("/api/step")
def api_step(body: dict = Body(...)):
    rule = body.get("rule")
    if not isinstance(rule, dict):
        raise HTTPException(400, "rule must be the compiled rule object")
    frames_in = (body.get("frames_so_far") or [])[-80:]
    period = float(body.get("frame_period_s") or 2.0)
    cam = camera_from_body("webcam", body)
    zones = cam or (body.get("zones") if isinstance(body.get("zones"), dict) else WEBCAM_ZONES)
    frames = []
    for fr in frames_in:
        dets = []
        by_prompt = fr.get("detections") or {}
        for o in rule.get("objects", []):
            for d in by_prompt.get(o["prompt"], []):
                # webcam frames go through image mode: its "id" is an index inside one frame, not a track
                dets.append(Detection(o["name"], d["box"], score=d.get("score")))
        frames.append((float(fr.get("t", 0)), dets, None, None))
    out = run_engine(rule, zones, frames, "norm", frame_period_s=period)
    out["status"] = rule_status(rule, "webcam", cam=cam)
    return out


@app.post("/api/explain")
def api_explain(request: Request, body: dict = Body(...)):
    event, rule = body.get("event"), body.get("rule")
    if not isinstance(event, dict) or not isinstance(rule, dict):
        raise HTTPException(400, "event and rule are required")
    clip = body.get("clip") or "webcam"
    cam = camera_from_body(clip, body)
    zones_for(clip, cam)
    return explain_cached(event, rule, clip, client_ip(request), allow_model=not body.get("template_only"),
                          exempt=operator_exempt(request), strict=True, cam=cam)


def _clock(s: float) -> str:
    s = int(round(s or 0))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _mmss(s: float) -> str:
    """Clip time as the page shows it on the alert card: 7 -> "00:07"."""
    s = int(round(s or 0))
    return f"{s // 60:02d}:{s % 60:02d}"


def report_events(events: list) -> list:
    """Page alerts -> report-input events. The time is the engine's alert start (start_s, when the condition
    had held min_duration_s, the time on the alert card), end_time its end_s; both mm:ss into the clip,
    as on the page. condition_since_s (the candidate start) is not the alert time."""
    out = []
    for i, e in enumerate(events, 1):
        m = e.get("measurements") or {}
        start = e.get("start_s", e.get("condition_since_s", 0))
        out.append({
            "id": f"evt-{i:02d}", "time": _mmss(start), "end_time": _mmss(e.get("end_s", start)),
            "rule_id": e.get("rule_id") or "custom", "severity": e.get("severity") or "medium",
            "duration_s": float(m.get("duration_s") or e.get("duration_s") or 1),
            "objects": [{"name": o["name"], "box": [float(v) for v in o["box"]]} for o in e.get("objects_at_start") or e.get("objects", [])
                        if o.get("box")],
            "measurements": {"distance_m": m.get("min_distance_m"), "count": m.get("max_count"), "height_m": None},
            "frame": f"evt-{i:02d}.jpg", "explanation": (e.get("explanation") or "")[:600]})
    return out


@app.post("/api/report")
def api_report(request: Request, body: dict = Body(...)):
    events = body.get("events") or []
    rules = {r.get("rule_id"): r for r in (body.get("rules") or []) if isinstance(r, dict)}
    if not events:
        raise HTTPException(400, "No alerts in this session yet. Run a rule on a clip or the webcam first.")
    if not spark_available():
        raise HTTPException(503, "Muse Spark is not configured on this server.")
    err = LIMITER.take(client_ip(request), "spark", exempt=operator_exempt(request))
    if err:
        raise HTTPException(429, err)
    events = events[:40]
    cams = []
    rin = {"site": "Frappua demo warehouse", "camera": "", "date": dt.date.today().isoformat(),
           "uptime_min": 0, "events": [], "rules": [], "manager_actions": []}
    seen_len: dict = {}
    for e in events:
        clip = e.get("clip") or "webcam"
        title = "Webcam" if clip == "webcam" else OWN_TITLE if clip == OWN else CLIPS.get(clip, {}).get("title", clip)
        if title not in cams:
            cams.append(title)
        seen_len[clip] = max(seen_len.get(clip, 0), float(e.get("clip_length_s") or e.get("end_s") or 0) + 1)
    rin["events"] = report_events(events)
    rin["camera"] = ", ".join(cams) + " (times are minutes:seconds into each clip)"
    rin["uptime_min"] = max(1, -(-int(sum(seen_len.values())) // 60))   # whole minutes, at least 1
    counts = defaultdict(int)
    for e in rin["events"]:
        counts[e["rule_id"]] += 1
    for rid in counts:
        r = rules.get(rid, {})
        rin["rules"].append({"id": rid, "title": r.get("title") or rid, "events": counts[rid],
                             "false_positives_marked": 0, "enabled": True})
    from .report import ReportError, ReportInputError, daily_report_result
    t0 = time.time()
    try:
        out = daily_report_result(rin, "en")
    except ReportInputError as e:
        raise HTTPException(400, str(e)[:400])
    except ReportError as e:
        raise HTTPException(502, "Muse Spark's report failed the checks twice: " + "; ".join(e.errors)[:400])
    except Exception as e:
        log.warning("report error: %s", e)
        raise HTTPException(502, f"Muse Spark request failed: {type(e).__name__}")
    return {"markdown": out["markdown"], "patterns": out["patterns"], "suggestions": out["suggestions"],
            "input": rin, "ms": int((time.time() - t0) * 1000)}


@app.post("/api/scout/step")
def api_scout_step(request: Request, body: dict = Body(...)):
    """Rule Scout, one step per request so each stays under 60 s. The browser keeps the signed session state.

    {clip} starts a run (1 per IP per hour); {state, sig} advances it by one model call or one tool round.
    """
    from . import scout
    ip = client_ip(request)
    state, sig = body.get("state"), body.get("sig")
    if state is None:
        clip = body.get("clip")
        clip_info(clip)
        if not spark_available():
            raise HTTPException(503, "Muse Spark is not configured on this server.")
        err = LIMITER.take(ip, "scout", exempt=operator_exempt(request))
        if err:
            raise HTTPException(429, err)
        state = scout.new_session(clip)
        return {"state": state, "sig": scout.sign_state(state), "view": scout.public_view(state)}
    if not isinstance(state, dict) or not scout.verify_state(state, sig):
        raise HTTPException(400, "This Rule Scout session is not valid any more. Start a new run.")
    if time.time() - float(state.get("started") or 0) > 1800:
        raise HTTPException(400, "This Rule Scout session is older than 30 minutes. Start a new run.")
    if not state.get("done"):
        err = LIMITER.take(ip, "scout_step", exempt=operator_exempt(request))
        if err:
            raise HTTPException(429, err)
        t0 = time.time()
        try:
            scout.step(state)
        except Exception as e:      # the state is unchanged when the model call fails: the page may retry
            log.warning("scout step error: %s", e)
            raise HTTPException(502, f"Muse Spark request failed: {type(e).__name__}. Press Continue to retry.")
        log.info("scout step %s round %s in %d ms", state["clip"], state["rounds"], (time.time() - t0) * 1000)
    return {"state": state, "sig": scout.sign_state(state), "view": scout.public_view(state)}


@app.get("/api/zones/{clip}")
def api_zones(clip: str):
    return zones_for(clip)


_static = os.environ.get("SM_STATIC_DIR")
if _static and Path(_static).is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")

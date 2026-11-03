"""Whole chain on a test clip: frames -> SAM 3.1 detections -> condition engine -> events.

    from safety_mode.pipeline import run_clip
    run = run_clip("busy-warehouse", ["r02", "r03"], machine_states={"forklift": "running:0-8"})
    for ev in run.events: ...
    run.timeline["r2"]          # one char per second: T true, . false, ? not evaluable

Frames: test-footage/frames/<clip>/NNN.jpg, 1 fps, NNN = second + 1.
Detections, per (frame, prompt), from the first source that has them:
  1. the compact offline cache committed to git (boxes and scores only):
     detector="video": test-footage/detections/<clip>.video.json (SAM 3.1 video mode, one
     request per clip and prompt, stable track_id per object; detect/sam_video.py);
     detector="image": test-footage/detections/<clip>.json (one request per frame and prompt,
     no track ids: "id" there is only the index inside the frame);
  2. SamClient: the full-response disk cache test-footage/cache/sam31/, then the API.
mode="offline" uses only 1, mode="api" only 2.
Default detector: DEFAULT_DETECTOR (env SM_DETECTOR overrides; see results-2026-10-29.md).
Masks are not decoded (decode_masks=False): the engine's `wearing` falls back to
box thirds, the rest of the engine uses boxes only.

One Engine per rule, so two rules may give the same object name different prompts.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from .engine import Detection, Engine, Event, Frame, Truth

REPO = Path(__file__).resolve().parents[1]
FOOTAGE = REPO / "test-footage"
FRAMES_DIR = FOOTAGE / "frames"
ZONES_DIR = FOOTAGE / "zones"
DETS_DIR = FOOTAGE / "detections"
RULES_DIRS = [REPO / "rules", FOOTAGE / "rules"]
CLIPS = ["pexels-6079421-forklift-busy-warehouse", "pexels-15170997-warehouse-team",
         "pexels-32243653-forklift-operator", "pexels-38707991-man-crosses-floor",
         "pexels-31751344-forklift-crosses-aisle", "pexels-32245123-site-worker-no-vest"]
DEFAULT_PROMPTS = ["person", "forklift", "hard hat", "safety vest"]
# options of every image request so far: the SamClient cache key includes them
SAM_OPTIONS = {"mask_encoding": "one_bit", "include_confidence": True}
_OPPOSITE = {"running": "stopped", "stopped": "running"}
DETECTORS = ("video", "image")
# video mode: closer to expected.md on all three clips and gives track ids (results-2026-10-29.md)
DEFAULT_DETECTOR = os.environ.get("SM_DETECTOR", "video")


_VIDEO_LOCK = threading.Lock()


# ------------------------------------------------------------------ names

def resolve_clip(name: str) -> str:
    if name in CLIPS:
        return name
    hits = [c for c in CLIPS if c.endswith(name) or name in c]
    if len(hits) != 1:
        raise ValueError(f"unknown or ambiguous clip {name!r}; known: {', '.join(CLIPS)}")
    return hits[0]


def resolve_rule(rid: Union[str, Path]) -> Path:
    """r02 / r2 / r2-vest / a path -> rule.json path (rules/, then test-footage/rules/)."""
    p = Path(rid)
    if p.suffix == ".json" and p.exists():
        return p
    s = str(rid)
    cands = [s]
    if s[:1] == "r" and s[1:].split("-")[0].isdigit():
        num, _, rest = s[1:].partition("-")
        cands.append(f"r{int(num):02d}" + (f"-{rest}" if rest else ""))
    for d in RULES_DIRS:
        for c in cands:
            if (d / f"{c}.json").exists():
                return d / f"{c}.json"
    raise ValueError(f"rule {rid!r} not found in {', '.join(str(d) for d in RULES_DIRS)}")


def frame_paths(clip: str, fps: int = 1) -> list[tuple[float, Path]]:
    if fps != 1:
        raise ValueError("test-footage/frames holds 1 fps frames only")
    d = FRAMES_DIR / clip
    files = sorted(d.glob("[0-9][0-9][0-9].jpg"))
    if not files:
        # frames are not in git; the committed detections JSON lists them, enough for offline runs
        cj = _committed_json(clip)
        if cj:
            files = [d / f"{k}.jpg" for k in sorted(json.loads(cj.read_text())["frames"])]
    if not files:
        raise FileNotFoundError(f"no frames in {d} and no {DETS_DIR / (clip + '.json')}")
    return [(float(int(f.stem) - 1), f) for f in files]


def _committed_json(clip: str) -> Optional[Path]:
    """Image-mode <clip>.json, else video-mode <clip>.video.json (clips added 30.10 have only video mode)."""
    for name in (f"{clip}.json", f"{clip}.video.json"):
        if (DETS_DIR / name).exists():
            return DETS_DIR / name
    return None


def image_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        cj = _committed_json(path.parent.name)
        return tuple(json.loads(cj.read_text())["frame_size"])
    from PIL import Image
    with Image.open(path) as im:
        return im.size


# ------------------------------------------------------------------ machine states

def parse_machine_spec(spec: str, clip_len_s: Optional[float] = None) -> Callable[[float], str]:
    """'running:0-8' -> running on s0..s8, the opposite state elsewhere.
    'running:0-8,stopped:9-24' -> explicit; 'running' -> whole clip."""
    parts = []
    for chunk in spec.split(","):
        state, _, rng = chunk.partition(":")
        state = state.strip()
        if state not in _OPPOSITE:
            raise ValueError(f"machine state must be running or stopped, got {state!r}")
        if rng:
            a, _, b = rng.partition("-")
            parts.append((state, float(a), float(b or a)))
        else:
            parts.append((state, float("-inf"), float("inf")))
    default = _OPPOSITE[parts[0][0]]

    def at(t: float) -> str:
        for state, a, b in parts:
            if a - 1e-6 <= t <= b + 1e-6:
                return state
        return default
    return at


def _machine_fn(machine_states) -> Callable[[float], dict]:
    fns = {}
    for name, v in (machine_states or {}).items():
        fns[name] = v if callable(v) else parse_machine_spec(v)
    return lambda t: {n: f(t) for n, f in fns.items()}


# ------------------------------------------------------------------ detections

class DetectionSource:
    """Boxes per (clip, frame, prompt): compact JSON first, then SamClient (cache, API).

    detector="video": records carry track_id (stable per object within a prompt);
    detector="image": records carry no track_id.
    """

    def __init__(self, mode: str = "auto", client=None, detector: Optional[str] = None):
        if mode not in ("auto", "offline", "api"):
            raise ValueError("mode must be auto, offline or api")
        detector = detector or DEFAULT_DETECTOR
        if detector not in DETECTORS:
            raise ValueError(f"detector must be one of {', '.join(DETECTORS)}")
        self.mode = mode
        self.detector = detector
        self._client = client
        self._compact: dict[str, dict] = {}
        self._video: dict[tuple, dict] = {}     # (clip, prompt) -> {stem: records} from SamClient
        self._locks: dict[tuple, threading.Lock] = {}

    @property
    def client(self):
        if self._client is None:
            from .detect import SamClient
            self._client = SamClient(**SAM_OPTIONS)
        return self._client

    @property
    def calls(self) -> int:
        return self._client.calls if self._client is not None else 0

    def compact_path(self, clip: str) -> Path:
        return DETS_DIR / (f"{clip}.video.json" if self.detector == "video" else f"{clip}.json")

    def compact(self, clip: str) -> dict:
        if clip not in self._compact:
            p = self.compact_path(clip)
            self._compact[clip] = json.loads(p.read_text()) if p.exists() else {"frames": {}}
        return self._compact[clip]

    def get(self, clip: str, frame: Path, prompt: str) -> list[dict]:
        if self.mode != "api":
            rec = self.compact(clip).get("frames", {}).get(frame.stem, {}).get(prompt)
            if rec is not None:
                return rec
            if self.mode == "offline":
                raise LookupError(f"{clip}/{frame.name} {prompt!r} is not in {self.compact_path(clip)}")
        if self.detector == "video":
            return self._video_records(clip, prompt).get(frame.stem, [])
        r = self.client.segment_image(frame, prompt)
        # no "id": in image mode SAM numbers objects inside one frame, it is not a track id
        return [{"box": [round(v, 1) for v in d.box],
                 "score": round(d.confidence, 3) if d.confidence is not None else None}
                for d in r.detections(decode_masks=False)]

    def _video_records(self, clip: str, prompt: str) -> dict:
        """One video request per (clip, prompt) through SamClient (its disk cache first)."""
        key = (clip, prompt)
        with _VIDEO_LOCK:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:              # one request per (clip, prompt) even with prefetch threads
            if key not in self._video:
                from .detect import sam_video
                r = sam_video.segment_clip(clip, [prompt], self.client)[prompt]
                info = sam_video.probe(sam_video.video_path(clip))
                frames = frame_paths(clip)
                size = image_size(frames[0][1])
                self._video[key], _ = sam_video.sample_and_bridge(clip, r, size, [f.stem for _, f in frames],
                                                                   info["fps"], info["frames"])
        return self._video[key]

    def prefetch(self, clip: str, frames: list[Path], prompts: list[str], workers: int = 6) -> dict:
        jobs = [(f, p) for f in frames for p in prompts]
        if self.mode != "offline":
            self.client         # create the one SamClient before the threads (its counter = paid calls)
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(lambda j: self.get(clip, *j), jobs))
        return {(f.stem, p): r for (f, p), r in zip(jobs, res)}


def export_detections(clip: str, prompts: list[str] = DEFAULT_PROMPTS, source: Optional[DetectionSource] = None,
                      workers: int = 6) -> Path:
    """Write/merge the compact cache from SamClient (cache or API): detections/<clip>.json (image
    detector) or detections/<clip>.video.json (video detector, detect/sam_video.py)."""
    src = source or DetectionSource("api")
    if src.detector == "video":
        from .detect import sam_video
        path, _ = sam_video.export_video_detections(clip, prompts, src.client)
        src._compact.pop(clip, None)
        return path
    frames = [f for _, f in frame_paths(clip)]
    got = src.prefetch(clip, frames, prompts, workers)
    path = DETS_DIR / f"{clip}.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    size = image_size(frames[0])
    data.update({
        "clip": clip, "model": "sam-3.1", "fps": 1, "frame_size": list(size),
        "source": "Meta Model API, image mode, one request per frame and prompt, long side <= 1280 px, "
                  "mask_encoding=one_bit, include_confidence=true; masks dropped",
        "box_units": "pixels of frames/<clip>/NNN.jpg, [x1, y1, x2, y2], x2/y2 exclusive",
        "updated": time.strftime("%Y-%m-%d"),
    })
    data["prompts"] = sorted(set(data.get("prompts", [])) | set(prompts))
    fr = data.setdefault("frames", {})
    for (stem, p), dets in got.items():
        fr.setdefault(stem, {})[p] = [{k: v for k, v in d.items() if k != "id"} for d in dets]
    data["frames"] = {k: dict(sorted(v.items())) for k, v in sorted(fr.items())}
    path.parent.mkdir(parents=True, exist_ok=True)
    # one line per frame keeps the file diffable and small
    lines = ["{"]
    head = {k: v for k, v in data.items() if k != "frames"}
    for k, v in head.items():
        lines.append(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},")
    lines.append('  "frames": {')
    items = list(data["frames"].items())
    for i, (k, v) in enumerate(items):
        lines.append(f"    {json.dumps(k)}: {json.dumps(v)}" + ("," if i < len(items) - 1 else ""))
    lines += ["  }", "}"]
    path.write_text("\n".join(lines) + "\n")
    return path


# ------------------------------------------------------------------ run

@dataclass
class ClipRun:
    clip: str
    rules: dict                                   # rule_id -> rule dict
    events: list[Event] = field(default_factory=list)
    timeline: dict = field(default_factory=dict)  # rule_id -> "T.?..." per second
    results: dict = field(default_factory=dict)   # rule_id -> list of ConditionResult per frame
    rejected: dict = field(default_factory=dict)
    not_evaluable: dict = field(default_factory=dict)
    detections: dict = field(default_factory=dict)  # (frame stem, prompt) -> boxes
    calls: int = 0
    frame_size: tuple = (0, 0)
    detector: str = ""


def to_engine_detections(rule: dict, dets: dict, stem: str) -> list[Detection]:
    """track_id only from video-mode records; the image-mode "id" is an index inside one frame."""
    out = []
    for obj in rule.get("objects", []):
        for d in dets.get((stem, obj["prompt"]), []):
            out.append(Detection(obj["name"], d["box"], score=d.get("score"), track_id=d.get("track_id")))
    return out


def run_clip(clip_name: str, rule_ids: list, zones_file=None, machine_states: Optional[dict] = None,
             fps: int = 1, source: Optional[DetectionSource] = None, max_gap_s: float = 1.0,
             detector: Optional[str] = None, episodes: str = "object") -> ClipRun:
    """Run rules on one clip. machine_states: {machine object name: 'running:0-8' | callable(t)->state}.
    detector: "video" | "image" (default DEFAULT_DETECTOR); ignored when source is given.
    episodes: "object" = one episode per tracked object (video detector), "rule" = one per rule."""
    clip = resolve_clip(clip_name)
    zones_path = Path(zones_file) if zones_file else ZONES_DIR / f"{clip}.json"
    zones = json.loads(zones_path.read_text())
    frames = frame_paths(clip, fps)
    size = image_size(frames[0][1])
    zw, zh = zones.get("frame_size") or size
    if abs(size[0] / size[1] - zw / zh) > 0.01:
        raise ValueError(f"frame {size} and zones frame_size {(zw, zh)} differ in aspect ratio")
    rules = {}
    for rid in rule_ids:
        r = json.loads(resolve_rule(rid).read_text(encoding="utf-8"))
        rules[r["rule_id"]] = r
    src = source or DetectionSource(detector=detector)
    calls0 = src.calls
    engines = {rid: Engine([r], zones, max_gap_s=max_gap_s, skip_rejected=True,
                                 episodes=episodes) for rid, r in rules.items()}
    run = ClipRun(clip, rules, frame_size=size, detector=src.detector)
    for eng in engines.values():
        run.rejected.update(eng.rejected)
        run.not_evaluable.update(eng.not_evaluable)
    # rejected rules (unsupported, zone missing on this camera) cost no detection calls
    prompts = sorted({o["prompt"] for rid, r in rules.items() if rid not in run.rejected for o in r["objects"]})
    dets = src.prefetch(clip, [f for _, f in frames], prompts)
    run.detections = dets
    ms = _machine_fn(machine_states)
    for rid, rule in rules.items():
        if rid in run.rejected:
            continue
        eng = engines[rid]
        chars, results = [], []
        for t, f in frames:
            # boxes are in pixels of the frame file; Frame.size normalises them, so a frame
            # of another resolution than zones.frame_size (same aspect) is rescaled correctly
            fr = Frame(t, to_engine_detections(rule, dets, f.stem), size=size, machine_states=ms(t) or None)
            eng.process(fr)
            res = eng.states[rid].last_result
            results.append(res)
            chars.append({Truth.TRUE: "T", Truth.FALSE: ".", Truth.NOT_EVALUABLE: "?"}[res.truth])
        eng.finish()
        run.events += eng.events
        run.timeline[rid] = "".join(chars)
        run.results[rid] = results
    run.calls = src.calls - calls0
    return run


def intervals(run: ClipRun, rule_id: str) -> list[tuple[int, int]]:
    return [(int(e.start_s), int(e.end_s)) for e in run.events if e.rule_id == rule_id]


def fmt_intervals(iv: list[tuple[int, int]]) -> str:
    return ", ".join(f"s{a}" if a == b else f"s{a}–s{b}" for a, b in iv) or "none"

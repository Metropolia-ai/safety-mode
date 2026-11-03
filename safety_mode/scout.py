"""Rule Scout: a Muse Spark agent with tool calling that proposes safety rules for one camera,
compiles them, runs them through the deterministic engine, inspects the alerts they cause and
accepts, revises or rejects each rule. The engine stays the only decider of alerts; the agent
only proposes and verifies.

    from safety_mode.scout import run_scout
    res = run_scout("busy-warehouse")
    res.rules       # accepted: [{rule_ref, text, rule, reason, alerts}]
    res.trace       # every tool call with a one-line summary

Tools exposed to the model (Chat Completions `tools`):
    get_camera_facts()                      zones, calibration, SAM prompts seen per second, clip length
    compile_rule(text, reason, revises?)    safety_mode.compiler.compile_rule (cached), -> rule_ref p1, p2...
    run_rule(rule_ref)                      engine over the cached detections: alerts + per-second timeline
    inspect_alert(event_id, second?)        structured facts for one second of an alert (no images)
    accept_rule(rule_ref, reason) / reject_rule(rule_ref, reason)

The session is a JSON-serialisable dict, so the web demo can run it one step per request
(each request stays under Vercel's 60 s): `new_session` -> `step` until `state["done"]`.
One step is either one model call or the execution of the tool calls of the last model call.

Budget guard: at most MAX_ROUNDS model calls and MAX_OUTPUT_TOKENS output tokens per run
(agent and compiler calls together), at most MAX_PROPOSALS compiled rules.

CLI:
    python -m safety_mode.scout busy-warehouse [--goal ...] [--max-steps 8] [--save]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import llm
from .engine import Detection, Engine, Frame
from .engine.geometry import Homography, distance_m, foot_point, mask_overlap, point_in_polygon
from .engine.primitives import is_operator
from .facts import person_like, position_words
from .llm import LLMConfig, config_from_env, make_client

log = logging.getLogger("safety_mode.scout")

REPO = Path(__file__).resolve().parents[1]
PROMPT = (Path(__file__).resolve().parent / "prompts" / "scout.md").read_text(encoding="utf-8")
CACHE_DIR = REPO / "web" / "cache" / "scout"

MAX_ROUNDS = 8
MAX_OUTPUT_TOKENS = 20_000
MAX_PROPOSALS = 8
MAX_CALLS_PER_ROUND = 8
MAX_TOKENS_PER_CALL = 8000
MIN_TOKENS_FOR_CALL = 1500
CLOSING_TOKENS = 7000                   # below this many output tokens left, the agent must decide, not propose
# Agent calls only (the compiler keeps the model's default). dev.meta.ai/docs/reasoning: minimal..max;
# without it Spark spent up to 4.5k reasoning tokens per round and one run hit the 20k budget undecided.
SCOUT_REASONING = "low"
PRICE_IN, PRICE_OUT = 1.25, 4.25        # $ per 1M tokens, api-notes.md (official price list)
DEFAULT_GOAL = ("Find 2 to 4 safety rules worth enforcing on this camera, check each against the footage, "
                "and keep only the ones whose alerts are real violations.")

MACHINE_WORDS = ("forklift", "truck", "machine", "vehicle", "loader", "press", "crane")
PPE_WORDS = ("hat", "helmet", "vest", "glove", "boot", "goggle", "glasses", "mask", "jacket")

# ------------------------------------------------------------------ tool schemas

def _fn(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required, "additionalProperties": False}}}


TOOLS = [
    _fn("get_camera_facts",
        "What this camera can see: zones (name, title, note), whether the floor is calibrated (needed for "
        "distances in metres), which SAM 3.1 object prompts have detections on this clip and in which seconds, "
        "in which zones people and forklifts stand, clip length, how machine state is set, what the engine can "
        "and cannot check. Call this first.", {}, []),
    _fn("compile_rule",
        "Compile one rule written in plain words, as a safety manager would write it, into rule.json with the "
        "Muse Spark rule compiler. Returns a rule_ref (p1, p2, ...) with the compiled objects, zones, condition, "
        "duration and severity, or 'unsupported' with a suggested rewrite, plus whether it can run on this camera. "
        "To revise a rule, compile the new wording with revises set to the old rule_ref; the old one is closed "
        "as replaced.",
        {"text": {"type": "string", "description": "The rule in one plain sentence."},
         "reason": {"type": "string", "description": "One sentence: why this rule fits this camera, or what the revision fixes."},
         "revises": {"type": "string", "description": "rule_ref of the rule this one replaces (only for revisions)."}},
        ["text", "reason"]),
    _fn("run_rule",
        "Run a compiled rule through the deterministic engine over the whole clip (cached SAM 3.1 detections, "
        "one frame per second). Returns the alerts (event_id, seconds, objects, measurements) and a per-second "
        "timeline of the condition: T true, . false, ? cannot be checked.",
        {"rule_ref": {"type": "string"}}, ["rule_ref"]),
    _fn("inspect_alert",
        "Structured facts for one alert at one second (default: the second the alert was raised): every detection "
        "of the rule's objects with score, position and zones, which ones the engine matched, floor distances, "
        "PPE overlap per person, operator-in-cab flags, detection counts and condition state in the neighbouring "
        "seconds. No images.",
        {"event_id": {"type": "string"},
         "second": {"type": "integer", "description": "Second of the clip inside the alert; default the alert's start."}},
        ["event_id"]),
    _fn("accept_rule", "Accept a rule that ran and whose alerts you checked. reason: one sentence with the evidence.",
        {"rule_ref": {"type": "string"}, "reason": {"type": "string"}}, ["rule_ref", "reason"]),
    _fn("reject_rule", "Reject a rule. reason: one sentence with the evidence (no alerts, fires all the time, "
        "cannot run on this camera, alerts are not real violations).",
        {"rule_ref": {"type": "string"}, "reason": {"type": "string"}}, ["rule_ref", "reason"]),
]

# ------------------------------------------------------------------ camera helpers (cached clip data)


def _web():
    from . import webapp
    return webapp


def _ranges(secs: list[int]) -> str:
    out, start, prev = [], None, None
    for s in secs:
        if start is None:
            start = prev = s
        elif s == prev + 1:
            prev = s
        else:
            out.append(f"s{start}" if start == prev else f"s{start}-s{prev}")
            start = prev = s
    if start is not None:
        out.append(f"s{start}" if start == prev else f"s{start}-s{prev}")
    return ", ".join(out) or "none"


def _norm(box, size):
    w, h = size
    return [box[0] / w, box[1] / h, box[2] / w, box[3] / h]


# SAM 3.1 video mode (track ids), the same detections and episodes as the demo page's /api/run
DETECTOR = "video"
EPISODES = "rule"


def camera_facts(clip: str) -> dict:
    W = _web()
    info = W.clip_info(clip)
    zf = W.zones_for(clip)
    dj = W.detections_for(clip, DETECTOR)
    size = tuple(dj["frame_size"])
    stems = sorted(dj["frames"])
    n = len(stems)
    prompts = {}
    for p in dj["prompts"]:
        secs = [int(s) - 1 for s in stems if dj["frames"][s].get(p)]
        scores = [d["score"] for s in stems for d in dj["frames"][s].get(p, []) if d.get("score") is not None]
        ids = {d.get("track_id") for s in stems for d in dj["frames"][s].get(p, []) if d.get("track_id") is not None}
        prompts[p] = {"seconds_seen": len(secs), "seen_at": _ranges(secs), "distinct_track_ids": len(ids),
                      "max_in_one_frame": max((len(dj["frames"][s].get(p, [])) for s in stems), default=0),
                      "mean_score": round(sum(scores) / len(scores), 2) if scores else None}
    movers = [p for p in dj["prompts"] if person_like(p) or any(k in p for k in MACHINE_WORDS)]
    zones = []
    for z in zf["zones"]:
        where = {}
        for p in movers:
            secs = [int(s) - 1 for s in stems
                    if any(point_in_polygon(foot_point(_norm(d["box"], size)), z["polygon"])
                           for d in dj["frames"][s].get(p, []))]
            where[p] = _ranges(secs)
        zones.append({"name": z["name"], "title": z.get("title") or z["name"], "note": (z.get("note") or "")[:300],
                      "standing_inside_by_second": where})
    cal = zf.get("floor_calibration")
    return {
        "camera": info["title"], "clip": clip, "clip_length_s": n, "fps": 1, "frame_size": list(size),
        "zones": zones,
        "floor_calibration": ("yes: distances in metres can be measured. " + (cal.get("assumption") or "")[:300])
        if cal else "no: distance rules (metres) cannot be checked on this camera",
        "sam_prompts_with_detections": prompts,
        "other_prompts": "No cached detections for any other prompt on this clip; synonyms map to these "
                         "(helmet -> hard hat, vest -> safety vest, worker -> person).",
        "machine_state": f"Not detected from the video; the demo sets it by hand: forklift {info['machine_note']}.",
        "engine": {"checks": ["in_zone", "count_in_zone", "distance_lt (needs calibration)", "wearing",
                              "machine_state (manual)"],
                   "cannot_check": ["pose (lying/standing)", "height_above_floor", "speed", "sound", "text"],
                   "person_in_cab": "a person riding a forklift is excluded from distance_lt",
                   "zone_membership": "by the bottom middle of the box (where the object stands)"},
    }


def _clip_frames(rule: dict, clip: str):
    """Engine frames for a rule on the cached detections, the same way webapp.api_run builds them."""
    W = _web()
    info = W.clip_info(clip)
    dj = W.detections_for(clip, DETECTOR)
    size = tuple(dj["frame_size"])
    status = W.rule_status(rule, clip, DETECTOR)
    pmap = status["prompts"]
    mach = W.parse_machine_spec(info["machine"])
    mnames = W.machine_names(rule.get("condition"))
    frames, raw = [], []
    for stem in sorted(dj["frames"]):
        t = float(int(stem) - 1)
        rec = dj["frames"][stem]
        dets, shown = [], []
        for o in rule.get("objects", []):
            src = pmap.get(o["prompt"])
            for d in (rec.get(src, []) if src else []):
                dets.append(Detection(o["name"], d["box"], score=d.get("score"), track_id=d.get("track_id")))
                shown.append({"name": o["name"], "prompt": o["prompt"], "score": d.get("score"), "id": d.get("track_id"),
                              "box": [round(v, 4) for v in _norm(d["box"], size)]})
        frames.append((t, dets, size, {m: mach(t) for m in mnames} or None))
        raw.append(shown)
    return status, frames, raw, size


def run_rule_on_clip(rule: dict, clip: str) -> dict:
    """Deterministic: engine over the cached detections. Same result as the page's /api/run (without explanations)."""
    W = _web()
    status, frames, raw, size = _clip_frames(rule, clip)
    out = W.run_engine(rule, W.zones_for(clip), frames, "px", episodes=EPISODES)
    out["status"], out["raw"], out["frame_size"] = status, raw, size
    return out


# ------------------------------------------------------------------ session state

def new_session(clip: str, goal: str = "", max_steps: int = MAX_ROUNDS) -> dict:
    W = _web()
    info = W.clip_info(clip)
    goal = (goal or DEFAULT_GOAL).strip()[:400]
    return {
        "v": 1, "clip": clip, "goal": goal, "max_steps": max(1, min(int(max_steps), MAX_ROUNDS)),
        "messages": [{"role": "user", "content": f"Camera: {info['title']} (clip id {clip}).\nGoal: {goal}"}],
        "trace": [], "proposals": {}, "decisions": {}, "pending": [], "rounds": 0, "nudged": False, "closing": False,
        "usage": {"agent_calls": 0, "compile_calls": 0, "compile_cached": 0, "prompt_tokens": 0,
                  "completion_tokens": 0, "reasoning_tokens": 0},
        "done": False, "stop_reason": None, "started": time.time(), "model": None,
    }


def _trace(state, kind, summary, **kw):
    state["trace"].append({"i": len(state["trace"]) + 1, "round": state["rounds"], "kind": kind,
                           "summary": summary, **kw})


def _add_usage(state, entries):
    u = state["usage"]
    for e in entries:
        u["prompt_tokens"] += e.get("prompt_tokens") or 0
        u["completion_tokens"] += e.get("completion_tokens") or 0
        u["reasoning_tokens"] += e.get("reasoning_tokens") or 0


def cost_usd(usage: dict) -> float:
    return round((usage["prompt_tokens"] * PRICE_IN + usage["completion_tokens"] * PRICE_OUT) / 1e6, 4)


def _undecided(state):
    return [r for r, p in state["proposals"].items() if r not in state["decisions"]]


# ------------------------------------------------------------------ the agent call

def _assistant_dict(msg) -> dict:
    d = {"role": "assistant", "content": getattr(msg, "content", None)}
    tcs = getattr(msg, "tool_calls", None) or []
    if tcs:
        d["tool_calls"] = [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                           for tc in tcs]
    return d


def _model_step(state, client, cfg: LLMConfig):
    used = state["usage"]["completion_tokens"]
    left = MAX_OUTPUT_TOKENS - used
    if left < MIN_TOKENS_FOR_CALL:
        _finish(state, "token budget")
        return
    last_round = state["rounds"] + 1 >= state["max_steps"]
    if (last_round or left < CLOSING_TOKENS) and not state.get("closing"):
        # leave room for the decisions: the run must not end with rules nobody judged
        state["closing"] = True
        und = _undecided(state)
        state["messages"].append({"role": "user", "content":
                                  ("This is your last round" if last_round else "The output budget is nearly used")
                                  + ": decide every open rule now with accept_rule or reject_rule"
                                  + (f" ({', '.join(und)})" if und else "") + ". Do not compile new rules."})
    messages = [{"role": "system", "content": PROMPT}] + state["messages"]
    effort = os.environ.get("SM_SCOUT_REASONING", SCOUT_REASONING) or cfg.reasoning
    extra = {"reasoning_effort": effort} if effort else {}
    resp = client.chat.completions.create(model=cfg.model, messages=messages, tools=TOOLS, tool_choice="auto",
                                          max_tokens=min(MAX_TOKENS_PER_CALL, left), **extra)
    n0 = len(llm.USAGE)
    llm._record(resp, cfg, "tools")
    _add_usage(state, llm.USAGE[n0:n0 + 1])
    state["usage"]["agent_calls"] += 1
    state["rounds"] += 1
    state["model"] = cfg.model
    choice = resp.choices[0]
    msg = _assistant_dict(choice.message)
    calls = msg.get("tool_calls", [])
    if len(calls) > MAX_CALLS_PER_ROUND:
        calls = calls[:MAX_CALLS_PER_ROUND]
        msg["tool_calls"] = calls
    state["messages"].append(msg)
    text = (msg.get("content") or "").strip()
    if text:
        _trace(state, "note", text[:600])
    if calls:
        state["pending"] = calls
        return
    if choice.finish_reason == "length":
        _finish(state, "output limit of one call")
        return
    und = _undecided(state)
    if und and not state["nudged"] and state["rounds"] < state["max_steps"]:
        state["nudged"] = True
        state["messages"].append({"role": "user", "content":
                                  f"Rules without a decision: {', '.join(und)}. Run, inspect and accept or "
                                  "reject each one with a tool call."})
        return
    _finish(state, "model finished" if not und else "model stopped with undecided rules")


def _finish(state, reason):
    state["done"] = True
    state["stop_reason"] = reason
    state["pending"] = []
    _trace(state, "stop", reason)


# ------------------------------------------------------------------ tools

def _short_rule(rule: dict) -> dict:
    return {k: rule.get(k) for k in ("title", "objects", "zones", "condition", "min_duration_s", "severity",
                                     "unsupported", "suggested_rewrite")}


def default_compile(clip: str, client, cfg) -> Callable[[str, str], tuple[dict, bool]]:
    """(text, rule_ref) -> (rule, cached). Uses the page's compile cache (demo/cache/compile.json) first."""
    W = _web()

    def compile_fn(text, ref):
        key = W.compile_key(text, clip)
        hit = W.COMPILE_CACHE.get(key)
        if hit:
            rule = W.cached_rule(hit, clip)
            rule["rule_id"] = ref
            return rule, True
        from .compiler import compile_rule
        rule = compile_rule(text, W.zones_for(clip)["zones"], cfg, client=client, rule_id=ref)
        W.COMPILE_CACHE.put(key, {"text": text, "clip": clip, "rule": rule})
        return copy.deepcopy(rule), False
    return compile_fn


def _tool_compile(state, args):
    """Reserve a rule_ref for a compile call. Returns (ref, None) or (None, (error result, summary))."""
    text = " ".join(str(args.get("text") or "").split())[:400]
    if not text:
        return None, ({"error": "text is empty"}, "compile: empty text")
    if state.get("closing"):
        return None, ({"error": "no new rules now: decide the open ones with accept_rule or reject_rule"},
                      "compile refused: closing round")
    if len(state["proposals"]) >= MAX_PROPOSALS:
        return None, ({"error": f"at most {MAX_PROPOSALS} rules per session"}, "compile refused: proposal limit")
    ref = f"p{len(state['proposals']) + 1}"
    revises = args.get("revises") or None
    if revises not in state["proposals"]:
        revises = None
    state["proposals"][ref] = {"text": text, "reason": " ".join(str(args.get("reason") or "").split())[:300],
                               "revises": revises, "rule": None, "error": None, "cached": False, "run": None,
                               "inspected": []}
    return ref, None


def _compile_result(state, ref, compile_fn):
    prop = state["proposals"][ref]
    try:
        rule, cached = compile_fn(prop["text"], ref)
        prop["rule"], prop["cached"] = rule, cached
    except Exception as e:  # CompileError, API errors: the agent sees the error and may reword
        prop["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return prop


def _compiled_view(state, ref):
    prop = state["proposals"][ref]
    if prop["error"]:
        return {"rule_ref": ref, "error": prop["error"]}
    rule = prop["rule"]
    st = _web().rule_status(rule, state["clip"], DETECTOR)
    missing = [p for p, v in (st.get("prompts") or {}).items() if not v]
    can_run = bool(rule.get("condition")) and not st["rejected"] and not st["not_evaluable"] and not missing
    why_not = (None if can_run else
               ("unsupported: " + (rule.get("unsupported") or "")) if not rule.get("condition") else
               st["rejected"] or st["not_evaluable"] or f"no detections on this clip for {', '.join(missing)}")
    return {"rule_ref": ref, "rule": _short_rule(rule), "can_run_on_this_camera": can_run, "why_not": why_not,
            "prompt_mapping": st.get("prompts"), "machine_note": st.get("machine")}


def _event_view(ref, i, ev, zones, size):
    objs = []
    for o in ev.get("objects", []):
        objs.append({"name": o["name"], "id": o.get("track_id"), "where": position_words(o["box"], size, zones),
                     "score": o.get("score")})
    m = {k: v for k, v in (ev.get("measurements") or {}).items() if k in
         ("duration_s", "min_distance_m", "max_count", "missing_ppe", "max_person_count", "objects_seen")}
    return {"event_id": f"{ref}.e{i}", "raised_at_s": ev["start_s"], "condition_since_s": ev["condition_since_s"],
            "last_true_s": ev["end_s"], "subject": ev.get("subject") or None, "objects": objs, "measurements": m}


def _tool_run(state, args):
    ref = str(args.get("rule_ref") or "")
    prop = state["proposals"].get(ref)
    if not prop:
        return {"error": f"unknown rule_ref {ref!r}; known: {', '.join(state['proposals']) or 'none'}"}, f"run {ref}: unknown"
    if not prop["rule"]:
        return {"error": f"{ref} did not compile: {prop['error']}"}, f"run {ref}: not compiled"
    view = _compiled_view(state, ref)
    rule = prop["rule"]
    if not view["can_run_on_this_camera"]:
        prop["run"] = {"ran": False, "why_not": view["why_not"], "alerts": 0}
        return {"rule_ref": ref, "ran": False, "why_not": view["why_not"]}, f"{ref} cannot run: {view['why_not']}"
    out = run_rule_on_clip(rule, state["clip"])
    zones = _web().zones_for(state["clip"])
    zl = [z for z in zones["zones"] if z["name"] in rule.get("zones", [])]
    evs = [_event_view(ref, i, e, zl, out["frame_size"]) for i, e in enumerate(out["events"], 1)]
    tl = out["timeline"]
    prop["run"] = {"ran": True, "alerts": len(evs), "true_s": tl.count("T"), "seconds": len(tl), "timeline": tl,
                   "events": [{"event_id": e["event_id"], "start_s": e["raised_at_s"], "end_s": e["last_true_s"],
                               "since_s": e["condition_since_s"], "subject": e["subject"]} for e in evs]}
    res = {"rule_ref": ref, "ran": True, "clip_length_s": len(tl), "seconds_condition_true": tl.count("T"),
           "seconds_not_checkable": tl.count("?"), "timeline": tl, "alerts": len(evs), "alert_list": evs[:10]}
    if len(evs) > 10:
        res["note"] = f"{len(evs) - 10} more alerts not listed"
    s = f"{ref} ran: {len(evs)} alert{'s' if len(evs) != 1 else ''}, condition true {tl.count('T')} of {len(tl)} s"
    return res, s


def _find_event(state, event_id):
    ref, _, num = str(event_id).partition(".e")
    prop = state["proposals"].get(ref)
    if not prop or not prop.get("rule") or not num.isdigit():
        return None, None, None
    out = run_rule_on_clip(prop["rule"], state["clip"])    # deterministic, a few ms
    i = int(num)
    if not 1 <= i <= len(out["events"]):
        return ref, None, out
    return ref, out["events"][i - 1], out


def inspect_facts(rule: dict, clip: str, out: dict, ev: dict, second: int) -> dict:
    W = _web()
    zf = W.zones_for(clip)
    size = out["frame_size"]
    H = Homography.from_calibration(zf.get("floor_calibration"))
    row = out["seconds"][second]
    hits = {tuple(round(v, 3) for v in h["box"]) for h in row.get("hits", [])}
    dets = []
    for d in out["raw"][second]:
        z_in = [z.get("title") or z["name"] for z in zf["zones"] if point_in_polygon(foot_point(d["box"]), z["polygon"])]
        dets.append({"object": d["name"], "id": d.get("id"), "prompt": d["prompt"], "score": d["score"],
                     "where": position_words(d["box"], size), "standing_in_zones": z_in,
                     "matched_by_engine": tuple(round(v, 3) for v in d["box"]) in hits,
                     "box_norm": [round(v, 3) for v in d["box"]]})
    people = [d for d in out["raw"][second] if person_like(d["name"]) or person_like(d["prompt"])]
    machines = [d for d in out["raw"][second] if any(k in d["prompt"].lower() for k in MACHINE_WORDS)]
    ppe = [d for d in out["raw"][second] if any(k in d["prompt"].lower() for k in PPE_WORDS)]
    pairs = []
    for i, p in enumerate(people):
        for j, m in enumerate(machines):
            pr = {"person": _label(p, i), "machine": _label(m, j), "operator_in_cab": is_operator(p["box"], m["box"])}
            if H is not None:
                pr["floor_distance_m"] = round(distance_m(p["box"], m["box"], H), 2)
            pairs.append(pr)
    ppe_on = []
    for i, p in enumerate(people):
        worn = {}
        for d in ppe:
            share = mask_overlap(p["box"], d["prompt"], d["box"])[1]
            if share > 0:
                worn[d["prompt"]] = max(worn.get(d["prompt"], 0), round(share, 2))
        ppe_on.append({"person": _label(p, i), "ppe_overlap_share": worn or "none (no PPE box on this person)",
                       "needed_share": 0.3})
    # condition and detection counts around the second: flicker shows up here
    lo, hi = max(0, second - 3), min(len(out["seconds"]) - 1, second + 3)
    around = []
    for s in range(lo, hi + 1):
        cnt = {}
        for d in out["raw"][s]:
            cnt[d["name"]] = cnt.get(d["name"], 0) + 1
        around.append({"s": s, "condition": out["seconds"][s]["truth"], "detections": cnt})
    m = (ev.get("measurements") or {})
    return {"second": second, "alert": {"raised_at_s": ev["start_s"], "condition_since_s": ev["condition_since_s"],
                                        "last_true_s": ev["end_s"], "held_s": m.get("duration_s"),
                                        "min_duration_s": rule.get("min_duration_s")},
            "detections": dets, "person_machine_pairs": pairs or None, "ppe_per_person": ppe_on if ppe else None,
            "neighbouring_seconds": around,
            "note": "#n is the SAM 3.1 track id: the same id in another second is the same object"}


def _label(d, i):
    """'person #3' by track id; without an id the position in the detections list."""
    return f"{d['name']} #{d['id']}" if d.get("id") is not None else i + 1


def _tool_inspect(state, args):
    eid = str(args.get("event_id") or "")
    ref, ev, out = _find_event(state, eid)
    if ev is None:
        known = [e["event_id"] for p in state["proposals"].values() if p.get("run") for e in p["run"].get("events", [])]
        return {"error": f"unknown event_id {eid!r}; known: {', '.join(known[:20]) or 'none (run a rule first)'}"}, \
            f"inspect {eid}: unknown"
    sec = args.get("second")
    if not isinstance(sec, int) or not ev["condition_since_s"] <= sec <= ev["end_s"]:
        sec = int(ev["start_s"])
    facts = inspect_facts(state["proposals"][ref]["rule"], state["clip"], out, ev, sec)
    state["proposals"][ref]["inspected"].append({"event_id": eid, "second": sec})
    matched = [d for d in facts["detections"] if d["matched_by_engine"]]
    bits = [f"{d['object']} {d['score']} {d['where']}" for d in matched[:3]]
    if facts["person_machine_pairs"]:
        ds = [p.get("floor_distance_m") for p in facts["person_machine_pairs"] if p.get("floor_distance_m") is not None]
        if ds:
            bits.append(f"closest person-machine {min(ds)} m")
    return facts, f"inspected {eid} at s{sec}: " + ("; ".join(bits) or "no matched objects")


def _tool_decide(state, args, decision):
    ref = str(args.get("rule_ref") or "")
    reason = " ".join(str(args.get("reason") or "").split())[:400]
    prop = state["proposals"].get(ref)
    if not prop:
        return {"error": f"unknown rule_ref {ref!r}"}, f"{decision} {ref}: unknown"
    if ref in state["decisions"]:
        return {"error": f"{ref} already has a decision: {state['decisions'][ref]['decision']}"}, \
            f"{decision} {ref}: already decided"
    if decision == "accept":
        run = prop.get("run")
        if not run or not run.get("ran"):
            return {"error": f"{ref} has not run on this clip; call run_rule first (the engine decides what fires)"}, \
                f"accept {ref} refused: not run"
        if run["alerts"] and not prop["inspected"]:
            return {"error": f"{ref} raised {run['alerts']} alerts; inspect at least one with inspect_alert first"}, \
                f"accept {ref} refused: no alert inspected"
    state["decisions"][ref] = {"decision": decision, "reason": reason}
    return {"rule_ref": ref, "decision": decision, "undecided": _undecided(state)}, f"{decision}ed {ref}: {reason}"


def _exec_tools(state, compile_fn):
    calls = state["pending"]
    state["pending"] = []
    parsed = []
    for tc in calls:
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
            if not isinstance(args, dict):
                args = {}
        except json.JSONDecodeError:
            args = {"_bad_json": tc["function"].get("arguments")}
        parsed.append((tc, tc["function"]["name"], args))
    # compiles first and in parallel (each is one Spark call unless cached)
    refs = {}
    for k, (tc, name, args) in enumerate(parsed):
        if name == "compile_rule":
            ref, err = _tool_compile(state, args)
            refs[k] = ref or err
    todo = [r for r in refs.values() if isinstance(r, str)]
    n0 = len(llm.USAGE)
    if todo:
        with ThreadPoolExecutor(min(4, len(todo))) as ex:
            list(ex.map(lambda r: _compile_result(state, r, compile_fn), todo))
    new_usage = llm.USAGE[n0:]
    _add_usage(state, new_usage)
    for r in todo:
        if state["proposals"][r]["cached"]:
            state["usage"]["compile_cached"] += 1
    state["usage"]["compile_calls"] += len(new_usage)
    # revisions close the rule they replace
    for r in todo:
        old = state["proposals"][r]["revises"]
        if old and old not in state["decisions"] and state["proposals"][r]["rule"]:
            state["decisions"][old] = {"decision": "revised", "reason": f"replaced by {r}: {state['proposals'][r]['reason']}",
                                       "by": r}
    for k, (tc, name, args) in enumerate(parsed):
        t0 = time.time()
        try:
            if name == "get_camera_facts":
                res, summary = camera_facts(state["clip"]), "read the camera facts"
                kind = "facts"
                f = res
                summary = (f"camera facts: {f['clip_length_s']} s, zones "
                           + ", ".join(z["title"] for z in f["zones"])
                           + f"; calibration {'yes' if f['floor_calibration'].startswith('yes') else 'no'}; prompts "
                           + ", ".join(f"{p} {v['seconds_seen']}/{f['clip_length_s']} s"
                                       for p, v in f["sam_prompts_with_detections"].items()))
            elif name == "compile_rule":
                r = refs[k]
                kind = "propose"
                if isinstance(r, str):
                    res = _compiled_view(state, r)
                    p = state["proposals"][r]
                    if p["error"]:
                        summary = f"{r} did not compile: {p['error']}"
                    elif not p["rule"].get("condition"):
                        summary = f"{r} unsupported: {p['rule'].get('unsupported')}"
                    else:
                        summary = f"{r} compiled: {p['rule'].get('title')}" + \
                                  ("" if res["can_run_on_this_camera"] else f" (cannot run here: {res['why_not']})")
                    if p["revises"]:
                        kind = "revise"
                else:
                    res, summary = r
            elif name == "run_rule":
                res, summary = _tool_run(state, args)
                kind = "run"
            elif name == "inspect_alert":
                res, summary = _tool_inspect(state, args)
                kind = "inspect"
            elif name in ("accept_rule", "reject_rule"):
                kind = name.split("_")[0]
                res, summary = _tool_decide(state, args, kind)
                if "error" in res:
                    kind = "refused"
            else:
                res, summary, kind = {"error": f"unknown tool {name}"}, f"unknown tool {name}", "error"
        except Exception as e:  # a tool bug must not kill the session; the model sees the error
            log.exception("tool %s failed", name)
            res, summary, kind = {"error": f"{type(e).__name__}: {str(e)[:300]}"}, f"{name} failed: {e}", "error"
        entry = {"tool": name, "args": {a: v for a, v in args.items() if a != "_bad_json"}}
        if kind in ("propose", "revise") and isinstance(refs.get(k), str):
            p = state["proposals"][refs[k]]
            entry.update(rule_ref=refs[k], text=p["text"], reason=p["reason"], revises=p["revises"],
                         cached=p["cached"], rule=_short_rule(p["rule"]) if p["rule"] else None,
                         can_run=res.get("can_run_on_this_camera"), why_not=res.get("why_not"))
        elif kind == "run" and "error" not in res:
            entry.update(rule_ref=res.get("rule_ref"), ran=res.get("ran"), alerts=res.get("alerts"),
                         timeline=res.get("timeline"), why_not=res.get("why_not"),
                         events=[{"event_id": e["event_id"], "start_s": e["raised_at_s"], "end_s": e["last_true_s"],
                                  "subject": e.get("subject")} for e in res.get("alert_list", [])])
        elif kind == "inspect" and "error" not in res:
            entry.update(event_id=args.get("event_id"), second=res["second"],
                         detections=[{k2: d[k2] for k2 in ("object", "id", "score", "where", "standing_in_zones",
                                                            "matched_by_engine")} for d in res["detections"]],
                         pairs=res["person_machine_pairs"], ppe=res["ppe_per_person"])
        elif kind in ("accept", "reject"):
            entry.update(rule_ref=args.get("rule_ref"), reason=state["decisions"][args["rule_ref"]]["reason"])
        entry["ms"] = int((time.time() - t0) * 1000)
        _trace(state, kind, summary, **entry)
        state["messages"].append({"role": "tool", "tool_call_id": tc["id"],
                                  "content": json.dumps(res, ensure_ascii=False, separators=(",", ":"))})
    if state["proposals"] and not _undecided(state):
        _finish(state, "every rule has a decision")
    elif state["rounds"] >= state["max_steps"]:
        _finish(state, f"step limit ({state['max_steps']} rounds)")
    elif state["usage"]["completion_tokens"] >= MAX_OUTPUT_TOKENS:
        _finish(state, "token budget")


def step(state: dict, client=None, cfg: Optional[LLMConfig] = None, compile_fn=None) -> dict:
    """Advance the session by one step (one model call, or the tool calls of the last model call). In place."""
    if state["done"]:
        return state
    cfg = cfg or config_from_env()
    if state["pending"]:
        if compile_fn is None:
            compile_fn = default_compile(state["clip"], client or make_client(cfg), cfg)
        _exec_tools(state, compile_fn)
    else:
        _model_step(state, client or make_client(cfg), cfg)
    return state


# ------------------------------------------------------------------ result

@dataclass
class ScoutResult:
    clip: str
    goal: str
    trace: list
    rules: list                     # accepted
    rejected: list                  # rejected and revised (replaced)
    undecided: list
    usage: dict
    cost_usd: float
    stop_reason: str
    rounds: int
    model: Optional[str]
    duration_s: float
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))

    def to_dict(self) -> dict:
        return asdict(self)


def result_of(state: dict) -> ScoutResult:
    acc, rej, und = [], [], []
    for ref, p in state["proposals"].items():
        d = state["decisions"].get(ref)
        item = {"rule_ref": ref, "text": p["text"], "proposal_reason": p["reason"], "revises": p["revises"],
                "rule": p["rule"], "compile_error": p["error"],
                "alerts": (p.get("run") or {}).get("alerts"), "events": (p.get("run") or {}).get("events"),
                "timeline": (p.get("run") or {}).get("timeline"),
                "decision": d["decision"] if d else None, "reason": d["reason"] if d else None}
        (acc if d and d["decision"] == "accept" else rej if d else und).append(item)
    return ScoutResult(state["clip"], state["goal"], state["trace"], acc, rej, und, dict(state["usage"]),
                       cost_usd(state["usage"]), state["stop_reason"] or "", state["rounds"], state.get("model"),
                       round(time.time() - state["started"], 1))


def run_scout(clip: str, goal_text: str = "", max_steps: int = MAX_ROUNDS, client_config: Optional[LLMConfig] = None,
              *, client=None, compile_fn=None, on_step: Optional[Callable[[dict], None]] = None) -> ScoutResult:
    cfg = client_config or config_from_env()
    client = client or make_client(cfg)
    state = new_session(clip, goal_text, max_steps)
    compile_fn = compile_fn or default_compile(clip, client, cfg)
    for _ in range(4 * MAX_ROUNDS + 4):          # hard stop, the session itself ends far earlier
        if state["done"]:
            break
        step(state, client, cfg, compile_fn)
        if on_step:
            on_step(state)
    if not state["done"]:
        _finish(state, "loop guard")
    return result_of(state)


# ------------------------------------------------------------------ signed state for the stateless web demo

def _secret() -> bytes:
    base = os.environ.get("SM_SCOUT_SECRET") or os.environ.get("SM_LLM_API_KEY") or "no-key"
    return hashlib.sha256(("rule-scout:" + base).encode()).digest()


def _canon(x):
    """JSON as the browser sends it back: JS turns 2.0 into 2, so whole floats are signed as ints."""
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_canon(v) for v in x]
    return x


def sign_state(state: dict) -> str:
    body = json.dumps(_canon(state), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hmac.new(_secret(), body, hashlib.sha256).hexdigest()


def verify_state(state: dict, sig: str) -> bool:
    return isinstance(sig, str) and hmac.compare_digest(sign_state(state), sig)


def public_view(state: dict) -> dict:
    """What the page needs: trace and outcome, without the conversation."""
    r = result_of(state)
    d = r.to_dict()
    d["done"] = state["done"]
    return d


# ------------------------------------------------------------------ CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description="Rule Scout: Muse Spark proposes and verifies rules for one clip.")
    ap.add_argument("clip", help="forklift-operator | warehouse-team | busy-warehouse")
    ap.add_argument("--goal", default="")
    ap.add_argument("--max-steps", type=int, default=MAX_ROUNDS)
    ap.add_argument("--save", action="store_true", help=f"write {CACHE_DIR}/<clip>.json")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    def show(st):
        for t in st["trace"][show.n:]:
            print(f"[{t['round']}] {t['kind']:8s} {t['summary']}", flush=True)
        show.n = len(st["trace"])
    show.n = 0
    res = run_scout(a.clip, a.goal, a.max_steps, on_step=show)
    u = res.usage
    print(f"\naccepted {len(res.rules)}, rejected/revised {len(res.rejected)}, undecided {len(res.undecided)}; "
          f"stop: {res.stop_reason}; rounds {res.rounds}; agent calls {u['agent_calls']}, compile calls "
          f"{u['compile_calls']} (cached {u['compile_cached']}); tokens in {u['prompt_tokens']} out "
          f"{u['completion_tokens']} (reasoning {u['reasoning_tokens']}); ~${res.cost_usd}; {res.duration_s} s")
    if a.save:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = CACHE_DIR / f"{a.clip}.json"
        p.write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("saved", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

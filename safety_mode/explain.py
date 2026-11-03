"""Alert explanation: one paragraph from structured event data only, no images.

explain_event(event, rule, lang)          -> str (Muse Spark; falls back to the template if the API fails)
explain_event_result(...)                 -> dict with explanation, severity_check, source, errors
explain_event_template(event, rule)       -> str, deterministic, from rule.explanation_template

The event may be in report-input shape (schemas/report-input.schema.json: id, time, duration_s,
objects[{name, box}], measurements{distance_m, count, height_m}) or an engine Event as a dict
(start_s, end_s, objects with normalised boxes, explanation_inputs). Pass the camera's zones file
to get "inside <zone>" in the object positions.

CLI: python -m safety_mode.explain <event.json> <rule.json> [--zones zones.json] [--lang en] [--template]
"""
import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import openai
from jsonschema import Draft202012Validator

from .facts import duration_words, lang_name, numbers_in, parse_json, person_like, position_words
from .llm import LLMConfig, chat_json, config_from_env, make_client

log = logging.getLogger("safety_mode.explain")

PROMPT = (Path(__file__).resolve().parent / "prompts" / "explain.md").read_text(encoding="utf-8")
SEVERITIES = ["low", "medium", "high", "critical"]
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string", "description": "one paragraph, 1-3 sentences, only the given facts"},
        "severity_check": {"type": "string", "enum": SEVERITIES,
                           "description": "how serious this event is judged from the facts"},
    },
    "required": ["explanation", "severity_check"],
    "additionalProperties": False,
}
VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)
MAX_WORDS = 80


class ExplainError(Exception):
    def __init__(self, msg, errors=None, raw=None):
        super().__init__(msg)
        self.errors = errors or []
        self.raw = raw


def _zones_list(zones):
    """zones file dict {frame_size, zones:[...]} or a list of zone dicts -> (list, frame_size)."""
    if not zones:
        return [], None
    if isinstance(zones, dict):
        fs = zones.get("frame_size")
        return zones.get("zones", []), tuple(fs) if fs else None
    return list(zones), None


def _measurements(event):
    """Non-null measurements with units in the key names."""
    m = event.get("measurements") or {}
    ei = event.get("explanation_inputs") or {}
    out = {}
    # engine events carry the last second's distance_m/count and the episode's min/max:
    # the text must quote the closest approach, not where the episode happened to end
    dist = m.get("min_distance_m", m.get("distance_m", ei.get("distance_m")))
    count = m.get("max_count", m.get("count", ei.get("count")))
    height = m.get("height_m", ei.get("height_m"))
    if dist is not None:
        out["min_floor_distance_m"] = round(float(dist), 2)
    if count is not None:
        out["count_in_zone"] = count
    if height is not None:
        out["height_above_floor_m"] = round(float(height), 2)
    missing = m.get("missing_ppe") or ei.get("missing_ppe")
    if missing:
        out["missing_ppe"] = missing if isinstance(missing, str) else ", ".join(str(p).replace("_", " ") for p in missing)
    return out


def _time_and_duration(event):
    ei = event.get("explanation_inputs") or {}
    t = event.get("time") or ei.get("time")
    d = event.get("duration_s", ei.get("duration_s"))
    if t is None and "condition_since_s" in event:
        s = int(round(event["condition_since_s"]))
        t = f"{s // 60:02d}:{s % 60:02d}"
    if d is None and "end_s" in event:
        d = event["end_s"] - event.get("condition_since_s", event.get("start_s", 0)) + 1
    return t, d


def build_facts(event, rule, zones=None, frame_size=None):
    """The only input the model sees. Everything in the explanation must come from here."""
    zone_list, fs = _zones_list(zones)
    frame_size = frame_size or fs
    zone_titles = {z["name"]: z.get("title") or z["name"] for z in zone_list}
    t, d = _time_and_duration(event)
    # where the objects were when the alert went up (the time the text gives), not at the episode's
    # last frame: a worker crossing right to left was "on the left side" at the end of A, right at 00:07
    objs = event.get("objects_at_start") or event.get("objects") or []
    counts = Counter(o["name"] for o in objs)
    facts = {
        "rule": {"id": rule.get("rule_id") or event.get("rule_id"), "title": rule.get("title"),
                 "severity": rule.get("severity"), "min_duration_s": rule.get("min_duration_s"),
                 "zones": [zone_titles.get(z, z) for z in rule.get("zones", [])],
                 "condition": rule.get("condition")},
        "event": {"id": event.get("id") or (f"event {event['event_id']}" if event.get("event_id") else None),
                  "time": t, "duration_s": d, "duration": duration_words(d) if d is not None else None,
                  "severity": event.get("severity") or rule.get("severity")},
        "objects": [_object_fact(o, frame_size, zone_list) for o in objs if o.get("box")],
        "object_counts": {k.replace("_", " "): v for k, v in counts.items()},
        "measurements": _measurements(event),
    }
    if not facts["event"]["id"]:
        del facts["event"]["id"]
    # tracked episode (SAM 3.1 video mode): the objects it is about, "person #2"
    if event.get("subject"):
        facts["event"]["subject"] = event["subject"]
        facts["event"]["object_ids"] = [f"{o['name'].replace('_', ' ')} #{o['track_id']}"
                                        for o in event.get("object_ids") or []]
    return facts


def _object_fact(o, frame_size, zone_list):
    out = {"name": o["name"].replace("_", " "), "position": position_words(o["box"], frame_size, zone_list)}
    if o.get("track_id") is not None:
        out["label"] = f"{out['name']} #{o['track_id']}"
    return out


def check_explanation(text, facts):
    """Guards on top of the schema: length, one paragraph, no numbers that are not in the facts."""
    errs = []
    if not text.strip():
        return ["explanation is empty"]
    if "\n\n" in text.strip():
        errs.append("write one paragraph, no blank lines")
    if len(text.split()) > MAX_WORDS:
        errs.append(f"too long: {len(text.split())} words, limit {MAX_WORDS}")
    allowed = numbers_in(json.dumps(facts, ensure_ascii=False))
    allowed |= numbers_in(re.sub(r"\D", " ", str(facts["rule"]["id"] or "")))  # "Rule 3" for r3
    extra = numbers_in(text) - allowed
    if extra:
        errs.append(f"numbers {sorted(extra)} are not in the facts; use only the numbers given")
    return errs


def explain_event_result(event, rule, lang="en", *, zones=None, frame_size=None,
                         client_config: LLMConfig | None = None, client=None, fallback=True):
    """Returns {"explanation", "severity_check", "source": "model"|"template", "errors"}."""
    facts = build_facts(event, rule, zones, frame_size)
    cfg = client_config or config_from_env()
    system = PROMPT.replace("{{LANG}}", lang_name(lang))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Facts:\n```json\n" + json.dumps(facts, ensure_ascii=False, indent=1) + "\n```"}]
    raw, errs = None, []
    try:
        client = client or make_client(cfg)
        for attempt in (1, 2):
            raw, mode = chat_json(client, cfg, messages, OUTPUT_SCHEMA, schema_name="alert_explanation")
            try:
                out = parse_json(raw)
                errs = [f"{e.json_path}: {e.message[:200]}" for e in VALIDATOR.iter_errors(out)]
                if not errs:
                    errs = check_explanation(out["explanation"], facts)
            except (json.JSONDecodeError, TypeError) as e:
                errs = [f"not valid JSON: {e}"]
            log.info("explain %s mode=%s attempt=%d %s", facts["rule"]["id"], mode, attempt,
                     "ok" if not errs else f"errors: {errs}")
            if not errs:
                if out["severity_check"] != facts["event"]["severity"]:
                    log.info("severity_check %s differs from rule severity %s",
                             out["severity_check"], facts["event"]["severity"])
                return {"explanation": out["explanation"].strip(), "severity_check": out["severity_check"],
                        "source": "model", "errors": []}
            messages += [{"role": "assistant", "content": raw or ""},
                         {"role": "user", "content": "Your answer is invalid:\n- " + "\n- ".join(errs)
                          + "\nReturn the corrected JSON object only."}]
    except (openai.APIError, RuntimeError) as e:
        errs = [f"API: {type(e).__name__}: {str(e)[:200]}"]
        log.warning("explain %s: model unavailable (%s)", facts["rule"]["id"], errs[0])
    if not fallback:
        raise ExplainError(f"no valid explanation: {errs}", errs, raw)
    return {"explanation": explain_event_template(event, rule), "severity_check": facts["event"]["severity"],
            "source": "template", "errors": errs}


def explain_event(event, rule, lang="en", **kw) -> str:
    """One paragraph for the alert. Falls back to explain_event_template when the model fails."""
    return explain_event_result(event, rule, lang, **kw)["explanation"]


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def explain_event_template(event, rule) -> str:
    """Deterministic: fill rule.explanation_template. Unknown placeholders get neutral words."""
    ei = event.get("explanation_inputs") or {}
    t, d = _time_and_duration(event)
    m = _measurements(event)
    objs = event.get("objects") or []
    persons = sum(1 for o in objs if person_like(o["name"]))
    others = sorted({o["name"].replace("_", " ") for o in objs if not person_like(o["name"])})
    values = {
        "time": t, "duration_s": d,
        "person_count": ei.get("person_count") or persons or None,
        "missing_ppe": m.get("missing_ppe") or "required PPE",
        "distance_m": m.get("min_floor_distance_m"),
        "count": m.get("count_in_zone"),
        "height_m": m.get("height_above_floor_m"),
        "object": ei.get("object") or (", ".join(others) if others else "Object"),
    }
    tpl = rule.get("explanation_template") or "{object}: rule \"" + (rule.get("title") or "?") + "\" at {time}."

    def sub(mt):
        v = values.get(mt[1])
        return "unknown" if v is None or v == "" else _fmt(v)

    text = re.sub(r"\{(\w+)\}", sub, tpl)
    return text[:1].upper() + text[1:]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m safety_mode.explain", description=__doc__.split("\n")[0])
    ap.add_argument("event", help="event JSON (report-input event or engine Event dict)")
    ap.add_argument("rule", help="rule JSON (rules/rNN.json)")
    ap.add_argument("--zones", help="zones file of the camera (positions get 'inside <zone>')")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--template", action="store_true", help="deterministic template only, no API call")
    ap.add_argument("--json", action="store_true", help="print the full result dict")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    event = json.loads(Path(a.event).read_text(encoding="utf-8"))
    rule = json.loads(Path(a.rule).read_text(encoding="utf-8"))
    zones = json.loads(Path(a.zones).read_text(encoding="utf-8")) if a.zones else None
    if a.template:
        res = {"explanation": explain_event_template(event, rule), "source": "template"}
    else:
        res = explain_event_result(event, rule, a.lang, zones=zones)
    print(json.dumps(res, ensure_ascii=False, indent=1) if a.json else res["explanation"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

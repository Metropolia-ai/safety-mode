"""Daily safety report (report-template.md) from the structured events of one day, no images.

daily_report(report_input, lang)         -> markdown str (Muse Spark)
daily_report_result(...)                 -> dict with markdown, patterns, suggestions, digest

The input is validated against schemas/report-input.schema.json. Counts, groups, end times and the weekday
are computed here, not by the model. The answer is checked: every high/critical event id is in the text,
no event id, rule number or clock time that is not in the input, no legal wording, one screen long.

CLI: python -m safety_mode.report <report-input.json> [--lang en] [--out report.md]
"""
import argparse
import datetime as dt
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import openai
from jsonschema import Draft202012Validator

from .facts import DEFAULT_FRAME, add_seconds, duration_words, lang_name, parse_json, position_words
from .llm import LLMConfig, chat_json, config_from_env, make_client

log = logging.getLogger("safety_mode.report")

ROOT = Path(__file__).resolve().parent.parent
INPUT_SCHEMA = json.loads((ROOT / "schemas" / "report-input.schema.json").read_text(encoding="utf-8"))
INPUT_VALIDATOR = Draft202012Validator(INPUT_SCHEMA)
PROMPT = (Path(__file__).resolve().parent / "prompts" / "report.md").read_text(encoding="utf-8")
SEV_ORDER = ["critical", "high", "medium", "low"]
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "markdown": {"type": "string", "description": "the report in the format of report-template.md"},
        "patterns": {"type": "array", "items": {"type": "string"},
                     "description": "repeat patterns found in the events (time, place, rule)"},
        "suggestions": {"type": "array", "items": {"type": "string"}, "description": "one or two practical actions"},
    },
    "required": ["markdown", "patterns", "suggestions"],
    "additionalProperties": False,
}
OUTPUT_VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)
MAX_WORDS = 450  # one screen
LEGAL = re.compile(r"\b(laws?|legal(?:ly)?|illegal(?:ly)?|liab\w*|negligen\w*|regulations?|regulatory|"
                   r"complian\w*|non-complian\w*|penalt\w*|lawsuits?|prosecut\w*|culpab\w*)\b", re.I)
ID_LIKE = re.compile(r"\b([A-Za-z]+)-(\d+)\b")
RULE_REF = re.compile(r"\b(?:[Rr]ule|[Ss]ääntö)\s+(\d+)\b|\b[Rr](\d+)\b")
CLOCK = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::\d{2})?(?![\d:])")


class ReportInputError(ValueError):
    pass


class ReportError(Exception):
    def __init__(self, msg, errors=None, raw=None):
        super().__init__(msg)
        self.errors = errors or []
        self.raw = raw


def validate_input(data):
    errs = [f"{e.json_path}: {e.message[:200]}" for e in INPUT_VALIDATOR.iter_errors(data)]
    if errs:
        raise ReportInputError("report input does not match report-input.schema.json: " + "; ".join(errs))


def _rule_num(rule_id):
    m = re.search(r"\d+", rule_id or "")
    return int(m[0]) if m else None


def build_digest(data, frame_size=None):
    """Everything the model may use. Numbers here are computed in code, so the model does not count."""
    fs = frame_size or DEFAULT_FRAME
    titles = {r["id"]: r["title"] for r in data["rules"]}
    evs = sorted(data["events"], key=lambda e: e["time"])
    events = []
    for e in evs:
        end = e.get("end_time") or add_seconds(e["time"], e["duration_s"])
        events.append({
            "id": e["id"], "time": e["time"], "end_time": end, "rule_id": e["rule_id"],
            "rule_number": _rule_num(e["rule_id"]), "rule_title": titles.get(e["rule_id"], e["rule_id"]),
            "severity": e["severity"], "duration_s": e["duration_s"], "duration": duration_words(e["duration_s"]),
            "objects": [{"name": o["name"].replace("_", " "), "position": position_words(o["box"], fs)}
                        for o in e["objects"]],
            "measurements": {k: v for k, v in e["measurements"].items() if v is not None},
            "frame": e["frame"], "explanation": e["explanation"]})
    by_sev = Counter(e["severity"] for e in evs)
    groups = []
    for sev in ("medium", "low"):
        per_rule = {}
        for e in events:
            if e["severity"] == sev:
                per_rule.setdefault(e["rule_id"], []).append(e)
        for rid, lst in per_rule.items():
            groups.append({"severity": sev, "rule_id": rid, "rule_number": _rule_num(rid),
                           "rule_title": titles.get(rid, rid), "events": len(lst), "ids": [x["id"] for x in lst],
                           "first_time": lst[0]["time"], "last_time": lst[-1]["time"],
                           "last_end_time": lst[-1]["end_time"]})
    try:
        weekday = dt.date.fromisoformat(data["date"]).strftime("%a")
    except ValueError:
        weekday = None
    up = data["uptime_min"]
    return {
        "site": data["site"], "camera": data["camera"], "date": data["date"], "weekday": weekday,
        "camera_uptime": f"{int(up // 60)} h {int(up % 60)} min",
        "totals": {"events": len(evs), **{s: by_sev.get(s, 0) for s in SEV_ORDER}},
        "high_and_critical_ids": [e["id"] for e in events if e["severity"] in ("high", "critical")],
        "groups": groups,
        "events": events,
        "rules": [{**r, "number": _rule_num(r["id"])} for r in data["rules"]],
        "manager_actions": data["manager_actions"],
    }


def _hhmm_set(times):
    out = set()
    for t in times:
        m = re.match(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", t or "")
        if not m:
            continue
        h, mi = int(m[1]), int(m[2])
        out.add((h, mi))
        if m[3] and int(m[3]) > 0:  # allow rounding up to the next minute ("removed at 16:35" for 16:34:40)
            out.add(((h + (mi + 1) // 60) % 24, (mi + 1) % 60))
    return out


def check_report(md, digest):
    """Guards on the markdown. Empty list = OK."""
    errs = []
    ids = {e["id"] for e in digest["events"]}
    frames = {e["frame"] for e in digest["events"]}
    known_prefixes = {m[1].lower() for i in ids for m in [ID_LIKE.match(i)] if m}
    missing = [i for i in digest["high_and_critical_ids"] if i not in md]
    if missing:
        errs.append(f"every high and critical event must be listed with its id; missing: {missing}")
    unknown = sorted({m[0] for m in ID_LIKE.finditer(md) if m[1].lower() in known_prefixes}
                     - ids - {f.rsplit(".", 1)[0] for f in frames})
    if unknown:
        errs.append(f"event ids {unknown} are not in the input; mention only input events, write ids in full, no ranges")
    nums = {r["number"] for r in digest["rules"]} | {e["rule_number"] for e in digest["events"]} \
        | {_rule_num(a["rule_id"]) for a in digest["manager_actions"]}
    bad_rules = sorted({int(a or b) for a, b in RULE_REF.findall(md)} - nums)
    if bad_rules:
        errs.append(f"rule numbers {bad_rules} are not in the input")
    allowed = _hhmm_set([e["time"] for e in digest["events"]] + [e["end_time"] for e in digest["events"]]
                        + [a["time"] for a in digest["manager_actions"]])
    bad_times = sorted({f"{h}:{mi}" for h, mi in CLOCK.findall(md) if (int(h), int(mi)) not in allowed})
    if bad_times:
        errs.append(f"clock times {bad_times} are not event start/end times or manager action times of the input")
    legal = sorted({m.group(0).lower() for m in LEGAL.finditer(md)})
    if legal:
        errs.append(f"no legal judgements or legal wording: remove {legal}")
    if len(md.split()) > MAX_WORDS:
        errs.append(f"too long: {len(md.split())} words, limit {MAX_WORDS} (one screen)")
    return errs


def daily_report_result(report_input, lang="en", *, frame_size=None, client_config: LLMConfig | None = None,
                        client=None):
    """Returns {"markdown", "patterns", "suggestions", "digest"}. Raises ReportInputError or ReportError."""
    validate_input(report_input)
    digest = build_digest(report_input, frame_size)
    cfg = client_config or config_from_env()
    client = client or make_client(cfg)
    system = PROMPT.replace("{{LANG}}", lang_name(lang))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Day data:\n```json\n" + json.dumps(digest, ensure_ascii=False, indent=1)
                 + "\n```"}]
    raw, errs = None, []
    for attempt in (1, 2):
        raw, mode = chat_json(client, cfg, messages, OUTPUT_SCHEMA, schema_name="daily_report")
        try:
            out = parse_json(raw)
            errs = [f"{e.json_path}: {e.message[:200]}" for e in OUTPUT_VALIDATOR.iter_errors(out)]
            if not errs:
                errs = check_report(out["markdown"], digest)
        except (json.JSONDecodeError, TypeError) as e:
            errs = [f"not valid JSON: {e}"]
        log.info("report %s %s mode=%s attempt=%d %s", digest["camera"], digest["date"], mode, attempt,
                 "ok" if not errs else f"errors: {errs}")
        if not errs:
            return {"markdown": out["markdown"].strip() + "\n", "patterns": out["patterns"],
                    "suggestions": out["suggestions"], "digest": digest}
        messages += [{"role": "assistant", "content": raw or ""},
                     {"role": "user", "content": "Your answer is invalid:\n- " + "\n- ".join(errs)
                      + "\nReturn the corrected JSON object only."}]
    raise ReportError(f"invalid report after retry: {errs}", errs, raw)


def daily_report(report_input, lang="en", **kw) -> str:
    return daily_report_result(report_input, lang, **kw)["markdown"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m safety_mode.report", description=__doc__.split("\n")[0])
    ap.add_argument("input", help="report input JSON (schemas/report-input.schema.json)")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--frame-size", default=None, help="WxH of the camera frame, default 1280x720")
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    ap.add_argument("--digest", action="store_true", help="print the digest the model would get; no API call")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    data = json.loads(Path(a.input).read_text(encoding="utf-8"))
    fs = tuple(int(v) for v in a.frame_size.lower().split("x")) if a.frame_size else None
    try:
        if a.digest:
            validate_input(data)
            print(json.dumps(build_digest(data, fs), ensure_ascii=False, indent=1))
            return 0
        res = daily_report_result(data, a.lang, frame_size=fs)
    except (ReportInputError, ReportError, openai.APIError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if a.out:
        Path(a.out).write_text(res["markdown"], encoding="utf-8")
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(res["markdown"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

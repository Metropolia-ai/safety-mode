"""Rule compiler: plain-language safety rule -> rule.json (schemas/rule.schema.json).

The model writes the rule; the JSON schema plus the semantic checks below decide whether it is usable.
One retry with the errors fed back, then CompileError.
"""
import json
import re
import logging
from pathlib import Path

from jsonschema import Draft202012Validator

from .llm import LLMConfig, chat_json, config_from_env, make_client

log = logging.getLogger("safety_mode.compiler")

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "schemas" / "rule.schema.json").read_text(encoding="utf-8"))
PROMPT = (Path(__file__).resolve().parent / "prompts" / "compiler.md").read_text(encoding="utf-8")
VALIDATOR = Draft202012Validator(SCHEMA)

# primitive -> fields that hold object names
PRIM_FIELDS = {"in_zone": ["object"], "count_in_zone": ["object"], "distance_lt": ["a", "b"],
               "wearing": ["person"], "pose": ["person"], "height_above_floor": ["object"],
               "machine_state": ["machine"]}


class CompileError(Exception):
    def __init__(self, msg, errors=None, raw=None):
        super().__init__(msg)
        self.errors = errors or []
        self.raw = raw


def primitives_used(cond):
    """Set of primitive names in a condition tree (combinators not included)."""
    out = set()
    if cond is None:
        return out
    (k, v), = cond.items()
    if k in ("all", "any"):
        for c in v:
            out |= primitives_used(c)
    elif k == "not":
        out |= primitives_used(v)
    else:
        out.add(k)
    return out


def check(rule, zone_names):
    """Schema errors + semantic errors. Empty list = valid."""
    errs = [f"{e.json_path}: {e.message[:300]}" for e in VALIDATOR.iter_errors(rule)]
    if errs:
        return errs
    objs = [o["name"] for o in rule["objects"]]
    if len(set(objs)) != len(objs):
        errs.append("objects[].name must be unique")
    unknown_zones = set(rule["zones"]) - set(zone_names)
    if unknown_zones:
        errs.append(f"zones {sorted(unknown_zones)} are not in the available zone list {sorted(zone_names)}; "
                    "use only available zone names; if the rule needs an area that is not marked, set condition to null "
                    "and explain in unsupported and suggested_rewrite")
    if rule["condition"] is None:
        if not rule["unsupported"]:
            errs.append("condition is null, so unsupported must explain why")
        if not rule["suggested_rewrite"]:
            errs.append("condition is null, so suggested_rewrite must propose an expressible wording")
        return errs
    if rule["unsupported"] is not None or rule["suggested_rewrite"] is not None:
        errs.append("condition is set, so unsupported and suggested_rewrite must be null")
    used_objs, used_zones = set(), set()

    def walk(n):
        (k, v), = n.items()
        if k in ("all", "any"):
            if not v:
                errs.append(f"'{k}' must not be empty")
            for c in v:
                walk(c)
        elif k == "not":
            walk(v)
        else:
            used_objs.update(v[f] for f in PRIM_FIELDS[k])
            if k == "wearing":
                used_objs.update(v["ppe"])
            if "zone" in v:
                used_zones.add(v["zone"])

    walk(rule["condition"])
    if used_objs - set(objs):
        errs.append(f"condition uses undeclared objects {sorted(used_objs - set(objs))}")
    if set(objs) - used_objs:
        errs.append(f"objects {sorted(set(objs) - used_objs)} are declared but not used in the condition "
                    "(each object costs one detector request per frame); remove them")
    if used_zones != set(rule["zones"]):
        errs.append(f"zones[] must list exactly the zones used in the condition: {sorted(used_zones)}")
    return errs


def _parse(raw):
    """json.loads, tolerating markdown fences or prose around one object (json_object mode is not enforced
    by every provider; Claude via OpenRouter wraps it)."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        i, j = raw.find("{"), raw.rfind("}")
        if i < 0 or j <= i:
            raise
        return json.loads(raw[i:j + 1])


def _zone_lines(zones):
    if not zones:
        return "(none: no zones are marked on this camera; the rule can only use the whole frame)"
    lines = []
    for z in zones:
        z = {"name": z} if isinstance(z, str) else z
        title = f' (title "{z["title"]}")' if z.get("title") else ""
        lines.append(f"- {z['name']}{title}")
    return "\n".join(lines)


PROSE_FIELDS = ("title", "unsupported", "suggested_rewrite", "explanation_template")


def humanize_zone_names(rule: dict, zones) -> dict:
    """People read title, unsupported, suggested_rewrite and the explanation template: zone keys
    (past_gate) never appear there, only zone titles ("Shop floor"). "past_gate (Shop floor)" -> "Shop floor",
    a bare key -> its title. JSON fields (zones, condition) keep the keys. Changes the rule in place."""
    titles = {}
    for z in zones or []:
        if isinstance(z, dict) and z.get("title"):
            titles[z["name"]] = z["title"]
    for k in PROSE_FIELDS:
        v = rule.get(k)
        if not isinstance(v, str):
            continue
        for name, title in titles.items():
            v = re.sub(rf"`?\b{re.escape(name)}\b`?\s*\((?:title\s*)?\"?{re.escape(title)}\"?\)", title, v, flags=re.I)
            v = re.sub(rf"`?\b{re.escape(name)}\b`?", title, v)
        rule[k] = v
    return rule


def compile_rule(text, zones, client_config: LLMConfig | None = None, *, client=None, rule_id="r1"):
    """Compile one rule. zones: list of zone dicts ({name, title, ...}) or names. Returns the rule dict."""
    cfg = client_config or config_from_env()
    client = client or make_client(cfg)
    zone_names = [z if isinstance(z, str) else z["name"] for z in zones]
    system = PROMPT.replace("{{ZONES}}", _zone_lines(zones))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"rule_id: {rule_id}\nRule: {text}"}]
    raw, errs = None, []
    for attempt in (1, 2):
        raw, mode = chat_json(client, cfg, messages, SCHEMA)
        try:
            rule = _parse(raw)
            errs = check(rule, zone_names) if isinstance(rule, dict) else ["top level is not a JSON object"]
        except (json.JSONDecodeError, TypeError) as e:
            errs = [f"not valid JSON: {e}"]
        log.info("%s/%s mode=%s attempt=%d %s", cfg.provider, cfg.model, mode, attempt,
                 "ok" if not errs else f"{len(errs)} error(s): {errs}")
        if not errs:
            rule["rule_id"] = rule_id
            return humanize_zone_names(rule, zones)
        messages += [{"role": "assistant", "content": raw or ""},
                     {"role": "user", "content": "Your JSON is invalid:\n- " + "\n- ".join(errs)
                      + "\nReturn the corrected JSON object only."}]
    raise CompileError(f"invalid rule after retry: {errs}", errs, raw)

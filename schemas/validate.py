#!/usr/bin/env python3
"""Checks for the Safety Mode schemas and data.

1. The schemas/*.schema.json schemas follow the Meta Model API strict mode
   (dev.meta.ai/docs/structured-output, captured 21.10.2026): the root is an object without anyOf/oneOf/allOf/enum/not;
   no allOf/oneOf anywhere; every object has additionalProperties:false and required = all properties keys;
   no recursion ($ref cycles); nesting depth <= 10.
2. rules/*.json are valid against rule.schema.json + semantic checks (object/zone names used in the condition
   are declared, condition = null <=> unsupported is filled in).
3. The JSON example from report-template.md is valid against report-input.schema.json.

Run: python schemas/validate.py   (needs jsonschema: pip install jsonschema)
Exit code 0 = all checks passed.
"""
import glob
import json
import os
import re
import sys

from jsonschema import Draft202012Validator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMAS = os.path.join(ROOT, "schemas")
MAX_DEPTH = 10  # the "Nesting depth" limit in the Meta docs
errors = []


def err(msg):
    errors.append(msg)
    print("  FAIL", msg)


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------- 1. strict mode ----------
def resolve(schema, ref):
    assert ref.startswith("#/$defs/"), ref
    return schema["$defs"][ref.split("/")[-1]]


def lint_strict(schema, name):
    if schema.get("type") != "object":
        err(f"{name}: root is not an object")
    for k in ("anyOf", "oneOf", "allOf", "enum", "not"):
        if k in schema:
            err(f"{name}: {k} at the root")

    def walk(node, path):
        if isinstance(node, dict):
            for k in ("allOf", "oneOf"):
                if k in node:
                    err(f"{name}: {k} in {path}")
            if node.get("type") == "object" or "properties" in node:
                if node.get("additionalProperties") is not False:
                    err(f"{name}: no additionalProperties:false in {path}")
                if sorted(node.get("required", [])) != sorted(node.get("properties", {})):
                    err(f"{name}: required != properties in {path}")
            for k, v in node.items():
                walk(v, f"{path}/{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(schema, "#")

    # recursion: the $ref graph between $defs has no cycles
    def refs(node):
        if isinstance(node, dict):
            if "$ref" in node:
                yield node["$ref"].split("/")[-1]
            for v in node.values():
                yield from refs(v)
        elif isinstance(node, list):
            for v in node:
                yield from refs(v)

    defs = schema.get("$defs", {})
    graph = {d: set(refs(v)) for d, v in defs.items()}
    state = {}

    def dfs(d, stack):
        state[d] = 1
        for n in graph.get(d, ()):
            if state.get(n) == 1:
                err(f"{name}: recursion {' -> '.join(stack + [n])}")
            elif n not in state:
                dfs(n, stack + [n])
        state[d] = 2

    n_err = len(errors)
    for d in defs:
        if d not in state:
            dfs(d, [d])
    if len(errors) > n_err:  # no depth count when there is recursion
        return None

    # instance depth: every object and array = one level, $ref is expanded
    def depth(node):
        if "$ref" in node:
            return depth(resolve(schema, node["$ref"]))
        if "anyOf" in node:
            return max(depth(v) for v in node["anyOf"])
        t = node.get("type")
        if t == "object":
            return 1 + max([depth(v) for v in node.get("properties", {}).values()] or [0])
        if t == "array":
            return 1 + depth(node.get("items", {}))
        return 0

    d = depth(schema)
    if d > MAX_DEPTH:
        err(f"{name}: depth {d} > {MAX_DEPTH}")
    return d


# ---------- 2. rule semantics ----------
PRIM_FIELDS = {  # primitive -> fields holding object names
    "in_zone": ["object"], "count_in_zone": ["object"], "distance_lt": ["a", "b"], "wearing": ["person"],
    "pose": ["person"], "height_above_floor": ["object"], "machine_state": ["machine"],
}


def check_rule(r, fn):
    objs = {o["name"] for o in r["objects"]}
    used_objs, used_zones = set(), set()

    def walk(n):
        (k, v), = n.items()
        if k in ("all", "any"):
            for c in v:
                walk(c)
        elif k == "not":
            walk(v)
        else:
            for f in PRIM_FIELDS[k]:
                used_objs.add(v[f])
            if k == "wearing":
                used_objs.update(v["ppe"])
            if "zone" in v:
                used_zones.add(v["zone"])

    if r["condition"] is None:
        if not r["unsupported"]:
            err(f"{fn}: condition=null, but unsupported is empty")
    else:
        if r["unsupported"] is not None or r["suggested_rewrite"] is not None:
            err(f"{fn}: has both condition and unsupported/suggested_rewrite")
        walk(r["condition"])
        if used_objs - objs:
            err(f"{fn}: undeclared objects in the condition {sorted(used_objs - objs)}")
        if used_zones - set(r["zones"]):
            err(f"{fn}: undeclared zones in the condition {sorted(used_zones - set(r['zones']))}")
        if objs - used_objs:
            err(f"{fn}: declared but not used {sorted(objs - used_objs)} (an extra SAM query per frame)")


def report_example():
    md = open(os.path.join(ROOT, "report-template.md"), encoding="utf-8").read()
    m = re.search(r"```json\s*(\{.*?\})\s*```", md, re.S)
    return json.loads(m.group(1))


def main():
    print("1. Schema strict mode")
    for p in sorted(glob.glob(os.path.join(SCHEMAS, "*.schema.json"))):
        s = load(p)
        Draft202012Validator.check_schema(s)
        n = len(errors)
        d = lint_strict(s, os.path.basename(p))
        print(f"  {'OK  ' if len(errors) == n else '    '} {os.path.basename(p)}: depth {d}, $defs {len(s.get('$defs', {}))}")

    print("2. rules/*.json against rule.schema.json")
    v = Draft202012Validator(load(os.path.join(SCHEMAS, "rule.schema.json")))
    for p in sorted(glob.glob(os.path.join(ROOT, "rules", "*.json"))):
        r = load(p)
        fn = os.path.basename(p)
        n = len(errors)
        for e in v.iter_errors(r):
            err(f"{fn}: {e.json_path}: {e.message[:200]}")
        if len(errors) == n:
            check_rule(r, fn)
        kind = "unsupported" if r.get("condition") is None else "ok"
        print(f"  {'OK  ' if len(errors) == n else '    '} {fn} ({r.get('rule_id')}, {kind})")

    print("3. Example from report-template.md against report-input.schema.json")
    v = Draft202012Validator(load(os.path.join(SCHEMAS, "report-input.schema.json")))
    n = len(errors)
    for e in v.iter_errors(report_example()):
        err(f"report-template.md: {e.json_path}: {e.message[:200]}")
    print(f"  {'OK  ' if len(errors) == n else '    '} report-template.md")

    print(f"\nResult: {'all passed' if not errors else f'{len(errors)} errors'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

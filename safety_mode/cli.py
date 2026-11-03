"""python -m safety_mode.compile "rule text" --zones test-footage/zones/<clip>.json [--rule-id r1] [--out rule.json]"""
import argparse
import json
import logging
import sys

from .compiler import CompileError, compile_rule
from .llm import config_from_env


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m safety_mode.compile",
                                description="Compile a plain-language safety rule into rule.json.")
    p.add_argument("text", help="the rule in plain language (any language)")
    p.add_argument("--zones", help="zones file, e.g. test-footage/zones/<clip>.json (omit = whole frame only)")
    p.add_argument("--rule-id", default="r1")
    p.add_argument("--out", help="write rule.json here instead of stdout")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s", stream=sys.stderr)

    zones = []
    if a.zones:
        with open(a.zones, encoding="utf-8") as f:
            zones = json.load(f)["zones"]
    cfg = config_from_env()
    try:
        rule = compile_rule(a.text, zones, cfg, rule_id=a.rule_id)
    except (CompileError, RuntimeError) as e:
        print(f"compile failed: {e}", file=sys.stderr)
        return 2
    out = json.dumps(rule, ensure_ascii=False, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(out + "\n")
    else:
        print(out)
    return 0

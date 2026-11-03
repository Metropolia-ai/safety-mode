"""Live test: compile the 10 rules of rules-examples.md and compare with rules/*.json.

Skipped unless SM_LLM_API_KEY is set. Run with -s to see the table:
  SM_LLM_BASE_URL=... SM_LLM_API_KEY=... SM_LLM_MODEL=... pytest -s tests/test_compiler_live.py
Compiled rules are written to tests/live-out/ (git-ignored) for review.
Without pytest: python tests/test_compiler_live.py [rule numbers...] [--repeat N]
  e.g. `python tests/test_compiler_live.py 4 --repeat 5` compiles r04 five times.
Token usage per call is summed at the end; cost uses the Spark price from api-notes.md.
"""
import json
import os
import re
from pathlib import Path

import pytest

from safety_mode import CompileError, compile_rule
from safety_mode.compiler import primitives_used
from safety_mode import llm
from safety_mode.llm import config_from_env

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "live-out"
PRICE_IN, PRICE_OUT = 1.25, 4.25  # USD per 1M tokens, Muse Spark (api-notes.md, official price list)


def usage_summary(calls):
    pin = sum(c["prompt_tokens"] for c in calls)
    pout = sum(c["completion_tokens"] for c in calls)
    reas = sum(c.get("reasoning_tokens", 0) for c in calls)
    cost = pin / 1e6 * PRICE_IN + pout / 1e6 * PRICE_OUT
    return f"calls={len(calls)} prompt_tokens={pin} completion_tokens={pout} (reasoning {reas}) est_cost=${cost:.4f}"


def example_rules():
    """[(n, text)] from the table in rules-examples.md."""
    md = (ROOT / "rules-examples.md").read_text(encoding="utf-8")
    return [(int(m[1]), m[2].strip()) for m in re.finditer(r"^\|\s*(\d+)\s*\|([^|]+)\|", md, re.M)]


def compare(got, exp):
    """Tolerant comparison. Returns a list of differences (empty = PASS)."""
    d = []
    if exp["unsupported"] is not None:
        # For a rule the engine cannot check, only the refusal matters: objects and condition are advisory.
        if got["unsupported"] is None:
            d.append("unsupported null != set")
        return d

    def norm(prompts):
        return sorted(p.strip().lower() for p in prompts)

    gp, ep = norm(o["prompt"] for o in got["objects"]), norm(o["prompt"] for o in exp["objects"])
    if gp != ep:
        d.append(f"prompts {gp} != {ep}")
    if sorted(got["zones"]) != sorted(exp["zones"]):
        d.append(f"zones {got['zones']} != {exp['zones']}")
    if got["severity"] != exp["severity"]:
        d.append(f"severity {got['severity']} != {exp['severity']}")
    if abs(got["min_duration_s"] - exp["min_duration_s"]) > 1:
        d.append(f"min_duration_s {got['min_duration_s']} != {exp['min_duration_s']}")
    top = lambda r: None if r["condition"] is None else next(iter(r["condition"]))
    if top(got) != top(exp):
        d.append(f"top operator {top(got)} != {top(exp)}")
    gu, eu = primitives_used(got["condition"]), primitives_used(exp["condition"])
    if gu != eu:
        d.append(f"primitives {sorted(gu)} != {sorted(eu)}")
    if (got["unsupported"] is None) != (exp["unsupported"] is None):
        d.append(f"unsupported {'set' if got['unsupported'] else 'null'} != {'set' if exp['unsupported'] else 'null'}")
    return d


def run_all(only=None, repeat=1):
    cfg = config_from_env()
    OUT.mkdir(exist_ok=True)
    rows = []
    start = len(llm.USAGE)
    rules = [(n, t) for n, t in example_rules() if not only or n in only]
    for n, text in [r for r in rules for _ in range(repeat)]:
        exp = json.loads((ROOT / "rules" / f"r{n:02d}.json").read_text(encoding="utf-8"))
        try:
            got = compile_rule(text, exp["zones"], cfg, rule_id=exp["rule_id"])
            diffs = compare(got, exp)
            (OUT / f"r{n:02d}.json").write_text(json.dumps(got, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except CompileError as e:
            got, diffs = None, [f"compile error: {e}"]
        rows.append((n, got, diffs))
    print(f"\nprovider={cfg.provider} model={cfg.model} mode={cfg.structured}")
    print(f"{'rule':<5} {'result':<6} {'kind':<12} diffs")
    for n, got, diffs in rows:
        kind = "-" if got is None else ("unsupported" if got["condition"] is None else "condition")
        print(f"r{n:02d}   {'PASS' if not diffs else 'FAIL':<6} {kind:<12} {'; '.join(diffs)}")
    print(f"PASS {sum(not d for _, _, d in rows)}/{len(rows)}")
    print("usage:", usage_summary(llm.USAGE[start:]))
    return rows


@pytest.mark.skipif(not os.environ.get("SM_LLM_API_KEY"), reason="SM_LLM_API_KEY not set")
def test_live_ten_rules():
    rows = run_all()
    assert len(rows) == 10
    by = {n: got for n, got, _ in rows}
    assert sum(not d for _, _, d in rows) >= 8
    assert by[4] is not None and by[4]["unsupported"], "r04 must be unsupported"
    assert by[9] is not None and by[9]["unsupported"], "r09 must be unsupported"


if __name__ == "__main__":
    import logging
    import sys
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    args = sys.argv[1:]
    rep = 1
    if "--repeat" in args:
        i = args.index("--repeat")
        rep = int(args[i + 1])
        del args[i:i + 2]
    run_all(only={int(a) for a in args} or None, repeat=rep)

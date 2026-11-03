"""Live test of explain.py and report.py on the configured model (Muse Spark by default).

Skipped unless SM_LLM_API_KEY is set. 2 calls if both answers pass the guards, at most 4.
  pytest -s tests/test_explain_report_live.py
  python tests/test_explain_report_live.py [--save]   # --save also writes safety_mode/samples/
Outputs go to tests/live-out/ (git-ignored). Token usage and cost are printed at the end.
"""
import json
import os
import re
import sys
from pathlib import Path

import pytest

from safety_mode import llm
from safety_mode.explain import explain_event_result
from safety_mode.report import daily_report_result

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "safety_mode" / "samples"
OUT = Path(__file__).resolve().parent / "live-out"
PRICE_IN, PRICE_OUT = 1.25, 4.25  # USD per 1M tokens, Muse Spark (api-notes.md, official price list)

pytestmark = pytest.mark.skipif(not os.environ.get("SM_LLM_API_KEY"), reason="SM_LLM_API_KEY not set")


def usage_summary(calls):
    pin = sum(c["prompt_tokens"] for c in calls)
    pout = sum(c["completion_tokens"] for c in calls)
    reas = sum(c.get("reasoning_tokens", 0) for c in calls)
    cost = pin / 1e6 * PRICE_IN + pout / 1e6 * PRICE_OUT
    return f"calls={len(calls)} prompt_tokens={pin} completion_tokens={pout} (reasoning {reas}) est_cost=${cost:.4f}"


def run(save=False):
    llm.USAGE.clear()
    OUT.mkdir(exist_ok=True)
    event = json.loads((SAMPLES / "event-r3.json").read_text(encoding="utf-8"))
    rule = json.loads((ROOT / "rules" / "r03.json").read_text(encoding="utf-8"))
    data = json.loads((SAMPLES / "report-input-2026-11-02.json").read_text(encoding="utf-8"))

    ex = explain_event_result(event, rule, "en", fallback=False)
    n_ex = len(llm.USAGE)
    rep = daily_report_result(data, "en")
    md = rep["markdown"]

    print("\n=== explanation r3 ===\n" + ex["explanation"] + f"\n(severity_check {ex['severity_check']})")
    print("\n=== report 2026-11-02 ===\n" + md)
    print("patterns:", rep["patterns"])
    print("suggestions:", rep["suggestions"])
    print("explain:", usage_summary(llm.USAGE[:n_ex]))
    print("report: ", usage_summary(llm.USAGE[n_ex:]))
    (OUT / "explanation-r3.txt").write_text(ex["explanation"] + "\n", encoding="utf-8")
    (OUT / "report-2026-11-02.md").write_text(md, encoding="utf-8")
    (OUT / "usage.json").write_text(json.dumps(llm.USAGE, indent=1), encoding="utf-8")
    if save:
        (SAMPLES / "explanation-r3.txt").write_text(ex["explanation"] + "\n", encoding="utf-8")
        (SAMPLES / "report-2026-11-02.md").write_text(md, encoding="utf-8")

    ids = {e["id"] for e in data["events"]}
    high = [e["id"] for e in data["events"] if e["severity"] in ("high", "critical")]
    assert ex["source"] == "model"
    assert "1.3" in ex["explanation"]
    for i in high:
        assert i in md, f"high event {i} missing from the report"
    mentioned = set(re.findall(r"\bevt-\d+\b", md))
    assert mentioned <= ids, f"report mentions events not in the input: {sorted(mentioned - ids)}"
    return ex, rep


def test_live_explain_and_report():
    run()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(save="--save" in sys.argv)

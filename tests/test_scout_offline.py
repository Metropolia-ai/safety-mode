"""Rule Scout offline: a fake Spark client replays a scripted tool sequence, a fake compiler maps
rule texts to rules/*.json, the engine runs on the committed detections. No network."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from safety_mode import llm, scout
from safety_mode.llm import LLMConfig

RULES = Path(__file__).resolve().parent.parent / "rules"
CFG = LLMConfig(api_key="test", model="muse-spark-1.3")

T_DIST = "A forklift must not come closer than 2 m to a pedestrian."
T_PPE = "Hard hat and safety vest are required past the gate."
T_PPE5 = "Hard hat and safety vest are required past the gate; alert after 5 seconds."
T_PRESS = "Nobody inside the yellow zone while the press machine runs."


def load(n):
    return json.loads((RULES / f"r{n:02d}.json").read_text(encoding="utf-8"))


def fake_compile(text, ref):
    rule = {T_DIST: load(3), T_PPE: load(2), T_PRESS: load(1)}.get(text)
    if text == T_PPE5:
        rule = load(2)
        rule["min_duration_s"] = 5
    if rule is None:
        raise ValueError("no canned rule")
    rule = copy.deepcopy(rule)
    rule["rule_id"] = ref
    return rule, True


def call(name, i, **args):
    return SimpleNamespace(id=f"c{i}-{name}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class FakeSpark:
    """Each reply is a list of tool calls or a string (final text). Records every request."""

    def __init__(self, *rounds, out_tokens=500):
        self.rounds = list(rounds)
        self.requests = []
        self.out_tokens = out_tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(copy.deepcopy(kw))
        r = self.rounds.pop(0)
        calls = r if isinstance(r, list) else None
        msg = SimpleNamespace(content=None if calls else r, tool_calls=calls)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="tool_calls" if calls else "stop")],
            usage=SimpleNamespace(prompt_tokens=2000, completion_tokens=self.out_tokens,
                                  completion_tokens_details=SimpleNamespace(reasoning_tokens=self.out_tokens - 100)))


def scripted():
    return FakeSpark(
        [call("get_camera_facts", 1)],
        [call("compile_rule", 2, text=T_DIST, reason="forklift and people share the aisle"),
         call("compile_rule", 3, text=T_PPE, reason="people work past the gate"),
         call("compile_rule", 4, text=T_PRESS, reason="yellow zone exists")],
        [call("run_rule", 5, rule_ref="p1"), call("run_rule", 6, rule_ref="p2"), call("run_rule", 7, rule_ref="p3"),
         call("accept_rule", 8, rule_ref="p1", reason="too early")],
        [call("inspect_alert", 9, event_id="p1.e1"), call("inspect_alert", 10, event_id="p2.e1", second=10)],
        [call("accept_rule", 11, rule_ref="p1", reason="5 alerts, pedestrian 0.9 m from the reach truck at s1"),
         call("compile_rule", 12, text=T_PPE5, reason="PPE flickers; longer hold", revises="p2"),
         call("reject_rule", 13, rule_ref="p3", reason="no press machine is detected on this clip")],
        [call("run_rule", 14, rule_ref="p4")],
        [call("inspect_alert", 15, event_id="p4.e1")],
        [call("accept_rule", 16, rule_ref="p4", reason="one alert s5-s22 on the person without a hard hat")],
    )


def test_scripted_session_accepts_revises_rejects():
    fake = scripted()
    res = scout.run_scout("warehouse-team", client_config=CFG, client=fake, compile_fn=fake_compile)
    assert res.stop_reason == "every rule has a decision"
    assert [r["rule_ref"] for r in res.rules] == ["p1", "p4"]
    dec = {r["rule_ref"]: r["decision"] for r in res.rejected}
    assert dec == {"p2": "revised", "p3": "reject"}
    assert res.rounds == 8 and res.usage["agent_calls"] == 8
    kinds = [t["kind"] for t in res.trace]
    assert kinds[0] == "facts"
    assert kinds.count("propose") == 3 and kinds.count("revise") == 1
    # accept before run_rule finished in the same round: p1 ran first (round order), but was not inspected yet
    refused = [t for t in res.trace if t["kind"] == "refused"]
    assert refused and "inspect" in refused[0]["summary"]
    p3_run = next(t for t in res.trace if t["kind"] == "run" and t["rule_ref"] == "p3")
    assert p3_run["ran"] is False and "press machine" in p3_run["why_not"]
    p1_run = next(t for t in res.trace if t["kind"] == "run" and t["rule_ref"] == "p1")
    # video mode (track ids), one episode per rule as on the page: s1-s6, s10-s12, s15, s18-s22
    assert p1_run["alerts"] == 4 and p1_run["timeline"].count("T") == 14
    assert [(e["start_s"], e["end_s"]) for e in p1_run["events"]] == [(1, 6), (10, 12), (15, 15), (18, 22)]
    assert p1_run["events"][0]["subject"] == "forklift #0, person #1 and person #2"
    insp = [t for t in res.trace if t["kind"] == "inspect"]
    assert insp[0]["second"] == 1 and insp[1]["second"] == 10
    assert any(d["matched_by_engine"] for d in insp[0]["detections"])
    assert insp[0]["pairs"] and "floor_distance_m" in insp[0]["pairs"][0]
    assert {d["id"] for d in insp[0]["detections"] if d["matched_by_engine"]} >= {0, 1, 2}
    assert {p["person"] for p in insp[0]["pairs"]} >= {"person #1", "person #2"}
    p4 = next(r for r in res.rules if r["rule_ref"] == "p4")
    assert p4["revises"] == "p2" and p4["rule"]["min_duration_s"] == 5
    ev = p4["events"][0]                        # 5 s hold: raised on the 5th true second
    assert ev["start_s"] == ev["since_s"] + 4
    # every tool call got its result back, in the next request, with the system prompt first
    second = fake.requests[1]["messages"]
    assert second[0]["role"] == "system" and "Rule Scout" in second[0]["content"]
    tool_msgs = [m for m in second if m["role"] == "tool"]
    assert len(tool_msgs) == 1 and tool_msgs[0]["tool_call_id"] == "c1-get_camera_facts"
    facts = json.loads(tool_msgs[0]["content"])
    assert facts["clip_length_s"] == 23 and "person" in facts["sam_prompts_with_detections"]
    assert fake.requests[0]["tools"] == scout.TOOLS
    assert res.usage["completion_tokens"] == 8 * 500
    assert res.cost_usd == pytest.approx((8 * 2000 * 1.25 + 8 * 500 * 4.25) / 1e6, abs=1e-4)


def test_accept_needs_a_run():
    fake = FakeSpark(
        [call("compile_rule", 1, text=T_DIST, reason="r")],
        [call("accept_rule", 2, rule_ref="p1", reason="looks fine")],
        [call("reject_rule", 3, rule_ref="p1", reason="not tested")])
    res = scout.run_scout("warehouse-team", client_config=CFG, client=fake, compile_fn=fake_compile)
    refused = [t for t in res.trace if t["kind"] == "refused"]
    assert "run_rule first" in refused[0]["summary"] or "not run" in refused[0]["summary"]
    assert res.rules == [] and res.rejected[0]["decision"] == "reject"


def test_step_limit_and_undecided():
    fake = FakeSpark(*[[call("get_camera_facts", i)] for i in range(3)])
    res = scout.run_scout("busy-warehouse", max_steps=2, client_config=CFG, client=fake, compile_fn=fake_compile)
    assert res.rounds == 2 and res.stop_reason.startswith("step limit")


def test_token_budget_stops_the_run():
    fake = FakeSpark(*[[call("get_camera_facts", i)] for i in range(8)], out_tokens=9000)
    res = scout.run_scout("busy-warehouse", client_config=CFG, client=fake, compile_fn=fake_compile)
    assert res.stop_reason == "token budget"
    assert res.usage["completion_tokens"] <= scout.MAX_OUTPUT_TOKENS + 9000
    # max_tokens of each call shrinks to what is left of the budget
    assert [r["max_tokens"] for r in fake.requests] == [8000, 8000, 2000]


def test_nudge_once_then_stop():
    fake = FakeSpark([call("compile_rule", 1, text=T_DIST, reason="r")], "Done.", "Still done.")
    res = scout.run_scout("warehouse-team", client_config=CFG, client=fake, compile_fn=fake_compile)
    assert res.stop_reason == "model stopped with undecided rules"
    assert [u["rule_ref"] for u in res.undecided] == ["p1"]
    assert any(m["role"] == "user" and "without a decision" in m["content"] for m in fake.requests[2]["messages"])


def test_unknown_refs_and_bad_args_do_not_crash():
    fake = FakeSpark(
        [call("run_rule", 1, rule_ref="p9"), call("inspect_alert", 2, event_id="zz"), call("nope", 3)],
        [SimpleNamespace(id="c4", function=SimpleNamespace(name="compile_rule", arguments="{not json"))],
        "stop")
    res = scout.run_scout("forklift-operator", client_config=CFG, client=fake, compile_fn=fake_compile)
    errs = [t["summary"] for t in res.trace]
    assert any("unknown" in s for s in errs) and any("empty text" in s for s in errs)


def test_signed_state_survives_a_js_round_trip():
    st = scout.new_session("busy-warehouse")
    st["usage"]["x"] = 2.0
    sig = scout.sign_state(st)
    js = json.loads(json.dumps(st).replace("2.0", "2"))     # what JSON.stringify sends back
    assert scout.verify_state(js, sig)
    js["max_steps"] = 99
    assert not scout.verify_state(js, sig)


def test_web_endpoint_steps(monkeypatch):
    from fastapi.testclient import TestClient
    from safety_mode import webapp
    monkeypatch.setenv("SM_LLM_API_KEY", "test")
    fake = scripted()
    monkeypatch.setattr(scout, "make_client", lambda cfg: fake)
    monkeypatch.setattr(scout, "default_compile", lambda clip, client, cfg: fake_compile)
    monkeypatch.setattr(webapp, "LIMITER", webapp.Limiter())
    monkeypatch.setenv("SM_LIMIT_SCOUT_PER_HOUR", "1")   # production default is 2
    c = TestClient(webapp.app)
    r = c.post("/api/scout/step", json={"clip": "warehouse-team"})
    assert r.status_code == 200
    j = r.json()
    assert c.post("/api/scout/step", json={"clip": "warehouse-team"}).status_code == 429   # 1 run per hour (env above)
    bad = c.post("/api/scout/step", json={"state": {**j["state"], "max_steps": 99}, "sig": j["sig"]})
    assert bad.status_code == 400
    for _ in range(30):
        if j["view"]["done"]:
            break
        r = c.post("/api/scout/step", json={"state": j["state"], "sig": j["sig"]})
        assert r.status_code == 200, r.text
        j = r.json()
    assert j["view"]["done"] and [x["rule_ref"] for x in j["view"]["rules"]] == ["p1", "p4"]
    assert "messages" not in j["view"]


def test_cached_traces_are_valid():
    """The committed live traces (web/cache/scout/*.json) have the fields the page reads."""
    files = sorted(scout.CACHE_DIR.glob("*.json"))
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d["clip"] == p.stem
        assert d["trace"] and d["usage"]["agent_calls"] >= 1
        for r in d["rules"]:
            assert r["decision"] == "accept" and r["rule"]["condition"]

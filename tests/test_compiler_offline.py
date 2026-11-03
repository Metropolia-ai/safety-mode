"""Offline tests: fake client with canned JSON, no network."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest

from safety_mode import CompileError, compile_rule
from safety_mode.llm import LLMConfig

RULES = Path(__file__).resolve().parent.parent / "rules"


def load(n):
    return json.loads((RULES / f"r{n:02d}.json").read_text(encoding="utf-8"))


class FakeClient:
    """Returns canned replies in order; records every request."""

    def __init__(self, *replies, reject_json_schema=False):
        self.replies = list(replies)
        self.calls = []
        self.reject_json_schema = reject_json_schema
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.reject_json_schema and kw["response_format"]["type"] == "json_schema":
            # built without an HTTP response so the test does not depend on the SDK's transport library
            err = openai.BadRequestError.__new__(openai.BadRequestError)
            Exception.__init__(err, "response_format json_schema is not supported")
            raise err
        content = self.replies.pop(0)
        if not isinstance(content, str):
            content = json.dumps(content)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def cfg(**kw):
    return LLMConfig(api_key="fake", base_url="https://fake.example/v1", model="fake-model", **kw)


def test_valid_first_try():
    fc = FakeClient(load(2))
    rule = compile_rule("Hard hat and vest are required past the warehouse gate", ["past_gate"], cfg(), client=fc, rule_id="r2")
    assert rule == load(2)
    assert len(fc.calls) == 1
    rf = fc.calls[0]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert "past_gate" in fc.calls[0]["messages"][0]["content"]


def test_invalid_json_then_valid():
    fc = FakeClient('{"rule_id": "r3", "title": ', load(3))
    rule = compile_rule("Forklift no closer than 2 m to a pedestrian", [], cfg(), client=fc, rule_id="r3")
    assert rule == load(3)
    assert len(fc.calls) == 2
    feedback = fc.calls[1]["messages"][-1]["content"]
    assert "not valid JSON" in feedback


def test_schema_error_then_valid():
    bad = copy.deepcopy(load(8))
    bad["severity"] = "urgent"
    bad["extra"] = 1
    fc = FakeClient(bad, load(8))
    rule = compile_rule("Forklift does not drive on the pedestrian lane", ["pedestrian_lane"], cfg(), client=fc, rule_id="r8")
    assert rule == load(8)
    feedback = fc.calls[1]["messages"][-1]["content"]
    assert "severity" in feedback or "extra" in feedback


def test_semantic_errors_fed_back():
    bad = copy.deepcopy(load(1))
    bad["objects"].append({"name": "rack", "prompt": "rack"})  # declared, unused
    bad["zones"] = ["press_zone"]  # not an available zone
    fc = FakeClient(bad, load(1))
    compile_rule("Nobody inside the yellow zone while the press is running", ["yellow_zone"], cfg(), client=fc, rule_id="r1")
    feedback = fc.calls[1]["messages"][-1]["content"]
    assert "rack" in feedback and "press_zone" in feedback


def test_fenced_json_is_accepted():
    fc = FakeClient("Here is the rule:\n```json\n" + json.dumps(load(10)) + "\n```")
    assert compile_rule("lying", [], cfg(), client=fc, rule_id="r10") == load(10)


def test_invalid_twice_raises():
    fc = FakeClient("not json", "still not json")
    with pytest.raises(CompileError) as e:
        compile_rule("x", [], cfg(), client=fc)
    assert len(fc.calls) == 2 and e.value.raw == "still not json"


def test_unsupported_path():
    fc = FakeClient(load(4))
    rule = compile_rule("Do not stand under a raised load", [], cfg(), client=fc, rule_id="r4")
    assert rule["condition"] is None and rule["unsupported"] and rule["suggested_rewrite"]


def test_unsupported_without_explanation_is_rejected():
    bad = copy.deepcopy(load(9))
    bad["unsupported"] = None
    fc = FakeClient(bad, load(9))
    rule = compile_rule("Pallets no more than three high in a stack at the ramp", ["ramp_zone"], cfg(), client=fc, rule_id="r9")
    assert rule["unsupported"]
    assert "unsupported must explain" in fc.calls[1]["messages"][-1]["content"]


def test_rule_id_is_forced():
    fc = FakeClient(load(10))
    assert compile_rule("lying", [], cfg(), client=fc, rule_id="r42")["rule_id"] == "r42"


def test_fallback_to_json_object():
    c = cfg()
    fc = FakeClient(load(7), load(7), reject_json_schema=True)
    rule = compile_rule("One person in the charging zone", ["charging_zone"], c, client=fc, rule_id="r7")
    assert rule == load(7)
    assert c.structured == "json_object"
    last = fc.calls[-1]
    assert last["response_format"] == {"type": "json_object"}
    assert "Output JSON schema" in last["messages"][0]["content"]
    # the config remembers the fallback: next compile goes straight to json_object
    n = len(fc.calls)
    compile_rule("One person in the charging zone", ["charging_zone"], c, client=fc, rule_id="r7")
    assert len(fc.calls) == n + 1


def test_provider_name_from_url():
    assert LLMConfig(base_url="https://api.meta.ai/v1").provider == "meta"
    assert LLMConfig(base_url="https://openrouter.ai/api/v1").provider == "openrouter"

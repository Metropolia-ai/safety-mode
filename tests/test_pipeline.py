"""Pipeline on the committed detections JSON (offline, no frames, no API)."""
import pytest

from safety_mode import pipeline
from safety_mode.pipeline import DetectionSource, intervals, parse_machine_spec, resolve_rule, run_clip


def test_resolve_rule_names():
    assert resolve_rule("r2").name == "r02.json"
    assert resolve_rule("r02").name == "r02.json"
    assert resolve_rule("r2-vest").name == "r02-vest.json"
    with pytest.raises(ValueError):
        resolve_rule("r99")


def test_machine_spec():
    at = parse_machine_spec("running:0-8")
    assert at(0) == at(8) == "running" and at(9) == "stopped"
    assert parse_machine_spec("stopped")(100) == "stopped"


def test_offline_run_forklift_operator(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "FRAMES_DIR", tmp_path)   # no frames: sizes and list from the JSON
    run = run_clip("forklift-operator", ["r02", "r03", "r01"], source=DetectionSource("offline"))
    assert run.calls == 0
    assert "r1" in run.rejected and "r3" in run.not_evaluable
    assert intervals(run, "r2") == [(2, 9)]
    assert run.timeline["r3"] == "?" * 10


def test_offline_missing_prompt_raises(monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(pipeline, "FRAMES_DIR", tmp_path / "frames")
    rule = json.loads(resolve_rule("r02").read_text())
    rule["objects"][2]["prompt"] = "prompt that was never sent"
    (tmp_path / "r.json").write_text(json.dumps(rule))
    with pytest.raises(LookupError):
        run_clip("forklift-operator", [tmp_path / "r.json"], source=DetectionSource("offline"))

"""Fill demo/cache/{compile,explain}.json for the 4 example rules on the sample clips and the webcam (Muse Spark, paid).

    export MODEL_API_KEY=<Meta Model API key>
    SM_CACHE_WRITE=1 python demo/precompute.py [--explain]
    python demo/precompute.py --desk      # the webcam "Try at your desk" chips only

Already cached entries cost nothing. --explain also runs each example on each clip and
caches the Spark explanations of the first 3 alerts (what a visitor sees first).
"""
import json
import os
import sys
import time

os.environ["SM_CACHE_WRITE"] = "1"
from fastapi.testclient import TestClient  # noqa: E402

from safety_mode import llm  # noqa: E402
from safety_mode.webapp import EXAMPLES, JUDGE_CLIPS, app  # noqa: E402

c = TestClient(app)
if "--desk" in sys.argv:
    # webcam "Try at your desk" chips: retry until the compiled rule asks SAM for the planned prompts
    from safety_mode import webapp
    from safety_mode.compiler import compile_rule
    for k, d in webapp.DESK_RULES.items():
        cam = None
        if d.get("zone"):
            z = {"name": webapp.DESK_ZONE_KEY[d["zone"]], "title": d["zone"], "polygon": [[0, 0], [1, 0], [1, 1]]}
            cam = webapp.camera_from_body("webcam", {"camera": {"zones": [z]}})
        key = webapp.compile_key(d["text"], "webcam", cam)
        hit = webapp.COMPILE_CACHE.get(key)
        if hit and sorted(o["prompt"] for o in hit["rule"]["objects"]) == sorted(d["prompts"]):
            print(f"desk {k}: cached")
            continue
        for attempt in range(4):
            rule = compile_rule(d["text"], (cam or webapp.WEBCAM_ZONES)["zones"], rule_id="custom")
            got = sorted(o["prompt"] for o in rule["objects"])
            print(f"desk {k} attempt {attempt + 1}: prompts {got} cond {json.dumps(rule['condition'])} dur {rule['min_duration_s']}")
            if got == sorted(d["prompts"]) and rule["condition"] and rule["min_duration_s"] == 2:
                webapp.COMPILE_CACHE.put(key, {"text": d["text"], "clip": "webcam", "rule": rule})
                break
    print(f"Spark calls {len(llm.USAGE)}")
    raise SystemExit(0)
for clip in sys.argv[1:] and [a for a in sys.argv[1:] if not a.startswith("--")] or JUDGE_CLIPS + ["webcam"]:
    for rid, text in EXAMPLES.items():
        t = time.time()
        r = c.post("/api/compile", json={"text": text, "clip": clip, "rule_id": rid})
        j = r.json()
        if r.status_code != 200:
            print(clip, rid, r.status_code, j)
            continue
        rule = j["rule"]
        print(f"{clip:18s} {rid} cached={j['cached']} {time.time() - t:5.1f}s "
              f"objects={[o['prompt'] for o in rule['objects']]} zones={rule['zones']} "
              f"cond={json.dumps(rule['condition'])[:140]} dur={rule['min_duration_s']} sev={rule['severity']} "
              f"status={ {k: v for k, v in j['status'].items() if v} }")
        if "--explain" in sys.argv and clip != "webcam":
            t = time.time()
            run = c.post("/api/run", json={"clip": clip, "rule": rule, "episodes": "rule"}).json()   # as the page
            print(f"    run {run['timeline']} events={[(e['start_s'], e['end_s']) for e in run['events']]} "
                  f"{time.time() - t:4.1f}s")
            for e in run["events"][:3]:
                print(f"      [{e['explanation_source']}] {e['explanation']}")
tok_in = sum(u["prompt_tokens"] for u in llm.USAGE)
tok_out = sum(u["completion_tokens"] for u in llm.USAGE)
print(f"Spark calls {len(llm.USAGE)}, tokens in {tok_in} out {tok_out}, ~${tok_in * 1.25e-6 + tok_out * 4.25e-6:.3f}")

# AI Vision Monitor: Safety Mode

Write a workplace safety rule in plain words, and an ordinary camera checks it.

Built by Frappua (Helsinki) for the Meta Global AI Developer Hackathon, 26 Oct to 4 Nov 2026.

**Built during the hackathon (26 Oct to 4 Nov 2026):** all code in `safety_mode/`, `tests/`, `web/`, `api/` and `demo/`, and the Muse Glimmer run in `jetson-bench/glimmer.md`.

- Live demo: https://safety-mode.vercel.app (no sign-in)
- Demo video (3 min): https://youtu.be/l5A9mPo_Wl8

## What it does

1. A safety manager types a rule: "Everyone on the warehouse floor must wear a hard hat and a high-visibility vest."
2. Muse Spark compiles it into a strict JSON rule, and the page shows what it understood: objects, zone, condition, hold time, severity. If a camera cannot check the rule, Spark refuses with a reason and a wording that can be checked.
3. SAM 3.1 segments the objects of the rule in each frame. In video mode every object keeps a track id, so an alert reads "person #2 without a vest for 10 s".
4. A deterministic engine evaluates the rule frame by frame and opens an alert when the condition has held long enough. The same detections always give the same alerts.
5. Muse Spark explains each alert in one paragraph and writes the session report. Our code checks both texts against the event data before the page shows them.
6. Rule Scout, a Muse Spark agent with tool calls, proposes rules for a camera, compiles them, runs them through the engine, looks at the alerts they raise and accepts, revises or rejects each rule with a one-sentence reason.

## Try it

Open https://safety-mode.vercel.app. No sign-in and no key: every feature is open to everyone, within hourly limits per address (section "Cost guards on the public page").

The four example rules run on the four sample clips ("Worker crosses the floor", "Site: vest check", "Forklift crosses the walkway", "Team at the racks"): press "Check rule", then "Run on this clip". Compilations, detections, the first explanations and the recorded Rule Scout runs come from a cache in this repository, so this path makes no Meta API calls.

There are three ways to test a rule of your own. These call Muse Spark and SAM 3.1 live:

1. **Your own clip.** Open "Your clip" and choose a video file. On its first frame, draw the area your rule talks about (click the corners, double-click to close, give it a name) or press "Whole frame". Then write a rule that uses that name, for example "Everyone at the desk must wear glasses." The file stays in your browser. When the rule runs, the page sends one frame per second of the first 30 seconds to SAM 3.1.
2. **Webcam at your desk.** Open "Webcam", allow the camera and press one of the "Try at your desk" rules: glasses, headphones, or no cup in the keyboard zone (the page asks you to draw that zone first). Take the glasses or headphones off, or put a cup in the zone. The page sends one frame every 2 seconds, and the alert comes on the first frame that shows the violation.
3. **Any rule on the sample clips.** Type a rule on one of the clips, for example "Nobody may stand on the shop floor without gloves." on "Worker crosses the floor". If it names an object we have no cached detections for, SAM 3.1 runs on the clip's frames while a progress line counts them.

The session report and a live Rule Scout run call Muse Spark live too. Rules with distances in metres need a floor calibration: of the sample clips only "Team at the racks" has one, and the page says so on the others.

## Architecture

```
rule in plain words --> Muse Spark: rule compiler (strict json_schema) --> rule.json
                          | schema and meaning checks, one retry, or "unsupported" + rewrite
                          v
camera frames ------> SAM 3.1: one request per object prompt --> boxes, masks, scores, track ids
(1 frame/s on clips,      (video mode for the sample clips, image mode for webcam and live frames;
 1 frame/2 s on webcam)    responses cached by image hash + prompt)
                          v
zones + floor calibration --> deterministic engine --> events (start, end, objects, metres)
                          | primitives, all/any/not, instance binding, state machine
                          v
event facts (no pixels) --> Muse Spark: explanation --> alert card
                          | max 80 words, one paragraph, no number that is not in the facts
                          v
session events --> Muse Spark: report --> markdown report
                     every high and critical event present, no invented IDs or times
```

The models write and explain. The engine decides whether to alert.

The page is `web/index.html` (one file, no build). It talks to a stateless FastAPI app (`safety_mode/webapp.py`, Vercel entry `api/index.py`), and the browser keeps the session.

### Engine

- Primitives: `in_zone`, `count_in_zone`, `distance_lt`, `wearing`, `machine_state`, combined with `all`, `any` and `not`. The schema also accepts `pose` and `height_above_floor`; the engine reports them as "cannot be checked" instead of guessing, with three-valued logic.
- State machine `idle -> candidate -> active -> closed`, one event per episode; a 1-frame gap does not close an event. With video-mode detections the engine keeps one episode per tracked object.
- Floor homography: four floor points per camera map pixels to metres. `distance_lt` measures between the nearest points of the two objects' bottom edges. A person box at least half inside a forklift box, with its bottom edge at least 10% of the forklift's height above the forklift's bottom edge, is the operator and does not count as a pedestrian near that forklift.
- Instance binding: a rule is true when one set of instances satisfies the whole tree. In "hard hat and vest past the gate", the person in the zone and the person without PPE must be the same person. `not` over an empty primitive stays false, so "not wearing" with no people in view never alerts.
- PPE on a person: at least 30% of the PPE mask (or box) inside the matching third of the person box: the top for hats, the middle for vests.

### Rule Scout

`safety_mode/scout.py` gives Muse Spark six function tools: `get_camera_facts`, `compile_rule`, `run_rule`, `inspect_alert`, `accept_rule`, `reject_rule`. The agent works from zones and cached SAM 3.1 detections, never from pixels. The engine still decides every alert. The session is a JSON dict, so the web page runs the agent one step per request; the page shows the recorded run for each clip and can start a live one.

### Cost guards on the public page

The live features are open to everyone, within limits per address (IP) and per day (`webapp.Limiter`). Cached results never count.

| Live call | Per address per hour | Per day (server instance) |
|---|---|---|
| Muse Spark calls (own rule, explanation, report) | 30 | 500 |
| SAM 3.1 frames (own clip, webcam, live frames of a sample clip) | 300 | 2500 |
| Rule Scout live runs | 2 | 30 |
| Rule Scout steps | 40 | 600 |

Over a limit the page shows when it resets ("Hourly limit for this address reached (…), try again in 12 minutes."), and the sample clips keep working from the cache. `GET /api/health` shows the limits and what is left for your address. The counters are kept in memory per server instance, so the limits are best effort.

## Meta models and where they are in the code

| Model | Used for | Code |
|---|---|---|
| Muse Spark (`muse-spark-1.3`, Meta Model API, Chat Completions) | rule compiler, strict `json_schema` | `safety_mode/compiler.py`, `safety_mode/llm.py`, prompt `safety_mode/prompts/compiler.md`, schema `schemas/rule.schema.json` |
| Muse Spark | alert explanations with fact checks and a template fallback | `safety_mode/explain.py`, `safety_mode/facts.py`, prompt `safety_mode/prompts/explain.md` |
| Muse Spark | session and daily report with output checks | `safety_mode/report.py`, prompt `safety_mode/prompts/report.md`, schema `schemas/report-input.schema.json` |
| Muse Spark | Rule Scout agent, function calling | `safety_mode/scout.py` |
| SAM 3.1 (`sam-3.1`, Meta Model API, `/v1/responses`) | image mode per text prompt; video mode with track ids | `safety_mode/detect/sam_api.py`, `safety_mode/detect/sam_video.py`, parser `safety_mode/detect/sam_parse.py` (masks via Meta's `meta-sam-parser`) |
| Muse Glimmer 30B (open weights, GGUF Q4_K_M, llama.cpp) | the same rule compiler, offline on a Jetson | `jetson-bench/bench/glimmer_server.sh`, `jetson-bench/bench/glimmer_run_live.py` |

The compiler talks to any OpenAI-compatible endpoint through the `SM_LLM_*` variables; that is how the same code runs on Muse Spark in the cloud and on Muse Glimmer on the board.

## On site, offline

A site that cannot send camera video to a cloud can run Safety Mode on an NVIDIA Jetson AGX Orin 64 GB:

- **Rule compiler on Muse Glimmer 30B** (Apache 2.0, GGUF Q4_K_M, 15.6 GiB, llama.cpp with strict `json_schema`): all 10 reference rules compiled correctly, including the two that must be refused. 53 s per rule on average with `reasoning_effort=low`, 18 GB of memory. Details: [`jetson-bench/glimmer.md`](jetson-bench/glimmer.md).
- **Detector SAM 3 on the same board**: 873 ms per frame with 3 prompts for SAM 3 INT8 (TensorRT), 853 ms for SAM 3.1 in BF16. One board serves 2 cameras at one frame every 2 s. Details: [`jetson-bench/results.md`](jetson-bench/results.md).

## Run it

Requirements: Python 3.10+, a Meta Model API key for the live parts.

```
pip install -e ".[web]"

# offline tests, no network
pytest tests --ignore=tests/test_compiler_live.py --ignore=tests/test_explain_report_live.py

# run rules on a sample clip with the committed SAM 3.1 detections (no API calls)
python -m safety_mode.run --clip warehouse-team --rules r02 r03 --timeline

# compile a rule with Muse Spark
export SM_LLM_API_KEY=<Meta Model API key>     # default base URL https://api.meta.ai/v1, model muse-spark-1.3
python -m safety_mode.compile "A forklift must not come closer than 2 m to a pedestrian." \
  --zones test-footage/zones/pexels-6079421-forklift-busy-warehouse.json

# Rule Scout on one clip
python -m safety_mode.scout busy-warehouse

# live tests against the Meta Model API
pytest -s tests/test_compiler_live.py tests/test_explain_report_live.py
```

The web page locally: download the sample clips from Pexels into `test-footage/raw/` (links in `test-footage/README.md`), run `demo/build.sh` (it assembles a deploy copy in `/tmp/safety-mode-deploy`), then `cd /tmp/safety-mode-deploy && MODEL_API_KEY=<key> SM_STATIC_DIR=. uvicorn api.index:app --port 8090`.

## Repository map

| Path | What is there |
|---|---|
| `safety_mode/` | compiler, SAM 3.1 client and parser (`detect/`), engine (`engine/`), pipeline, explanations, reports, Rule Scout, web API |
| `tests/` | offline tests for all of the above; live tests against the Meta Model API |
| `web/`, `api/`, `demo/` | the demo page, its Vercel entry, deploy script, cached compilations and browser checks (`demo/e2e.py`) |
| `schemas/`, `rules/` | strict JSON schemas for rules and report input; 10 reference rules |
| `test-footage/` | Pexels clip list, hand-drawn zones and floor calibration, committed SAM 3.1 detections, expected alerts |
| `jetson-bench/` | on-device measurements of SAM 3, SAM 3.1 and Muse Glimmer on a Jetson AGX Orin |
| `api-notes.md`, `rules-examples.md`, `report-template.md` | API notes, reference rules, report format |

## Limits we know about

- Pose ("person lying") and height above the floor are accepted but marked "cannot be checked".
- Distances need a four-point floor calibration per camera; far from the camera the error grows.
- A seated forklift operator sits in the cab, not on the floor, so floor-zone rules can miss him.
- Machine state (running or stopped) is set by hand in the demo.

## Licence

The sample clips are from Pexels and stay under the Pexels licence (https://www.pexels.com/license/); the raw videos are not in this repository, the detections are. Faces on two of the clips are blurred on the page. `meta-sam-parser`, Muse Glimmer and other dependencies keep their own licences. The code in this repository is licensed under the Apache License 2.0 (`LICENSE`), copyright 2026 Frappua.

No API keys or secrets are in this repository.

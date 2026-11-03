# safety_mode: rule compiler

A rule in words (any language) → `rule.json` following `schemas/rule.schema.json`. Calls go through the OpenAI SDK. The provider is set by environment variables, and the code for Meta and for OpenRouter is the same.

## Files
- `compiler.py`: `compile_rule(text, zones, client_config, rule_id=...)`. It checks the answer against the schema and for meaning (the objects and zones of the condition are declared, there are no extra objects, zones come only from the camera's list, `condition: null` ⇔ `unsupported` and `suggested_rewrite` are filled). On an error there is one retry, and the errors are sent to the model. A second error raises `CompileError`.
- `llm.py`: client from env. It tries `json_schema` strict first. If the provider answers 400/422, it switches to `json_object` with the schema in the system prompt. The real check is the same in both modes: schema and meaning in `compiler.py`.
- `prompts/compiler.md`: system prompt (primitives, camera zones, the "unsupported instead of inventing" rule, durations, severity).
- `cli.py`: `python -m safety_mode.compile "A forklift must not come closer than 2 m to a pedestrian." --zones test-footage/zones/pexels-6079421-forklift-busy-warehouse.json`.

## Variables
| Variable | Default | Current (Meta, since 27.10) |
|---|---|---|
| `SM_LLM_BASE_URL` | `https://api.meta.ai/v1` | default |
| `SM_LLM_API_KEY` | none | the Meta Model API key (`MODEL_API_KEY`) |
| `SM_LLM_MODEL` | `muse-spark-1.3` | default |
| `SM_LLM_STRUCTURED` | `auto` | `json_schema` (strict accepted) |
| `SM_LLM_PROVIDER` | by host | `meta` |
| `SM_LLM_REASONING` | not sent | not sent (the model picks the level) |

Running on Spark: `export SM_LLM_API_KEY=<Meta Model API key>`, then `pytest -s tests/test_compiler_live.py` or `python tests/test_compiler_live.py [rule numbers] [--repeat N]`. The tokens of every call go to the log and to `llm.USAGE`. At the end of the run the total and a cost estimate are printed.

`max_tokens` was raised from 2000 to 8000. On Spark it limits reasoning and answer together (dev.meta.ai/docs/reasoning). At 2000 Spark twice spent the whole limit on reasoning for r04 and returned `content: null`.

## Tests
- `pytest tests/test_compiler_offline.py`: no network, fake client (valid answer, broken JSON and retry, schema and meaning errors, unsupported, fallback to json_object, JSON inside markdown).
- `tests/test_compiler_live.py`: only with `SM_LLM_API_KEY`. Without pytest: `python tests/test_compiler_live.py 4 --repeat 5`. 10 rules from `rules-examples.md` against `rules/*.json`: the set of SAM prompts, zones, severity, `min_duration_s` ±1 s, the top operator, the set of primitives, matching unsupported. Results are written to `tests/live-out/` (not in git).

## Live run 27.10 (Meta Model API, `muse-spark-1.3`, strict json_schema)
Meta accepted the schema `schemas/rule.schema.json` in strict mode without changes, so the fallback to `json_object` was not needed. The prompt `prompts/compiler.md` was not changed.

| Rule | Result | What did not match |
|---|---|---|
| r01–r03, r05–r08, r10 | PASS | — |
| r04 | FAIL, but `unsupported` | prompts `load, person` instead of `forklift, person, raised load` |
| r09 | PASS, `unsupported` | — |

9 out of 10, r04 and r09 come back as `unsupported`. The only r04 mismatch is in the objects. The rule text ("Nobody stands under a raised load") has no forklift, and the prompt `raised load` breaks the prompt rule "no state adjectives". We did not tune the prompt for this. The reference `rules/r04.json` itself probably needs a fix.

r04 stability: 5 calls in a row, 5 out of 5 `unsupported` (all with the same objects `load, person`). In the second full run r04 was also `unsupported`. In the first one (limit 2000) both answers for r04 were cut off at the limit, which gave a compile error. The other 9 matched the result of the second run.

Tokens for all calls on 27.10 (27 calls: the first full run with limit 2000, 5× r04, the second full run):

| Run | Calls | Input | Output (of which reasoning) | Estimate, $ |
|---|---|---|---|---|
| 10 rules, `max_tokens` 2000 | 12 | 138 032 | 15 334 (not measured) | 0.238 |
| r04 × 5 | 5 | 57 515 | 11 228 (10 118) | 0.120 |
| 10 rules, `max_tokens` 8000 | 10 | 114 951 | 11 547 (10 153) | 0.193 |
| Total | 27 | 310 498 | 38 109 | 0.550 |

Price from `api-notes.md`: $1.25 / $4.25 per 1M tokens input/output (official price list, 27.10). One rule costs about $0.02. Input is about 11 500 tokens per call, although the system prompt is about 1 800 tokens. Meta probably adds the strict schema to the input (not checked). 85–90 % of the output is reasoning.

## Live run 26.10 (OpenRouter, `openai/gpt-5-mini`, strict json_schema)
| Rule | Result | What did not match |
|---|---|---|
| r01–r03, r05–r08, r10 | PASS | — |
| r04 | FAIL, but `unsupported` | prompts `load, person` instead of `forklift, person, raised load` |
| r09 | PASS, `unsupported` | — |

9 out of 10, r04 and r09 come back as `unsupported`, and the test passes. In the previous run of the same prompt r04 once compiled into the condition `distance_lt + height_above_floor` (an approximation instead of a refusal). In 4 separate calls out of 4 it was `unsupported`. So the refusal on r04 is unstable for this model, about 1 time in 5.

Other models:
- `openai/gpt-4.1-mini`: 8 out of 10 after prompt edits, but r04 compiles a condition instead of `unsupported` almost every time.
- `anthropic/claude-sonnet-4.5`: does not accept the strict schema ("compiled grammar is too large"). The fallback to `json_object` worked, but the run cost ~$0.68, so we did not repeat it.

OpenRouter spend for everything on 26.10: ~$0.96 (of which Sonnet ~$0.68). The gpt-5-mini run ≈ $0.024.

## Condition engine (`safety_mode/engine/`)
Deterministic, with no models and no network: frame detections + camera zones + `rule.json` → events.

```python
from safety_mode.engine import Engine, Frame, Detection
eng = Engine(["rules/r02.json"], "test-footage/zones/pexels-6079421-forklift-busy-warehouse.json",
             machine_states={"press": "running"})
for t, dets in enumerate(frames):        # Detection(name from rule.objects, box in pixels, mask, score, track_id)
    for ev in eng.process(Frame(t, dets)):   # events opened, updated and closed on this frame (copies)
        ...
eng.finish()                              # close the active ones at the end of the clip
```

- `geometry.py`: point in polygon, foot point, floor homography from the 4 `floor_calibration` points, distance along the floor, box overlap ratio, PPE on a person.
- `primitives.py`: 7 primitives. `pose` and `height_above_floor` return `NotEvaluable`. So does `distance_lt` without calibration.
- `evaluate.py`: walks the `all/any/not` tree by the schema, checks structure and names.
- `state.py`: `idle → candidate → active → closed`, one event per episode.
- `diagnose.py` (30.10): what the engine measured, per object per second: the foot point, which rule zones it is in, the PPE ratio on a person, the floor distance and the two nearest points for every forklift–person pair (`geometry.nearest_floor_points`). It does not affect the decision. `/api/run` returns it per second (`diag`) together with the state (`state`), and the page and `test-footage/verify/render_verify.py` draw the proof from it.
- On load: `condition: null` (r04, r09), a schema error and a zone the camera does not have give `RuleRejected` (or `engine.rejected` with `skip_rejected=True`). Rules with `pose`/`height_above_floor` (r06, r10) and `distance_lt` without calibration load and end up in `engine.not_evaluable`.

Conventions (from `test-footage/expected.md`):
- A frame "lasts" `frame_period_s` (1 s). The alarm fires on the frame where the duration first reaches ≥ `min_duration_s`: a 3 s rule that is true from s0 gives an event with `start_s = 2`, `condition_since_s = 0`. `end_s` = the last true frame.
- An episode closes when the condition stays not true for longer than `max_gap_s`. The default is 1 (since 27.10): one missed frame does not close the event, as in `expected.md`. `max_gap_s=0` closes on the first false frame. Only a raised event survives a gap: the `min_duration_s` window is filled by consecutive true frames.
- `in_zone` and `count_in_zone` use the foot point (the middle of the bottom edge of the box).
- `distance_lt`: the bottom edges of both boxes are mapped to metres by the homography, and the distance is taken between the nearest points of these segments. A pair is excluded if the box of one object is ≥ 50 % inside the box of the other and its bottom edge is above the other's bottom edge by ≥ 10 % of the height (an operator in a cab or on a platform; `primitives.is_operator`). Without the second condition a pedestrian at the mast of a reach truck was also counted as an operator.
- `wearing`: PPE is on a person if ≥ 30 % of the PPE mask (without a mask: the PPE box) lies in the right third of the person's box. The third is chosen by name: hat/helmet → top, vest/jacket → middle, boot → bottom, otherwise the whole box.
- Per-object episodes (since 29.10): if all detections of the objects that the rule links have a `track_id` (SAM 3.1 video mode), the episode belongs to those objects: to a person in r02, to a forklift and person pair in r03. Each set has its own state machine, and the window and the gap are counted per object. The event gets `object_ids` (`[{"name": "person", "track_id": 2}]`) and `subject` ("person #2"), and so does `explanation_inputs.subject`. A person who left the frame for longer than `max_gap_s` starts a new episode on return. Without ids, and for rules without linked objects (`count_in_zone`), there is one episode per rule, as before. `Engine(..., episodes="rule")` turns this on even with ids.
- Binding: a rule is true if there is a set of instances (one person, one forklift…) on which the tree is true. In r02 `in_zone(person)` and `not wearing(person)` refer to the same person. If an object is not in the frame, the primitive is "empty": it is false, and `not` does not flip it (`not wearing` with no people does not alarm). "Nobody in the zone" is written as `count_in_zone(..., "==", 0)`, not through `not in_zone`.
- Not evaluable (NE) by Kleene: `all` is false if at least one child is false, `any` is true if at least one is true, otherwise NE. For the timer NE = not true.

Tests: `pytest tests/test_engine_*.py` (47, offline): geometry, every primitive, timing, scenarios r01, r02, r03, r05, r07, r08 on the zone files, rejection of r04/r09, NE for r06/r10. `charging_zone` (r07) is in none of the zone files, so the test defines it itself.

## Detection: SAM 3.1 via the Meta API (`safety_mode/detect/`, 27.10)

Files:
- `sam_api.py`: `SamClient(mask_encoding=..., include_confidence=...)`, methods `segment_image(path|bytes, prompt)` and `segment_video(mp4, prompt)`. One prompt = one request (`responses.create`, model `sam-3.1`). The image is scaled down to 1280 px on the long side, and the coordinates are mapped back to source pixels.
- `sam_parse.py`: `parse_output(text, prompt, scale=...)` → a list of `Detection{prompt, frame_index, track_id, box, mask, mask_box, frame_size, confidence}`. `box` = [x1, y1, x2, y2] in source pixels, x2/y2 exclusive. `mask` = uint8 0/1 the size of the box (a local raster), `full_mask(W, H)` pastes it into the frame.
- `overlay.py`: overlay (mask fill, box outline, label "prompt #id confidence"), JPEG up to 200 KB.
- `batch.py`: runs over the test frames, writes `test-footage/detections/` and `summary.json`.

Running (key in the variable `MODEL_API_KEY`):
```
export MODEL_API_KEY=<Meta Model API key>
python -m safety_mode.detect test-footage/frames/pexels-6079421-forklift-busy-warehouse/005.jpg --prompt person --prompt forklift --overlay /tmp/out.jpg
python -m safety_mode.detect.batch            # 60 requests with an empty cache
pytest tests/test_sam_parse.py tests/test_sam_api.py   # no network
```
CLI flags: `--mask-encoding one_bit`, `--confidence`, `--offline` (cache only), `--json`.

Format facts (details in `api-notes.md`, checked against dev.meta.ai and live answers):
- One line per frame `<Nf>`, entries `id<|box;...|><|mask;...|>` separated by commas. An empty `output_text` = nothing found.
- The box is inclusive on x2/y2, in pixels of what was sent. The mask is local to the box: H×W = the box size.
- The mask decoder is official: `pip install meta-sam-parser` (Meta, 0.1.0). The format is custom: base85 with its own alphabet and a range coder, so there is no point writing it ourselves. The dependencies `meta-sam-parser`, `pillow`, `numpy` are in `pyproject.toml` (since 27.10).
- `metadata.include_confidence="true"` gives confidence, all values ≥ 0.507. `one_bit` is 20–25 times shorter than `lossless`.
- Mask decoding in pure Python is slow: ~0.9 s per 337×276 mask. If only the box is needed, use `detections(decode_masks=False)`.

Cache: `test-footage/cache/sam31/<sha256 of the source bytes>_<hash of prompt, model and options>.json` (in `.gitignore`), holding the full answer, sizes and time. Running the same frame with the same prompt again is free. The number of files in the cache = the number of paid requests from this machine.

Spend on 27.10: 62 image requests (1 first by hand, 1 through the CLI, 60 batch) ≈ $0.155 at $2.50 per 1000, plus 1 video request for 254 frames ≈ $0.051 at $0.20 per 1000 frames (minimum 50 frames per request). Total ≈ $0.21. Time: an image 0.5–4.2 s, 60 requests in 6 threads in 37 s; a 10 s video: upload 5.4 s + segmentation 12.4 s.

Results against expectations: `test-footage/expected.md`, section "SAM 3.1 through the Meta API, 27.10".

### Video mode and track id (`sam_video.py`, 29.10)
```
python -m safety_mode.detect.sam_video [--clip busy-warehouse] [--prompts person forklift ...] [--offline]
python -m safety_mode.detect.compare [--seconds]      # video against images per second, IoU ≥ 0.5
```
- The clip is scaled down to 1280 px on the long side (all frames, no sound, `test-footage/cache/video/<clip>-1280.mp4`; the file hash is part of the cache key, so do not recreate the file) and goes to `input_video` once per prompt. The full answer is in `cache/sam31/`.
- In the answer the video frames start at 0, and empty frames are skipped. The 1 fps frame `NNN.jpg` (s = NNN − 1) is taken from the middle of the second: video frame `ceil((s + 0.5) · fps) − 1`, fps from ffprobe (checked by image difference on all 58 frames).
- Output: `test-footage/detections/<clip>.video.json` in git: per second and per prompt `{box, score, track_id}` in frame pixels, `frame_map`, the track summary `tracks`, `requests`. Ids of different prompts are not linked to each other.
- Id stitching (`tracks.py`): a new id of the same prompt within 1 s after the old one disappeared and at its place (IoU ≥ 0.4 or foot points closer than 0.3 m along the floor) gets the old id. The list of stitches is in `bridges`, the SAM id in `sam_track_id`. It is applied on export and in `DetectionSource`.
- The explanation (`explain.build_facts`) gets `event.subject`, `event.object_ids` and `objects[].label`. Spark writes "Person #1 was inside Past the gate…".
- `<clip>.json` (image mode) no longer has the `id` field: there it was the object number inside one frame, not a track.
- Spend on 29.10: 14 requests, 7 722 frames ≈ $1.54. Time 15–44 s per request.
- Video errors we saw: the hood of a reach-truck operator as `hard hat` (warehouse-team), the operator in the cab on busy-warehouse as two tracks (body and legs). Details: `test-footage/results-2026-10-29.md`.

Tests: `pytest tests/test_sam_video.py` (frame mapping, scale, cache in git).

## Explanation and report (Muse Spark, 27.10)

Both tasks get only structured data, no frames. Object positions are turned into words in code: the third of the frame by the box centre ("on the left side / in the centre / on the right side of the frame"), distance by the bottom edge ("near the camera / mid-distance / far from the camera"), the zone by the foot point if a zones file is passed.

Files:
- `explain.py`: `explain_event(event, rule, lang="en", zones=None)` → a paragraph; `explain_event_result(...)` → `{explanation, severity_check, source, errors}`; `explain_event_template(event, rule)` → deterministic text from the rule's `explanation_template`. The event is in the `report-input` form (id, time, duration_s, objects with box in pixels, measurements) or an engine Event as a dict (`explanation_inputs`). The output is strict json_schema `{explanation, severity_check}`. Checks: schema, one paragraph, ≤ 80 words, the text has no numbers that are not in the facts. An error → one retry with the errors. A second error, an API error or no key → the template (`source: "template"`).
- `report.py`: `daily_report(report_input, lang="en")` → markdown following `report-template.md`; `daily_report_result(...)` → also `patterns`, `suggestions`, `digest`. The input is checked against `schemas/report-input.schema.json` (`ReportInputError` before the model call). The totals by severity, the medium/low groups by rule, the event end time, the weekday and uptime are computed by code (`digest`), not by the model. Answer checks: all high/critical ids are in the text; no ids, rule numbers or times (HH:MM) that are not in the input; no legal words (law, liable, compliance…); ≤ 450 words. An error → one retry, a second → `ReportError`. There is no fallback template for the report.
- `facts.py`: shared helpers (positions in words, numbers in the text, duration in words).
- `prompts/explain.md`, `prompts/report.md`: system prompts. `report.md` has the "must / must not" list from `report-template.md`.
- `samples/`: `event-r3.json` (the r3 event from `report-template.md`), `report-input-2026-11-02.json` (the full day from the report example: 7 events, r6 with no events, r9 with 3 false ones and a switch-off at 11:00), `explanation-r3.txt` and `report-2026-11-02.md` (Spark answers after the prompt fix).

CLI:
```
python -m safety_mode.explain safety_mode/samples/event-r3.json rules/r03.json [--zones <zones.json>] [--lang fi] [--template] [--json]
python -m safety_mode.report safety_mode/samples/report-input-2026-11-02.json [--lang fi] [--out report.md] [--frame-size 1920x1080] [--digest]
```
`--template` and `--digest` work without network. Boxes in the report input are in pixels, and the default frame size is 1280×720 (our test clips). Another camera needs `--frame-size`, otherwise the positions in words will be wrong.

Tests: `pytest tests/test_explain_report_offline.py` (19, no network: positions, facts without box, retry, an invented number, the template after two bad answers and with the API unavailable, input check, digest, all events in the report prompt, every report check). `tests/test_explain_report_live.py` with `SM_LLM_API_KEY`: the r3 explanation and the full report, all high ids in the report, no ids that are not from the input. `python tests/test_explain_report_live.py --save` rewrites `samples/`.

Live run 27.10, `muse-spark-1.3`, strict json_schema accepted on the first try, no retries:

| Run | Calls | Input | Output (reasoning) | Estimate, $ |
|---|---|---|---|---|
| v1: explanation + report | 2 | 4 068 | 5 796 (5 103) | 0.030 |
| v2 after the prompt fix | 2 | 4 207 | 4 122 (3 514) | 0.023 |
| Total | 4 | 8 275 | 9 918 | 0.053 |

Price from `api-notes.md`. An explanation costs about $0.007–0.011, a report about $0.016–0.019.

What we fixed after v1: the explanation inserted the rule title as part of a phrase ("The camera detected Forklift within 2 m of a pedestrian involving a forklift and a person at centre of the frame"). The report attached a position to every object and presented a single event (a pallet at the exit) as a pattern. Added to the prompts: use your own words, give the position once and only where it helps, a pattern = at least two events. The positions in words got prepositions. In v2 there are no invented events, numbers or legal judgements. Small issues remain: the duration appears twice in items 2 and 7, the reason for switching off r9 is repeated, and "Remind arrivals" assumes these are visiting people (the input does not say so).

## Pipeline: frames → SAM 3.1 → engine (`safety_mode/pipeline.py`, 27.10)

```
python -m safety_mode.run --clip busy-warehouse --rules r02 r03 --machine forklift=running:0-8 --timeline
python -m safety_mode.run --clip all --export-detections [--prompts person forklift "hard hat" "safety vest"]
```
- `run_clip(clip, rule_ids, zones_file=None, machine_states=None, fps=1)`: frames `test-footage/frames/<clip>/NNN.jpg` (NNN = second + 1), prompts = the union of `objects[].prompt` of the accepted rules, `decode_masks=False` (`wearing` by box thirds). A separate `Engine` for each rule, detection name = the `objects[].name` with that prompt. The box is in pixels of the frame file and is normalised by its size (`Frame.size`).
- Detections: first the offline cache in git (box and confidence, no masks), then the `cache/sam31/` cache and the API. `--source offline` uses only JSON and works without frames too. Since 29.10 the default is video mode (`<clip>.video.json`, `track_id`). `SM_DETECTOR=image` or `DetectionSource(detector="image")` brings back image mode (`<clip>.json`, no ids). `run_clip(..., episodes="rule")` gives one episode per rule even with ids.
- The page (`webapp.py`, `/api/run`) also takes video mode if it has all the prompts of the rule, otherwise image mode. The response has `detector`, and boxes and hits have `id`. The page sends `episodes: "rule"` (one alarm per rule episode, label "person #1" from the objects at the alarm second). Rule Scout counts the same way: video mode (`scout.DETECTOR`), one episode per rule (`scout.EPISODES`).
- Rules: `r02`, `r2`, a path, or the variants from `test-footage/rules/` (`r01-forklift`, `r02-vest`). `--machine NAME=running:0-8` = running on s0–s8, stopped on the rest.
- A rejection (`unsupported`, a zone the camera does not have) spends no SAM requests.

Result on three clips (details in `test-footage/results-2026-10-27.md`):

| Clip | Matched | Did not match |
|---|---|---|
| busy-warehouse | r03, r05, r08, r01-forklift: no alarms | r02: expected s2–s24, got nothing. The foot point of the seated operator (bottom of the box in the cab) is outside `past_gate` |
| warehouse-team | r02 s3–s22, r02-vest s3–s12, r01-forklift s1–s4/s18/s22, r06 "not evaluable" | r03: 10 out of 20 seconds. Misses come from the reach truck not being found (s3, s4, s11) and from distances of 2.2–5 m by calibration where the by-eye marking gave < 2 m |
| forklift-operator | r02 s2–s9 (vest), r03 "not evaluable", r01 rejected | — |

Since 27.10 the vest prompt is `safety vest` instead of `high-visibility vest`: 16 out of 16 vests on warehouse-team 003–012 against 14 (`expected.md`, "Prompt synonyms").

Tests: `pytest tests/test_pipeline.py tests/test_pipeline_video.py tests/test_engine_tracks.py` (offline, from detection JSON; episodes by id: two people, one leaves for a long time, a short absence, a forklift and person pair).

Video mode against image mode on the same rules (`test-footage/results-2026-10-29.md`): not worse in any row, r03 on warehouse-team 14 out of 20 seconds against 10 (the reach truck is found on s3, s4, s11). With per-object episodes r02 on warehouse-team gives person #1 s3–s12, #5 s10–s12, #10 s16 instead of one s3–s22: after s13 the condition is true on different ids for 1–3 seconds each.

## Rule Scout: a Muse Spark agent with tool calls (`safety_mode/scout.py`, 29.10)

The agent gets a camera (a clip). It proposes rules in words by itself, compiles them, runs them through the engine, looks at the alarm facts and accepts, rewrites or rejects each rule. Alarms are still decided only by the engine. The agent proposes and checks.

Tools (Chat Completions `tools`, `tool_choice: auto`):
| Tool | What it does |
|---|---|
| `get_camera_facts()` | zones with titles and notes, whether there is floor calibration, which SAM prompts were found and in which seconds, who stands in which zone per second, clip length, how the forklift state is set, what the engine cannot do |
| `compile_rule(text, reason, revises?)` | `compiler.compile_rule` (first the page cache `demo/cache/compile.json`), returns `rule_ref` p1, p2…; with `revises` the old rule is closed as "replaced" |
| `run_rule(rule_ref)` | the engine on the committed detections, like `/api/run`: alarms with `event_id`, seconds, objects and measurements, a `T.?` line per second |
| `inspect_alert(event_id, second?)` | facts of one second, no images: all detections of the rule's objects (score, position in words, zones, whether it matched the engine), person–forklift distances along the floor, the "operator in the cab" flag, the PPE ratio on each person, the condition state and the number of detections ±3 s |
| `accept_rule` / `reject_rule(rule_ref, reason)` | a decision with a one-sentence reason |

Checks in code, not in the prompt: `accept` without `run_rule`, and `accept` of a rule with alarms without `inspect_alert`, return an error to the agent. The session ends when every rule has a decision. Budget: at most 8 agent calls and 20 000 output tokens per run (agent and compiler together), at most 8 rules. On the last round, or with less than 7 000 tokens left, the agent is told "decide, no new rules needed", and compiling is blocked. `reasoning_effort: low` is set only for the agent (`SM_SCOUT_REASONING`). Without it Spark spent up to 4 500 reasoning tokens per round, and one run ended on the budget with no decisions.

The session is stored in a JSON dictionary. A step = one model call or the execution of the tools from the last call. On the page, `POST /api/scout/step`: `{clip}` starts a run (limit `SM_LIMIT_SCOUT_PER_HOUR=2` per IP, `SM_DAILY_SCOUT_RUNS=30` per instance), `{state, sig}` makes the next step, and each request fits into the Vercel 60 s. The state is signed with HMAC (key from `SM_SCOUT_SECRET` or from the API key), so steps or the round limit cannot be forged. Replaying a signed step is limited by `SM_LIMIT_SCOUT_STEPS_PER_HOUR=40`.

Running: `python -m safety_mode.scout busy-warehouse [--max-steps 8] [--save]`. `--save` writes `web/cache/scout/<clip>.json`, and the page takes recorded runs from there at no cost. With `SM_CACHE_WRITE=1` the rules compiled by the agent go into `demo/cache/compile.json`, and the "Try this rule" button on the page compiles them from the cache. Prompt: `prompts/scout.md`.

Tests: `pytest tests/test_scout_offline.py` (9, no network): a fake Spark plays an 8-round scenario (facts, 3 rules, run, inspection, accept refused without inspection, revision, acceptance), refusal without a run, the round limit, the token budget and the `max_tokens` reduction, the reminder about undecided rules, junk in the arguments, the state signature after browser JSON, the step endpoint through TestClient, the format of recorded runs.

### Runs 30.10, evening: sample clips A, B, C (`muse-spark-1.3`, recorded to the page)
| Clip | Proposed | Accepted | Rejected and why | Calls (agent + compiler) | Input / output tokens | $ |
|---|---|---|---|---|---|---|
| Worker crosses the floor (`--max-steps 6`) | 3 | hard hat on "Shop floor", vest on "Shop floor" (person #0, s5–s21, alarm s7) | "nobody on the floor while the forklift is running": there is no forklift in the frame, this is a normal walk-through | 5 + 3 | 54 782 / 6 259 | 0.10 |
| Forklift crosses the walkway (`--max-steps 10`, rerecorded) | 3 | "a forklift stands on the crossing for longer than 2 s" (forklift #0, s5–s7, alarm s6) | "a forklift and a pedestrian on the crossing at the same time" and "nobody on the crossing while the forklift is running": person #0 = the driver in the cab (`operator_in_cab`) | 5 + 3 | 55 138 / 6 556 | 0.10 |
| Site: vest check (`--max-steps 5`, `MAX_PROPOSALS=2`) | 2 | vest on "Site floor" (person #1, s0–s5, alarm s2) | hard hat: the condition is true for 1 s | 5 + 2 | 41 061 / 3 551 | 0.07 |

For Team at the racks the 29.10 run was kept. The first B run (`--max-steps 5`) ended with no decision on one rule. Rerecording with `--max-steps 10` cost 15 calls instead of 12: the first successful run was started without `--save` (5 + 2), the repeat was 5 + 3. In total the page cache on the evening of 30.10 took 41 Spark calls: 18 for `demo/precompute.py` (8 compilations, 10 explanations) and 23 for three Scout runs. `MAX_PROPOSALS=2` for C was set from the command line (`python -c "import safety_mode.scout as s; s.MAX_PROPOSALS = 2; s.main([...])"`) to keep the run short.

### Live runs 29.10, evening: video mode with ids (`muse-spark-1.3`)
Since the evening of 29.10 the agent sees video mode detections: every object has a track id (`distinct_track_ids` in the camera facts, `id` on detections in `inspect_alert`, pairs "person #2 / forklift #0"), and the alarm names the objects (`subject`). One episode per rule, as on the page. These runs are recorded to the page (`web/cache/scout/`):

| Clip | Proposed | Accepted | Rejected and why | Calls (agent + compiler) | Input / output tokens | $ | Time |
|---|---|---|---|---|---|---|---|
| Forklift aisle | 3 | "Forklifts must not stop in the Door apron for more than 2 seconds." (forklift #0, condition s0–s3, alarm s1, score 0.92) | 2 m to a pedestrian: not a single true second; a pedestrian on the door apron: 1 s (s24) with a 2 s duration | 5 + 2 | 43 628 / 6 218 | 0.08 | 24 s |
| Team at the racks | 3 | 2 m to a pedestrian (4 alarms: s1–s6, s10–s12, s15, s18–s22; 0.88–0.9 m); hard hat past the gate (s3–s22, person #1); vest past the gate (s3–s12, person #1) | none | 6 + 2 | 55 108 / 5 464 | 0.09 | 21 s |
| Operator close-up (`--max-steps 5`) | 3 | none | hard hat: true only on s0; "nobody past the gate while the forklift is running": true for all 10 s on the driver (person #0), who is not a pedestrian; "forklift without an operator": the compiler answered unsupported | 5 + 2 | 42 150 / 8 015 | 0.09 | 36 s |
| Total | | | | 22 calls (one rule per clip from the compile cache) | 140 886 / 19 697 | 0.26 | |

The accepted rules make sense, and there were no reruns. The difference from the daytime runs (image mode): on Team at the racks the agent accepted all three rules, not two, because this time it did not propose "nobody in the yellow zone". 2 m to a pedestrian gives 4 alarms instead of 5 (video mode finds the reach truck on s3, s4, s11). An inaccuracy in a reason: for the hard hat the agent writes "Person #1 … for 22 s", although person #1 is in the frame only on s1–s12. The s1–s22 rule episode is held by different people, and the label is taken from the alarm second.

### Live runs 29.10, day: image mode (`muse-spark-1.3`)
These were on the page until the evening of 29.10:

| Clip | Proposed | Accepted | Rejected and why | Calls (agent + compiler) | Input / output tokens | $ |
|---|---|---|---|---|---|---|
| Forklift aisle | 4 | "Forklifts must not stand in the Door apron for more than 2 seconds." (condition s0–s3, alarm on s1, score 0.93) | 2 m to a pedestrian: not a single true second; a person in the yellow zone while the forklift is running: the person is there only on s24, the forklift runs s0–s8; a person on the door apron: 1 s with a 5 s duration | 7 + 3 | 65 928 / 7 292 | 0.11 |
| Team at the racks | 3 | 2 m to a pedestrian (5 alarms, 0.91 m on s1); vest past the gate (condition s1–s12, alarm on s3) | "Nobody in the yellow zone": the agent read in the zone note that this is the parking spot of a reach truck that leaves by s2, and judged people entering later to be a normal walk-through | 6 + 3 | 66 261 / 10 357 | 0.13 |
| Operator close-up (`--max-steps 5`) | 3 | none | hard hat: 2 separate seconds with a 3 s duration; "nobody past the gate while the forklift is running": true for all 10 s, the zone = the whole frame, this is the scene, not a violation; "at most one person": 1 s | 5 + 3 | 53 397 / 5 947 | 0.09 |

Total on 29.10, with debugging: 60 Spark calls (2 trial calls on the tool call format, 4 runs on Forklift aisle, one each on the two other clips), input 399 691, output 68 794 tokens, about $0.79 at the price from `api-notes.md`.

What we fixed along the way (three discarded runs on Forklift aisle):
1. The first run "rewrote the rule so that it would fire": "No person may enter Past the gate" gave no alarms, so the agent replaced it with "…even for a second" for the sake of the single second s24 and accepted it. This is also the only revision in all the runs. Added to the prompt: the threshold and the duration come from practice, not from the clip; a rule with no alarms is rejected, not weakened; people present on a normal floor are not a violation.
2. The second run ended on the 20 000 token budget with no decisions (4 500 reasoning tokens per round). Added `reasoning_effort: low` and a last "decisions only" round.
3. In the third one the compiler gave "No forklifts in door apron" a 60 s duration (the compiler prompt rule for things in aisles), and the agent rejected everything. Added to the prompt: compare the compiled rule with the intended one, and write the duration into the wording.

A live run from the button on production was checked on the evening of 29.10 (`demo/e2e.py --scout-live`, Forklift aisle, video mode): 34 s, 15 `/api/scout/step` requests of 0.2–9.9 s each (Vercel limit 60 s, `SM_LLM_TIMEOUT=28`), 7 agent calls and 3 compiler calls, 65 868 / 8 389 tokens, $0.12. Accepted 0 out of 4: the agent itself set "a forklift stands on the apron for longer than 5 s", and the condition holds for 4 s. Response times of the other page requests: `demo/README.md`, "Response time".

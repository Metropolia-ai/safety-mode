# Live demo page

**URL: https://safety-mode.vercel.app** (Vercel, project `safety-mode`, scope `<vercel-scope>`, Hobby plan).

## What a visitor sees (since 30.10, evening; live SAM since 31.10; own clip and desk rules since 31.10, daytime; open to everyone since 2.11)
1. Clips in the selector, in order: "Worker crosses the floor" (A, Pexels 38707991, head blurred), "Site: vest check" (C, 32245123), "Forklift crosses the walkway" (B, 31751344, 0–12 s), "Team at the racks", "Webcam", "Your clip" (the visitor's own file, section "Own clip and zones"). The page opens on A with example 1. "Operator close-up" and "Forklift aisle" are removed from the selector but open via `?clip=forklift-operator` / `?clip=busy-warehouse`.
2. Examples: 1 "Everyone on the warehouse floor must wear a hard hat and a high-visibility vest." (`r2`), 2 "Everyone on the floor must wear a high-visibility vest." (`r2-vest`), 3 "A forklift must not come closer than 2 m to a pedestrian." (`r3`), 4 "No forklift in the walkway crossing." (`r8`). Each clip opens with its own example: A→1, C→2, B→4, Team→3. When the clip changes, the example changes too unless the visitor has typed a rule of their own, and the "How I understood it" card is re-checked from the cache.
3. "Check rule", "Run on this clip", the proof overlay on the video, the seconds strip and the alert feed work as before. The line under the strip names all violators of the episode in order of appearance ("person #1 …; then person #5 …" when it changes), not only the one present in the second the alert fired.
4. "Session report", "Rule Scout" (recorded runs for all four clips) and "Webcam" are unchanged.
5. Since 2.11 every live feature is open to everyone with no key: an own rule, live SAM 3.1 on the clips, the own clip, the webcam, the report and the live Rule Scout. Any rule runs on the clips: objects not in the cache (gloves, pallets, ladders) are found by SAM 3.1 live on the clip frames ("How it works"). Only the limits per address and per day protect the Meta API budget (section "Spending limits"). From 31.10 to 2.11 these features needed an access key typed into a footer field; that field and its messages are gone.

What each example shows (production 30.10, `e2e.py --matrix`, video mode, one episode per rule):

| Clip | 1 hard hat and vest | 2 vest | 3 2 m to a pedestrian | 4 forklift on the crossing |
|---|---|---|---|---|
| Worker crosses the floor (A) | **alert 7–21 s**, person #0 (candidate 5–6 s) | alert 7–21 s, person #0 | not checked: no floor calibration | Spark refuses: no crossing zone |
| Site: vest check (C) | alert 2–5 s, person #1 (both wear hard hats) | **alert 2–5 s**, person #1 (candidate 0–1 s; at 3 s the vest of #0 fell into box #1, the 1 s gap tolerance keeps the alert open) | not checked | refused: no crossing zone |
| Forklift crosses the walkway (B) | Spark refuses: no "warehouse floor" on this camera | alert 7–8 s on the forklift driver (person #0): the compiler took the whole frame; the wording "Pedestrians on the floor…" gave the same result (1 attempt), the example was kept | not checked | **alert 6–7 s**, forklift #0 (candidate 5 s) |
| Team at the racks | alert 3–22 s: #1, then #5, #2 and 3 more | alert 3–12 s, person #1 | **4 alerts: 1–6, 10–12, 15, 18–22 s** | refused: no crossing zone |

Bold marks the default example. On the A and B strips: idle → candidate → alert → idle. C starts with a candidate (the violator is in the frame from 0 s) and ends with a candidate at 10 s (#0 behind the scaffolding, vest not found, 1 s). On Team, rule 3 lasts 1 s, there is no candidate, and the clip ends inside the last alert.

Per-second expectations: `test-footage/expected.md` (sections 4–6 for A, B, C); clip selection: `test-footage/candidates/REPORT-2026-10-30.md`.

### Faces in clips A and B
`demo/blur_heads.py`: the top 22 % of each person box, enlarged by 15 %, from the boxes in `detections/<clip>.video.json`, interpolated to 25 fps, pixelated and blurred; H.264 1280×720, faststart, 2.20 MB. Before the first and after the last box of a walking person, the box keeps moving for 2 s at the track speed taken from the first (last) two boxes and is clipped by the frame border. For the driver (B), the first and last box still hold for 0.5 s. The 1 fps frames and the detections do not change. `build.sh` rebuilds the file in `test-footage/cache/blurred/` if the source, the detections or the script are newer.

Check on 31.10: the first SAM box on A is at 5.4 s, but the person comes out from behind the rack on the right earlier. Before the fix the blur started at frame 123 (4.92 s), and the head was uncovered on frames 110–122 (4.40–4.88 s), with the face recognisable on 114–122 (4.56–4.88 s). Now the blur sits on the head position from 3.4 s, and frames 110–122 are covered. The exit at the left edge (20.5–23 s) was covered and still is: from 21 s the head is behind a rack post, and only a shoulder and an arm are in the frame. The head band was not widened: over the whole walk the head is fully covered, and the band reaches the shoulders. Contact sheets with timecodes (all frames 3.5–7.5 and 20.5–23 s before and after, every other frame near the head 5–23 s, 1 frame/s over the whole clip): `test-footage/verify/blur-A-2026-10-31/`. B is byte-identical after the fix.

On B the forklift driver is blurred (1.90 MB): a person box inside the forklift box counts as the driver, and the top half of the box is blurred with a 40 % margin. In s8 SAM returned only the edge of the driver at the right border of the frame, so the box is extended to the upper middle of the forklift box. The first and last box hold for 0.5 s. Checked on 8 frames 4.9–9.0 s.

## How it works
- Stateless server: `safety_mode/webapp.py` (FastAPI), Vercel entry `api/index.py`. The browser holds the session (detections, events, alerts). No WebSocket, each request up to 60 s.
- Page: `web/index.html`, one file with no build step.
- The clips run on committed SAM 3.1 video mode detections (`test-footage/detections/*.video.json`, track id, 1 frame/s; if a prompt is not there, image mode `*.json`), and the SAM API is not called for them. Only safe synonyms: `helmet` and variants → `hard hat`; `vest`, `hi-vis vest`, `high-visibility vest` → `safety vest`; `worker`, `pedestrian`, `people`, `employee` → `person`; `lift truck`, `forklift truck` → `forklift` (`webapp.ALIASES`). `hat`, `vehicle`, `operator`, `man`, `pallet truck` are no longer substituted.
- A prompt not in the cache (since 31.10, `status.live` in the `/api/compile` response) goes to SAM 3.1 live by the same path as the webcam. The page grabs clip frames with a hidden `<video>` at s + 0.5 s (like the `ffmpeg fps=1` frames in the cache), draws them into a canvas at most 1280 px wide (the cache frame size), JPEG 0.85 (lower if above 560 KB), and sends one frame at a time to `/api/detect` with only the missing prompts, 4 requests in parallel, with the line "SAM 3.1 live: 12/24 frames". Then `/api/run` with `extra_detections` `{"<second>": {"<prompt>": [{box 0..1, score}]}}`: the server merges them with the cache by second (cached prompts keep their track id, live ones come as image mode without ids, and the engine then keeps one episode per rule) and returns 409 if live data is missing for even one second-prompt pair. Live detections are stored in `sessionStorage` by clip and prompt: running the same rule again (or another rule with the same object) does not call SAM. On the "How I understood it" card such a prompt has a `live` tag and a line saying SAM 3.1 runs live; the clip caption reads "… · SAM 3.1 live".
- The forklift state (`machine_state`) is set by hand (v1): "running for the whole clip" on all clips except Forklift aisle (0–8 s). No example uses it. Rule Scout does (on A it proposed "nobody on the floor while the forklift is running" and rejected it itself: there is no forklift in the frame).
- Webcam: `/api/detect` (SAM 3.1, one request per prompt, in parallel), then `/api/step` recomputes the engine over all session frames from scratch. Each frame counts as one SAM frame against the limits.
- User rules get their own `rule_id` in the session (`custom-1`, `custom-2`, …), so the report does not merge the alerts of two own rules into one.
- Explanations: Muse Spark for the first 3 alerts of a run, the rest from the rule template (the feed says where the text comes from). The facts for Spark include `subject` and object labels with ids.
- Rule Scout: recorded runs `web/cache/scout/<clip>.json` (static, no cost), live run step by step via `/api/scout/step`, each step as a separate request.
- Layout: in two columns, each column scrolls on its own if it is taller than the window (open Rule Scout panel, vertical clip on a laptop screen), and the alert feed is at least 300 px; in one column the page grows.
- Cache in the repo `demo/cache/`: `compile.json` (4 examples × 4 sample clips and the webcam; key = text + zone names, so A, C and the webcam share a result, and zone titles in a refusal text are filled in for the current camera, `webapp.cached_rule`) and `explain.json` (explanations of the first alerts of the examples). New results on Vercel go to the instance's `/tmp` and live as long as the instance does.

Endpoints: `POST /api/compile {text, clip, rule_id?, camera?}`, `POST /api/run {clip, rule, episodes?: "rule"|"object", detector?: "video"|"image", extra_detections?, camera?}` (the page sends `episodes: "rule"`; without the flag, episodes are per object), `POST /api/scout/step {clip} | {state, sig}`, `POST /api/detect {jpeg, prompts[]}`, `POST /api/step {rule, frames_so_far, frame_period_s, camera?}`, `POST /api/explain {event, rule, clip, camera?}`, `POST /api/report {events, rules, clip}`, `GET /api/health`.

## Own clip and zones (since 31.10, daytime)
The "Your clip" tab after "Webcam".
- The visitor picks a video file. The file is not uploaded: the browser decodes it itself, and `<video>` plays a `blob:` URL. Under the clip there is the line "Your clip. name, size, length; the first 30 s are checked" and a "Choose another" button. The first 30 s are checked (frames at s + 0.5 s, 1 frame/s, the same `frameGrabber` as live SAM on the clips: canvas no wider than 1280 px, JPEG 0.85). The file picker panel says that the file stays in the browser and that frames go to the Meta API when a rule runs. If the browser cannot play the file, the line under the rule asks for MP4 (H.264) or WebM.
- Zones: the "Zones and floor calibration on this camera" block has "Draw a zone" and "Whole frame" buttons. "Draw a zone" moves the clip to the first frame; a click adds a point, a double click or Enter closes the zone (3 points needed), Esc cancels. A click closer than 1.5 % to the frame edge puts the point on the edge: for a person cut off by the frame, the foot point lies on the edge. Each zone has a name field and "delete". The zone key is built from the name (`walkway crossing` → `walkway_crossing`, a duplicate → `_2`), and the compiler gets `walkway_crossing (title "walkway crossing")` like for the clips. Zones are stored in `localStorage` by file name and size (`sm-zones:own:<name>:<bytes>`): the same file opens with its zones next time. Without a zone "Check rule" does not run: "Draw at least one zone on your clip first, or press “Whole frame”".
- There is no floor calibration, so a rule with metres gets "Your clip has no floor calibration…", like the clips without calibration. `machine_state` is not checked either.
- Server (`webapp.camera_from_body`): `camera {zones: [{name, title, polygon}], frame_size, seconds}`; zone name `[a-z][a-z0-9_]*`, 3–64 points, coordinates are clipped to 0..1, up to 8 zones, `seconds` 1–30, duplicate names are rejected. The compile cache key includes zone names and titles. `/api/run` with `clip: "own"` takes only `extra_detections` (all prompts live, no track ids, one episode per rule); the response is the same as for the clips: strip, proof overlay, caption, explanations (the first 3 from Spark), `title: "Your clip"`. In the report the camera is called "Your clip".
- Live detections are stored in `sessionStorage` under the key `own:<name>:<bytes>` and the prompt: a second rule with the same object on the same file does not call SAM.
- Rule Scout does not run on this tab. Instead of the panel there is one sentence: Scout needs SAM 3.1 results for all objects in the clip, and an own clip only has the objects of the visitor's rules. Running Scout on an own clip would mean storing detections on the server between steps.
- Cost of a run: frames × prompts × $0.0025. 22 s and 2 prompts = 44 SAM requests ≈ $0.11; the ceiling of 30 s × 4 prompts = $0.30; plus compilation and up to 3 explanations ≈ $0.04. A repeat on the same file costs $0.

Webcam: the same "Draw a zone" on a camera snapshot (the button pauses the image and stops checking; after the zone is closed the camera continues). Drawn zones are added to the "Warehouse floor" zone (the whole frame) and are stored in `localStorage` (`sm-zones:webcam`); without them everything works as before. In the "Zones and floor calibration" card this zone has been labelled "Whole frame. The whole webcam frame counts as the zone." since 31.10 (`SETUP.webcam.titles` in `web/index.html`), while its `title` stays "Warehouse floor": it is part of the compile cache key of rules with drawn zones (the cup rule in `demo/cache/compile.json`). Screenshot `demo/screenshots/prod/webcam-zone-card-card.png`. `/api/step` and `/api/explain` receive `camera.zones`.

### "Try at your desk" rules
Under the rule field on the webcam tab there are three buttons. Texts and prompts are in `webapp.DESK_RULES`, compilations in `demo/cache/compile.json` (`python demo/precompute.py --desk`, 3 Spark calls; the compile key of the cup rule includes the zone `keyboard_zone` titled "Keyboard zone", and the page offers to draw it). Hold time 2 s: with one frame every 2 s, the alert fires on the first frame with a violation.

| Button | Rule | Engine condition |
|---|---|---|
| Wear glasses | Everyone in view must wear glasses, alert after 2 seconds. | `not wearing(person, [glasses])` |
| Wear headphones | Everyone in view must wear headphones, alert after 2 seconds. | `not wearing(person, [headphones])` |
| No cup in the keyboard zone | No cup in the keyboard zone for more than 2 seconds. | `in_zone(cup, keyboard_zone)`; the button first asks to draw a "Keyboard zone" |

Not taken: Spark rejected "No phone in hand while in view": the engine has no "in hand" primitive (`primitives.py`: `in_zone`, `count_in_zone`, `distance_lt`, `wearing`, `machine_state`; `pose` and `height_above_floor` are not implemented). "No phone in view…" was also rejected: presence in the frame without a zone cannot be expressed. The first compilation of the glasses rule without "alert after 2 seconds" gave 3 s, and once the prompt "safety glasses", which SAM did not find on these glasses. That is why the texts include "alert after 2 seconds" and the results are cached.

Check on production 31.10 (`e2e.py --desk`, Chrome fake camera from Pexels clips with the clip time in the corner, one frame every 2 s; F = a frame sent to SAM):

| Rule, clip | Alert (frames) | What a person sees | Mismatch |
|---|---|---|---|
| Cup, 6785501 (26.8 s, cup by the laptop, 11–17.5 s at the mouth) | F0–F5, F8–F15 (14 of 16) | cup on the desk F0–F4, F8–F15; lifted F5–F7 | F5: the cup has just been lifted and is still above the zone. The run before this: alert F0–F4, F8–F15, a full match |
| Glasses, 7579570 (22.5 s, glasses off from ~9.5 s) | F0, F5–F10, F12–F13 | no glasses F4–F10 and F13 (glasses in hand) | F0 is false: SAM did not find the thin rimless glasses; F4 missed (glasses in hand near the face). Local run: F5–F10, no false alerts |
| Headphones, 7504974 (7.4 s, wearing headphones) + 7579570 from 10 s (no headphones) | F4–F9 (6 of 12) | no headphones F4–F9 | none. Two clips are joined because Pexels had no static shot of a person taking off over-ear headphones. Clip 8128281 (a gamer puts on a headset) was dropped: the screen behind him shows drawn people, and SAM counted them as people without headphones |

Frame sheets: `demo/screenshots/prod/desk-<rule>/NN.jpg` (what went to SAM), screenshots `desk-<rule>-ready.png`, `-alert.png`, `-end.png`, logs `e2e-log-desk-<rule>.txt`.

### Own clip check on production 31.10
`e2e.py --own-clip`: the file `test-footage/raw/pexels-7579570-desk-glasses-off.mp4` via `set_input_files`, the zone "desk" drawn with four mouse clicks and a double click, the name typed into the field, the rule "Everyone at the desk must wear glasses.".

| Step | Result |
|---|---|
| File opened | 0.09 s, "Your clip. …, 2.2 MB, 00:22" |
| Compilation (Spark) | 7.8 s: `in_zone(person, desk)` and `not wearing(person, [glasses])`, 3 s, "Where: desk" |
| Run → strip | 20.6 s, 22 `/api/detect` requests of 0.9–2.6 s each |
| Strip | `.cc.......ccAAAAAAAAAA`: alert 12–21 s (true from 10 s, 3 s required), candidate 1–2 s (glasses pushed up on the forehead); a person sees the glasses off from ~9.5 s |
| Feed | "Glasses required at desk · Your clip, 00:12", the Muse Spark explanation names the "desk zone" |
| Repeat Run | 0.4 s, 0 SAM requests |
| Report | 17.2 s, "camera Your clip", "00:12–00:21 … Glasses required at desk", no zone keys |

Screenshots `own-clip-01-pick.png` … `own-clip-07-report.png`, log `e2e-log-own-clip.txt`. After the deploy, `--no-key --matrix --scout` (no key, 25 checks, $0) and `--live-rule` (with the key) were repeated: everything passed, as on the morning of 31.10.

How to run:
```
R=test-footage/raw; DT="drawtext=text='%{pts\:hms}':x=8:y=8:fontsize=22:fontcolor=yellow:box=1:boxcolor=black@0.7"
ffmpeg -i $R/pexels-6785501-desk-cup.mp4 -vf "fps=10,scale=1280:720,$DT" -pix_fmt yuv420p /tmp/fakecam-desk-cup.y4m
ffmpeg -i $R/pexels-7579570-desk-glasses-off.mp4 -vf "fps=10,scale=1280:-2,pad=1280:720:(ow-iw)/2:(oh-ih)/2,$DT" -pix_fmt yuv420p /tmp/fakecam-desk-glasses.y4m
ffmpeg -i $R/pexels-7504974-desk-headphones-adjust.mp4 -ss 10 -i $R/pexels-7579570-desk-glasses-off.mp4 -filter_complex "[0:v]fps=10,scale=-2:720,pad=1280:720:(ow-iw)/2:0,setsar=1[a];[1:v]fps=10,scale=1280:-2,pad=1280:720:0:(oh-ih)/2,setsar=1[b];[a][b]concat=n=2:v=1[c];[c]$DT[o]" -map "[o]" -pix_fmt yuv420p /tmp/fakecam-desk-headphones.y4m
/tmp/pw/bin/python demo/e2e.py https://safety-mode.vercel.app demo/screenshots/prod --only-new --own-clip            # ~$0.15
/tmp/pw/bin/python demo/e2e.py https://safety-mode.vercel.app demo/screenshots/prod --only-new --desk --desk-rule=cup --desk-frames=16   # glasses, headphones the same way; $0.04-0.08
```
`--only-new` skips the regular pass over the four clips.

## Open access and the operator override (since 2.11)
- No key is needed for anything. Every request that spends Meta API money counts against the limits in "Spending limits": `/api/compile` outside the cache (Spark), `/api/run` explanations (Spark, the first 3 alerts), `/api/explain` outside the cache (Spark), `/api/report` (Spark), `/api/detect` (one SAM frame per request, whatever the number of prompts), `/api/scout/step` (one run at the start, one step per further request). Cached answers never count.
- Over a limit the API answers 429 with one sentence that says when it resets, for example "Hourly limit for this address reached (30 Muse Spark calls), try again in 12 minutes. The sample clips with cached results still work." The daily cap: "The demo's daily limit of SAM 3.1 frames is used up, it resets in 5 hours. …". The page shows the sentence in the line under the rule; the webcam stops on 429. In `/api/run`, an alert whose explanation would go over the Spark limit gets the template text with that sentence as the note.
- `GET /api/health` shows to everyone `limits` (per hour per address and per day per instance), `remaining` (what is left for the calling address this hour) and `remaining_today` (what is left on this instance today). No secrets.
- Operator override via `SM_ACCESS_KEY` (Vercel, production, sensitive; the value is not in git). A request that carries this value in the header read by `webapp.operator_exempt` (or as `?key=` on `/api/health` only) skips the per-address hourly limits; the daily caps still apply. `/api/health` then adds `"exempt": true`. If `SM_ACCESS_KEY` is unset, nothing is exempt. The page has no field for it and never sends it; it is for scripts and checks run by the operator.
- To change the value: `python3 -c "import secrets;print('SM_ACCESS_KEY='+secrets.token_urlsafe(24))"` into the env file, then `vercel env rm SM_ACCESS_KEY production --yes`, `printf '%s' "$SM_ACCESS_KEY" | vercel env add SM_ACCESS_KEY production --sensitive`, `vercel --prod`.

## Refusals and the report (30.10, evening)
- Zone keys (`past_gate`) are not shown in human-facing text: the compiler gets zones as `past_gate (title "Shop floor")` with the instruction "use only the title in prose", and `compiler.humanize_zone_names`, after compilation and for old cache entries, replaces the key with the title in `title`, `unsupported`, `suggested_rewrite`, `explanation_template`. The page does the same for "not checked" reasons and captions (`humanZones`). "Where" for a refusal: "The place this rule names is not marked on this camera". Three refusals were recompiled (3 Spark calls). Screenshot `demo/screenshots/prod/judge-refusal-man-crosses-floor-r8.png`.
- Session report: event time = the engine alert's `start_s`, end = `end_s`, as mm:ss from the start of the clip, as on the alert card (`webapp.report_events`; previously it took `condition_since_s` as HH:MM:SS, and the model wrote "00:00–00:05"). The report input schema has a new optional field `end_time`, and the report template has the event line "<time>–<end_time>". Live check on production: A "00:07–00:21", C "00:02–00:05", B "00:06–00:07" (2 Spark calls: the first showed the correct start without an end, the second came after the template fix). Screenshot `09-report.png`.

## Load times and sizes (production, 30.10, evening)
| What | Value |
|---|---|
| Page `index.html` | 73 KB, 0.41 s (curl) |
| First clip ready to play in headless Chrome, fresh profile | 0.59 s (warm instance), 2.58 s (first visit after a deploy); DOM 0.34–0.54 s |
| A `man-crosses-floor.mp4` (blurred) | 2.19 MB, 0.36 s (2.20 MB on 31.10 after the blur fix) |
| B after blurring the driver | 1.90 MB |
| C `site-worker-no-vest.mp4` | 4.85 MB, 0.45 s |
| B `forklift-crosses-aisle.mp4` | 2.40 MB, 0.35 s |
| Team `warehouse-team.mp4` | 16.3 MB, 1.12 s |
| `/api/compile` cached examples | 0.15–0.45 s |
| `/api/run` | 0.16–0.25 s; once 9.95 s (first request to a cold instance) |

## Response times (production, 29.10, Chrome from Helsinki, `e2e.py`)
| Request | 29.10 | 27.10 |
|---|---|---|
| `/api/compile`, cached example | 0.14–0.4 s | 0.4–0.9 s |
| `/api/compile`, own rule (live Spark) | 5.5 s | 4.3–14 s |
| `/api/run`, explanations cached | 0.16–0.26 s; once 11.9 s (cause not investigated, repeating the same request gave 0.16–0.25 s) | 0.5–0.9 s |
| `/api/run`, new alerts (up to 3 Spark explanations in parallel) | not measured | 8.6 s |
| `/api/explain` (webcam alerts) | 6.9–8.0 s | |
| `/api/report` | 11.0 s | 13.5–16.9 s |
| `/api/detect`, 3 prompts | 1.4–2.0 s | 1.5–2.6 s |
| `/api/step` | 0.16–0.18 s | |
| `/api/scout/step` (live Rule Scout, 15 requests) | 0.2–9.9 s | |
| Whole live Rule Scout, Forklift aisle | 34 s, 7 + 3 Spark calls, $0.12 | |
| First alert after "Run" on Team at the racks, rule 2 | 4.4 s (of which 3 s of clip) | 3.4–5 s |
| First webcam alert after "Start" | 4.3 s | 3.6–5 s |

Each live Rule Scout step fits into the Vercel 60 s limit with a 6× margin: the longest is 9.9 s (an agent round with rule compilation).

## Spending limits
Since 2.11 the limits are the only protection of the live features. The same values are the defaults in `webapp.Limiter` and are set in Vercel (production):

| Vercel variable | Value | 31.10–2.11 (behind the key) | What it does |
|---|---|---|---|
| `SM_LIMIT_SPARK_PER_HOUR` | 30 | 120 | Spark calls per hour per address (compilation, explanation, report; the cache does not count) |
| `SM_LIMIT_SAM_FRAMES_PER_HOUR` | 300 | 600 | SAM frames per hour per address (webcam, own clip and live clip frames); frame = one `/api/detect`, however many prompts it has |
| `SM_LIMIT_SCOUT_PER_HOUR` | 2 | 6 | live Rule Scout runs per hour per address |
| `SM_LIMIT_SCOUT_STEPS_PER_HOUR` | 40 | 120 | Rule Scout steps per hour per address (a signed state can be replayed, this is its ceiling; a run takes about 15 steps) |
| `SM_DAILY_SPARK_CALLS` | 500 | 1500 | Spark calls per day per instance |
| `SM_DAILY_SAM_FRAMES` | 2500 | 5000 | SAM frames per day per instance |
| `SM_DAILY_SCOUT_RUNS` | 30 | 60 | live Rule Scout runs per day per instance |
| `SM_DAILY_SCOUT_STEPS` | 600 | 1200 | Rule Scout steps per day per instance |
| `SM_WEBCAM_MAX_FRAMES` | 40 | 40 | frames in one webcam session (enforced by the page) |
| `SM_LLM_TIMEOUT`, `SM_LLM_MAX_RETRIES` | 28, 0 | the same | so that a Spark request fits into 60 s |
| `SM_ACCESS_KEY` | operator value | the access key | operator override of the per-address limits (section above) |

The counters live in the memory of the function instance. Vercel may keep several instances, so both the hourly per-address limit and the daily limit apply per instance, not to the whole project; a shared counter would need external storage (KV), and there is none. The address is the first entry of `X-Forwarded-For`, which Vercel sets itself. The day turns over at midnight UTC (the instance clock).

## Cost of one session (estimate)
Prices from `api-notes.md` (official price list, 27.10): Spark $1.25 / $4.25 per 1M input/output tokens, SAM 3.1 $2.50 per 1000 image requests.

| Action | Cost |
|---|---|
| Examples on the clips (compilation, run, explanations) | $0 (cache) |
| Own rule: compilation | ~$0.02 |
| Run of an own rule with alerts: up to 3 explanations | ~$0.02–0.03 |
| Session report | ~$0.02–0.03 |
| Webcam: frame = 3 SAM requests | $0.0075 per frame, $0.30 for a full 40-frame session |
| Own rule with a new object on a clip: SAM 3.1 live, one frame per second × new prompts | A 24 frames $0.06, Team 23 frames $0.06, C 11 frames $0.03, B 12 frames $0.03 per prompt; plus compilation and up to 3 explanations ≈ $0.04; a repeat in the same session $0 (browser cache) |
| Live Rule Scout run (8–10 Spark calls) | ~$0.08–0.13 (runs of 29.10) |
| Own clip: one frame per second × prompts, up to 30 s | 22 s × 2 prompts $0.11, ceiling 30 s × 4 prompts $0.30; plus compilation and up to 3 explanations ≈ $0.04; a repeat on the same file $0 |
| "Try at your desk" rule on the webcam | compilation $0 (cache); frame = 1–2 SAM requests, $0.0025–0.005 |

A typical session (2 own rules with a new object, a report, 15 webcam frames): about $0.35–0.45. Ceiling per address per hour: 300 frames × 1–4 prompts ($0.75–3) + 30 Spark calls (~$0.60) + 2 Rule Scout runs (~$0.25), so about $1.60–3.85. Ceiling per day per instance: 2500 frames × 1–4 prompts ($6.25–25) + 500 Spark calls (~$10) + 30 Rule Scout runs (~$3.90), so about $20–39.

## Checked in the browser
`demo/e2e.py`: real headless Google Chrome (Playwright, `channel="chrome"`; the bundled Chromium has no H.264). Without flags: page load, clip order, the default example on each sample clip, Check, Run, alert seconds and strip phases (`EXPECTED` in the script), screenshots `judge-<clip>-alert.png` (paused inside an alert) and `judge-<clip>-end.png`. `--matrix`: 4 examples × 4 clips. `--scout`: recorded Rule Scout runs, "Try this rule". `--webcam`: the webcam is replaced with the forklift-operator clip. `--custom`, `--report`, `--scout-live` spend Spark calls.

Run on production 30.10, 18:15–18:20 (`--webcam --matrix --scout`, then `--matrix --scout` after fixing a race in the script itself): all checks passed, no console errors, the webcam raised an alert 5.6 s after start. Log `demo/screenshots/prod/e2e-log.txt`, screenshots `demo/screenshots/prod/judge-*.png`, `scout-<clip>.png`, `10-webcam-ready.png`, `11-webcam-alert.png`. The older screenshots (`02`–`09`, `12`–`15`, `proof-*`) were taken before the evening of 30.10, on the previous clip set.

Run on production 31.10, 09:00–09:14, after the key and live SAM (`--no-key --matrix --scout` without the key, then `--live-rule --webcam --report` with the key, then `--report` after fixing the report prompt): all checks passed except one in the first run with the key (the report wrote "00:07–00:07": a group of one alert took the start time twice; fixed in `prompts/report.md`, the repeated `--report` passed, "00:07–00:21"). The logs of this run are not kept: the run of 2.11 below replaced them.

| Check, production 31.10 | Result |
|---|---|
| No key: own rule on A | 401, the server's sentence under the rule, the key field focused and highlighted; example 1 then runs from the cache, alert 7–21 s |
| No key: 4 examples × 4 clips, Rule Scout from the recording | as on 30.10, no differences |
| A "Nobody may stand on the shop floor without gloves." | Spark compilation 10.7–18.3 s; SAM 3.1 live, 24 frames in 10.5–10.9 s (request 0.9–3.2 s); from Run to the strip 20.8 s (another ~10 s waits for the Spark explanation of the first alert inside `/api/run`); alert 7–21 s, "person #0 without glove 5–21 s", Muse Spark explanation |
| B "No pallet may stay on the walkway crossing for more than 5 seconds." | compilation 14.4–15.9 s; SAM 3.1 live, 12 frames in 6.2–6.6 s; from Run to the strip 16.1–17.6 s; alert 4–11 s, "pallet (up to 9 at once) in the walkway crossing 0–11 s". The screenshot shows these are pallets on the lower rack levels whose bottom points fall into the hand-drawn crossing zone: the mechanics are right, but it does not work as an example |
| Repeat Run of the same rule | 0 SAM requests, "SAM 3.1 live results from earlier in this session reused", 0.4 s |
| Webcam with the key | alert 3.4–4.3 s after Start |
| Report with the key | 16.3–20.0 s |

Screenshots: `live-rule-<clip>-progress.png` (the line "SAM 3.1 live: k/n"), `live-rule-<clip>-run.png`, `live-rule-<clip>-result.png`, `09-report.png`.

Not checked: a real webcam with a live person, Safari and Firefox, mobile screens, several function instances under load; live SAM on Team at the racks (1280×2276 frames, the heaviest JPEGs) and on C.

Run on production 2.11, 23:43–23:59, after the key gate was removed, all as a plain visitor with no key and no header. Every run checks first that `/api/health` shows the limits and that the page has no key field. All checks passed, no console errors.

| Run | Time | Result | Cost (estimate, prices above) |
|---|---|---|---|
| `--matrix --scout` | 5 min | 4 examples × 4 clips as on 30.10, recorded Rule Scout, "Try this rule"; log `e2e-log.txt` | 2 Spark calls, ~$0.02: matrix alerts with no cached explanation now get a Muse Spark explanation instead of the template |
| `--live-rule --webcam --report` | 5 min | A gloves: compilation 7.0 s, 24 live frames, Run to strip 49.4 s, alert 7–21 s; B pallets: compilation 13.2 s, 12 live frames, Run to strip 37.1 s, alert 4–11 s; repeat Run 0 SAM requests; webcam alert 6.1 s after Start; report 12.1 s with 00:07–00:21; log `e2e-log-live-rule.txt` | 60 SAM requests + ~6 Spark calls, ~$0.25 |
| `--only-new --own-clip` | 1.3 min | compilation 9.4 s, 22 frames × 2 prompts, Run to strip 23.2 s, alert 12–21 s, repeat 0 requests, report 19.6 s; log `e2e-log-own-clip.txt` | 44 SAM requests + 3 Spark calls, ~$0.17 |
| `--only-new --desk --desk-rule=cup|glasses|headphones --desk-frames=16` | 3 × 1 min | alert 1.6 s (cup), 1.7 s (glasses), 9.8 s (headphones) after Start; logs `e2e-log-desk-<rule>.txt` | 83 SAM requests + ~6 Spark calls, ~$0.27 |

Total about $0.70. Also checked by hand on production: `/api/health` with the operator value in the header or in `?key=` returns `"exempt": true`, with a wrong value no `exempt`; a `/api/detect` request with a forged `X-Forwarded-For` counted against the real address (Vercel replaces the header), 1 SAM request; the third and fourth Rule Scout start in an hour returned 429 "Hourly limit for this address reached (2 live Rule Scout runs), try again in 60 minutes. …". The `remaining` numbers in the logs jump between runs because requests land on different instances, each with its own counters.

## How to update
```
cd <repo>
demo/build.sh                         # clean copy without .git in /tmp/safety-mode-deploy (Hobby blocks the git author)
cd /tmp/safety-mode-deploy && /usr/local/bin/vercel --prod --yes --scope <vercel-scope>
env -u SM_ACCESS_KEY /tmp/pw/bin/python demo/e2e.py https://safety-mode.vercel.app demo/screenshots/prod --matrix --scout   # $0; exit code 1 if anything does not match
/tmp/pw/bin/python demo/e2e.py https://safety-mode.vercel.app demo/screenshots/prod --live-rule --webcam --report [--scout-live]   # as a plain visitor, ~$0.25
/tmp/pw/bin/python demo/proof_shots.py https://safety-mode.vercel.app demo/screenshots/prod   # proof overlay, proof-*.png
```
- `build.sh` takes the clips from `test-footage/raw/` (they are not in git, Pexels links are in `test-footage/README.md`), copies the video stream without re-encoding, strips the audio and moves moov to the start of the file; for A it takes the blurred file (`demo/blur_heads.py`). Currently A 2.2, C 4.9, B 2.4, Team 16.3 MB, plus the hidden Operator close-up 10.6 and Forklift aisle 17.3 MB. If a clip is larger than 40 MB, the script re-encodes it to 1280 px, H.264, CRF 26.
- The `.vercel` folder in the copy is kept between builds (the project link).
- Key: `MODEL_API_KEY` in Vercel (production, sensitive); the Spark key comes from it too. To change it: `vercel env rm MODEL_API_KEY production`, then `printf '%s' "$KEY" | vercel env add MODEL_API_KEY production --sensitive`, then `vercel --prod`.
- To refresh the example cache after changing the compiler prompt: delete `demo/cache/compile.json`, then `export MODEL_API_KEY=<key>; SM_CACHE_WRITE=1 python demo/precompute.py --explain` (about $0.30; on the evening of 30.10 it took 18 Spark calls: 8 compilations and 10 explanations, 3 of them redone after zone titles of the current camera appeared in rule titles).
- Locally: `cd /tmp/safety-mode-deploy && SM_STATIC_DIR=. uvicorn api.index:app --port 8090` with `MODEL_API_KEY` in the environment.
- Playwright: `python3 -m venv /tmp/pw && /tmp/pw/bin/pip install playwright`; the browser is the system `/usr/bin/google-chrome`. Fake camera: `ffmpeg -ss 1 -t 12 -i test-footage/raw/pexels-32243653-forklift-operator.mp4 -vf scale=960:540,fps=10 -pix_fmt yuv420p /tmp/fakecam.y4m`.

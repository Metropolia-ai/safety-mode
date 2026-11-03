# Expected rule results on the test clips

Labelled by hand on 21.10.2026 from the frames `frames/<clip>/NNN.jpg` (1 fps, frame NNN = second NNN−1) and the zones from `zones/<clip>.json`. All frames of the three clips were reviewed, plus zoomed crops where the decision is debatable. The rules come from `../rules-examples.md`.

**This is a proxy.** At the hackathon, detection runs through SAM 3.1 in the Meta Model API. The second table below is local SAM 3.1 FP32 from the Jetson bench, not the API. The first API run (27.10) is compared with these labels at the end of this file.

## Conventions the tables rely on
- **Second = frame.** 1 frame per second, each frame "costs" 1 s. A condition true on k consecutive frames lasts k s. The alert fires in the second where the duration first reaches ≥ `min_duration_s`. With a 1 s window this is the first frame. With 3 s it is the third consecutive frame (condition from s0 → alert at s2).
- **One alert per episode.** "Must fire" = the seconds where the event is active (from the moment it fires to the last frame with a true condition). A one-frame gap (the object disappears for 1 s) does not close the event.
- **`in_zone` by the ground point:** the middle of the bottom edge of the box (a person's feet, a vehicle's wheels). Not by the overlap of the box with the zone: on a handheld camera and next to racks, box overlap gives false alerts. This is marked as a risk below.
- **`distance_lt` on the floor:** the ground points of both objects go through the homography from `floor_calibration`. The distance is in metres between the nearest points of the bottom edges. Without calibration (`null`) the rule is not evaluated.
- **The operator is not a pedestrian.** For `distance_lt(forklift, person)`, a person is excluded if their box is ≥ 50 % inside the forklift box and their feet are above the bottom edge of the forklift (by ≥ 10 % of its height; added 27.10: the reach truck box with the mast covers pedestrians next to it). Otherwise rule 3 fires all the time on the driver. For rule 2 the operator is an ordinary person: they also need a hard hat and a vest.
- **"Uncertain"** = seconds where the frame does not allow a confident yes/no (zone border ±0.3 m, 2 m ± calibration error). An alert there, or no alert, is not counted as an error.
- `machine_state` in v1 is set by hand. For each clip below, the accepted value is given.

## 1. pexels-6079421-forklift-busy-warehouse (25 s, s0–s24)

What is in the frame: a Toyota forklift with a film-wrapped pallet on the forks reverses out of the roller door opening towards the camera (s0–s8), then stands in the aisle left of the yellow line (s9–s24). At s23–s24 the operator starts to get out of the cab. The operator wears a baseball cap, no hard hat and no vest. A pedestrian in a grey T-shirt walks **beyond** the door threshold, deep in the corridor (15+ m behind the forklift by calibration), and does not enter the `past_gate` zone. Another person in a turquoise T-shirt stands at the racks in the far background on the left (~16 m from the camera, ≥5 m from the forklift). The operator's hand on the grab handle while reversing (s1–s6) looks like "a person behind the forklift". There is no separate person there. The camera is handheld: up to s4 the scene matches frame 001, by s9 it shifts ~50 px to the right, more at the bottom right (parallax). The lane polygon on s9–s24 lands on the forklift body (see risks).

`machine_state(forklift)`: running = moving s0–s8, stopped s9–s24 (from the frames; set by hand in v1).

**Rule 1 (variant): nobody in `yellow_zone` while the forklift is moving**, min 1 s, high

| Seconds | Alert | Why |
|---|---|---|
| s0–s24 | must not fire | There is not a single person in the strip between the yellow line and the racks. The operator sits in the cab left of the line. |
| — | risk | On s9–s24 (camera drift) the zone polygon lies on the forklift body. If `in_zone` is computed by box overlap, the operator's box will catch the zone. By the ground point it will not. The forklift is standing at that time, so with `machine_state=stopped` there is no alert in any case. |

**Rule 2: hard hat and vest past the gate (`past_gate`)**, min 3 s, medium

| Seconds | Alert | Why |
|---|---|---|
| s0–s1 | must not fire | The condition is true (operator without hard hat and vest), but the 3 s window is not filled yet. |
| s2–s24 | **must fire** (one event) | The operator without hard hat and vest is visible in every frame. The person at the racks on the left also has no vest (the hard hat cannot be made out in the far background). |
| — | do not count | The pedestrian beyond the door (past the threshold) is outside `past_gate`. If they get into the event, it is a zone labelling error, not a detection error. |

**Rule 3: a forklift no closer than 2 m to a pedestrian**, min 1 s, high

| Seconds | Alert | Why |
|---|---|---|
| s0–s24 | must not fire | The pedestrian beyond the door is 15+ m behind the forklift and moving away. The person at the racks is ≥5 m away. The operator is excluded (inside the forklift box). |
| — | risk | Without the operator exclusion the rule fires for all 25 s. The operator's hand on the grab handle (s1–s6) can give a separate person mask right at the forklift: SAM 3.1 finds 4–5 person on s0–s5 with 3 real people (second table). |

**Rule 5: the fire exit is kept clear (`fire_exit_zone`)**, min 60 s, medium

| Seconds | Alert | Why |
|---|---|---|
| s0–s24 | must not fire | The clip is 25 s < the 60 s window. A candidate (condition true) is expected on s0–s5: the pallet on the forks passes through the zone in front of the door. From s6 the pallet is closer to the camera than the zone. |
| — | risk | The pallet with boxes left of the opening stands right against the corner of the zone: outside the zone by the ground point, inside by box overlap. |

**Rule 8: no forklifts on the pedestrian lane (`pedestrian_lane`)**, min 2 s, medium

| Seconds | Alert | Why |
|---|---|---|
| s0–s24 | must not fire | The forklift is left of the yellow line the whole time (right wheels ~0.5 m from the line on s0–s4). |
| — | risk | On s9–s24, because of camera drift, the forklift box overlaps the lane polygon. The ground point (the middle of the rear edge) does not. Box overlap → false alert s10–s24. |

## 2. pexels-15170997-warehouse-team (23 s, s0–s22, vertical)

What is in the frame: the camera is almost still (handheld), a reach truck drives along the aisle: at s0 it stands at the racks on the right, s2–s9 centre-left, s10–s13 at the far left wall, s14–s22 in the centre again. People: **A** wears an orange reflective coverall and a hood/balaclava, no hard hat visible. **B** wears an orange vest over black and a hard hat with a liner. **C** wears a black jacket, no vest, a hood, no hard hat. The reach truck **operator** wears a hood. At s22 on the right, at the racks, there is one more person in a vest (far background) and sparks (welding/cutting). B, A and C together carry and place a rack frame (s6–s9), then lay it on the floor. At s21 B passes right next to the camera and covers almost the whole frame.

Correction to `README.md`: it says "two in orange vests and hard hats". From the frames, a hard hat is clearly visible only on B. A wears a reflective coverall (hi-vis class, like a vest) and a hood.

`machine_state(reach truck)`: running for the whole clip (the reach truck is working, drives s0–s4, s9–s20; set by hand in v1).

**Rule 1 (variant): nobody in `yellow_zone` (the reach truck parking spot in frame 001) while the machine is working**, min 1 s, high

| Seconds | Alert | Why |
|---|---|---|
| s2 | **must fire** | B stands in the zone at the spot the reach truck has just left. |
| s4 | **must fire** | C stands in the zone at the rack frame. |
| s18 | **must fire** | A walks through the zone. |
| s22 | **must fire** | The person in a vest in the far background at the racks stands in the zone. |
| s1, s3, s5, s16, s19 | uncertain | The feet of B/C/A are on the zone border (±0.3 m). |
| s0, s6–s15, s17, s20, s21 | must not fire | Nobody in the zone. At s0 the reach truck itself stands there with the operator on the platform (the operator is not on the floor). |
| — | limitation | The zone is static, the reach truck leaves it by s2: it is "around the machine" only on s0–s1. An honest variant of rule 1 needs a zone tied to the machine (a buffer around the mask). That is not a v1 primitive. |

**Rule 2: hard hat and vest past the gate (`past_gate` = the whole floor)**, min 3 s, medium

| Seconds | Alert | Why |
|---|---|---|
| s0–s1 | must not fire | s0: only the operator on the reach truck (feet on the platform at the top edge of the zone, uncertain). s1: condition true, window not filled. |
| s3–s22 | **must fire** (one event; s21 is a 1-frame gap) | C without vest and hard hat is visible s1–s12, s15, s19–s20. A without a hard hat is visible almost all the time. The operator has no vest. At s21 only B is visible in close-up (in PPE). |

Variant for the demo, **"vest only"** (hard hats in the far background are not found reliably; A's coverall counts as hi-vis):

| Seconds | Alert | Why |
|---|---|---|
| s3–s12 | **must fire** | C without a vest is in frame s1–s12 in a row. |
| s15, s19–s20 | must not fire | C is visible for 1–2 frames in a row, the 3 s window is not filled. |
| s0–s2, s13–s14, s16–s18, s21–s22 | must not fire | Nobody without a vest, except possibly the operator on the reach truck: no vest is visible on them (s0, s5–s6, s17). If SAM finds the operator, there will be an alert. We count it as uncertain. |

**Rule 3: a forklift (reach truck) no closer than 2 m to a pedestrian**, min 1 s, high

| Seconds | Alert | Why |
|---|---|---|
| s1–s13 | **must fire** | People work right next to the reach truck: s1–s2 B and C at its side, s3–s9 B/C/A at 0.5–1.8 m (the frame is placed right at the mast), s10–s13 A and B kneeling at the reach truck by the far wall. |
| s15–s20 | **must fire** | s15 B and C crouch at the forks, s16–s17 B bends over in front of the reach truck (~1–1.5 m), s18 B walks past, s19 C at the forks, s20 C ~0.7 m from the forks. |
| s22 | **must fire** | A passes ~1.5 m from the reach truck. |
| s0 | must not fire | Only the operator (excluded). |
| s14 | must not fire (uncertain) | The nearest person (B) is ~2.7 m away by calibration. |
| s21 | must not fire | B in close-up at the camera, ~5 m from the reach truck. The others are hidden. |
| — | accuracy | The calibration of this clip is ±20–30 % (see `zones/…json`), so the seconds with a distance of 1.5–2.5 m (s8, s9, s14, s17, s18) can go either way. In the demo this clip shows rule 3 "on the edge", not as a clean example. |

**Rule 6: no climbing on racks (a person higher than 1.2 m above the floor in `rack_zone`)**, min 2 s, high

| Seconds | Alert | Why |
|---|---|---|
| s0–s22 | must not fire | Nobody climbs a rack. The people at the racks (s1–s3) stand on the floor. The frame placed on s6–s9 is a rack part in their hands, not a climb. |
| — | risk | `height_above_floor` by the bottom edge of the mask: the operator on the reach truck platform (~0.3 m) in the zone at s0 is below the threshold. If SAM merges a person with the raised frame, the bottom edge does not change, so there will be no alert. |

## 3. pexels-32243653-forklift-operator (10 s, s0–s9)

What is in the frame: several shots of one drive (shot changes at s1, s3, s7). The operator wears a red hard hat and a grey T-shirt, **no vest**, and carries a film-wrapped roll on the forks. At s0 at the left edge there is a blurred person in a light blue shirt with a phone (far background). At s9 at the door there is a pink scooter/cart, not a person. The floor is visible only from s3 and from different viewpoints. There is no calibration (`floor_calibration: null`), and there is one zone, the whole frame.

**Rule 1**: not applicable, the clip has no `yellow_zone`. Expected: no alerts on s0–s9. The compiler/engine must report that the zone is not marked, not silently evaluate.

**Rule 2: hard hat and vest past the gate (`past_gate` = the whole frame)**, min 3 s, medium

| Seconds | Alert | Why |
|---|---|---|
| s0–s1 | must not fire | The condition is true (hard hat present, no vest), the window is not filled. |
| s2–s9 | **must fire** (one event) | The operator is visible in every frame, no vest. The hard hat is present, so the missing PPE in the explanation = the vest only. At s7 the operator is small behind the mast (SAM 3.1 did not find a person at s7, this is a 1-frame gap). |
| — | uncertain | The blurred person at the edge at s0: PPE not visible. A single frame, it does not affect the event. |

**Rule 3: a forklift no closer than 2 m to a pedestrian**, min 1 s, high

| Seconds | Alert | Why |
|---|---|---|
| s0–s9 | must not fire | There are no pedestrians nearby (except the blurred background at s0), and without floor calibration `distance_lt` is not evaluated. The engine must mark the rule as "not computable on this camera". |

## 4. pexels-38707991-man-crosses-floor (25 s, s0–s23, tripod)

What is in the frame: a warehouse with racks, the camera is still. At s5 a person in dark clothes enters from the right, without a hard hat and without a vest. They walk across the floor from right to left (ground point x 0.72 → 0.12) and are still visible at the left edge at s21. There are no other people. The face is recognisable: on the page the head is blurred. Zone `past_gate` ("Shop floor"): the whole floor, y ≥ 0.62. No calibration.

**Example 1 (hard hat and vest on the floor), example 2 (vest on the floor)**, min 3 s, medium

| Seconds | Alert | Why |
|---|---|---|
| s0–s4 | must not fire | No people. |
| s5–s6 | must not fire (candidate) | A person in the zone without hard hat and vest, the 3 s window is not filled. |
| s7–s21 | **must fire** (one event, person #0) | The person walks across the floor without hard hat and vest. |
| s22–s23 | must not fire | The person has left. A false box at the top edge is outside the zone and does not affect the alert. |

Examples 3 and 4: not checked (no floor calibration, no crossing zone).

## 5. pexels-31751344-forklift-crosses-aisle (0–12 s of the source, s0–s11, tripod)

What is in the frame: the front cross aisle of a warehouse, behind it an aisle between two rows of racks. A forklift enters from the left, crosses the continuation of the aisle between the racks and leaves to the right. Zone `pedestrian_lane` ("Walkway crossing"): a trapezoid (0.22, 0.66), (0.74, 0.66), (0.92, 1), (0.06, 1), not painted on the floor. No calibration.

**Example 4 (no forklift on the walkway crossing)**, min 2 s, medium

| Seconds | Alert | Why |
|---|---|---|
| s0–s3 | must not fire | No forklift. |
| s4 | uncertain | Only the forks come in, SAM 3.1 does not find them. There must be no alert yet. |
| s5 | must not fire (candidate) | Forklift #0 in the zone, the 2 s window is not filled. |
| s6–s7 | **must fire** (one event, forklift #0) | The forklift is on the crossing. |
| s8–s11 | must not fire | The rear goes past the right edge, outside the zone. Then the frame is empty. |

Examples 1–3: no pedestrians. SAM 3.1 finds only the forklift driver (person #0, s5–s8). There is no floor zone and no calibration, so the floor rules and the 2 m rule are not checked here.

## 6. pexels-32245123-site-worker-no-vest (11 s, s0–s10, tripod)

What is in the frame: a construction site by scaffolding. Person #1 without a vest walks to the edge of the frame and disappears after s5. Person #0 in an orange vest stays behind the scaffolding posts. The faces are small, the people cannot be recognised. Zone `past_gate` ("Site floor"): y ≥ 0.70, the full width. No calibration.

**Example 2 (vest on the floor)**, min 3 s, medium

| Seconds | Alert | Why |
|---|---|---|
| s0–s1 | must not fire (candidate) | #1 without a vest. #0 in a vest (0.53). |
| s2 | **must fire** | #1 without a vest for the third second in a row. |
| s3 | **must fire** (holds a 1 s gap) | The vest of #0 fell into the middle third of the #1 box (0.39): the condition is false, `max_gap_s` holds the alert. An overlap error without masks. |
| s4–s5 | **must fire** | #1 walks to the edge without a vest. |
| s6–s9 | must not fire | Only #0 in a vest (0.52–0.59). |
| s10 | must not fire (candidate) | #0 is hidden by the scaffolding posts, the vest is not found. One second, the window is not filled. |

Example 1 (hard hat and vest): both have hard hats, SAM 3.1 finds them on s0–s9 (#0) and s0–s5 (#1). So the expectation is the same as for example 2: alert s2–s5 on #1.

## Summary: where the alert must fire

| Clip | Rule 1 | Rule 2 | Rule 3 | Other |
|---|---|---|---|---|
| busy-warehouse | no | s2–s24 | no | 5: no (clip < 60 s), 8: no |
| warehouse-team | s2, s4, s18, s22 (+5 uncertain) | s3–s22; "vest only" s3–s12 | s1–s13, s15–s20, s22 | 6: no |
| forklift-operator | not applicable | s2–s9 (vest) | no (no calibration) | — |
| man-crosses-floor | — | s7–s21 (and "vest only") | no calibration | — |
| forklift-crosses-aisle | — | no people | no calibration | 8 (example 4): s6–s7 |
| site-worker-no-vest | — | "vest only" s2–s5 | no calibration | — |

What the clips do not have and what cannot be checked: a person in the yellow zone at a **running** machine as a clean example (rule 1 on warehouse-team relies on a static zone), a person lying down (rule 10), a forklift approaching a pedestrian on a clip with good calibration (busy-warehouse gives only a negative example). See `README.md`, "Missing".

## What SAM 3.1 finds per second (local bench, proxy)

Source: `../jetson-bench/logs/sam3.1-fp32_<clip>.json`, SAM 3.1 FP32 on Jetson AGX Orin, one prompt per frame, masks with score ≥ 0.5. The numbers are how many masks were found. Output of `python3 presence_from_bench.py` (it also checks that repeats of looped frames give the same numbers: no mismatches).

Matching seconds: the bench cuts the clip itself with ffmpeg `fps=1` (long side 1280, i.e. warehouse-team is 720×1280, not 1280×2276 as in `frames/`), loops N unique frames and spends the first 5 frames on warmup. So `found_per_frame[i]` = second `(i + 5) mod N`, not the first N records in a row.

Important: this is a local model with a 0.5 threshold. SAM 3.1 in the Meta API does not return a score (`../api-notes.md`), so the threshold cannot be set there. The API numbers may differ.

### pexels-6079421-forklift-busy-warehouse (N=25, warmup=5, SAM 3.1 FP32, score>=0.5)

| s | person | forklift | high-visibility vest | hard hat |
|---|---|---|---|---|
| 0 | 4 | 1 | 0 | 0 |
| 1 | 5 | 1 | 0 | 0 |
| 2 | 4 | 1 | 0 | 0 |
| 3 | 4 | 1 | 0 | 0 |
| 4 | 4 | 1 | 0 | 0 |
| 5 | 4 | 1 | 0 | 0 |
| 6 | 2 | 1 | 0 | 0 |
| 7 | 2 | 1 | 0 | 0 |
| 8 | 2 | 1 | 0 | 0 |
| 9 | 2 | 1 | 0 | 0 |
| 10 | 3 | 1 | 0 | 0 |
| 11 | 2 | 1 | 0 | 0 |
| 12 | 3 | 1 | 0 | 0 |
| 13 | 2 | 1 | 0 | 0 |
| 14 | 2 | 1 | 0 | 0 |
| 15 | 2 | 1 | 0 | 0 |
| 16 | 2 | 1 | 0 | 0 |
| 17 | 2 | 1 | 0 | 0 |
| 18 | 2 | 1 | 0 | 0 |
| 19 | 3 | 1 | 0 | 0 |
| 20 | 2 | 1 | 0 | 0 |
| 21 | 2 | 1 | 0 | 0 |
| 22 | 2 | 1 | 0 | 0 |
| 23 | 2 | 1 | 0 | 0 |
| 24 | 2 | 1 | 0 | 0 |

### pexels-15170997-warehouse-team (N=23, warmup=5, SAM 3.1 FP32, score>=0.5)

| s | person | forklift | high-visibility vest | hard hat |
|---|---|---|---|---|
| 0 | 1 | 1 | 0 | 0 |
| 1 | 4 | 1 | 1 | 0 |
| 2 | 3 | 1 | 1 | 1 |
| 3 | 2 | 0 | 1 | 0 |
| 4 | 3 | 0 | 0 | 0 |
| 5 | 4 | 1 | 1 | 0 |
| 6 | 3 | 0 | 1 | 0 |
| 7 | 4 | 1 | 2 | 0 |
| 8 | 4 | 1 | 2 | 0 |
| 9 | 4 | 1 | 2 | 0 |
| 10 | 3 | 1 | 2 | 1 |
| 11 | 3 | 1 | 1 | 0 |
| 12 | 5 | 1 | 2 | 0 |
| 13 | 1 | 1 | 2 | 0 |
| 14 | 2 | 1 | 2 | 0 |
| 15 | 3 | 1 | 2 | 0 |
| 16 | 2 | 1 | 2 | 0 |
| 17 | 2 | 1 | 1 | 0 |
| 18 | 2 | 1 | 2 | 0 |
| 19 | 3 | 2 | 0 | 0 |
| 20 | 2 | 1 | 1 | 0 |
| 21 | 1 | 1 | 1 | 0 |
| 22 | 3 | 1 | 2 | 2 |

### pexels-32243653-forklift-operator (N=10, warmup=5, SAM 3.1 FP32, score>=0.5)

| s | person | forklift | high-visibility vest | hard hat |
|---|---|---|---|---|
| 0 | 2 | 0 | 0 | 1 |
| 1 | 1 | 1 | 0 | 1 |
| 2 | 1 | 1 | 0 | 1 |
| 3 | 1 | 2 | 0 | 1 |
| 4 | 1 | 1 | 0 | 1 |
| 5 | 1 | 2 | 0 | 2 |
| 6 | 1 | 2 | 0 | 2 |
| 7 | 0 | 1 | 0 | 0 |
| 8 | 1 | 0 | 0 | 1 |
| 9 | 1 | 0 | 0 | 1 |

How to read this against the expectations:
- busy-warehouse: `high-visibility vest` and `hard hat` = 0 in all seconds, so rule 2 will fire on the operator. `person` 4–5 on s0–s5 with three real people: the extra mask is the operator's hand on the grab handle or a fragment. For rule 3 it is a false alert candidate if the operator exclusion fails. `forklift` = 1 everywhere.
- warehouse-team: `hard hat` is almost always 0 (B's hard hat is not found), so the full variant of rule 2 will also fire on B. Show the "vest only" variant. `vest` 1–2: B's vest and A's coverall. `forklift` 0 on s3, s4, s6: the reach truck is hidden by people and the frame in these seconds. For rule 3 this is a miss on s3–s4, s6 (the event holds thanks to the neighbouring seconds).
- forklift-operator: `hard hat` 1–2 almost everywhere, `vest` 0, so rule 2 "no vest" will fire. `person` 0 on s7 is a 1-frame gap, the event does not break. `forklift` 0 on s0 (close-up of the cab) and s8–s9 (the roll hides the forklift).

## SAM 3.1 through the Meta API, 27.10

Run `python -m safety_mode.detect.batch`: frames 001, 005, 009, 013, 017, 021 (forklift-operator only 001, 005, 009), prompts person, forklift, high-visibility vest, hard hat, 60 `sam-3.1` requests with `mask_encoding=one_bit` and `include_confidence=true`. The frame was sent with the long side no more than 1280 px (warehouse-team 720×1280, as in the bench), and the coordinates were converted to `frames/` pixels. Overlays: `detections/<clip>_<frame>_<prompt>.jpg`, raw numbers and boxes: `detections/summary.json`, API responses in the cache `cache/sam31/` (not in git).

The request has no threshold, but all confidences from the API are ≥ 0.507. It looks like the API itself cuts below 0.5, like the bench. So the numbers below can be compared with the bench directly.

#### pexels-6079421-forklift-busy-warehouse

Cell: API / Jetson bench, API confidence in brackets.

| frame (s) | person | forklift | high-visibility vest | hard hat |
|---|---|---|---|---|
| 001 (s0) | 3 / 4 (0.61, 0.66, 0.62) **≠** | 1 / 1 (0.84) | 0 / 0 | 0 / 0 |
| 005 (s4) | 4 / 4 (0.81, 0.65, 0.87, 0.53) | 1 / 1 (0.91) | 0 / 0 | 0 / 0 |
| 009 (s8) | 2 / 2 (0.82, 0.80) | 1 / 1 (0.94) | 0 / 0 | 0 / 0 |
| 013 (s12) | 2 / 3 (0.91, 0.61) **≠** | 1 / 1 (0.93) | 0 / 0 | 0 / 0 |
| 017 (s16) | 2 / 2 (0.85, 0.85) | 1 / 1 (0.94) | 0 / 0 | 0 / 0 |
| 021 (s20) | 1 / 2 (0.75) **≠** | 1 / 1 (0.94) | 0 / 0 | 0 / 0 |

#### pexels-15170997-warehouse-team

Cell: API / Jetson bench, API confidence in brackets.

| frame (s) | person | forklift | high-visibility vest | hard hat |
|---|---|---|---|---|
| 001 (s0) | 1 / 1 (0.62) | 1 / 1 (0.51) | 0 / 0 | 0 / 0 |
| 005 (s4) | 3 / 3 (0.96, 0.96, 0.87) | 0 / 0 | 0 / 0 | 0 / 0 |
| 009 (s8) | 4 / 4 (0.96, 0.96, 0.84, 0.96) | 1 / 1 (0.54) | 2 / 2 (0.69, 0.92) | 0 / 0 |
| 013 (s12) | 4 / 5 (0.59, 0.93, 0.95, 0.94) **≠** | 1 / 1 (0.73) | 2 / 2 (0.64, 0.78) | 0 / 0 |
| 017 (s16) | 2 / 2 (0.97, 0.93) | 1 / 1 (0.77) | 2 / 2 (0.89, 0.61) | 1 / 0 (0.52) **≠** |
| 021 (s20) | 2 / 2 (0.95, 0.98) | 1 / 1 (0.89) | 1 / 1 (0.66) | 0 / 0 |

#### pexels-32243653-forklift-operator

Cell: API / Jetson bench, API confidence in brackets.

| frame (s) | person | forklift | high-visibility vest | hard hat |
|---|---|---|---|---|
| 001 (s0) | 2 / 2 (0.90, 0.96) | 0 / 0 | 0 / 0 | 1 / 1 (0.75) |
| 005 (s4) | 1 / 1 (0.80) | 1 / 1 (0.71) | 0 / 0 | 1 / 1 (0.85) |
| 009 (s8) | 1 / 1 (0.77) | 0 / 0 | 0 / 0 | 1 / 1 (0.62) |

55 of 60 cells match. Four mismatches are in `person`: the API finds 1 person fewer, in the far background (busy-warehouse s0, s12, s20; warehouse-team s12). The fifth is B's hard hat on warehouse-team s16 (API 1, bench 0). The forklift and the vest match the bench in all 15 frames.

Against the human expectations (overlays reviewed: busy-warehouse 001, 021 person; warehouse-team 005 person, 009 vest, 017 hard hat; forklift-operator 005 hard hat):
- **The vest on warehouse-team is found** on s8, s12, s16 on two people: B's vest (0.92) and A's reflective coverall (0.69). C (black jacket) has no vest in any frame. Rule 2 "vest only" will fire on C, as expected. Risk: at s4 B in a vest is clearly visible, but `vest` = 0 both in the API and in the bench. With `min_duration_s` 3 s, one such second will not raise an alert.
- **The operator's hard hat on forklift-operator is found** in all three frames (s0, s4, s8; 0.62–0.85), and the box sits exactly on the red hard hat. `vest` = 0 everywhere, so rule 2 "no vest" will fire.
- **The forklift is found**: busy-warehouse 6 of 6 (0.84–0.94). The warehouse-team reach truck 5 of 6, missed at s4 (hidden by people, as in the bench), low confidence (0.51–0.89). forklift-operator only at s4, 0 at s0 (cab close-up) and s8 (the roll hides it), as in the bench.
- busy-warehouse: the operator without a hard hat has `hard hat` = 0 and `vest` = 0 in all frames, so rule 2 will fire. A small `person` box to the right of the operator at s0 and s4 (around (789, 299) and (870, 302) in frame pixels) is, by position, the operator's hand on the grab handle, as in the bench. For rule 3 it is cut off by the exclusion "box ≥ 50 % inside the forklift". The pedestrian in the door opening is found at s4, not found at s0.
- warehouse-team `hard hat`: B's hard hat is found once out of six (s16, 0.52). The conclusion is the same as from the bench: the full variant of rule 2 cannot be shown on this clip, only "vest".

Result: for the demo the API behaves like local SAM 3.1, and the alert intervals in the tables above do not need to change. Not checked: the frames between the selected ones (15 of 58 were run), prompts with synonyms ("safety vest", "helmet").

Video (`input_video`, forklift-operator, re-encoded to 1280×720, 25 fps, 254 frames, prompt person): 254 lines `<0f>`…`<253f>` came back in a row. The person with id 0 is in all 254 frames, including the shot changes at s1, s3, s7. A second id 1 is on frames 1–13 (the blurred person on the left at s0). Confidence 0.79–0.96. Upload 5.4 s, segmentation 12.4 s, response 100 KB. Raw response: `../tests/fixtures/sam31/pexels-32243653-forklift-operator_video1280_person.txt`.

### Prompt synonyms (27.10, 80 requests)

Vest: warehouse-team frames 003–012, four prompts. Reference from the overlays: B's vest (orange vest) is visible in all 10 frames, A's coverall on 006, 008–012, 16 instances in total. C and the operator have no vest.

| Prompt | Found of 16 | Misses | False |
|---|---|---|---|
| `high-visibility vest` | 14 | 005 (B), 012 (B behind A) | 0 |
| `safety vest` | **16** | — | 0 |
| `reflective vest` | 15 | 005 (B) | 0 |
| `hi-vis vest` | 15 | 012 (B behind A) | 0 |

`safety vest` was also run on all 58 frames. On busy-warehouse and forklift-operator it finds 0 (there are no vests, no false ones). On warehouse-team outside 003–012 it differs from `high-visibility vest` in two frames: 014 (`high-visibility vest` has an extra second box at the edge of B's vest) and 016 (`safety vest` has an extra box on the reflective strip of B's hard hat liner; it does not fall into the middle third of the person and does not affect `wearing`).

Hard hat: frames 001–010 of warehouse-team and forklift-operator, `hard hat` / `helmet` / `safety helmet`.

| Clip | `hard hat` | `helmet` | `safety helmet` |
|---|---|---|---|
| forklift-operator (hard hat in all 10 frames) | 9/10, miss 008 (s7) | 9/10, miss 008 | 7/10, misses 007–009 |
| warehouse-team (hard hat only on B, visible on 002–010) | 0/9 | 3/9 (002, 009, 010) plus boxes on hoods: C on 010 (false), A and the operator on 002, 009, 010 (no hard hat visible under the hood, counted as false) | 0/9 |

**Decision:** vest `high-visibility vest` → `safety vest` (the miss at s4, which could make the "vest only" rule alert on B, is gone). The hard hat stays `hard hat`: `helmet` finds B more often, but fires on hoods, and a false hard hat hides a violation. The prompt was changed in `rules/r02.json`, `test-footage/rules/r02-vest.json`, `rules-examples.md`, `safety_mode/prompts/compiler.md`, `pipeline.DEFAULT_PROMPTS`, `detect/batch.py`. The per-frame tables above stay with `high-visibility vest`: that is how the bench and the first API run went. A live compiler run with the new prompt was not done.

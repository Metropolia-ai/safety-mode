# Frame-by-frame alert check, 30.10.2026

Question: do the alerts fire correctly? This file covers every second of every example on every clip where the rule is checked. Each second was viewed by eye and compared with what the engine decided.

## How we checked
- `test-footage/verify/render_verify.py` runs the rules through the same `/api/run` as the page: SAM 3.1 video mode with track ids, one episode per rule, rules from `rules/` and `test-footage/rules/` (their conditions match what the compiler produces for the page examples).
- Each frame at 1 fps shows the rule zones, every box with its id, and the anchor points. For r03 it also shows a line between the closest points of the bottom edges of the forklift and the person, with the distance in metres. For r02 the person is red (PPE not found) or green. The header shows the engine state for that second and the measurements. `NNN.jpg` = second NNN−1, `sheet.jpg` = contact sheet with 6 columns, `seconds.json` = the same data as text.
- Verdict: "agree" if a person looking at this frame would say the same as the engine (alert, window filling, or silence). "Disagree" if not. "Unclear" if the frame does not honestly allow a decision.
- Not checked: r01 and r03 on forklift-operator (there is no yellow zone and no floor calibration there, and the page says so), and r11 on the two clips without a zone at the door.

## Summary by rule
| Rule | Clip | Condition by second | Alerts (engine caption) | Agree | Disagree | Unclear |
|---|---|---|---|---|---|---|
| r02 | forklift-operator | `TTTTTTTTTT` | s2–s9 (person #0) | 10 | 0 | 0 |
| r02 | warehouse-team | `.TTTTTTTTTTTT.TTT.TTTTT` | s3–s22 (person #1) | 19 | 2 | 2 |
| r02 | busy-warehouse | `.....TTTTTTTTTTTTTTTTTTTT` | s7–s24 (person #3 and person #5) | 20 | 5 | 0 |
| r02-vest | forklift-operator | `TTTTTTTTTT` | s2–s9 (person #0) | 10 | 0 | 0 |
| r02-vest | warehouse-team | `.TTTTTTTTTTTT..T...TT.T` | s3–s12 (person #1) | 20 | 2 | 1 |
| r02-vest | busy-warehouse | `.....TTTTTTTTTTTTTTTTTTTT` | s7–s24 (person #3 and person #5) | 20 | 5 | 0 |
| r03 | warehouse-team | `.TTTTTT...TTT..T..TTT.T` | s1–s6 (forklift #0, person #1 and person #2), s10–s12 (forklift #0, person #2 and person #5), s15–s15 (forklift #0, person #2 and person #7), s18–s22 (forklift #0 and person #2) | 17 | 1 | 5 |
| r03 | busy-warehouse | `.........................` | none | 25 | 0 | 0 |
| r01-forklift | warehouse-team | `.TT.T.............T...T` | s1–s4 (person #2), s18–s18 (person #11), s22–s22 (person #7 and person #12) | 20 | 2 | 1 |
| r01-forklift | busy-warehouse | `.........................` | none | 25 | 0 | 0 |
| r11 | busy-warehouse | `TTTT.....................` | s1–s3 (forklift #0) | 4 | 5 | 16 |

`T` = condition true, `.` = false. The busy-warehouse numbers for r02 and r02-vest are after the floor zone fix (below). Before the fix they were 2 agree and 23 disagree: there was no alert at all.

## What we found

1. **Example 4 (r11) on Forklift aisle is wrong in meaning.** The rule says "do not stand". The alert at s1–s3 comes while the forklift reverses through the zone (0.4–0.5 m/s by calibration) and never stops. The engine measures time present in the zone. There is no "standing" primitive. On 28.10 the door zone was made deeper, from 3 to 6.5 m, to fit this rule: with 3 m the forklift does not enter it at all. Then the forklift stands for 16 s (s9–s24) with its front part about 1.2 m inside the zone, and the engine stays silent, because it checks one point at the rear edge of the box. The result is backwards: an alert while it drives, silence while it stands.
2. **Rule 2 on Team at the racks is right by second, but the caption is wrong.** The condition is true for almost the whole clip, and that is correct: almost all the time someone on the floor is without a hard hat or vest (#1 without anything s1–s12, then #5 without a hard hat, #7 without a vest, #11 without a hard hat). But there is one episode per rule, and the card names it "person #1" for 20 s, although #1 is in frame only s1–s12. The long amber bar is correct. What was missing is an explanation of who exactly broke the rule. For two seconds (s13, s17) there is no violator, and the amber holds through the 1 s gap tolerance.
3. **Rule 2 on Forklift aisle used to stay silent during a violation.** The driver without a hard hat and vest is visible the whole clip (expected.md: alert s2–s24). The engine still does not count the seated driver: SAM splits him into two tracks (body and legs), and the anchor point of both is in the cab. The far person at the racks (#5, ~16 m, no vest) stood on floor that the zone did not cover, although the zone note says "all visible floor". The zone was fixed (below), and now the alert s7–s24 holds on this person. At 16 m his hard hat cannot be made out. The missing vest is visible.
4. **r03 on Team at the racks is mostly right.** Where people are right next to the reach truck (0.1–1.9 m), there is an alert. 5 seconds are unclear: the engine measures 2.1–4.4 m, by eye it is 2–3 m. Calibration is ±20–30 %, and the 2 m threshold is inside the error. One clear error: s13, the camera tilted and the calibration drifted (4.51 m with a person at the reach truck).
5. **r01 on Team at the racks is right in geometry and meaningless in substance.** The "yellow zone" there is drawn by hand around the place where the reach truck stands at 0 s. There is nothing yellow on the floor. The reach truck leaves by 2 s, and "forklift is running" is set by hand for the whole clip. People's feet do fall inside the rectangle, but a viewer will not understand why this is a violation.
6. **Common to all rules:** a second without a violation inside an alert (1 s gap tolerance) had the same amber colour on the old bar as a violation. On the page since 30.10 it looks different from a second with a violation.

## What was fixed
- **Floor zone on busy-warehouse** (`past_gate`, now "Warehouse floor"): the far edge on the left was raised from y=0.5347 to 0.50 (x 0–0.367). The shop floor between the workbenches and the racks is visible there, and person #5 stands on it (box bottom y=0.51–0.524). This is a markup fix to match the zone's own definition and `expected.md`. Thresholds were not touched. Effect: r02 and r02-vest on this clip give an alert at s7–s24 instead of silence.
- **Captions for people**: zones and example texts were renamed to plain titles.
- **The Muse Spark explanation gave the wrong distance**: the facts used the distance from the last second of the episode, not the closest one (r03 on warehouse-team: "within 1.54 m", although the approach went down to 0.26 m). Now `explain.py` takes the episode minimum. The explanations were regenerated.
- **Page** (https://safety-mode.vercel.app, since 30.10): the video shows the same overlay as the check frames (anchor points, red and green person colour for PPE, a line with metres for r03, zone name, state of the second). A gap second inside an alert is hatched, not filled. Under the bar there is a line "when, who, in which seconds" or "why there is no alert". Screenshots: `demo/screenshots/prod/proof-*.png`.
- **The engine was not changed.** Each error found needs a new primitive or a new model, not a threshold change. They are listed below as limitations.

## Limitations that remain (not fixed)
| Limitation | Where it shows | What is needed to fix it |
|---|---|---|
| No "standing" primitive: "do not stand longer than N s" compiles into "is in the zone longer than N s" | r11, s0–s4 | a speed primitive by track id (anchor point displacement on the floor per second) and a compiler rule |
| Vehicle in zone = one point at the box edge closest to the camera; the body extending away from the camera is not counted | r11, s9–s24 | vehicle footprint on the floor (bottom edge + length by vehicle class) or a mask |
| A seated driver does not stand on the floor: his anchor point is in the cab | r02 busy-warehouse s2–s6 | link the driver to the vehicle (`is_operator`) and use the vehicle anchor point; r02 also needs a forklift detector |
| Distance is measured from the near edge of the vehicle box; if the person is behind the vehicle, the number is too high by the vehicle length | r03 busy-warehouse (pedestrian far away, does not affect the verdict) | the same, vehicle footprint |
| Floor calibration is for a fixed camera; a handheld camera tilts and shifts | r03 s13 warehouse-team, r11 s9–s24 | a fixed camera (a real installation will have one) or frame stabilisation |
| The min_duration_s window resets if the person is hidden for 1 s; the gap tolerance only holds an alert that is already raised | r02-vest warehouse-team s19–s22 | keep the window through a short gap by track id (changes the `expected.md` convention, needs a decision) |
| False hard hat: a hood and a knitted cap are found as `hard hat` | warehouse-team #7, #13 | a prompt change or a second pass on the head; did not affect the verdicts on this clip |
| PPE on far people (≤ 50 px) is not checked reliably | busy-warehouse #5 | camera closer or higher resolution |
| One episode per rule names the people in the alert second, not everyone over the episode | r02 warehouse-team: "person #1" for 20 s | on the page the line under the bar lists everyone by second (done) |

## Examples that give a misleading result
- Example 4 on Forklift aisle as it is now: an alert while the forklift drives, and silence while it stands. The Spark explanation for this alert says "Forklift #0 stopped… The stop lasted 4 s", so it repeats the rule error. On a camera without the zone, the compiler itself writes that "stop" needs speed, which the primitives do not have. On Forklift aisle it silently compiles "stop" into "is in the zone". Rule Scout also accepted this rule on this clip.
- Example 1 on Team at the racks: a drawn "yellow zone" with no marking on the floor.
- Example 2 on Forklift aisle as "the system caught the driver without a hard hat": the alert holds on a tiny far person, and the engine does not see the driver.
- The rule 2 alert card on Team at the racks without the line under the bar: the title "· person #1" and the Spark explanation "Person #1 … for 22 s", although #1 is in frame only s1–s12.

## Second by second
### r02 on forklift-operator

Alerts: s2–s9 (person #0). Frames: `test-footage/verify/forklift-operator_r02/`, sheet `sheet.jpg`.

| s | Engine | Measured | Verdict | Why |
|---|---|---|---|---|
| s0 | candidate 1 of 3 s (cond. true) | person #0 without vest; person #1 without hard hat, vest | agree | driver in a hard hat, no vest (grey T-shirt); the 3 s window is not filled yet |
| s1 | candidate 2 of 3 s (cond. true) | person #0 without vest | agree | same |
| s2 | ALERT (cond. true) | person #0 without vest | agree | the same driver without a vest; he has a hard hat, the alert is for the vest. He sits in the cab, and the zone here is the whole frame |
| s3 | ALERT (cond. true) | person #0 without vest | agree | same |
| s4 | ALERT (cond. true) | person #0 without vest | agree | same |
| s5 | ALERT (cond. true) | person #0 without vest | agree | same |
| s6 | ALERT (cond. true) | person #0 without vest | agree | same |
| s7 | ALERT (cond. true) | person #0 without vest | agree | same |
| s8 | ALERT (cond. true) | person #0 without vest | agree | same |
| s9 | ALERT (cond. true) | person #0 without vest | agree | same |

### r02 on warehouse-team

Alerts: s3–s22 (person #1). Frames: `test-footage/verify/warehouse-team_r02/`, sheet `sheet.jpg`.

| s | Engine | Measured | Verdict | Why |
|---|---|---|---|---|
| s0 | idle (cond. false) | everyone on the floor in PPE, or nobody | agree | only the reach truck driver in the cab is in frame; his anchor point is in the cab, above the floor |
| s1 | candidate 1 of 3 s (cond. true) | person #1 without hard hat, vest; person #3 without hard hat | agree | #1 in a dark jacket without hard hat and vest, #3 in orange overalls without a hard hat; the window is filling |
| s2 | candidate 2 of 3 s (cond. true) | person #1 without hard hat, vest | agree | same |
| s3 | ALERT (cond. true) | person #1 without hard hat, vest | agree | #1 without hard hat and vest stands on the floor. Errors on other people do not affect the verdict: at s5 #5 has a vest, but the engine says NO (box cut off by the frame edge, overlap 0.28 < 0.30); at s9–s10 #2 has a hard hat, the engine loses it when he tilts his head |
| s4 | ALERT (cond. true) | person #1 without hard hat, vest | agree | same |
| s5 | ALERT (cond. true) | person #1 without hard hat, vest; person #5 without hard hat, vest | agree | same |
| s6 | ALERT (cond. true) | person #1 without hard hat, vest | agree | same |
| s7 | ALERT (cond. true) | person #1 without hard hat, vest | agree | same |
| s8 | ALERT (cond. true) | person #1 without hard hat, vest; person #5 without hard hat | agree | same |
| s9 | ALERT (cond. true) | person #1 without hard hat, vest; person #2 without hard hat; person #5 without hard hat | agree | same |
| s10 | ALERT (cond. true) | person #1 without hard hat, vest; person #2 without hard hat; person #5 without hard hat | agree | same |
| s11 | ALERT (cond. true) | person #1 without hard hat, vest | agree | same |
| s12 | ALERT (cond. true) | person #1 without hard hat, vest; person #5 without hard hat | agree | same |
| s13 | ALERT (cond. false) | everyone on the floor in PPE, or nobody | disagree | nobody without PPE on the floor: #1 has left, #2 wears a hard hat and vest, #7 is in the cab. The 1 s gap tolerance holds the alert |
| s14 | ALERT (cond. true) | person #5 without hard hat | agree | #5 in hi-vis overalls and a balaclava, no hard hat |
| s15 | ALERT (cond. true) | person #2 without vest; person #5 without hard hat; person #7 without vest | agree | #7 in a dark jacket without a vest sits by the frame, #5 without a hard hat (the engine loses the vest on #2: 0.18, he bent down) |
| s16 | ALERT (cond. true) | person #5 without hard hat | unclear | #5 is a strip at the left edge, the head is not visible; "without hard hat" cannot be checked from this frame |
| s17 | ALERT (cond. false) | everyone on the floor in PPE, or nobody | disagree | no violators on the floor (#2 in PPE, #7 in the cab); the 1 s gap holds the alert |
| s18 | ALERT (cond. true) | person #11 without hard hat | agree | #11 in hooded overalls, no hard hat |
| s19 | ALERT (cond. true) | person #7 without vest; person #11 without hard hat | agree | #7 has stepped off the reach truck, no vest; his "hard hat" is really a hood (false SAM detection), #11 without a hard hat |
| s20 | ALERT (cond. true) | person #2 without hard hat; person #7 without vest | agree | #7 without a vest |
| s21 | ALERT (cond. true) | person #12 without hard hat | unclear | #12 passes right next to the camera, the head is cut off by the frame; vest present, hard hat not visible |
| s22 | ALERT (cond. true) | person #7 without hard hat, vest | agree | #7 without a vest in the background |

### r02 on busy-warehouse

Alerts: s7–s24 (person #3 and person #5). Frames: `test-footage/verify/busy-warehouse_r02/`, sheet `sheet.jpg`.

| s | Engine | Measured | Verdict | Why |
|---|---|---|---|---|
| s0 | idle (cond. false) | everyone on the floor in PPE, or nobody | agree | forklift driver without hard hat and vest, but the 3 s window is not filled yet |
| s1 | idle (cond. false) | everyone on the floor in PPE, or nobody | agree | same |
| s2 | idle (cond. false) | everyone on the floor in PPE, or nobody | disagree | the driver without hard hat and vest is visible, there should be an alert (expected.md: s2–s24). The engine does not count him: SAM splits the seated driver into two tracks (#2 body, #3 legs), the anchor point of both is in the cab, outside the floor zone |
| s3 | idle (cond. false) | everyone on the floor in PPE, or nobody | disagree | same |
| s4 | idle (cond. false) | everyone on the floor in PPE, or nobody | disagree | same |
| s5 | candidate 1 of 3 s (cond. true) | person #5 without hard hat, vest | disagree | same; #5 has appeared at the racks on the left, the window is filling |
| s6 | candidate 2 of 3 s (cond. true) | person #3 without hard hat, vest; person #5 without hard hat, vest | disagree | same |
| s7 | ALERT (cond. true) | person #3 without hard hat, vest; person #5 without hard hat, vest | agree | there is an alert, as a person would say, but it holds on #5: a person ~16 m from the camera, in a turquoise T-shirt without a vest (a hard hat cannot be made out at this size); at s7 and s24 also the driver's legs (#3). The engine still does not count the driver himself |
| s8 | ALERT (cond. true) | person #5 without hard hat, vest | agree | same |
| s9 | ALERT (cond. true) | person #5 without hard hat, vest | agree | same |
| s10 | ALERT (cond. true) | person #5 without hard hat, vest | agree | same |
| s11 | ALERT (cond. true) | person #5 without hard hat, vest | agree | same |
| s12 | ALERT (cond. true) | person #5 without hard hat, vest | agree | same |
| s13 | ALERT (cond. true) | person #5 without hard hat, vest | agree | same |
| s14 | ALERT (cond. true) | person #5 without hard hat, vest | agree | same |
| s15 | ALERT (cond. true) | person #5 without hard hat, vest | agree | same |
| s16 | ALERT (cond. true) | person #5 without hard hat, vest | agree | same |
| s17 | ALERT (cond. true) | person #5 without hard hat, vest | agree | same |
| s18 | ALERT (cond. true) | person #5 without hard hat, vest | agree | same |
| s19 | ALERT (cond. true) | person #5 without hard hat, vest | agree | same |
| s20 | ALERT (cond. true) | person #5 without hard hat, vest | agree | same |
| s21 | ALERT (cond. true) | person #5 without hard hat, vest | agree | same |
| s22 | ALERT (cond. true) | person #5 without hard hat, vest | agree | same |
| s23 | ALERT (cond. true) | person #5 without hard hat, vest | agree | same |
| s24 | ALERT (cond. true) | person #3 without hard hat, vest; person #5 without hard hat, vest | agree | same |

### r02-vest on forklift-operator

Alerts: s2–s9 (person #0). Frames: `test-footage/verify/forklift-operator_r02-vest/`, sheet `sheet.jpg`.

| s | Engine | Measured | Verdict | Why |
|---|---|---|---|---|
| s0 | candidate 1 of 3 s (cond. true) | person #0 without vest; person #1 without vest | agree | driver without a vest, the 3 s window is not filled yet |
| s1 | candidate 2 of 3 s (cond. true) | person #0 without vest | agree | same |
| s2 | ALERT (cond. true) | person #0 without vest | agree | driver without a vest for the whole clip |
| s3 | ALERT (cond. true) | person #0 without vest | agree | same |
| s4 | ALERT (cond. true) | person #0 without vest | agree | same |
| s5 | ALERT (cond. true) | person #0 without vest | agree | same |
| s6 | ALERT (cond. true) | person #0 without vest | agree | same |
| s7 | ALERT (cond. true) | person #0 without vest | agree | same |
| s8 | ALERT (cond. true) | person #0 without vest | agree | same |
| s9 | ALERT (cond. true) | person #0 without vest | agree | same |

### r02-vest on warehouse-team

Alerts: s3–s12 (person #1). Frames: `test-footage/verify/warehouse-team_r02-vest/`, sheet `sheet.jpg`.

| s | Engine | Measured | Verdict | Why |
|---|---|---|---|---|
| s0 | idle (cond. false) | everyone on the floor in PPE, or nobody | agree | only the driver in the cab |
| s1 | candidate 1 of 3 s (cond. true) | person #1 without vest | agree | #1 without a vest, the window is filling |
| s2 | candidate 2 of 3 s (cond. true) | person #1 without vest | agree | same |
| s3 | ALERT (cond. true) | person #1 without vest | agree | #1 without a vest. s5: #5 has a vest, the engine says NO (overlap 0.28), does not affect the verdict |
| s4 | ALERT (cond. true) | person #1 without vest | agree | same |
| s5 | ALERT (cond. true) | person #1 without vest; person #5 without vest | agree | same |
| s6 | ALERT (cond. true) | person #1 without vest | agree | same |
| s7 | ALERT (cond. true) | person #1 without vest | agree | same |
| s8 | ALERT (cond. true) | person #1 without vest | agree | same |
| s9 | ALERT (cond. true) | person #1 without vest | agree | same |
| s10 | ALERT (cond. true) | person #1 without vest | agree | same |
| s11 | ALERT (cond. true) | person #1 without vest | agree | same |
| s12 | ALERT (cond. true) | person #1 without vest | agree | same |
| s13 | ALERT (cond. false) | everyone on the floor in PPE, or nobody | disagree | nobody without a vest on the floor; the 1 s gap holds the alert |
| s14 | idle (cond. false) | everyone on the floor in PPE, or nobody | agree | everyone on the floor wears a vest or hi-vis overalls |
| s15 | candidate 1 of 3 s (cond. true) | person #2 without vest; person #7 without vest | agree | #7 without a vest for 1 s, the window is not filled |
| s16 | idle (cond. false) | everyone on the floor in PPE, or nobody | agree | #7 sits in the reach truck, the anchor point is in the cab; everyone on the floor wears a vest |
| s17 | idle (cond. false) | everyone on the floor in PPE, or nobody | agree | same |
| s18 | idle (cond. false) | everyone on the floor in PPE, or nobody | agree | same |
| s19 | candidate 1 of 3 s (cond. true) | person #7 without vest | agree | #7 has stepped onto the floor without a vest, the window is filling |
| s20 | candidate 2 of 3 s (cond. true) | person #7 without vest | agree | same |
| s21 | idle (cond. false) | everyone on the floor in PPE, or nobody | unclear | #7 is hidden by a person passing by the camera; the engine resets the 3 s window |
| s22 | candidate 1 of 3 s (cond. true) | person #7 without vest | disagree | #7 is on the floor without a vest s19–s22, one second hidden. A person would count 4 s, but the engine restarted the window and gave no alert |

### r02-vest on busy-warehouse

Alerts: s7–s24 (person #3 and person #5). Frames: `test-footage/verify/busy-warehouse_r02-vest/`, sheet `sheet.jpg`.

| s | Engine | Measured | Verdict | Why |
|---|---|---|---|---|
| s0 | idle (cond. false) | everyone on the floor in PPE, or nobody | agree | driver without a vest, the window is not filled |
| s1 | idle (cond. false) | everyone on the floor in PPE, or nobody | agree | same |
| s2 | idle (cond. false) | everyone on the floor in PPE, or nobody | disagree | the driver without a vest is not counted (anchor point in the cab) |
| s3 | idle (cond. false) | everyone on the floor in PPE, or nobody | disagree | same |
| s4 | idle (cond. false) | everyone on the floor in PPE, or nobody | disagree | same |
| s5 | candidate 1 of 3 s (cond. true) | person #5 without vest | disagree | same |
| s6 | candidate 2 of 3 s (cond. true) | person #3 without vest; person #5 without vest | disagree | same |
| s7 | ALERT (cond. true) | person #3 without vest; person #5 without vest | agree | alert on #5 (far person without a vest) and on the driver's legs at s7, s24 |
| s8 | ALERT (cond. true) | person #5 without vest | agree | same |
| s9 | ALERT (cond. true) | person #5 without vest | agree | same |
| s10 | ALERT (cond. true) | person #5 without vest | agree | same |
| s11 | ALERT (cond. true) | person #5 without vest | agree | same |
| s12 | ALERT (cond. true) | person #5 without vest | agree | same |
| s13 | ALERT (cond. true) | person #5 without vest | agree | same |
| s14 | ALERT (cond. true) | person #5 without vest | agree | same |
| s15 | ALERT (cond. true) | person #5 without vest | agree | same |
| s16 | ALERT (cond. true) | person #5 without vest | agree | same |
| s17 | ALERT (cond. true) | person #5 without vest | agree | same |
| s18 | ALERT (cond. true) | person #5 without vest | agree | same |
| s19 | ALERT (cond. true) | person #5 without vest | agree | same |
| s20 | ALERT (cond. true) | person #5 without vest | agree | same |
| s21 | ALERT (cond. true) | person #5 without vest | agree | same |
| s22 | ALERT (cond. true) | person #5 without vest | agree | same |
| s23 | ALERT (cond. true) | person #5 without vest | agree | same |
| s24 | ALERT (cond. true) | person #3 without vest; person #5 without vest | agree | same |

### r03 on warehouse-team

Alerts: s1–s6 (forklift #0, person #1 and person #2), s10–s12 (forklift #0, person #2 and person #5), s15–s15 (forklift #0, person #2 and person #7), s18–s22 (forklift #0 and person #2). Frames: `test-footage/verify/warehouse-team_r03/`, sheet `sheet.jpg`.

| s | Engine | Measured | Verdict | Why |
|---|---|---|---|---|
| s0 | idle (cond. false) | forklift or people not found | agree | SAM did not find the reach truck, but there are no pedestrians near it |
| s1 | ALERT (cond. true) | forklift #0–person #2: 0.90 m (< 2 m) | agree | people stand right next to the reach truck: 0.26–1.6 m by calibration, by eye also under 2 m |
| s2 | ALERT (cond. true) | forklift #0–person #1: 0.53 m (< 2 m) | agree | same |
| s3 | ALERT (cond. true) | forklift #0–person #2: 1.12 m (< 2 m) | agree | same |
| s4 | ALERT (cond. true) | forklift #0–person #2: 0.26 m (< 2 m) | agree | same |
| s5 | ALERT (cond. true) | forklift #0–person #1: 0.72 m (< 2 m) | agree | same |
| s6 | ALERT (cond. true) | forklift #0–person #2: 1.54 m (< 2 m) | agree | same |
| s7 | ALERT (cond. false) | forklift #0–person #1: 2.65 m (>= 2 m) | unclear | 2.65 m; they carry a frame next to the reach truck, by eye about 2 m. Calibration ±20–30 %, the threshold is inside the error |
| s8 | idle (cond. false) | forklift #0–person #2: 3.78 m (>= 2 m) | unclear | 3.78 m; by eye 2–3 m |
| s9 | idle (cond. false) | forklift #0–person #1: 4.90 m (>= 2 m) | agree | 4.9 m, the reach truck has moved further in |
| s10 | ALERT (cond. true) | forklift #0–person #5: 0.88 m (< 2 m) | agree | people at the reach truck: 0.88–1.9 m |
| s11 | ALERT (cond. true) | forklift #0–person #1: 0.90 m (< 2 m) | agree | same |
| s12 | ALERT (cond. true) | forklift #0–person #2: 1.90 m (< 2 m) | agree | same |
| s13 | ALERT (cond. false) | forklift #0–person #2: 4.51 m (>= 2 m) | disagree | #2 stands at the reach truck, by eye under 2 m, engine 4.51 m: the camera tilted (the scene moved up ~40 px), the calibration was made for a fixed camera. The 1 s gap holds the amber in this second |
| s14 | idle (cond. false) | forklift #0–person #2: 2.06 m (>= 2 m) | unclear | 2.06 m, by eye about 2 m |
| s15 | ALERT (cond. true) | forklift #0–person #2: 0.60 m (< 2 m) | agree | #2 and #7 at the frame right next to the reach truck: 0.6–0.93 m |
| s16 | ALERT (cond. false) | forklift #0–person #2: 4.35 m (>= 2 m) | unclear | 4.35 m, by eye 2–3 m; the gap holds the amber |
| s17 | idle (cond. false) | forklift #0–person #2: 3.06 m (>= 2 m) | unclear | 3.06 m, #2 bent down in front of the reach truck, by eye 2–3 m |
| s18 | ALERT (cond. true) | forklift #0–person #2: 1.89 m (< 2 m) | agree | people pass right next to the reach truck: 0.12–1.9 m |
| s19 | ALERT (cond. true) | forklift #0–person #2: 0.12 m (< 2 m) | agree | same |
| s20 | ALERT (cond. true) | forklift #0–person #7: 0.49 m (< 2 m) | agree | same |
| s21 | ALERT (cond. false) | forklift #0–person #12: 6.16 m (>= 2 m) | agree | only #12 at the camera, far from the reach truck (the gap holds the amber) |
| s22 | ALERT (cond. true) | forklift #0–person #13: 0.57 m (< 2 m) | agree | #13 passes in front of the reach truck, 0.57 m |

### r03 on busy-warehouse

Alerts: none. Frames: `test-footage/verify/busy-warehouse_r03/`, sheet `sheet.jpg`.

| s | Engine | Measured | Verdict | Why |
|---|---|---|---|---|
| s0 | idle (cond. false) | forklift #0–person #0: 17.50 m (>= 2 m) | agree | the pedestrian in the doorway and the person at the racks are far away (14–28 m by the engine). The engine correctly excludes the driver. The number is too high by the forklift length (~2.5 m): it is measured from the box edge closest to the camera, and the pedestrian is behind the forklift; does not affect the verdict |
| s1 | idle (cond. false) | forklift #0–person #0: 18.39 m (>= 2 m) | agree | same |
| s2 | idle (cond. false) | forklift #0–person #0: 17.91 m (>= 2 m) | agree | same |
| s3 | idle (cond. false) | forklift #0–person #0: 17.93 m (>= 2 m) | agree | same |
| s4 | idle (cond. false) | forklift #0–person #0: 18.84 m (>= 2 m) | agree | same |
| s5 | idle (cond. false) | forklift #0–person #0: 19.24 m (>= 2 m) | agree | same |
| s6 | idle (cond. false) | forklift #0–person #0: 20.13 m (>= 2 m) | agree | same |
| s7 | idle (cond. false) | forklift #0–person #0: 20.37 m (>= 2 m) | agree | same |
| s8 | idle (cond. false) | forklift #0–person #5: 20.88 m (>= 2 m) | agree | same |
| s9 | idle (cond. false) | forklift #0–person #0: 23.23 m (>= 2 m) | agree | same |
| s10 | idle (cond. false) | forklift #0–person #0: 24.04 m (>= 2 m) | agree | same |
| s11 | idle (cond. false) | forklift #0–person #0: 24.10 m (>= 2 m) | agree | same |
| s12 | idle (cond. false) | forklift #0–person #0: 22.84 m (>= 2 m) | agree | same |
| s13 | idle (cond. false) | forklift #0–person #0: 22.86 m (>= 2 m) | agree | same |
| s14 | idle (cond. false) | forklift #0–person #0: 22.84 m (>= 2 m) | agree | same |
| s15 | idle (cond. false) | forklift #0–person #0: 21.13 m (>= 2 m) | agree | same |
| s16 | idle (cond. false) | forklift #0–person #0: 18.22 m (>= 2 m) | agree | same |
| s17 | idle (cond. false) | forklift #0–person #0: 17.41 m (>= 2 m) | agree | same |
| s18 | idle (cond. false) | forklift #0–person #0: 16.29 m (>= 2 m) | agree | same |
| s19 | idle (cond. false) | forklift #0–person #0: 15.58 m (>= 2 m) | agree | same |
| s20 | idle (cond. false) | forklift #0–person #0: 14.31 m (>= 2 m) | agree | same |
| s21 | idle (cond. false) | forklift #0–person #0: 14.93 m (>= 2 m) | agree | same |
| s22 | idle (cond. false) | forklift #0–person #0: 15.26 m (>= 2 m) | agree | same |
| s23 | idle (cond. false) | forklift #0–person #0: 15.04 m (>= 2 m) | agree | same |
| s24 | idle (cond. false) | forklift #0–person #0: 18.70 m (>= 2 m) | agree | same |

### r01-forklift on warehouse-team

Alerts: s1–s4 (person #2), s18–s18 (person #11), s22–s22 (person #7 and person #12). Frames: `test-footage/verify/warehouse-team_r01-forklift/`, sheet `sheet.jpg`.

| s | Engine | Measured | Verdict | Why |
|---|---|---|---|---|
| s0 | idle (cond. false) | nobody in zone | agree | nobody in the drawn rectangle |
| s1 | ALERT (cond. true) | person #2 in zone | agree | the feet of #2 (and #1 at s2) are inside the drawn rectangle |
| s2 | ALERT (cond. true) | person #1 in zone; person #2 in zone | agree | same |
| s3 | ALERT (cond. false) | nobody in zone | disagree | nobody in the rectangle; the 1 s gap holds the alert |
| s4 | ALERT (cond. true) | person #1 in zone | agree | #1 stands in the rectangle |
| s5 | ALERT (cond. false) | nobody in zone | disagree | nobody in the rectangle; the 1 s gap holds the alert |
| s6 | idle (cond. false) | nobody in zone | agree | nobody in the rectangle |
| s7 | idle (cond. false) | nobody in zone | agree | same |
| s8 | idle (cond. false) | nobody in zone | agree | same |
| s9 | idle (cond. false) | nobody in zone | agree | same |
| s10 | idle (cond. false) | nobody in zone | agree | same |
| s11 | idle (cond. false) | nobody in zone | agree | same |
| s12 | idle (cond. false) | nobody in zone | agree | same |
| s13 | idle (cond. false) | nobody in zone | agree | same |
| s14 | idle (cond. false) | nobody in zone | agree | same |
| s15 | idle (cond. false) | nobody in zone | agree | same |
| s16 | idle (cond. false) | nobody in zone | agree | same |
| s17 | idle (cond. false) | nobody in zone | agree | same |
| s18 | ALERT (cond. true) | person #11 in zone | agree | #11 stands in the rectangle |
| s19 | ALERT (cond. false) | nobody in zone | unclear | the feet of #11 are right on the boundary; the gap holds the alert |
| s20 | idle (cond. false) | nobody in zone | agree | nobody in the rectangle |
| s21 | idle (cond. false) | nobody in zone | agree | same |
| s22 | ALERT (cond. true) | person #7 in zone; person #12 in zone | agree | #7 and #12 stand in the rectangle |

### r01-forklift on busy-warehouse

Alerts: none. Frames: `test-footage/verify/busy-warehouse_r01-forklift/`, sheet `sheet.jpg`.

| s | Engine | Measured | Verdict | Why |
|---|---|---|---|---|
| s0 | idle (cond. false) | nobody in zone | agree | nobody in the aisle between the yellow line and the racks |
| s1 | idle (cond. false) | nobody in zone | agree | same |
| s2 | idle (cond. false) | nobody in zone | agree | same |
| s3 | idle (cond. false) | nobody in zone | agree | same |
| s4 | idle (cond. false) | nobody in zone | agree | same |
| s5 | idle (cond. false) | nobody in zone | agree | same |
| s6 | idle (cond. false) | nobody in zone | agree | same |
| s7 | idle (cond. false) | nobody in zone | agree | same |
| s8 | idle (cond. false) | nobody in zone | agree | same |
| s9 | idle (cond. false) | nobody in zone | agree | same |
| s10 | idle (cond. false) | nobody in zone | agree | same |
| s11 | idle (cond. false) | nobody in zone | agree | same |
| s12 | idle (cond. false) | nobody in zone | agree | same |
| s13 | idle (cond. false) | nobody in zone | agree | same |
| s14 | idle (cond. false) | nobody in zone | agree | same |
| s15 | idle (cond. false) | nobody in zone | agree | same |
| s16 | idle (cond. false) | nobody in zone | agree | same |
| s17 | idle (cond. false) | nobody in zone | agree | same |
| s18 | idle (cond. false) | nobody in zone | agree | same |
| s19 | idle (cond. false) | nobody in zone | agree | same |
| s20 | idle (cond. false) | nobody in zone | agree | same |
| s21 | idle (cond. false) | nobody in zone | agree | same |
| s22 | idle (cond. false) | nobody in zone | agree | same |
| s23 | idle (cond. false) | nobody in zone | agree | same |
| s24 | idle (cond. false) | nobody in zone | agree | same |

### r11 on busy-warehouse

Alerts: s1–s3 (forklift #0). Frames: `test-footage/verify/busy-warehouse_r11/`, sheet `sheet.jpg`.

| s | Engine | Measured | Verdict | Why |
|---|---|---|---|---|
| s0 | candidate 1 of 2 s (cond. true) | forklift #0: in Area in front of the loading door | disagree | the forklift is in the zone, but it is reversing (0.4–0.5 m/s by calibration), and the engine already counts "standing" time |
| s1 | ALERT (cond. true) | forklift #0: in Area in front of the loading door | disagree | the alert says "stands longer than 2 s", but the forklift drives all this time and does not stop. The engine measures time in the zone, not a stop; the zone depth was raised on 28.10 from 3 m to 6.5 m to fit this rule, with 3 m the forklift would not enter it |
| s2 | ALERT (cond. true) | forklift #0: in Area in front of the loading door | disagree | same |
| s3 | ALERT (cond. true) | forklift #0: in Area in front of the loading door | disagree | same |
| s4 | ALERT (cond. false) | forklift #0: outside Area in front of the loading door | disagree | the anchor point is already outside the zone; the 1 s gap holds the amber |
| s5 | idle (cond. false) | forklift #0: outside Area in front of the loading door | agree | the forklift is still driving, the rear edge is outside the zone |
| s6 | idle (cond. false) | forklift #0: outside Area in front of the loading door | agree | same |
| s7 | idle (cond. false) | forklift #0: outside Area in front of the loading door | agree | same |
| s8 | idle (cond. false) | forklift #0: outside Area in front of the loading door | agree | same |
| s9 | idle (cond. false) | forklift #0: outside Area in front of the loading door | unclear | the forklift stands (0.06–0.16 m/s). By the floor model its rear edge is 1.1–1.4 m beyond the near boundary of the zone, and its front part is about 1.2 m inside it; the engine checks only the point at the rear edge and stays silent. A person could say "stands in front of the door for 16 s". By s9 the camera had shifted by ~50 px, the zone polygon is static |
| s10 | idle (cond. false) | forklift #0: outside Area in front of the loading door | unclear | same |
| s11 | idle (cond. false) | forklift #0: outside Area in front of the loading door | unclear | same |
| s12 | idle (cond. false) | forklift #0: outside Area in front of the loading door | unclear | same |
| s13 | idle (cond. false) | forklift #0: outside Area in front of the loading door | unclear | same |
| s14 | idle (cond. false) | forklift #0: outside Area in front of the loading door | unclear | same |
| s15 | idle (cond. false) | forklift #0: outside Area in front of the loading door | unclear | same |
| s16 | idle (cond. false) | forklift #0: outside Area in front of the loading door | unclear | same |
| s17 | idle (cond. false) | forklift #0: outside Area in front of the loading door | unclear | same |
| s18 | idle (cond. false) | forklift #0: outside Area in front of the loading door | unclear | same |
| s19 | idle (cond. false) | forklift #0: outside Area in front of the loading door | unclear | same |
| s20 | idle (cond. false) | forklift #0: outside Area in front of the loading door | unclear | same |
| s21 | idle (cond. false) | forklift #0: outside Area in front of the loading door | unclear | same |
| s22 | idle (cond. false) | forklift #0: outside Area in front of the loading door | unclear | same |
| s23 | idle (cond. false) | forklift #0: outside Area in front of the loading door | unclear | same |
| s24 | idle (cond. false) | forklift #0: outside Area in front of the loading door | unclear | same |

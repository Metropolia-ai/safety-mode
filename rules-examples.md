# Safety Mode: rule examples "the way an HSE manager writes them"

The rules are written the way an occupational safety officer says them, without computer vision terms. Each rule comes with what Muse Spark must "compile" it into: objects for SAM 3.1, zones, condition, time window, severity. This is the test set for the rule compiler and for the demo page.

| # | Rule in words (input) | Objects (SAM 3.1 text prompts) | Zone | Condition | Window | Severity |
|---|---|---|---|---|---|---|
| 1 | Nobody inside the yellow zone while the press is running | person, press machine | yellow_zone (marked) | person ∩ yellow_zone AND machine_state=running | 1 s | high |
| 2 | Everyone on the warehouse floor wears a hard hat and a high-visibility vest (page example 1: "Everyone on the warehouse floor must wear a hard hat and a high-visibility vest."; example 2, vest only: "Everyone on the floor must wear a high-visibility vest.") | person, hard hat, safety vest | past_gate ("Warehouse floor": the floor the camera sees) | person ∩ past_gate AND NOT (hard hat on person AND vest on person) | 3 s (to ignore flicker) | medium |
| 3 | A forklift no closer than 2 m to a pedestrian (page example 3: "A forklift must not come closer than 2 m to a pedestrian.") | forklift, person | whole frame | distance(forklift, person) < 2 m (by floor calibration) | 1 s | high |
| 4 | Nobody stands under a raised load | forklift with raised load, person | whole frame | person under the load bbox while the forks are above 1.5 m | 1 s | high |
| 5 | The fire exit is always kept clear | pallet, box, cart | fire_exit_zone | any object ∩ fire_exit_zone | 60 s | medium |
| 6 | No climbing on racks | person, rack | rack_zone | person higher than 1.2 m above the floor inside rack_zone | 2 s | high |
| 7 | At most one person in the battery charging zone | person | charging_zone | count(person ∩ charging_zone) > 1 | 5 s | low |
| 8 | No forklifts on the pedestrian lane (page example 4: "No forklift in the walkway crossing.", zone "Walkway crossing" on clip B) | forklift | pedestrian_lane | forklift ∩ pedestrian_lane | 2 s | medium |
| 9 | Pallet stacks at the ramp no higher than three pallets | pallet stack | ramp_zone | stack height > 3 pallets | 30 s | low |
| 10 | If a person lies on the floor for more than 10 seconds, raise an alarm | person lying | whole frame | person in a horizontal pose > 10 s | 10 s | critical |
| 11 | A forklift does not stand in the area in front of the loading door for more than 2 seconds (was example 4 on the page until the evening of 30.10; removed together with the Forklift aisle clip) | forklift | fire_exit_zone ("Area in front of the loading door", ≈6.5 m in front of the door on busy-warehouse) | forklift ∩ fire_exit_zone | 2 s | medium |

## What the rule tells the compiler
Muse Spark gets the rule text, the list of zones marked on the frame (names and polygons) and the dictionary of available detectors. It returns JSON that follows the schema:

```json
{
  "rule_id": "r2",
  "title": "PPE on the warehouse floor",
  "objects": [{"name": "person", "prompt": "person"}, {"name": "hard_hat", "prompt": "hard hat"}, {"name": "vest", "prompt": "safety vest"}],
  "zones": ["past_gate"],
  "condition": {"all": [{"in_zone": {"object": "person", "zone": "past_gate"}}, {"not": {"wearing": {"person": "person", "ppe": ["hard_hat", "vest"]}}}]},
  "min_duration_s": 3,
  "severity": "medium",
  "explanation_template": "{person_count} person(s) on the warehouse floor without {missing_ppe} for {duration_s} s, at {time}.",
  "unsupported": null,
  "suggested_rewrite": null
}
```

The zone names in JSON (`past_gate`, `fire_exit_zone`) stayed the same so that old rules still load. People see the zone titles (`title`): "Warehouse floor", "Area in front of the loading door". Page examples since the evening of 30.10: 1 = rule 2, 2 = rule 2 with the vest only, 3 = rule 3, 4 = rule 8. The yellow zone rule and rule 11 (area at the door) were removed from the examples; the files `rules/r01.json`, `rules/r11.json` remain. Zone titles on the sample clips: "Shop floor" (A), "Site floor" (C), "Walkway crossing" (B), "Warehouse floor" (Team). Frame-by-frame check: `test-footage/verification-2026-10-30.md` (old clips), `test-footage/candidates/REPORT-2026-10-30.md` (A, B, C).

Primitive arguments are named (`{"in_zone": {"object": ..., "zone": ...}}`), because the Muse Spark strict mode cannot describe positional arrays of mixed types. The exact schema is in `schemas/rule.schema.json`, all 10 rules are in `rules/r01.json`–`r10.json`.

The condition is limited to a small set of primitives: `in_zone`, `count_in_zone`, `distance_lt`, `wearing`, `pose`, `height_above_floor`, `machine_state`, `all`, `any`, `not`. Anything that does not fit them, the compiler returns as `unsupported` with an explanation, and the manager edits the wording. This way the model does not "invent" parts of the rule.

## How to check it
For each rule `test-footage/` has a clip and an expected result: in which seconds the alert must fire and in which it must not. Demo metric: rules 2 and 3 are caught on the samples, and there are no false alerts on the quiet clip.

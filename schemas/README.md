# Safety Mode JSON schemas

- `rule.schema.json` is the output of the rule compiler (Muse Spark, `response_format={"type":"json_schema","json_schema":{"name":"safety_rule","strict":true,"schema":<this file>}}`). Rules 1–10 from `../rules-examples.md` in this format are `../rules/r01.json … r10.json`.
- `report-input.schema.json` is the daily report input (the structured events of the day, see `../report-template.md`). Our engine writes it, not the model. The schema still uses the same strict style, so it can also be passed as `response_format`.
- `validate.py` checks three things: the schemas fit strict mode, `rules/*.json` are valid and consistent, and the JSON example from `report-template.md` is valid. Run: `python schemas/validate.py` (needs `jsonschema`).

## Meta Model API strict mode (checked against the original source 21.10.2026)
`dev.meta.ai/docs/structured-output`, section "Enforce strict mode" and the note "Recursive schemas aren't supported":
- the root is a plain object, with no `anyOf/oneOf/allOf/enum/not` at the root; `allOf` and `oneOf` nowhere, `anyOf` is allowed below the root;
- every object has `additionalProperties: false`, and `required` lists all keys of `properties`;
- the rules are checked inside `$defs` / `definitions` too (so `$defs` + `$ref` are allowed);
- **recursive schemas are not supported, a request with them → HTTP 400**, "flatten into a fixed-depth representation";
- limits: nesting depth 10 levels, 5000 properties, 120 000 characters of names/enum/const, ≤ 200 000 nodes after `$ref` expansion;
- `strict` is false by default, but the output is still constrained by the schema; `strict: true` only adds a server check that the schema fits the supported subset. We set `strict: true` so that schema errors are caught at once.

**Decision on the condition tree:** the documentation forbids recursion, so the tree is limited to **3 levels**: `cond_l1` (a primitive or `all/any/not` over `cond_l2`) → `cond_l2` (a primitive or `all/any/not` over `cond_l3`) → `cond_l3` (primitives only). This is enough for all 10 rules (the deepest is rule 2: `all → not → wearing`). Instance depth as counted by `validate.py` (each object and array is one level) = 8 ≤ 10. The docs do not say how Meta counts "nesting depth" (by instance or by schema text); `muse-spark-1.3` accepted the schema in strict mode on the first live request (27.10, `safety_mode/README.md`).

Schema keywords are limited to a safe set: `type`, `properties`, `required`, `additionalProperties`, `items`, `enum`, `anyOf`, `$defs`, `$ref`, `description`, `title`. No `minItems`, `pattern`, `minimum`, `$schema`, `$id`. The docs list no supported keywords beyond those above, and there is no reason to risk a 400. Nullable fields use `anyOf: [{type: X}, {type: null}]`, not `type: [X, "null"]` (the docs say nothing about type arrays).

## Condition format
The primitives are exactly those from `rules-examples.md`. Arguments are **named** (an object), not positional arrays as in the example in `rules-examples.md` (`{"in_zone": ["person", "past_gate"]}`). The positional `wearing(person, [ppe...])` is an array of mixed types, and it cannot be described strictly without `prefixItems`. The r2 example in `rules-examples.md` is the old notation; the current one is `../rules/r02.json`.

| Primitive | Arguments | True when |
|---|---|---|
| `in_zone` | `object`, `zone` | the object's footpoint (middle of the box bottom edge) is inside the zone polygon |
| `count_in_zone` | `object`, `zone`, `op` (`> >= < <= ==`), `n` | number of objects in the zone `op n` |
| `distance_lt` | `a`, `b`, `m` | floor distance between the objects < m metres (by `floor_calibration`) |
| `wearing` | `person`, `ppe[]` | all PPE from the list is on the person (hard hat ∩ upper third of the mask, vest ∩ middle third) |
| `pose` | `person`, `pose` (`lying`, `standing`) | the person's pose |
| `height_above_floor` | `object`, `m` | the lowest point of the object is more than m metres above the floor |
| `machine_state` | `machine`, `state` (`running`, `stopped`) | machine state (v1: by hand or by a separate prompt) |
| `all` / `any` | array of next-level nodes | all / at least one |
| `not` | next-level node | negation |

A node is an object with exactly one key (the name of a primitive or combinator). The names in the arguments are `objects[].name` and `zones[]` of the same rule. `validate.py` checks that all mentioned objects and zones are declared and that declared objects are used (each object = a separate SAM request per frame).

Rule fields: `rule_id`, `title` (in the rule's language; English for the demo), `objects[{name, prompt}]` (prompt is one noun for SAM 3.1), `zones[]` (empty = whole frame), `condition` (a tree or `null`), `min_duration_s`, `severity` (`low|medium|high|critical`), `explanation_template` (placeholders `{time} {duration_s} {person_count} {missing_ppe} {distance_m} {count} {height_m} {object}`; the schema does not fix the placeholder list), `unsupported` and `suggested_rewrite` (a string or `null`). If the rule cannot be expressed with the primitives → `condition: null`, `unsupported` explains why, and `suggested_rewrite` offers a wording that can be expressed.

## Result on the 10 rules
- Expressible: 1, 2, 3, 5, 6, 7, 8, 10.
- `unsupported`: **4** ("under a raised load" is a vertical relation with no primitive for it; the rewrite "nobody closer than 1 m to the forklift while the load is above 1.5 m" is expressible with `height_above_floor` + `distance_lt`) and **9** (stack height in pallets comes neither from an object count nor from the height of the lowest point; the rewrite "no more than 6 pallets in the ramp zone" changes the meaning, and the text says so).
- Rule 6 is formally expressible (`in_zone` + `height_above_floor`). With one camera, though, the engine knows a person's height above the floor only from the footpoint, and a person who climbed a rack has none. This is a question for the engine implementation, not for the schema. The rack base in the zone will be needed as the floor reference.
- Rule 6 in the `rules-examples.md` table mentions the object `rack`; r06 does not have it. The rack is given by the zone `rack_zone`, so an extra SAM prompt is not needed.

## Strict mode, checked live
`rule.schema.json` with `strict: true` passed without a 400 on the first live request (27.10), with its nesting depth and its `title`/`description` keywords as they are.

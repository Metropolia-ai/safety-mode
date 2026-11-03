# Safety Mode: daily report format

Muse Spark writes the report from a structured list of the day's events. One screen, read in a minute, printed to PDF for the safety meeting. Report language = rule language (Finnish or English).

---

**Safety report — Warehouse A, camera 3 — Mon 2026-11-02**

**Summary.** 7 events, 2 high, 5 medium. One repeat pattern: PPE missing past the gate at shift change (06:55–07:10), 4 of 5 medium events. No critical events. Camera uptime 23 h 48 min.

**High**
1. 10:42:17 — Rule 3 *Forklift within 2 m of a pedestrian*. Forklift (right aisle) passed 1.3 m from a worker carrying a box. Duration 2 s. Frame: `evt-0412.jpg`.
2. 14:05:03 — Rule 1 *Nobody in the yellow zone while the press runs*. One person inside the zone for 6 s while the machine state was running. Frame: `evt-0418.jpg`.

**Medium**
3–6. 06:55–07:10 — Rule 2 *PPE past the gate*. Four workers entered without a vest; hard hats present. Frames: `evt-0401..0404.jpg`.
7. 16:20:40 — Rule 5 *Fire exit clear*. Pallet left in the fire-exit zone for 14 min; removed at 16:35.

**Patterns and suggestions**
- Vest violations cluster at shift change. Suggest a vest rack at the gate or a reminder on the shift board.
- Forklift–pedestrian near miss happened in the right aisle where the pedestrian lane is not marked. Consider marking it; rule 8 can then be enabled.

**Rule health.** Rule 6 (climbing racks) produced 0 events and 0 false positives. Rule 9 (pallet stack height) was disabled at 11:00 by the manager: 3 false positives from a stack partly outside the frame.

---

## Report data structure (input for Muse Spark)
```json
{
  "site": "Warehouse A", "camera": "cam-3", "date": "2026-11-02", "uptime_min": 1428,
  "events": [
    {"id": "evt-0412", "time": "10:42:17", "rule_id": "r3", "severity": "high", "duration_s": 2,
     "objects": [{"name": "forklift", "box": [412, 180, 760, 520]}, {"name": "person", "box": [790, 210, 860, 500]}],
     "measurements": {"distance_m": 1.3, "count": null, "height_m": null}, "frame": "evt-0412.jpg",
     "explanation": "Forklift passed 1.3 m from a worker carrying a box."}
  ],
  "rules": [{"id": "r3", "title": "Forklift within 2 m of a pedestrian", "events": 1, "false_positives_marked": 0, "enabled": true}],
  "manager_actions": [{"time": "11:00", "action": "disabled", "rule_id": "r9", "reason": "false positives"}]
}
```
The report must give the number of events by severity, list high events in full, group repeated medium events, find patterns (time, place), suggest one or two actions and note rule health. It must not invent events that are not in the input, and it must not give legal assessments.

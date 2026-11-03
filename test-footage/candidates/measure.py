#!/usr/bin/env python3
"""Per-second measured values from timelines.json: for every person/forklift box the track id,
foot point (bottom-centre, 0..1), whether it is in the rule zone, and for PPE the best share of a
hard hat box in the person's upper third / vest box in the middle third (engine threshold 0.3).
Writes measurements.json and prints one line per second."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from safety_mode.engine.geometry import foot_point, point_in_polygon, mask_overlap  # noqa: E402

tl = json.loads((HERE / "timelines.json").read_text())
out = {}
for clip, v in tl.items():
    zones = json.loads((HERE / "zones" / f"{clip}.json").read_text())
    W, H = zones["frame_size"]
    rid, rule = next(iter(v["rules"].items()))
    zone = next(z for z in zones["zones"])["polygon"]
    rows = []
    for s, stem in enumerate(sorted(v["detections"])):
        fr = v["detections"][stem]
        objs = []
        for p in ("person", "forklift"):
            for r in fr.get(p, []):
                b = [r["box"][0] / W, r["box"][1] / H, r["box"][2] / W, r["box"][3] / H]
                fx, fy = foot_point(b)
                o = {"obj": f"{p} #{r['track_id']}", "foot": [round(fx, 3), round(fy, 3)],
                     "in_zone": point_in_polygon((fx, fy), zone)}
                if p == "person":
                    for ppe in ("hard hat", "safety vest"):
                        if ppe in fr:
                            shares = [mask_overlap(b, "hard_hat" if ppe == "hard hat" else "vest",
                                                   [q["box"][0] / W, q["box"][1] / H, q["box"][2] / W, q["box"][3] / H])[1]
                                      for q in fr[ppe]]
                            o[ppe] = round(max(shares), 2) if shares else 0.0
                objs.append(o)
        rows.append({"s": s, "state": v["rules"][rid]["states"][s], "truth": v["rules"][rid]["truth"][s], "objects": objs})
    out[clip] = {"rule": rid, "rows": rows}
    print("==", clip, rid)
    for r in rows:
        print(f"s{r['s']:<3}{r['state']:<10}", "; ".join(
            f"{o['obj']} foot {o['foot']} {'IN' if o['in_zone'] else 'out'}" +
            "".join(f" {k} {o[k]}" for k in ("hard hat", "safety vest") if k in o) for o in r["objects"]))
(HERE / "measurements.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

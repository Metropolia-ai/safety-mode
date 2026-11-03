"""What the engine measured on one frame, for drawing proof on the page and on verification frames.

Not part of the decision: the engine decides with evaluate.py; this module recomputes the same
quantities per object so a viewer can check them (foot points, zone membership, PPE overlap per
person, floor distance and the two nearest points per forklift/person pair).

    diag = frame_diagnostics(rule["condition"], ctx)     # ctx: primitives.Context of the frame
"""
from __future__ import annotations

from .geometry import foot_point, mask_overlap, nearest_floor_points, point_in_polygon
from .evaluate import node_key
from .primitives import is_operator


def _prims(node, out=None) -> list:
    out = [] if out is None else out
    key = node_key(node)
    if key in ("all", "any"):
        for c in node[key]:
            _prims(c, out)
    elif key == "not":
        _prims(node[key], out)
    else:
        out.append((key, node[key]))
    return out


def _r(v, n=4):
    return round(float(v), n)


def frame_diagnostics(condition: dict | None, ctx) -> dict:
    """{"objects": [...], "pairs": [...]} with normalised image coordinates.

    objects: every detection of an object the rule binds: name, index, id, box, foot (bottom-centre),
      zones (rule zones the foot point is in), ppe {item: {"share", "worn"}} for wearing() persons.
    pairs: distance_lt pairs a/b: distance_m (None without calibration), operator (pair skipped by the
      engine: person riding the machine), lt (distance < m), m, and p1/p2: the nearest points of the two
      bottom edges on the floor, mapped back to the image.
    """
    if not condition:
        return {"objects": [], "pairs": []}
    prims = _prims(condition)
    zone_by_obj: dict = {}
    ppe_by_person: dict = {}
    dist_specs = []
    for key, a in prims:
        if key in ("in_zone", "count_in_zone"):
            zone_by_obj.setdefault(a["object"], []).append(a["zone"])
        elif key == "wearing":
            ppe_by_person.setdefault(a["person"], [])
            ppe_by_person[a["person"]] += [p for p in a["ppe"] if p not in ppe_by_person[a["person"]]]
        elif key == "distance_lt":
            dist_specs.append((a["a"], a["b"], float(a["m"])))
    names = list(dict.fromkeys(list(zone_by_obj) + list(ppe_by_person) + [n for s in dist_specs for n in s[:2]]))
    objects = []
    for n in names:
        for d in ctx.dets.get(n, []):
            fp = foot_point(d.box)
            o = {"name": n, "index": d.index, "id": d.track_id, "box": [_r(v) for v in d.box],
                 "foot": [_r(fp[0]), _r(fp[1])]}
            if n in zone_by_obj:
                o["zones"] = [z for z in zone_by_obj[n] if z in ctx.zones and point_in_polygon(fp, ctx.zones[z])]
            if n in ppe_by_person:
                ppe = {}
                for item in ppe_by_person[n]:
                    best = 0.0
                    for p in ctx.dets.get(item, []):
                        best = max(best, mask_overlap(d.box, item, p.box, p.mask)[1])
                    worn = any(mask_overlap(d.box, item, p.box, p.mask)[0] for p in ctx.dets.get(item, []))
                    ppe[item] = {"share": _r(best, 2), "worn": worn}
                o["ppe"] = ppe
                o["all_worn"] = all(v["worn"] for v in ppe.values())
            objects.append(o)
    pairs = []
    for a, b, m in dist_specs:
        for da in ctx.dets.get(a, []):
            for db in ctx.dets.get(b, []):
                if a == b and db.index == da.index:
                    continue
                p = {"a": {"name": a, "index": da.index, "id": da.track_id},
                     "b": {"name": b, "index": db.index, "id": db.track_id}, "m": m,
                     "operator": is_operator(db.box, da.box) or is_operator(da.box, db.box)}
                if ctx.H is None:
                    p.update(distance_m=None, lt=None, p1=None, p2=None)
                else:
                    dist, q1, q2 = nearest_floor_points(da.box, db.box, ctx.H)
                    i1, i2 = ctx.H.to_image(q1), ctx.H.to_image(q2)
                    p.update(distance_m=round(dist, 2), lt=(dist < m) and not p["operator"],
                             p1=[_r(i1[0]), _r(i1[1])], p2=[_r(i2[0]), _r(i2[1])])
                pairs.append(p)
    pairs.sort(key=lambda p: (p["operator"], p["distance_m"] if p["distance_m"] is not None else 1e9))
    return {"objects": objects, "pairs": pairs}

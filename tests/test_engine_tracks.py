"""Episodes per tracked object (track_id from SAM 3.1 video mode)."""
from safety_mode.engine import Detection, Engine, Frame

ZONES = {"frame_size": [100, 100], "floor_calibration": None,
         "zones": [{"name": "z", "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]}]}


def rule(min_s=2):
    return {"rule_id": "rv", "title": "vest past the gate",
            "objects": [{"name": "person", "prompt": "person"}, {"name": "vest", "prompt": "safety vest"}],
            "zones": ["z"],
            "condition": {"all": [{"in_zone": {"object": "person", "zone": "z"}},
                                  {"not": {"wearing": {"person": "person", "ppe": ["vest"]}}}]},
            "min_duration_s": min_s, "severity": "high",
            "explanation_template": "{person_count} person(s) without {missing_ppe} for {duration_s} s at {time}",
            "unsupported": None, "suggested_rewrite": None}


def person(tid, x=0.1):
    return Detection("person", [x, 0.6, x + 0.1, 0.9], track_id=tid)


def run(pattern_a, pattern_b, ids=True, **kw):
    """pattern per second: '1' = that person is in the zone without a vest, '0' = not in the frame."""
    eng = Engine([rule(kw.pop("min_s", 2))], ZONES, box_units="norm", **kw)
    for t, (a, b) in enumerate(zip(pattern_a, pattern_b)):
        dets = []
        if a == "1":
            dets.append(person(2 if ids else None, 0.1))
        if b == "1":
            dets.append(person(5 if ids else None, 0.6))
        eng.process(Frame(t, dets))
    eng.finish()
    return eng.events


def test_two_people_one_leaves_for_long():
    # person #2 leaves after s3 for 3 s and comes back: a new episode; person #5 stays all the time
    evs = run("1111000111", "1111111111")
    by = {}
    for e in evs:
        by.setdefault(e.subject, []).append((e.condition_since_s, e.start_s, e.end_s))
    assert by == {"person #2": [(0, 1, 3), (7, 8, 9)], "person #5": [(0, 1, 9)]}
    e5 = next(e for e in evs if e.subject == "person #5")
    assert e5.object_ids == [{"name": "person", "track_id": 5}]
    assert [o["track_id"] for o in e5.objects] == [5]
    assert e5.measurements["duration_s"] == 10
    assert e5.explanation_inputs["subject"] == "person #5"
    assert e5.explanation_inputs["person_count"] == 1


def test_short_absence_keeps_the_episode():
    # one missing second (max_gap_s 1, the default) does not split person #2's episode
    evs = run("1110111", "0000000")
    assert [(e.subject, e.start_s, e.end_s) for e in evs] == [("person #2", 1, 6)]


def test_without_track_ids_one_episode_per_rule():
    evs = run("1111000111", "1111111111", ids=False)
    assert len(evs) == 1 and evs[0].subject == "" and evs[0].object_ids == []
    assert (evs[0].start_s, evs[0].end_s) == (1, 9)


def test_episodes_rule_one_episode_named_at_activation():
    evs = run("1111000111", "1111111111", episodes="rule")
    assert len(evs) == 1 and (evs[0].start_s, evs[0].end_s) == (1, 9)
    # the single episode names the tracked people true when it was raised
    assert evs[0].subject == "person #2 and person #5"
    assert evs[0].object_ids == [{"name": "person", "track_id": 2}, {"name": "person", "track_id": 5}]
    assert run("0011111", "0000000", episodes="rule")[0].subject == "person #2"


def test_window_counts_per_person():
    # rule true every second, but each person only 1 s in a row: no person reaches 2 s
    evs = run("10101", "01010")
    assert evs == []
    assert run("10101", "01010", ids=False)[0].start_s == 1


def test_distance_pair_subject():
    """r03 on the busy-warehouse floor: the episode belongs to the pair forklift #0 / person #3."""
    import json
    from pathlib import Path
    from safety_mode.engine.geometry import Homography
    root = Path(__file__).resolve().parent.parent
    zf = json.loads((root / "test-footage/zones/pexels-6079421-forklift-busy-warehouse.json").read_text())
    H = Homography.from_calibration(zf["floor_calibration"])

    def floor_box(x0, x1, y, h):
        (u0, v), (u1, _) = H.to_image((x0, y)), H.to_image((x1, y))
        return [u0, v - h, u1, v]

    eng = Engine([root / "rules/r03.json"], zf, box_units="norm")
    fk = Detection("forklift", floor_box(0.0, 1.5, 2.0, 0.2), track_id=0)
    near = Detection("person", floor_box(3.0, 3.4, 2.0, 0.15), track_id=3)       # 1.5 m
    far = Detection("person", floor_box(8.0, 8.4, 2.0, 0.15), track_id=4)        # 6.5 m
    names = {o["name"] for o in json.loads((root / "rules/r03.json").read_text())["objects"]}
    assert names == {"forklift", "person"}
    eng.process(Frame(0, [fk, near, far]))
    eng.finish()
    (ev,) = eng.events
    assert ev.subject == "forklift #0 and person #3"
    assert ev.object_ids == [{"name": "forklift", "track_id": 0}, {"name": "person", "track_id": 3}]

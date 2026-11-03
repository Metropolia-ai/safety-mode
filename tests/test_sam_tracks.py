"""Track id bridging across short breaks (detect/tracks.py) and the committed bridged cache."""
import json

from safety_mode.detect.tracks import bridge_prompt
from safety_mode.engine.geometry import Homography
from safety_mode.pipeline import DETS_DIR, ZONES_DIR


def rec(tid, box):
    return {"box": box, "score": 0.9, "track_id": tid}


A = [100, 100, 200, 400]
A2 = [110, 105, 210, 405]       # IoU with A ~0.8
FAR = [700, 100, 800, 400]


def test_new_id_on_the_same_box_takes_the_old_id():
    secs = {"001": [rec(1, A)], "002": [rec(1, A)], "003": [rec(4, A2), rec(6, FAR)]}
    out, br = bridge_prompt(secs, "person")
    assert [d["track_id"] for d in out["003"]] == [1, 6]
    assert out["003"][0]["sam_track_id"] == 4 and "sam_track_id" not in out["003"][1]
    assert br == [{"prompt": "person", "second": 2, "from": 4, "to": 1, "last_seen_s": 1, "iou": 0.79, "foot_m": None}]


def test_one_missing_second_is_bridged_two_are_not():
    one = {"001": [rec(1, A)], "002": [], "003": [rec(4, A2)]}
    assert bridge_prompt(one, "person")[0]["003"][0]["track_id"] == 1
    two = {"001": [rec(1, A)], "002": [], "003": [], "004": [rec(4, A2)]}
    assert bridge_prompt(two, "person")[0]["004"][0]["track_id"] == 4


def test_old_id_still_present_is_not_taken():
    secs = {"001": [rec(1, A)], "002": [rec(1, A), rec(4, A2)]}
    out, br = bridge_prompt(secs, "person")
    assert [d["track_id"] for d in out["002"]] == [1, 4] and br == []


def test_one_to_one_best_overlap_wins():
    secs = {"001": [rec(1, A)], "002": [rec(4, [150, 100, 250, 400]), rec(5, A2)]}
    out, _ = bridge_prompt(secs, "person")
    assert {d.get("sam_track_id", d["track_id"]): d["track_id"] for d in out["002"]} == {4: 4, 5: 1}


def test_foot_point_on_the_floor_bridges_without_overlap():
    zf = json.loads((ZONES_DIR / "pexels-15170997-warehouse-team.json").read_text())
    H = Homography.from_calibration(zf["floor_calibration"])
    w, h = 1280, 2276
    old = [600, 1000, 640, 1400]                       # narrow box, foot point (620, 1400)
    fx, fy = H.to_floor((620 / w, 1400 / h))
    u, _ = H.to_image((fx + 0.1, fy))                  # 0.1 m to the side on the floor
    new = [u * w - 20, 1300, u * w + 20, 1400]        # half hidden: only the legs, IoU far below 0.4
    out, br = bridge_prompt({"001": [rec(1, old)], "002": [rec(4, new)]}, "person", frame_size=(w, h), H=H)
    assert out["002"][0]["track_id"] == 1 and br[0]["iou"] < 0.4 and br[0]["foot_m"] <= 0.3


def test_committed_cache_has_unique_ids_per_second_and_logged_bridges():
    for f in DETS_DIR.glob("*.video.json"):
        v = json.loads(f.read_text())
        for stem, rec_ in v["frames"].items():
            for p, dets in rec_.items():
                ids = [d["track_id"] for d in dets]
                assert len(ids) == len(set(ids)), (f.name, stem, p)
        changed = {(p, d["sam_track_id"]) for r in v["frames"].values() for p, ds in r.items() for d in ds
                   if "sam_track_id" in d}
        assert changed == {(b["prompt"], b["from"]) for b in v["bridges"]}

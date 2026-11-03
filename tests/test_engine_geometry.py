"""Engine geometry: polygons, foot point, floor homography, distances, PPE overlap."""
import json
from pathlib import Path

import numpy as np
import pytest

from safety_mode.engine.geometry import (Homography, distance_m, foot_point, mask_overlap,
                                         overlap_ratio, point_in_polygon, ppe_region)

ZONES = Path(__file__).resolve().parent.parent / "test-footage" / "zones"
BUSY = json.loads((ZONES / "pexels-6079421-forklift-busy-warehouse.json").read_text())
TEAM = json.loads((ZONES / "pexels-15170997-warehouse-team.json").read_text())


def test_point_in_polygon_square_and_edge():
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert point_in_polygon((0.5, 0.5), sq)
    assert not point_in_polygon((1.5, 0.5), sq)
    assert point_in_polygon((1.0, 0.5), sq)  # on the edge counts as inside


def test_point_in_polygon_concave_past_gate():
    poly = next(z["polygon"] for z in BUSY["zones"] if z["name"] == "past_gate")
    assert point_in_polygon((0.5, 0.9), poly)       # shop floor near the camera
    assert not point_in_polygon((0.42, 0.56), poly)  # gate opening (notch in the polygon)
    assert not point_in_polygon((0.5, 0.3), poly)    # wall above the floor


def test_foot_point_is_bottom_centre():
    assert foot_point([10, 20, 30, 100]) == (20.0, 100)


@pytest.mark.parametrize("cal", [BUSY["floor_calibration"], TEAM["floor_calibration"]])
def test_homography_round_trip(cal):
    H = Homography.from_calibration(cal)
    for px, m in zip(cal["points_px"], cal["points_m"]):
        assert H.to_floor(px) == pytest.approx(m, abs=1e-9)
        assert H.to_image(m) == pytest.approx(px, abs=1e-9)
    # arbitrary floor point outside the calibration rectangle
    p = (1.7, 6.3)
    assert H.to_floor(H.to_image(p)) == pytest.approx(p, abs=1e-9)


def test_homography_none_without_calibration():
    assert Homography.from_calibration(None) is None


def test_distance_between_bottom_edges():
    H = Homography.from_calibration(BUSY["floor_calibration"])
    def box_at(x0, x1, y_m, h=0.1):
        (u0, v), (u1, _) = H.to_image((x0, y_m)), H.to_image((x1, y_m))
        return [u0, v - h, u1, v]
    a = box_at(0.0, 1.0, 2.0)
    b = box_at(2.5, 3.0, 2.0)
    assert distance_m(a, b, H) == pytest.approx(1.5, abs=1e-6)
    c = box_at(0.2, 0.6, 4.0)
    assert distance_m(a, c, H) == pytest.approx(2.0, abs=1e-6)


def test_overlap_ratio():
    assert overlap_ratio([0, 0, 1, 1], [0, 0, 2, 2]) == 1.0
    assert overlap_ratio([0, 0, 2, 2], [0, 0, 1, 1]) == 0.25
    assert overlap_ratio([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0


def test_ppe_region_by_name():
    assert ppe_region("hard_hat") == "upper"
    assert ppe_region("helmet") == "upper"
    assert ppe_region("vest") == "middle"
    assert ppe_region("hi_vis_jacket") == "middle"
    assert ppe_region("safety_boots") == "lower"
    assert ppe_region("gloves") == "whole"


def test_mask_overlap_with_masks_and_box_fallback():
    person = [0.4, 0.2, 0.6, 0.8]          # thirds: 0.2-0.4, 0.4-0.6, 0.6-0.8
    hat_box = [0.45, 0.18, 0.55, 0.26]
    assert mask_overlap(person, "hard_hat", hat_box)[0]           # box fallback
    assert not mask_overlap(person, "vest", hat_box)[0]           # hat box is not on the torso
    mask = np.zeros((100, 100), bool)
    mask[45:55, 42:58] = True                                     # torso, middle third
    assert mask_overlap(person, "vest", [0.42, 0.45, 0.58, 0.55], mask)[0]
    assert not mask_overlap(person, "hard_hat", [0.42, 0.45, 0.58, 0.55], mask)[0]


def test_nearest_floor_points_match_distance():
    """The drawn distance line (diagnose.py) ends at the two points distance_m measures between."""
    import math
    import random
    from safety_mode.engine.geometry import Homography, distance_m, nearest_floor_points
    H = Homography.from_calibration(BUSY["floor_calibration"])
    rnd = random.Random(7)
    for _ in range(300):
        ax = sorted(rnd.uniform(0, 1) for _ in range(2))
        bx = sorted(rnd.uniform(0, 1) for _ in range(2))
        a = [ax[0], 0.2, ax[1], rnd.uniform(0.65, 1)]
        b = [bx[0], 0.2, bx[1], rnd.uniform(0.65, 1)]
        d, p, q = nearest_floor_points(a, b, H)
        assert abs(d - distance_m(a, b, H)) < 1e-9
        assert abs(math.dist(p, q) - d) < 1e-9


def test_frame_diagnostics_ppe_and_pairs():
    from safety_mode.engine.diagnose import frame_diagnostics
    from safety_mode.engine.evaluate import normalise_detections, zones_to_polygons
    from safety_mode.engine.geometry import Homography
    from safety_mode.engine.primitives import Context
    from safety_mode.engine.types import Detection
    dets = [Detection("person", [0.30, 0.50, 0.36, 0.80], track_id=1),
            Detection("vest", [0.30, 0.61, 0.36, 0.69]),
            Detection("forklift", [0.40, 0.45, 0.55, 0.78], track_id=0)]
    ctx = Context(normalise_detections(dets, None, "norm"), zones_to_polygons(BUSY),
                  Homography.from_calibration(BUSY["floor_calibration"]), {})
    ppe = {"all": [{"in_zone": {"object": "person", "zone": "past_gate"}},
                   {"not": {"wearing": {"person": "person", "ppe": ["vest"]}}}]}
    d = frame_diagnostics(ppe, ctx)
    (p,) = d["objects"]
    assert p["id"] == 1 and p["zones"] == ["past_gate"] and p["ppe"]["vest"]["worn"] and p["all_worn"]
    d = frame_diagnostics({"distance_lt": {"a": "forklift", "b": "person", "m": 2}}, ctx)
    (pair,) = d["pairs"]
    assert pair["a"]["id"] == 0 and pair["b"]["id"] == 1 and not pair["operator"]
    assert pair["lt"] == (pair["distance_m"] < 2) and len(pair["p1"]) == 2

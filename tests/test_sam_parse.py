"""Offline tests of the SAM 3.1 output parser on raw responses saved from the Meta API (27.10.2026)."""
from pathlib import Path

import numpy as np
import pytest

from safety_mode.detect import SamParseError, parse_output
from safety_mode.detect.sam_parse import decode_mask

FIX = Path(__file__).parent / "fixtures" / "sam31"
BUSY_LOSSLESS = FIX / "pexels-6079421-forklift-busy-warehouse_005_person.txt"
BUSY_ONEBIT = FIX / "pexels-6079421-forklift-busy-warehouse_005_person_onebit_conf.txt"
TEAM_VEST = FIX / "pexels-15170997-warehouse-team_009_high-visibility-vest_onebit_conf.txt"
VIDEO = FIX / "pexels-32243653-forklift-operator_video1280_person.txt"


def _check_masks(dets):
    for d in dets:
        assert d.mask is not None and d.mask.dtype == np.uint8
        h, w = d.mask_shape
        assert d.mask.shape == (h, w)
        # box-local raster: one mask pixel per box pixel (box corners are inclusive on the wire)
        assert (round(d.mask_box[3] - d.mask_box[1]), round(d.mask_box[2] - d.mask_box[0])) == (h, w)
        assert set(np.unique(d.mask)) <= {0, 1}
        fill = d.mask.mean()
        assert 0.15 < fill < 0.95, fill
        # mask touches all four box edges (the box is the mask's bounding box)
        ys, xs = np.nonzero(d.mask)
        assert ys.min() <= 1 and xs.min() <= 1 and ys.max() >= h - 2 and xs.max() >= w - 2


def test_lossless_image_person():
    dets = parse_output(BUSY_LOSSLESS.read_text(), "person")
    assert len(dets) == 4
    assert {d.track_id for d in dets} == {0, 1, 2, 3}
    assert all(d.frame_index == 0 and d.frame_size == (1280, 720) for d in dets)
    assert all(d.confidence is None for d in dets)
    op = next(d for d in dets if d.track_id == 0)
    assert op.box == [752, 286, 856, 409]          # x2/y2 inclusive 855/408 on the wire
    assert op.mask_shape == (123, 104)
    _check_masks(dets)


def test_one_bit_with_confidence():
    dets = parse_output(BUSY_ONEBIT.read_text(), "person")
    assert len(dets) == 4
    assert all(d.mask_payload.startswith("!") for d in dets)
    assert all(0.5 <= d.confidence <= 1 for d in dets)
    _check_masks(dets)


def test_same_object_lossless_vs_one_bit_agree():
    a = {d.track_id: d for d in parse_output(BUSY_LOSSLESS.read_text(), "person")}
    b = parse_output(BUSY_ONEBIT.read_text(), "person")
    # two separate requests: ids are not guaranteed to match, so pair by box overlap
    for d in b:
        best = max(a.values(), key=lambda x: _iou(x.box, d.box))
        assert _iou(best.box, d.box) > 0.9


def test_scale_to_source_pixels_and_full_mask():
    # warehouse-team frames are 1280x2276, sent to SAM as 720x1280
    sx, sy = 1280 / 720, 2276 / 1280
    dets = parse_output(TEAM_VEST.read_text(), "high-visibility vest", scale=(sx, sy))
    assert len(dets) == 2
    for d in dets:
        assert d.frame_size == (720, 1280)
        assert 0 <= d.box[0] < d.box[2] <= 1280 + 1 and 0 <= d.box[1] < d.box[3] <= 2276 + 2
        full = d.full_mask(1280, 2276)
        x1, y1, x2, y2 = (int(round(v)) for v in d.box)
        assert full.sum() > 0
        assert full[:, :max(x1 - 1, 0)].sum() == 0 and full[:max(y1 - 1, 0), :].sum() == 0
        assert full[y1:y2, x1:x2].sum() == full.sum()


def test_video_lines_and_stable_ids():
    text = VIDEO.read_text()
    dets = parse_output(text, "person", decode_masks=False)
    frames = sorted({d.frame_index for d in dets})
    assert len(text.splitlines()) == len(frames) == 254
    assert frames == list(range(254))
    assert all(a.frame_index <= b.frame_index for a, b in zip(dets, dets[1:]))
    assert sum(1 for d in dets if d.track_id == 0) == 254
    d = dets[100]
    m = decode_mask(d.mask_payload, *d.mask_shape)
    assert m.shape == d.mask_shape and m.sum() > 0


def test_empty_output_is_no_detections():
    assert parse_output("", "forklift") == []
    assert parse_output("\n", "forklift") == []


def test_garbage_raises():
    with pytest.raises(SamParseError):
        parse_output("I could not find anything", "person")
    with pytest.raises(SamParseError):
        parse_output("<0f>0<|box;x1=1;y1=1;x2=2;y2=2;w=3;h=3|><|mask;x=0;y=0;data=bad|>", "person")


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0

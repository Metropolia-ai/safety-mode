"""SAM 3.1 video mode: frame alignment, sampling to 1 fps, committed video cache (no network)."""
import json

from safety_mode.detect import sam_video
from safety_mode.detect.sam_api import SamResult
from safety_mode.pipeline import DETS_DIR


def test_second_to_frame_middle_of_second():
    # checked by image difference against frames/<clip>/NNN.jpg on all three clips
    assert [sam_video.second_to_frame(s, 25, 254) for s in (0, 1, 9)] == [12, 37, 237]
    assert [sam_video.second_to_frame(s, 24, 604) for s in (0, 1, 24)] == [11, 35, 587]
    assert [sam_video.second_to_frame(s, 30, 683) for s in (0, 1, 22)] == [14, 44, 674]
    assert sam_video.second_to_frame(30, 25, 254) == 253          # clamped to the last frame


def test_short_names():
    assert sam_video.short_name("pexels-6079421-forklift-busy-warehouse") == "busy-warehouse"
    assert sam_video.short_name("pexels-15170997-warehouse-team") == "warehouse-team"
    assert sam_video.video_path("pexels-32243653-forklift-operator").name == "forklift-operator-1280.mp4"


def test_sample_seconds_scales_and_keeps_ids():
    # 2 fps toy video, 640x360 sent, frames at 1280x720: frame 0 (s0) and frame 2 (s1) are sampled
    text = ("<0f>0<|box;x1=10;y1=20;x2=29;y2=59;w=640;h=360;c=0.9|>\n"
            "<1f>0<|box;x1=11;y1=20;x2=30;y2=59;w=640;h=360;c=0.9|>\n"
            "<2f>3<|box;x1=100;y1=100;x2=109;y2=119;w=640;h=360;c=0.8|>,0<|box;x1=12;y1=20;x2=31;y2=59;w=640;h=360;c=0.9|>\n")
    r = SamResult(text, "person", (1280, 720), (640, 360), 0.0, True, {})
    out = sam_video.sample_seconds(r, (1280, 720), ["001", "002"], fps=2, n_frames=3)
    assert out["001"] == [{"box": [20.0, 40.0, 60.0, 120.0], "score": 0.9, "track_id": 0}]
    assert [d["track_id"] for d in out["002"]] == [0, 3]           # sorted by id
    assert out["002"][0]["box"] == [24.0, 40.0, 64.0, 120.0]


def test_committed_video_cache_shape():
    for clip in ("pexels-6079421-forklift-busy-warehouse", "pexels-15170997-warehouse-team",
                 "pexels-32243653-forklift-operator"):
        v = json.loads((DETS_DIR / f"{clip}.video.json").read_text())
        img = json.loads((DETS_DIR / f"{clip}.json").read_text())
        assert v["mode"] == "video" and v["frame_size"] == img["frame_size"]
        assert sorted(v["frames"]) == sorted(img["frames"])
        assert {"person", "forklift", "hard hat", "safety vest"} <= set(v["prompts"])
        for rec in v["frames"].values():
            for dets in rec.values():
                assert all(isinstance(d["track_id"], int) for d in dets)
    # forklift-operator: the operator is one id over the whole clip
    v = json.loads((DETS_DIR / "pexels-32243653-forklift-operator.video.json").read_text())
    assert v["tracks"]["person"]["0"] == {"first_s": 0, "last_s": 9, "seconds": 10}

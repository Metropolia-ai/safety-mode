#!/usr/bin/env python3
"""Thin wrapper: runs safety_mode.pipeline on test-footage/candidates/ without touching pipeline.py.

pipeline.py and detect/sam_video.py read module-level paths (FRAMES_DIR, DETS_DIR, ZONES_DIR, CLIPS,
FOOTAGE, VIDEO_DIR); they are re-pointed here at runtime. SAM 3.1 full responses go to
candidates/cache/sam31/ (gitignored), compact detections to candidates/detections/<clip>.video.json.

  python test-footage/candidates/run_candidates.py --export      # paid: one video request per clip and prompt
  python test-footage/candidates/run_candidates.py               # offline: rules -> timelines, report.json
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
from clips import CLIPS, SAM_FPS                      # noqa: E402
from safety_mode import pipeline                      # noqa: E402
from safety_mode.detect import sam_video              # noqa: E402
from safety_mode.detect.sam_api import SamClient      # noqa: E402

pipeline.FRAMES_DIR = HERE / "frames"
pipeline.DETS_DIR = HERE / "detections"
pipeline.ZONES_DIR = HERE / "zones"
pipeline.CLIPS = list(CLIPS)
sam_video.FOOTAGE = HERE          # homography() and _frame_stems_and_size() read FOOTAGE/zones, FOOTAGE/frames
sam_video.VIDEO_DIR = HERE / "cache" / "video"
sam_video.RAW_DIR = HERE / "raw"
sam_video.DETS_DIR = HERE / "detections"

RULES = REPO / "rules"
# clip -> [(rule file, prompts the rule needs)]
PLAN = {
    "pexels-38707991-man-crosses-floor": [RULES / "r02.json"],
    "pexels-4280746-hatless-visitor": [HERE / "rules" / "r02-hat.json"],
    "pexels-31751344-forklift-crosses-aisle": [RULES / "r08.json"],
    "pexels-32245123-site-worker-no-vest": [REPO / "test-footage" / "rules" / "r02-vest.json"],
    "pexels-32268407-forklift-roll-worker": [HERE / "rules" / "r01-forks.json"],
    "pexels-13422071-scrubber-worker": [RULES / "r02.json"],
}


def prompts_for(clip):
    return sorted({o["prompt"] for rp in PLAN[clip] for o in json.loads(rp.read_text())["objects"]})


def client(offline=False):
    return SamClient(cache_dir=HERE / "cache" / "sam31", offline=offline, **pipeline.SAM_OPTIONS)


def export(clips):
    c = client()
    billed = 0
    for clip in clips:
        before = c.calls
        path, _ = sam_video.export_video_detections(clip, prompts_for(clip), c)
        n = sam_video.probe(sam_video.video_path(clip))["frames"]
        paid = c.calls - before
        billed += paid * sam_video.billed_frames(n)
        print(f"{clip}: {paid} paid video requests x {n} frames ({SAM_FPS} fps) -> {path.name}")
    print(f"paid requests {c.calls}, frames billed {billed}, ~${billed * sam_video.PRICE_PER_1000_FRAMES / 1000:.3f}")


def run(clips):
    out = {}
    src = pipeline.DetectionSource("offline", client=client(offline=True), detector="video")
    for clip in clips:
        rules = PLAN[clip]
        r = pipeline.run_clip(clip, [str(p) for p in rules], source=src, episodes="rule")
        n = len(r.timeline[next(iter(r.timeline))])
        clip_out = {"rules": {}, "detections": {}}
        for rid, tl in r.timeline.items():
            alert = set()
            for e in r.events:
                if e.rule_id == rid:
                    alert |= set(range(int(e.start_s), int(e.end_s) + 1))
            states = ["ALERT" if s in alert else ("candidate" if tl[s] == "T" else "idle") for s in range(n)]
            clip_out["rules"][rid] = {
                "title": r.rules[rid]["title"], "min_duration_s": r.rules[rid]["min_duration_s"],
                "truth": tl, "states": states,
                "events": [{"start_s": e.start_s, "end_s": e.end_s, "since_s": e.condition_since_s,
                            "subject": e.subject} for e in r.events if e.rule_id == rid],
                "matched": [[{"name": o.name, "track_id": getattr(o, "track_id", None)} for o in res.objects]
                            if res is not None else [] for res in r.results[rid]],
            }
        for (stem, p), d in sorted(r.detections.items()):
            clip_out["detections"].setdefault(stem, {})[p] = d
        out[clip] = clip_out
        for rid, v in clip_out["rules"].items():
            print(f"{clip} {rid}: {v['truth']}  events {[(e['start_s'], e['end_s']) for e in v['events']]}")
    (HERE / "timelines.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--clip", action="append")
    a = ap.parse_args()
    clips = a.clip or list(CLIPS)
    export(clips) if a.export else run(clips)

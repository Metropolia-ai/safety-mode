"""Video mode vs image mode, per second and prompt: IoU matches and counts.

    python -m safety_mode.detect.compare [--iou 0.5] [--seconds]

Reads test-footage/detections/<clip>.json (image mode) and <clip>.video.json (video mode),
both in pixels of frames/<clip>/NNN.jpg. A video box matches an image box when IoU >= --iou
(greedy by IoU, one to one).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sam_api import REPO

DETS_DIR = REPO / "test-footage" / "detections"


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(img: list, vid: list, thr: float) -> tuple[list, list, list]:
    """(pairs [(i, v, iou)], unmatched image indices, unmatched video indices)."""
    cands = sorted(((iou(a["box"], b["box"]), i, j) for i, a in enumerate(img) for j, b in enumerate(vid)),
                   reverse=True)
    used_i, used_j, pairs = set(), set(), []
    for o, i, j in cands:
        if o < thr:
            break
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j, o))
    return pairs, [i for i in range(len(img)) if i not in used_i], [j for j in range(len(vid)) if j not in used_j]


def compare_clip(clip: str, prompts: list[str] | None = None, thr: float = 0.5) -> dict:
    img = json.loads((DETS_DIR / f"{clip}.json").read_text())
    vid = json.loads((DETS_DIR / f"{clip}.video.json").read_text())
    prompts = prompts or [p for p in vid["prompts"] if p in img["prompts"]]
    out = {}
    for p in prompts:
        tot = {"image": 0, "video": 0, "matched": 0, "iou_sum": 0.0, "image_only": [], "video_only": []}
        per_s = {}
        for stem in sorted(set(img["frames"]) & set(vid["frames"])):
            a = img["frames"][stem].get(p, [])
            b = vid["frames"][stem].get(p, [])
            pairs, io, vo = match(a, b, thr)
            s = int(stem) - 1
            tot["image"] += len(a)
            tot["video"] += len(b)
            tot["matched"] += len(pairs)
            tot["iou_sum"] += sum(o for _, _, o in pairs)
            tot["image_only"] += [(s, a[i]) for i in io]
            tot["video_only"] += [(s, b[j]) for j in vo]
            per_s[s] = (len(a), len(b), len(pairs))
        tot["mean_iou"] = round(tot["iou_sum"] / tot["matched"], 3) if tot["matched"] else None
        tot["per_second"] = per_s
        out[p] = tot
    return out


def main(argv=None) -> int:
    from ..pipeline import CLIPS
    ap = argparse.ArgumentParser(prog="python -m safety_mode.detect.compare")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--seconds", action="store_true", help="print per-second counts image/video/matched")
    a = ap.parse_args(argv)
    print("| clip | prompt | image boxes | video boxes | matched | image only | video only | mean IoU |")
    print("|---|---|---|---|---|---|---|---|")
    for clip in CLIPS:
        if not (DETS_DIR / f"{clip}.video.json").exists():
            continue
        for p, t in compare_clip(clip, thr=a.iou).items():
            print(f"| {clip} | {p} | {t['image']} | {t['video']} | {t['matched']} | {len(t['image_only'])} | "
                  f"{len(t['video_only'])} | {t['mean_iou']} |")
            if a.seconds:
                print("   " + " ".join(f"s{s}:{i}/{v}/{m}" for s, (i, v, m) in t["per_second"].items() if (i, v) != (m, m)))
                for kind in ("image_only", "video_only"):
                    for s, d in t[kind]:
                        print(f"     {kind} s{s} box {d['box']} score {d.get('score')}"
                              + (f" id {d['track_id']}" if "track_id" in d else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

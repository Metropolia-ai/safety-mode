#!/usr/bin/env python3
"""How many objects the local SAM 3.1 (FP32, Jetson bench) found in each second of a clip.

Reads jetson-bench/logs/sam3.1-fp32_<clip>.json (prompts.<prompt>.found_per_frame: 60 frames at 1 fps,
masks with score >= 0.5) and prints a Markdown table: second x prompt.

Index mapping: the bench cuts the clip into N unique frames (1 fps), loops them and spends the first
`warmup` frames on warm-up (not included in found_per_frame). So found_per_frame[i] is
frame (i + warmup) mod N, i.e. second (i + warmup) mod N. warmup = frames - n from the log line
"[sam3.1-fp32] <clip> frames=65 ..." (bench_sam.py default --warmup 5).

  python3 presence_from_bench.py            # all three clips
  python3 presence_from_bench.py --check    # plus a check that repeated frames give the same counts
"""
import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "jetson-bench", "logs")
CLIPS = {  # clip -> number of unique 1 fps frames (see test-footage/README.md and the "Quality" section of results.md)
    "pexels-6079421-forklift-busy-warehouse": 25,
    "pexels-15170997-warehouse-team": 23,
    "pexels-32243653-forklift-operator": 10,
}
PROMPTS = ["person", "forklift", "high-visibility vest", "hard hat"]


def warmup_of(clip, n):
    log = os.path.join(LOGS, f"sam3.1-fp32_{clip}.log")
    m = re.search(r"frames=(\d+)", open(log, encoding="utf-8").read())
    return int(m.group(1)) - n


def per_second(clip, check=False):
    d = json.load(open(os.path.join(LOGS, f"sam3.1-fp32_{clip}.json"), encoding="utf-8"))
    n_uniq = CLIPS[clip]
    frames_dir = os.path.join(ROOT, "test-footage", "frames", clip)
    if os.path.isdir(frames_dir):  # frames are not in git; if they exist locally, check the count
        assert len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")]) == n_uniq, clip
    res = {}
    for p in PROMPTS:
        found = d["prompts"][p]["found_per_frame"]
        if isinstance(found, str):
            found = json.loads(found)
        w = warmup_of(clip, len(found))
        sec = {}
        for i, v in enumerate(found):
            s = (i + w) % n_uniq
            if s in sec and sec[s] != v and check:
                print(f"  ! {clip} {p}: second {s} gave {sec[s]} and {v} on the repeat")
            sec.setdefault(s, v)
        res[p] = sec
    return res, w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    for clip, n in CLIPS.items():
        res, w = per_second(clip, a.check)
        print(f"\n### {clip} (N={n}, warmup={w}, SAM 3.1 FP32, score>=0.5)\n")
        print("| s | " + " | ".join(PROMPTS) + " |")
        print("|---|" + "---|" * len(PROMPTS))
        for s in range(n):
            print(f"| {s} | " + " | ".join(str(res[p][s]) for p in PROMPTS) + " |")


if __name__ == "__main__":
    main()

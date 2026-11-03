#!/usr/bin/env python3
"""Mask IoU against the SAM 3.1 reference (out/<ref>/<clip>/masks_<prompt>_<idx>.npz, frames 5-14).
Union IoU: union of all masks in the frame (score >= the run threshold) for the reference and for the model.
Computed over frames where the reference found something; a frame where the model found nothing = miss (IoU 0).
iou_union_mean counts misses as 0; iou_union_hit uses only frames where the model found something.
Instance: for each reference mask, the best IoU among the model masks (0 if there are none), averaged over masks.
extra: frames where the reference is empty but the model found something.
Usage: python iou.py [--ref sam3.1-fp32]  -> logs/iou.csv + summary on stdout."""
import argparse, csv, glob, statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ["pexels-15170997-warehouse-team", "pexels-6079421-forklift-busy-warehouse", "pexels-32243653-forklift-operator"]
PROMPTS = ["person", "high-visibility_vest"]
MODELS = ["efficientsam3-rv-m", "efficientsam3-tv-m", "efficientsam3-ev-m", "embedl-sam3-int8", "sam3.1-bf16"]


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="sam3.1-fp32")
    ap.add_argument("--out", default=str(ROOT / "logs" / "iou.csv"))
    args = ap.parse_args()
    rows = []
    for clip in CLIPS:
        for p in PROMPTS:
            ref_files = sorted(glob.glob(str(ROOT / "out" / args.ref / clip / f"masks_{p}_*.npz")))
            for m in MODELS:
                ref_n, misses, extra, u_all, u_hit, inst, n_frames = 0, 0, 0, [], [], [], 0
                for rf in ref_files:
                    of = ROOT / "out" / m / clip / Path(rf).name
                    if not of.exists():
                        continue
                    r, o = np.load(rf), np.load(of)
                    assert str(r["frame"]) == str(o["frame"]), (rf, of)
                    n_frames += 1
                    ru, ou = r["union"], o["union"]
                    if not ru.any():
                        extra += int(ou.any())
                        continue
                    ref_n += 1
                    v = iou(ru, ou) if ou.any() else 0.0
                    u_all.append(v)
                    if ou.any():
                        u_hit.append(v)
                    else:
                        misses += 1
                    for rm in r["masks"]:
                        inst.append(max([iou(rm, om) for om in o["masks"]], default=0.0))
                if not n_frames:
                    continue
                f = lambda xs: round(statistics.mean(xs), 3) if xs else ""
                rows.append({"clip": clip, "prompt": p.replace("_", " "), "model": m, "ref": args.ref,
                             "frames": n_frames, "ref_found_frames": ref_n, "misses": misses, "extra": extra,
                             "iou_union_mean": f(u_all), "iou_union_hit": f(u_hit),
                             "ref_instances": len(inst), "iou_instance_mean": f(inst)})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['clip'][:22]:22s} {r['prompt']:20s} {r['model']:20s} ref {r['ref_found_frames']}/{r['frames']} "
              f"miss {r['misses']} extra {r['extra']} union {r['iou_union_mean']} hit {r['iou_union_hit']} "
              f"inst({r['ref_instances']}) {r['iou_instance_mean']}")
    print("saved", args.out)


if __name__ == "__main__":
    main()

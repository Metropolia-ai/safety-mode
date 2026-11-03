#!/usr/bin/env python3
"""Union IoU of person masks, TensorRT engine run (out/<tag>/<clip>/masks_person_XX.npz, frames 5-14) against the
PyTorch EfficientSAM3 EV-M FP32 run and the SAM 3.1 FP32 reference (same frames, same score threshold 0.5).
Frames where the reference is empty are skipped; a frame where the engine finds nothing counts as IoU 0.
python iou_trt.py [--tag efficientsam3-ev-m-trt] [--clip pexels-15170997-warehouse-team]"""
import argparse, statistics
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="efficientsam3-ev-m-trt"); ap.add_argument("--clip", default="pexels-15170997-warehouse-team")
a = ap.parse_args()
for ref in ["efficientsam3-ev-m", "sam3.1-fp32"]:
    vals, empty = [], 0
    for i in range(5, 15):
        r = np.load(ROOT / "out" / ref / a.clip / f"masks_person_{i:02d}.npz")["union"]
        t = np.load(ROOT / "out" / a.tag / a.clip / f"masks_person_{i:02d}.npz")["union"]
        if not r.any():
            empty += 1; continue
        u = np.logical_or(r, t).sum(); vals.append(float(np.logical_and(r, t).sum() / u))
    print(f"{a.tag} vs {ref} [person, frames 5-14]: union IoU mean {statistics.mean(vals):.3f} over {len(vals)} frames "
          f"(min {min(vals):.3f}); reference empty on {empty}")

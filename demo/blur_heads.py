#!/usr/bin/env python3
"""Blur the head of every person (and forklift driver) in a clip for the demo page (Pexels people are recognisable).

    python demo/blur_heads.py pexels-38707991-man-crosses-floor OUT.mp4
    python demo/blur_heads.py pexels-31751344-forklift-crosses-aisle OUT.mp4

Head region = top 22 % of each person box, grown by 15 % around its centre. Boxes come from the
committed SAM 3.1 video-mode detections (test-footage/detections/<clip>.video.json, one box per
second per track), placed at the time of the video frame they were measured on (frame_map, SAM video
fps) and linearly interpolated per track to every frame of the raw clip. For up to 2 s before a walker's
first box and after its last one the box moves on along the track's velocity from its first (last) two
boxes, clamped to the frame (A: the walker's face shows at 4.4-4.9 s, SAM's first box is at 5.4 s); a
driver's first and last box are held for 0.5 s (B: a moved box would cover the empty aisle). A person box inside a forklift box (the driver) gets
the top half instead, grown 40 %; a second with a forklift and no person uses the upper-middle of the forklift box. Raw frames and
detections are not changed; the output is a new H.264 file (same size, yuv420p, faststart, no audio).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

REPO = Path(__file__).resolve().parents[1]
HEAD_FRAC = 0.22
GROW = 0.15
EDGE_S = 0.5      # driver tracks: first/last box held this long, seconds
EXTRAP_S = 2.0    # walker tracks: first/last box moved along the track's velocity this long, seconds
# a person box inside a forklift box is the driver: SAM draws it up to the cab roof and the head sits
# lower, so the band is the top half of the box, grown 40 % (checked on B, s5-s8)
DRIVER_FRAC = 0.5
DRIVER_GROW = 0.4


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return st["width"], st["height"], float(num) / float(den)


def track_samples(det):
    """{track_id: [(t, box)]} with t = time of the SAM video frame the box was measured on.

    A second with a forklift but no person box gets a stand-in for the driver: the upper-middle part
    of the forklift box (middle 50 % of its width, top 45 % of its height), so its top 22 % covers the cab."""
    vfps = det["video"]["fps"]
    tracks = {}
    for stem, rec in det["frames"].items():
        t = det["frame_map"][stem] / vfps
        W = det["frame_size"][0]
        for d in rec.get("person", []):
            cab = [f["box"] for f in rec.get("forklift", []) if _inside(d["box"], f["box"])]
            box = list(d["box"])
            if cab and (box[0] <= 2 or box[2] >= W - 2):
                # driver cut by the frame edge: SAM's box is only the visible sliver and may miss the head
                # (B, s8); widen it to the upper-middle of the forklift box
                x1, y1, x2, y2 = cab[0]
                box[0], box[2] = min(box[0], x1 + 0.25 * (x2 - x1)), max(box[2], x2 - 0.25 * (x2 - x1))
            tracks.setdefault(d["track_id"], []).append((t, np.array(box, float), bool(cab)))
        if not rec.get("person"):
            for d in rec.get("forklift", []):
                x1, y1, x2, y2 = d["box"]
                w, h = x2 - x1, y2 - y1
                tracks.setdefault(f"forklift-{d['track_id']}", []).append(
                    (t, np.array([x1 + 0.25 * w, y1, x2 - 0.25 * w, y1 + 0.45 * h], float), False))
    return {k: sorted(v, key=lambda x: x[0]) for k, v in tracks.items()}


def _inside(a, b, share=0.5):
    """At least `share` of box a lies inside box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area >= share


def box_at(samples, t, W, H):
    """(box, driver) at time t: linear between samples. Outside them a walker's first/last box moves on
    with the velocity of the first/last two samples for EXTRAP_S, clamped to the frame (W, H in detection
    pixels): people step out from behind racks a second before SAM's first box. A driver's (or a
    single-sample track's) first/last box is held for EDGE_S."""
    ts = [s[0] for s in samples]
    moving = len(samples) > 1 and not any(s[2] for s in samples)
    edge = EXTRAP_S if moving else EDGE_S
    if t < ts[0] - edge or t > ts[-1] + edge:
        return None
    if t <= ts[0] or t >= ts[-1]:
        ref = samples[0] if t <= ts[0] else samples[-1]
        if not moving:
            return ref[1], ref[2]
        a, b = samples[:2] if t <= ts[0] else samples[-2:]
        v = (b[1] - a[1]) / (b[0] - a[0])
        return np.clip(ref[1] + v * (t - ref[0]), 0, [W, H, W, H]), ref[2]
    i = next(i for i in range(1, len(ts)) if ts[i] >= t)
    a, b = samples[i - 1], samples[i]
    w = (t - a[0]) / (b[0] - a[0])
    return a[1] + w * (b[1] - a[1]), a[2] or b[2]


def head_region(box, scale_x, scale_y, W, H, driver=False):
    x1, y1, x2, y2 = box
    frac, grow = (DRIVER_FRAC, DRIVER_GROW) if driver else (HEAD_FRAC, GROW)
    hy2 = y1 + frac * (y2 - y1)
    cx, cy, w, h = (x1 + x2) / 2, (y1 + hy2) / 2, (x2 - x1) * (1 + grow), (hy2 - y1) * (1 + grow)
    r = [int((cx - w / 2) * scale_x), int((cy - h / 2) * scale_y),
         int(np.ceil((cx + w / 2) * scale_x)), int(np.ceil((cy + h / 2) * scale_y))]
    r = [max(0, r[0]), max(0, r[1]), min(W, r[2]), min(H, r[3])]
    return r if r[2] - r[0] > 2 and r[3] - r[1] > 2 else None


def main(clip, out):
    det = json.loads((REPO / "test-footage" / "detections" / f"{clip}.video.json").read_text())
    src = REPO / "test-footage" / "raw" / f"{clip}.mp4"
    W, H, fps = probe(src)
    sx, sy = W / det["frame_size"][0], H / det["frame_size"][1]
    tracks = track_samples(det)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(src), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-crf", "20", "-preset", "slow",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    n = blurred = 0
    while True:
        buf = dec.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        t = n / fps
        regions = [head_region(b, sx, sy, W, H, drv) for b, drv in
                   (x for x in (box_at(s, t, *det["frame_size"]) for s in tracks.values()) if x is not None)]
        regions = [r for r in regions if r]
        if regions:
            im = Image.frombuffer("RGB", (W, H), buf, "raw", "RGB", 0, 1).copy()
            for r in regions:
                crop = im.crop(r)
                k = max(8, (r[2] - r[0]) // 6)
                # pixelate, then blur: no recognisable features survive, edges stay soft
                small = crop.resize((max(1, crop.width // k), max(1, crop.height // k)), Image.BILINEAR)
                im.paste(small.resize(crop.size, Image.NEAREST).filter(ImageFilter.GaussianBlur(k / 2)), r[:2])
            buf = im.tobytes()
            blurred += 1
        enc.stdin.write(buf)
        n += 1
    enc.stdin.close()
    enc.wait()
    dec.wait()
    print(f"{clip}: {n} frames at {fps:g} fps, head blurred on {blurred}, -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))

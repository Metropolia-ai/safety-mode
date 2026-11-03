#!/usr/bin/env python3
"""raw/<clip>-src.mp4 (Pexels 1280x720 file) -> raw/<clip>.mp4 (trimmed, native fps),
cache/video/<short>-1280.mp4 (10 fps, what SAM 3.1 gets), frames/<clip>/NNN.jpg (1 fps).

Frame NNN (second s = NNN - 1) is cut from the 10 fps video at index ceil((s + 0.5) * 10) - 1,
the same mapping safety_mode/detect/sam_video.second_to_frame uses, so 1 fps frames and the
video-mode detections are the same picture."""
import math, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clips import CLIPS, SAM_FPS

def run(*a):
    subprocess.run(a, check=True)

def short(clip):
    p = clip.split("-")
    return "-".join(p[2:])

for clip, c in CLIPS.items():
    src, raw = HERE / "raw" / f"{clip}-src.mp4", HERE / "raw" / f"{clip}.mp4"
    vid = HERE / "cache" / "video" / f"{short(clip)}-1280.mp4"
    if not raw.exists():
        t = ["-ss", str(c["trim"][0]), "-to", str(c["trim"][1])] if c["trim"] else []
        run("ffmpeg", "-v", "error", "-y", "-i", str(src), *t, "-an", "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", str(raw))
    if not vid.exists():
        run("ffmpeg", "-v", "error", "-y", "-i", str(raw), "-vf", f"fps={SAM_FPS}", "-an", "-c:v", "libx264",
            "-crf", "18", "-pix_fmt", "yuv420p", str(vid))
    n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_read_frames", "-of", "csv=p=0", str(vid)], capture_output=True, text=True).stdout)
    secs = n // SAM_FPS
    fd = HERE / "frames" / clip
    fd.mkdir(parents=True, exist_ok=True)
    idx = [min(math.ceil((s + 0.5) * SAM_FPS) - 1, n - 1) for s in range(secs)]
    sel = "+".join(f"eq(n\\,{k})" for k in idx)
    run("ffmpeg", "-v", "error", "-y", "-i", str(vid), "-vf", f"select='{sel}'", "-fps_mode", "passthrough",
        "-q:v", "2", "-start_number", "1", str(fd / "%03d.jpg"))
    print(clip, n, "frames at", SAM_FPS, "fps ->", len(list(fd.glob("*.jpg"))), "1 fps frames")

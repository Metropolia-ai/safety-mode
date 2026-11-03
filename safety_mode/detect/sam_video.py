"""SAM 3.1 video mode: one ``input_video`` request per clip and prompt, stable track ids.

    python -m safety_mode.detect.sam_video [--clip busy-warehouse] [--prompts person forklift ...] [--offline]

Per clip and prompt the whole clip (long side 1280 px, all frames, no audio) is uploaded and
segmented once; SAM 3.1 returns every frame with a track id that stays the same for the
same object across the clip. The full responses live in the SamClient disk cache
(test-footage/cache/sam31/, gitignored); the downscaled MP4s in test-footage/cache/video/.

The pipeline runs at 1 fps on test-footage/frames/<clip>/NNN.jpg (NNN = second + 1). Those
frames were cut from the middle of each second: frame NNN is video frame
ceil((s + 0.5) * fps) - 1 for s = NNN - 1 (checked by image difference on all 58 frames,
24, 25 and 30 fps). ``second_to_frame`` does that with the fps from ffprobe.

Export: test-footage/detections/<clip>.video.json, per second per prompt
[{"box": [x1, y1, x2, y2] in pixels of frames/<clip>/NNN.jpg, "score", "track_id"}].
Track ids are bridged across short breaks (detect/tracks.py): "sam_track_id" keeps the SAM id
where it was changed, "bridges" lists every bridge.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from .sam_api import REPO, SamClient
from .sam_parse import parse_output
from .tracks import bridge_prompt

FOOTAGE = REPO / "test-footage"
VIDEO_DIR = FOOTAGE / "cache" / "video"
RAW_DIR = FOOTAGE / "raw"
DETS_DIR = FOOTAGE / "detections"
MAX_SIDE = 1280
# same request options as the image cache (pipeline.SAM_OPTIONS)
SAM_OPTIONS = {"mask_encoding": "one_bit", "include_confidence": True}
BRIDGE_MAX_GAP_S = 1.0      # same as the engine's default max_gap_s
PRICE_PER_1000_FRAMES = 0.20
MIN_FRAMES_BILLED = 50


def short_name(clip: str) -> str:
    """pexels-6079421-forklift-busy-warehouse -> forklift-busy-warehouse -> busy-warehouse-1280.mp4."""
    parts = clip.split("-")
    tail = "-".join(parts[2:]) if len(parts) > 2 and parts[1].isdigit() else clip
    return {"forklift-busy-warehouse": "busy-warehouse"}.get(tail, tail)


def video_path(clip: str) -> Path:
    return VIDEO_DIR / f"{short_name(clip)}-{MAX_SIDE}.mp4"


def probe(path: Path) -> dict:
    """width, height, fps, frames (decoded count) of the first video stream."""
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, _, den = s["r_frame_rate"].partition("/")
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": float(num) / float(den or 1), "frames": int(s["nb_read_frames"])}


def prepare_video(clip: str) -> Path:
    """test-footage/cache/video/<short>-1280.mp4 from raw/<clip>.mp4 (kept: the cache key is its sha256)."""
    out = video_path(clip)
    if out.exists():
        return out
    raw = RAW_DIR / f"{clip}.mp4"
    if not raw.exists():
        raise FileNotFoundError(f"{raw} is missing (download link in test-footage/README.md)")
    src = probe(raw)
    if src["width"] >= src["height"]:
        scale = f"{MAX_SIDE}:-2" if src["width"] > MAX_SIDE else "iw:ih"
    else:
        scale = f"-2:{MAX_SIDE}" if src["height"] > MAX_SIDE else "iw:ih"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-vf", f"scale={scale}", "-an", "-c:v", "libx264",
                    "-crf", "23", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", str(out)], check=True)
    return out


def second_to_frame(s: float, fps: float, n_frames: int) -> int:
    """Video frame index that frames/<clip>/NNN.jpg shows, s = NNN - 1 (middle of the second)."""
    k = math.ceil((s + 0.5) * fps - 1e-9) - 1
    return min(max(k, 0), n_frames - 1)


def billed_frames(n_frames: int) -> int:
    return max(n_frames, MIN_FRAMES_BILLED)


def segment_clip(clip: str, prompts: list[str], client: Optional[SamClient] = None, workers: int = 4) -> dict:
    """{prompt: SamResult}; one video request per prompt (cache first)."""
    client = client or SamClient(**SAM_OPTIONS)
    path = prepare_video(clip)
    with ThreadPoolExecutor(max(1, min(workers, len(prompts)))) as ex:
        res = list(ex.map(lambda p: client.segment_video(path, p), prompts))
    return dict(zip(prompts, res))


def per_frame(result, frame_size: tuple[int, int]) -> dict[int, list]:
    """{video frame index: [sam_parse.Detection]} with boxes in pixels of frame_size."""
    sw, sh = result.sent_size
    scale = (frame_size[0] / sw, frame_size[1] / sh)
    out: dict[int, list] = {}
    for d in parse_output(result.text, result.prompt, scale=scale, decode_masks=False):
        out.setdefault(d.frame_index, []).append(d)
    return out


def sample_seconds(result, frame_size: tuple[int, int], stems: list[str], fps: float, n_frames: int) -> dict:
    """{stem: [{"box", "score", "track_id"}]} at the video frame each 1 fps frame shows."""
    pf = per_frame(result, frame_size)
    out = {}
    for stem in stems:
        k = second_to_frame(int(stem) - 1, fps, n_frames)
        out[stem] = [{"box": [round(v, 1) for v in d.box], "score": round(d.confidence, 3)
                      if d.confidence is not None else None, "track_id": d.track_id}
                     for d in sorted(pf.get(k, []), key=lambda d: d.track_id)]
    return out


def homography(clip: str):
    """Floor homography of the clip's zones file (None without calibration)."""
    from ..engine.geometry import Homography
    zp = FOOTAGE / "zones" / f"{clip}.json"
    if not zp.exists():
        return None
    return Homography.from_calibration(json.loads(zp.read_text()).get("floor_calibration"))


def sample_and_bridge(clip: str, result, frame_size, stems, fps, n_frames, bridge: bool = True):
    """sample_seconds + bridge_prompt: ({stem: records}, bridges)."""
    sampled = sample_seconds(result, frame_size, stems, fps, n_frames)
    if not bridge:
        return sampled, []
    return bridge_prompt(sampled, result.prompt, max_gap_s=BRIDGE_MAX_GAP_S, frame_size=frame_size,
                         H=homography(clip))


def _frame_stems_and_size(clip: str) -> tuple[list[str], tuple[int, int]]:
    """1 fps stems and frame size: from frames/ if present, else from the image-mode JSON."""
    d = FOOTAGE / "frames" / clip
    files = sorted(d.glob("[0-9][0-9][0-9].jpg"))
    if files:
        from PIL import Image
        with Image.open(files[0]) as im:
            return [f.stem for f in files], im.size
    f = DETS_DIR / f"{clip}.json"
    cj = json.loads((f if f.exists() else DETS_DIR / f"{clip}.video.json").read_text())
    return sorted(cj["frames"]), tuple(cj["frame_size"])


def tracks_summary(frames: dict, prompts: list[str]) -> dict:
    """{prompt: {track_id: {"first_s", "last_s", "seconds"}}} over the sampled seconds."""
    out: dict = {}
    for p in prompts:
        tr: dict = {}
        for stem in sorted(frames):
            s = int(stem) - 1
            for d in frames[stem].get(p, []):
                t = tr.setdefault(str(d["track_id"]), {"first_s": s, "last_s": s, "seconds": 0})
                t["last_s"] = s
                t["seconds"] += 1
        out[p] = tr
    return out


def write_compact(path: Path, data: dict) -> None:
    """One line per frame, like detections/<clip>.json (diffable, small)."""
    lines = ["{"]
    for k, v in data.items():
        if k == "frames":
            continue
        lines.append(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},")
    lines.append('  "frames": {')
    items = list(data["frames"].items())
    for i, (k, v) in enumerate(items):
        lines.append(f"    {json.dumps(k)}: {json.dumps(v)}" + ("," if i < len(items) - 1 else ""))
    lines += ["  }", "}"]
    path.write_text("\n".join(lines) + "\n")


def export_video_detections(clip: str, prompts: list[str], client: Optional[SamClient] = None,
                            workers: int = 4) -> tuple[Path, SamClient]:
    """Write/merge test-footage/detections/<clip>.video.json. Returns (path, client) (client.calls = paid)."""
    client = client or SamClient(**SAM_OPTIONS)
    results = segment_clip(clip, prompts, client, workers)
    vp = video_path(clip)
    info = probe(vp)
    stems, size = _frame_stems_and_size(clip)
    path = DETS_DIR / f"{clip}.video.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    frames = data.get("frames", {})
    requests = data.get("requests", {})
    bridges = [b for b in data.get("bridges", []) if b["prompt"] not in results]
    for p, r in results.items():
        recs, br = sample_and_bridge(clip, r, size, stems, info["fps"], info["frames"])
        bridges += br
        for stem, dets in recs.items():
            frames.setdefault(stem, {})[p] = dets
        requests[p] = {"frames_in_response": len(r.text.splitlines()), "elapsed_s": r.elapsed_s,
                       "upload_s": r.meta.get("upload_s"), "file_id": r.meta.get("file_id")}
    all_prompts = sorted(set(data.get("prompts", [])) | set(prompts))
    frames = {k: dict(sorted(v.items())) for k, v in sorted(frames.items())}
    out = {
        "clip": clip, "model": "sam-3.1", "mode": "video", "fps": 1, "frame_size": list(size),
        "video": {"file": str(vp.relative_to(REPO)), "width": info["width"], "height": info["height"],
                  "fps": info["fps"], "frames": info["frames"]},
        "source": "Meta Model API, video mode: one input_video request per clip and prompt (file upload, "
                  "long side 1280 px, all frames), mask_encoding=one_bit, include_confidence=true; masks dropped",
        "frame_index": "1 fps frame NNN (second s = NNN - 1) = video frame ceil((s + 0.5) * fps) - 1",
        "box_units": "pixels of frames/<clip>/NNN.jpg, [x1, y1, x2, y2], x2/y2 exclusive",
        "track_id": "SAM 3.1 object id, stable across the clip for one prompt (ids of different prompts are unrelated); "
                    "a new id that starts within 1 s after an old one stopped, on an overlapping box (IoU >= 0.4) or "
                    "within 0.3 m on the floor, takes the old id (detect/tracks.py); sam_track_id = the SAM id then",
        "updated": time.strftime("%Y-%m-%d"),
        "prompts": all_prompts,
        "requests": dict(sorted(requests.items())),
        "frame_map": {s: second_to_frame(int(s) - 1, info["fps"], info["frames"]) for s in stems},
        "bridges": sorted(bridges, key=lambda b: (b["prompt"], b["second"], b["to"])),
        "tracks": tracks_summary(frames, all_prompts),
        "frames": frames,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_compact(path, out)
    return path, client


def main(argv=None) -> int:
    from ..pipeline import CLIPS, DEFAULT_PROMPTS, resolve_clip
    ap = argparse.ArgumentParser(prog="python -m safety_mode.detect.sam_video")
    ap.add_argument("--clip", action="append", help="clip name or suffix; default all three")
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    ap.add_argument("--offline", action="store_true", help="cache only, never call the API")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    clips = [resolve_clip(c) for c in a.clip] if a.clip else CLIPS
    client = SamClient(offline=a.offline, **SAM_OPTIONS)
    frames_billed = 0
    for clip in clips:
        before = client.calls
        path, _ = export_video_detections(clip, a.prompts, client, a.workers)
        n = probe(video_path(clip))["frames"]
        paid = client.calls - before
        frames_billed += paid * billed_frames(n)
        tr = json.loads(path.read_text())["tracks"]
        ids = ", ".join(f"{p}: {len(tr.get(p, {}))}" for p in a.prompts)
        print(f"{clip}: {n} frames, {paid} paid requests, track ids per prompt {ids} -> {path.relative_to(REPO)}")
        for b in json.loads(path.read_text())["bridges"]:
            print(f"   bridge {b['prompt']} #{b['from']} -> #{b['to']} at s{b['second']} (last seen s{b['last_seen_s']}, "
                  f"IoU {b['iou']}, foot {b['foot_m']} m)")
    print(f"paid requests {client.calls}, frames billed {frames_billed}, "
          f"~${frames_billed * PRICE_PER_1000_FRAMES / 1000:.2f} at ${PRICE_PER_1000_FRAMES}/1000 frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

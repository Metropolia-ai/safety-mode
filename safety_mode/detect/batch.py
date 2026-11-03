"""Run SAM 3.1 on a set of test frames for several prompts, write overlays and a count table.

python -m safety_mode.detect.batch [--frames 1,5,9,13,17,21] [--workers 6]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .overlay import draw, save_jpeg
from .sam_api import REPO, SamClient

CLIPS = ["pexels-6079421-forklift-busy-warehouse", "pexels-15170997-warehouse-team",
         "pexels-32243653-forklift-operator"]
PROMPTS = ["person", "forklift", "hard hat", "safety vest"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="1,5,9,13,17,21")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(REPO / "test-footage" / "detections"))
    ap.add_argument("--min-conf", type=float, default=0.5)
    a = ap.parse_args(argv)
    frames_dir = REPO / "test-footage" / "frames"
    jobs = []
    for clip in CLIPS:
        for n in (int(x) for x in a.frames.split(",")):
            f = frames_dir / clip / f"{n:03d}.jpg"
            if f.exists():
                jobs += [(clip, f, p) for p in PROMPTS]
    cl = SamClient(mask_encoding="one_bit", include_confidence=True)

    def run(job):
        clip, f, p = job
        r = cl.segment_image(f, p)
        ds = r.detections()
        name = f"{clip}_{f.stem}_{p.replace(' ', '-')}.jpg"
        title = f"{clip.split('-', 2)[-1]} {f.stem} | {p}: {len(ds)}"
        size = save_jpeg(draw(f, ds, title=title), Path(a.out) / name)
        return {"clip": clip, "frame": f.stem, "prompt": p, "n": len(ds),
                "n_conf": sum(1 for d in ds if (d.confidence or 0) >= a.min_conf),
                "conf": [round(d.confidence, 2) for d in ds if d.confidence is not None],
                "boxes": [[round(v) for v in d.box] for d in ds],
                "elapsed_s": r.elapsed_s, "cached": r.cached, "overlay_kb": size // 1024}

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run, jobs))
    out = Path(a.out) / "summary.json"
    out.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(json.dumps({k: r[k] for k in ("clip", "frame", "prompt", "n", "n_conf", "conf", "elapsed_s", "cached")}))
    print(f"jobs {len(jobs)}, paid calls {cl.calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

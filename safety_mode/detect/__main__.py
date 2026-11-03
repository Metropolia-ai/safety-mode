"""python -m safety_mode.detect <image> --prompt person [--prompt forklift] [--overlay out.jpg]"""
from __future__ import annotations

import argparse
import json
import sys

from .overlay import draw, save_jpeg
from .sam_api import SamClient


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m safety_mode.detect",
                                 description="SAM 3.1 (Meta Model API) on one image, cached on disk.")
    ap.add_argument("image")
    ap.add_argument("--prompt", action="append", required=True,
                    help="one noun phrase per request; repeat for several objects")
    ap.add_argument("--overlay", help="write an overlay JPEG here")
    ap.add_argument("--mask-encoding", choices=["lossless", "one_bit"], default=None)
    ap.add_argument("--confidence", action="store_true", help="metadata.include_confidence=true")
    ap.add_argument("--offline", action="store_true", help="cache only, never call the API")
    ap.add_argument("--json", action="store_true", help="print detections as JSON lines")
    a = ap.parse_args(argv)
    try:
        cl = SamClient(mask_encoding=a.mask_encoding, include_confidence=a.confidence, offline=a.offline)
        dets = []
        for p in a.prompt:
            r = cl.segment_image(a.image, p)
            ds = r.detections()
            dets += ds
            src = "cache" if r.cached else f"API {r.elapsed_s:.2f} s"
            print(f"{p}: {len(ds)} found ({src})", file=sys.stderr)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for d in dets:
        row = {"prompt": d.prompt, "track_id": d.track_id, "box": [round(v, 1) for v in d.box],
               "mask_px": int(d.mask.sum()) if d.mask is not None else None, "confidence": d.confidence}
        print(json.dumps(row) if a.json else
              f"{d.prompt:22s} #{d.track_id:<3d} box={row['box']} mask_px={row['mask_px']}")
    if a.overlay:
        n = save_jpeg(draw(a.image, dets), a.overlay)
        print(f"overlay: {a.overlay} ({n // 1024} KB)", file=sys.stderr)
    print(f"paid calls this run: {cl.calls}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

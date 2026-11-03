#!/usr/bin/env python3
"""Contact sheets (sheets/<clip>.jpg, <= 300 KB) and zone overlays (zones/<clip>-overlay.jpg).
Each tile = one second: zones (test-footage/zones/draw_zones.draw), SAM 3.1 boxes with prompt #track id,
and a bar with the rule state at that second: grey idle, amber candidate, red ALERT."""
import io, json, sys
from pathlib import Path
from PIL import Image, ImageDraw
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "zones"))
import draw_zones as dz                               # noqa: E402  (draw(), shrink(), save_small(), font())
dz.COLORS.setdefault("fork_zone", (255, 140, 0))
PCOL = {"person": (0, 255, 255), "hard hat": (255, 0, 255), "safety vest": (0, 255, 0), "forklift": (255, 255, 0)}
SCOL = {"idle": (90, 90, 90), "candidate": (255, 170, 0), "ALERT": (230, 30, 30)}


def tile(clip, stem, zones, dets, states, width=480):
    im = dz.draw(zones, Image.open(HERE / "frames" / clip / f"{stem}.jpg"))
    d = ImageDraw.Draw(im)
    f = dz.font(22)
    for p, recs in dets.get(stem, {}).items():
        for r in recs:
            x1, y1, x2, y2 = r["box"]
            d.rectangle([x1, y1, x2, y2], outline=PCOL.get(p, (255, 255, 255)), width=4)
            d.text((x1 + 3, max(0, y1 - 24)), f"{p} #{r.get('track_id')}", font=f, fill=PCOL.get(p, (255, 255, 255)),
                   stroke_width=2, stroke_fill=(0, 0, 0))
    t = dz.shrink(im, width)
    s = int(stem) - 1
    bar = Image.new("RGB", (t.width, 34), (0, 0, 0))
    bd = ImageDraw.Draw(bar)
    x = 0
    w = t.width // max(1, len(states))
    for rid, st in states.items():
        bd.rectangle([x, 0, x + w, 34], fill=SCOL[st[s]])
        bd.text((x + 6, 4), f"s{s} {rid} {st[s]}", font=dz.font(20), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        x += w
    out = Image.new("RGB", (t.width, t.height + 34))
    out.paste(t, (0, 0)); out.paste(bar, (0, t.height))
    return out


def main():
    tl = json.loads((HERE / "timelines.json").read_text())
    (HERE / "sheets").mkdir(exist_ok=True)
    for clip, v in tl.items():
        zones = json.loads((HERE / "zones" / f"{clip}.json").read_text())
        ref = HERE.parent / zones["reference_frame"]
        n, q = dz.save_small(dz.shrink(dz.draw(zones, Image.open(ref))), HERE / "zones" / f"{clip}-overlay.jpg")
        states = {rid: r["states"] for rid, r in v["rules"].items()}
        stems = sorted(v["detections"])
        tiles = [tile(clip, s, zones, v["detections"], states) for s in stems]
        cols = 6
        rows = (len(tiles) + cols - 1) // cols
        tw, th = tiles[0].size
        sh = Image.new("RGB", (cols * tw, rows * th), "white")
        for i, t in enumerate(tiles):
            sh.paste(t, ((i % cols) * tw, (i // cols) * th))
        side = 1800 if rows <= 4 else 1300
        n2, q2 = dz.save_small(dz.shrink(sh, side), HERE / "sheets" / f"{clip}.jpg")
        print(clip, f"overlay {n // 1024} KB, sheet {n2 // 1024} KB q={q2}")


if __name__ == "__main__":
    main()

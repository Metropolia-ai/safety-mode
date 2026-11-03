#!/usr/bin/env python3
"""Draws the zones and floor calibration points from zones/<clip>.json over a frame (PIL only).

By default writes zones/<clip>-overlay.jpg for each zones/*.json: the reference frame,
semi-transparent labelled polygons, four calibration points with metres.
The long side is scaled down to 1280 px, JPEG quality is chosen so the file is < 300 KB.

  python3 draw_zones.py                      # overlays for all clips
  python3 draw_zones.py --frame 013          # the same drawing on another frame (handheld camera: drift check)
  python3 draw_zones.py --sheet /tmp/sheets  # contact sheets of all frames with zones (for labelling expected.md)
"""
import argparse
import glob
import io
import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FOOTAGE = os.path.dirname(HERE)
MAX_SIDE = 1280
MAX_BYTES = 300 * 1024

COLORS = {  # RGB by zone name; the rest get grey
    "past_gate": (60, 130, 255),
    "fire_exit_zone": (235, 50, 50),
    "pedestrian_lane": (40, 200, 90),
    "yellow_zone": (255, 215, 0),
    "rack_zone": (200, 80, 230),
}


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
              "C:/Windows/Fonts/arialbd.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def px(pt, w, h):
    return (pt[0] * w, pt[1] * h)


def draw(cfg, img):
    """Returns a copy of img with the zones and the calibration. Coordinates in the JSON are normalised 0..1."""
    img = img.convert("RGB")
    w, h = img.size
    s = max(w, h) / 1280  # scale for line width and font
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    f = font(int(22 * s))
    placed = []  # rectangles of labels already drawn

    def free_spot(x, y, tw, th):
        """Moves a label down while it overlaps labels already placed."""
        while any(x < x2 and x + tw > x1 and y < y2 and y + th > y1 for x1, y1, x2, y2 in placed):
            y += th + 6 * s
        placed.append((x, y, x + tw, y + th))
        return y

    # 1) polygons; identical ones (yellow_zone = pedestrian_lane) are filled once
    filled = set()
    for z in cfg["zones"]:
        c = COLORS.get(z["name"], (160, 160, 160))
        pts = [px(p, w, h) for p in z["polygon"]]
        key = tuple(map(tuple, z["polygon"]))
        if key not in filled:
            d.polygon(pts, fill=c + (40 if z["name"] == "past_gate" else 90,))
            filled.add(key)
        d.line(pts + [pts[0]], fill=c + (255,), width=max(2, int(3 * s)))
    # 2) floor calibration: points and metres
    cal = cfg.get("floor_calibration")
    if cal:
        pts = [px(p, w, h) for p in cal["points_px"]]
        d.line(pts + [pts[0]], fill=(255, 255, 255, 230), width=max(1, int(2 * s)))
        fs = font(int(18 * s))
        r = 7 * s
        for p, m in zip(pts, cal["points_m"]):
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
            t = f"({m[0]:g}, {m[1]:g}) m"
            tb = d.textbbox((0, 0), t, font=fs)
            x, y = p[0] + r + 3, p[1] - r - 2
            placed.append((x, y, x + tb[2], y + tb[3]))
            d.text((x, y), t, font=fs, fill=(255, 255, 255, 255), stroke_width=max(1, int(2 * s)),
                   stroke_fill=(0, 0, 0, 255))
    # 3) zone labels on top of everything: at the polygon centre, below narrow polygons
    for z in cfg["zones"]:
        c = COLORS.get(z["name"], (160, 160, 160))
        pts = [px(p, w, h) for p in z["polygon"]]
        tb = d.textbbox((0, 0), z["name"], font=f)
        tw, th = tb[2] - tb[0] + 8 * s, tb[3] - tb[1] + 10 * s
        cx = sum(p[0] for p in pts) / len(pts)
        ys = [p[1] for p in pts]
        y = sum(ys) / len(ys) - th / 2
        if max(ys) - min(ys) < 3 * th:
            y = max(ys) + 6 * s
        x = min(max(cx - tw / 2, 4), w - tw - 4)
        y = free_spot(x, min(max(y, 4), h - th - 4), tw, th)
        d.rectangle([x, y, x + tw, y + th], fill=(0, 0, 0, 170))
        d.text((x + 4 * s, y + 2 * s), z["name"], font=f, fill=c + (255,))
    return Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")


def shrink(img, side=MAX_SIDE):
    w, h = img.size
    k = side / max(w, h)
    return img.resize((round(w * k), round(h * k)), Image.LANCZOS) if k < 1 else img


def save_small(img, path):
    for q in (88, 82, 76, 70, 62, 55):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q, optimize=True)
        if buf.tell() <= MAX_BYTES:
            break
    with open(path, "wb") as fh:
        fh.write(buf.getvalue())
    return buf.tell(), q


def sheet(cfg, out_dir):
    """Contact sheet: all frames of the clip with zones, label = second (NNN.jpg - 1)."""
    frames = sorted(glob.glob(os.path.join(FOOTAGE, "frames", cfg["clip"], "*.jpg")))
    tiles = []
    for fp in frames:
        t = shrink(draw(cfg, Image.open(fp)), 480)
        ImageDraw.Draw(t).text((6, 4), f"s{int(os.path.basename(fp)[:3]) - 1}", font=font(26), fill=(255, 255, 0),
                               stroke_width=2, stroke_fill=(0, 0, 0))
        tiles.append(t)
    tw, th = tiles[0].size
    cols = 4 if tw > th else 6
    per = cols * (3 if tw > th else 2)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(0, len(tiles), per):
        part = tiles[i:i + per]
        rows = (len(part) + cols - 1) // cols
        sh = Image.new("RGB", (cols * tw, rows * th), "white")
        for j, t in enumerate(part):
            sh.paste(t, ((j % cols) * tw, (j // cols) * th))
        p = os.path.join(out_dir, f"{cfg['clip']}-{i // per}.jpg")
        sh.save(p, quality=80)
        print(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", help="frame number NNN instead of the reference frame (output: <clip>-overlay-NNN.jpg)")
    ap.add_argument("--sheet", metavar="DIR", help="write contact sheets of all frames to DIR")
    a = ap.parse_args()
    for jp in sorted(glob.glob(os.path.join(HERE, "*.json"))):
        cfg = json.load(open(jp, encoding="utf-8"))
        if a.sheet:
            sheet(cfg, a.sheet)
            continue
        ref = cfg["reference_frame"]
        suffix = ""
        if a.frame:
            ref = os.path.join("frames", cfg["clip"], f"{int(a.frame):03d}.jpg")
            suffix = f"-{int(a.frame):03d}"
        if not os.path.exists(os.path.join(FOOTAGE, ref)):
            print(f"no frame {ref}, skipping")
            continue
        img = Image.open(os.path.join(FOOTAGE, ref))
        assert list(img.size) == cfg["frame_size"], (jp, img.size)
        out = os.path.join(HERE, f"{cfg['clip']}-overlay{suffix}.jpg")
        n, q = save_small(shrink(draw(cfg, img)), out)
        print(f"{out}  {n // 1024} KB  q={q}")


if __name__ == "__main__":
    main()

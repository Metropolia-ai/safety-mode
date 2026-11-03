"""Per-second verification frames for the example rules (what the engine saw and decided).

    python test-footage/verify/render_verify.py [--only r03]

Runs every example rule on its clips through the page's own /api/run (FastAPI TestClient, video mode,
one episode per rule, no Spark: explanations fall back to the template), then draws on each 1 fps frame:
the rule's zones (filled, plain title), every detection box with label and track id, foot points,
for distance rules the line between the nearest points of the forklift's and the person's bottom edges
with the metres, for PPE rules the person box tinted red (PPE missing) or green (all worn), and a header
with the engine state of that second and the measured values. Writes
test-footage/verify/<clip>_<rule>/NNN.jpg, sheet.jpg (6 columns) and seconds.json.
Frames are not in git (test-footage/frames/), so this needs the local frames.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.pop("SM_LLM_API_KEY", None)
os.environ.pop("MODEL_API_KEY", None)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from safety_mode.webapp import CLIPS, app, zones_for  # noqa: E402

RUNS = [  # (rule label, rule file, clips)
    ("r02", "rules/r02.json", ["forklift-operator", "warehouse-team", "busy-warehouse"]),
    ("r02-vest", "test-footage/rules/r02-vest.json", ["forklift-operator", "warehouse-team", "busy-warehouse"]),
    ("r03", "rules/r03.json", ["warehouse-team", "busy-warehouse"]),
    ("r01-forklift", "test-footage/rules/r01-forklift.json", ["warehouse-team", "busy-warehouse"]),
    ("r11", "rules/r11.json", ["busy-warehouse"]),
]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
ZCOL = {"past_gate": (47, 111, 222), "fire_exit_zone": (198, 40, 40), "pedestrian_lane": (31, 154, 87),
        "yellow_zone": (230, 190, 0), "rack_zone": (140, 80, 200)}
OCOL = {"forklift": (255, 140, 0), "person": (240, 240, 240)}
PPECOL = {"hard_hat": (255, 230, 0), "vest": (0, 220, 255), "safety_vest": (0, 220, 255)}
RED, GREEN = (229, 57, 53), (67, 160, 71)


def font(sz, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, sz)


def label(draw, xy, text, f, fg=(255, 255, 255), bg=(0, 0, 0)):
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), text, font=f)
    draw.rectangle([l - 3, t - 2, r + 3, b + 2], fill=bg)
    draw.text((x, y), text, font=f, fill=fg)


def fmt_state(st):
    if st["phase"] == "alert":
        return f"ALERT (condition since s{int(st['since_s'])}, raised s{int(st['start_s'])})"
    if st["phase"] == "candidate":
        return f"candidate {st['held_s']:g} s of {st['min_s']:g} s"
    return "idle"


def oid(o):
    return f"{o['name'].replace('_', ' ')} #{o['id']}" if o.get("id") is not None else f"{o['name']} [{o['index']}]"


def measured_lines(rule, sec, ztitle):
    d = sec["diag"]
    out = []
    for o in d["objects"]:
        bits = []
        if "zones" in o:
            zs = [ztitle.get(z, z) for z in rule["zones"]]
            bits.append(("in " + ", ".join(ztitle.get(z, z) for z in o["zones"])) if o["zones"]
                        else "outside " + ", ".join(zs))
        if "ppe" in o:
            bits.append(", ".join(f"{k.replace('_', ' ')} {v['share']:.2f} {'ok' if v['worn'] else 'NO'}"
                                  for k, v in o["ppe"].items()))
        if bits:
            out.append(f"{oid(o)}: " + "; ".join(bits))
    for p in d["pairs"][:4]:
        if p["distance_m"] is None:
            continue
        tag = "operator, skipped" if p["operator"] else ("< %g m" % p["m"] if p["lt"] else ">= %g m" % p["m"])
        out.append(f"{oid(p['a'])} - {oid(p['b'])}: {p['distance_m']:.2f} m ({tag})")
    return out


def render(clip, rlabel, rule, run, outdir: Path):
    zf = zones_for(clip)
    ztitle = {z["name"]: z["title"] for z in zf["zones"]}
    frames_dir = REPO / "test-footage" / "frames" / CLIPS[clip]["file"]
    outdir.mkdir(parents=True, exist_ok=True)
    is_ppe = "wearing" in json.dumps(rule["condition"])
    is_dist = "distance_lt" in json.dumps(rule["condition"])
    thumbs = []
    rows = []
    for sec in run["seconds"]:
        t = int(sec["t"])
        im = Image.open(frames_dir / f"{t + 1:03d}.jpg").convert("RGB")
        W0, H0 = im.size
        scale = 960 / max(W0, H0) if W0 > H0 else 640 / W0
        im = im.resize((int(W0 * scale), int(H0 * scale)), Image.LANCZOS)
        W, H = im.size
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        f_s, f_m = font(max(12, W // 60)), font(max(14, W // 48), True)
        P = lambda pt: (pt[0] * W, pt[1] * H)  # noqa: E731
        for z in zf["zones"]:
            if z["name"] not in rule.get("zones", []):
                continue
            c = ZCOL.get(z["name"], (120, 120, 120))
            od.polygon([P(p) for p in z["polygon"]], fill=c + (70,), outline=c + (255,), width=3)
        hits = {(h["name"], h["id"]) for h in sec["hits"]}
        diag = sec["diag"]
        dets = sec["dets"]
        # PPE boxes and other detections
        for dd in dets:
            if dd["name"] in ("person", "forklift") or dd["name"] in [o["name"] for o in diag["objects"]]:
                continue
            c = PPECOL.get(dd["name"], (200, 200, 200))
            od.rectangle([*P(dd["box"][:2]), *P(dd["box"][2:])], outline=c + (255,), width=2)
        for o in diag["objects"]:
            x1, y1, x2, y2 = (*P(o["box"][:2]), *P(o["box"][2:]))
            c = OCOL.get(o["name"], (200, 200, 200))
            if "ppe" in o:
                c = GREEN if o["all_worn"] else RED
                od.rectangle([x1, y1, x2, y2], fill=c + (60,))
            od.rectangle([x1, y1, x2, y2], outline=c + (255,), width=4 if (o["name"], o["id"]) in hits else 2)
        im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
        dr = ImageDraw.Draw(im)
        for dd in dets:
            if dd["name"] in [o["name"] for o in diag["objects"]]:
                continue
            c = PPECOL.get(dd["name"], (200, 200, 200))
            label(dr, (dd["box"][0] * W, max(0, dd["box"][1] * H - 16)),
                  f"{dd['name'].replace('_', ' ')}{' #%s' % dd['id'] if dd.get('id') is not None else ''} {dd['score'] or 0:.2f}",
                  f_s, fg=(0, 0, 0), bg=c)
        for z in zf["zones"]:
            if z["name"] in rule.get("zones", []):
                c = ZCOL.get(z["name"], (120, 120, 120))
                xs = [p[0] for p in z["polygon"]]
                ys = [p[1] for p in z["polygon"]]
                # label inside the polygon, near its top
                ty = min(ys) + 0.02
                tw_ = dr.textlength(z["title"], font=f_m)
                label(dr, (min(max(0, (min(xs) + max(xs)) / 2 * W - tw_ / 2), W - tw_ - 6), ty * H), z["title"], f_m, bg=c)
        # distance lines (non-operator pairs, nearest first)
        if is_dist:
            shown = 0
            for p in diag["pairs"]:
                if p["operator"] or p["p1"] is None or shown >= 3:
                    continue
                col = RED if p["lt"] else (180, 180, 180)
                a, b = P(p["p1"]), P(p["p2"])
                dr.line([a, b], fill=col, width=5 if shown == 0 else 2)
                mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                label(dr, (mx + 6, my - 10), f"{p['distance_m']:.2f} m", f_m if shown == 0 else f_s, bg=col if p["lt"] else (60, 60, 60))
                shown += 1
        for o in diag["objects"]:
            x1, y1 = o["box"][0] * W, o["box"][1] * H
            c = OCOL.get(o["name"], (200, 200, 200))
            if "ppe" in o:
                c = GREEN if o["all_worn"] else RED
            txt = oid(o)
            if "ppe" in o:
                txt += "  " + " ".join(f"{k.replace('safety_', '').replace('_', ' ')} {'ok' if v['worn'] else 'NO'}"
                                       for k, v in o["ppe"].items())
            label(dr, (x1, max(0, y1 - 18)), txt, f_s, fg=(0, 0, 0) if c in (OCOL["person"], OCOL["forklift"]) else (255, 255, 255), bg=c)
            fx, fy = P(o["foot"])
            inz = bool(o.get("zones"))
            r = 7
            dr.ellipse([fx - r, fy - r, fx + r, fy + r], fill=(0, 230, 118) if inz else (255, 255, 255),
                       outline=(0, 0, 0), width=2)
        # header
        st = sec["state"]
        phase = st["phase"]
        hdr_col = {"alert": (200, 120, 0), "candidate": (90, 90, 140), "idle": (40, 40, 40)}[phase]
        lines = [f"{clip} | {rlabel} | s{t:02d}",
                 f"engine: {fmt_state(st)} | condition {'TRUE' if sec['truth'] == 'T' else ('not evaluable' if sec['truth'] == '?' else 'false')}"
                 + (" on " + ", ".join(oid(h) for h in sec["hits"]) if sec["hits"] else "")]
        lines += measured_lines(rule, sec, ztitle) or ["(no bound objects detected this second)"]
        wrapped = []
        tmp = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        for i, ln in enumerate(lines):
            f = f_m if i < 2 else f_s
            cur = ""
            for w in ln.split(" "):
                cand = (cur + " " + w).strip()
                if tmp.textlength(cand, font=f) > W - 16 and cur:
                    wrapped.append((cur, f))
                    cur = "   " + w
                else:
                    cur = cand
            wrapped.append((cur, f))
        hdr_h = 10 + sum(f.size + 6 for _, f in wrapped)
        canvas = Image.new("RGB", (W, H + hdr_h), hdr_col)
        canvas.paste(im, (0, hdr_h))
        cd = ImageDraw.Draw(canvas)
        y = 5
        for ln, f in wrapped:
            cd.text((8, y), ln, font=f, fill=(255, 255, 255))
            y += f.size + 6
        canvas.save(outdir / f"{t + 1:03d}.jpg", quality=72)
        thumbs.append((t, phase, sec["truth"], canvas))
        rows.append({"s": t, "truth": sec["truth"], "state": st, "measured": measured_lines(rule, sec, ztitle),
                     "hits": sec["hits"]})
    # contact sheet, 6 columns
    tw = 420 if thumbs[0][3].width > thumbs[0][3].height else 300
    th = max(int(c.height * tw / c.width) for *_, c in thumbs)
    cols = 6
    nrow = -(-len(thumbs) // cols)
    cap = 34
    sheet = Image.new("RGB", (cols * tw, nrow * (th + cap)), (20, 20, 20))
    sd = ImageDraw.Draw(sheet)
    fc = font(22, True)
    for i, (t, phase, truth, c) in enumerate(thumbs):
        x, y = (i % cols) * tw, (i // cols) * (th + cap)
        sheet.paste(c.resize((tw, int(c.height * tw / c.width)), Image.LANCZOS), (x, y + cap))
        col = {"alert": (255, 170, 0), "candidate": (160, 160, 255), "idle": (200, 200, 200)}[phase]
        sd.text((x + 6, y + 4), f"s{t:02d} {phase.upper() if phase == 'alert' else phase} {truth}", font=fc, fill=col)
    sheet.save(outdir / "sheet.jpg", quality=80)
    (outdir / "seconds.json").write_text(json.dumps({"clip": clip, "rule": rlabel, "events": [
        {k: e[k] for k in ("start_s", "end_s", "condition_since_s", "subject")} for e in run["events"]],
        "timeline": run["timeline"], "seconds": rows}, ensure_ascii=False, indent=1) + "\n")


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    c = TestClient(app)
    for rlabel, rfile, clips in RUNS:
        if only and rlabel != only:
            continue
        rule = json.loads((REPO / rfile).read_text())
        for clip in clips:
            run = c.post("/api/run", json={"clip": clip, "rule": rule, "episodes": "rule"}).json()
            if run.get("rejected"):
                print(clip, rlabel, "rejected:", run["rejected"])
                continue
            render(clip, rlabel, rule, run, HERE / f"{clip}_{rlabel}")
            print(f"{clip:18s} {rlabel:13s} {run['timeline']}  events="
                  f"{[(e['start_s'], e['end_s'], e['subject']) for e in run['events']]} ne={run.get('not_evaluable')}")


if __name__ == "__main__":
    main()

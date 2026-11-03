"""python -m safety_mode.run --clip busy-warehouse --rules r02 r03 [--machine forklift=running:0-8]

Runs the whole chain on a test clip (frames -> SAM 3.1 -> engine) and prints the events.
--export-detections writes test-footage/detections/<clip>.json (the offline cache) instead.
"""
from __future__ import annotations

import argparse
import sys

from .pipeline import (CLIPS, DEFAULT_PROMPTS, DetectionSource, export_detections, resolve_clip,
                       run_clip)


def _objects(ev) -> str:
    counts: dict = {}
    for o in ev.objects:
        counts[o["name"]] = counts.get(o["name"], 0) + 1
    return ", ".join(f"{k}×{v}" if v > 1 else k for k, v in counts.items())


def _measurements(ev) -> str:
    m = ev.measurements
    out = []
    if m.get("missing_ppe_seen"):
        out.append("missing " + "/".join(m["missing_ppe_seen"]))
    if "min_distance_m" in m:
        out.append(f"min {m['min_distance_m']} m")
    if m.get("max_person_count"):
        out.append(f"persons ≤{m['max_person_count']}")
    if "max_count" in m:
        out.append(f"count ≤{m['max_count']}")
    out.append(f"{m.get('duration_s', '?')} s")
    return "; ".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m safety_mode.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--clip", required=True, help=f"clip name or a unique part of it ({', '.join(CLIPS)}); "
                                                   "'all' with --export-detections")
    ap.add_argument("--rules", nargs="+", default=[], help="r02, r2-vest, r01-forklift or a path")
    ap.add_argument("--zones", help="zones file (default test-footage/zones/<clip>.json)")
    ap.add_argument("--machine", action="append", default=[],
                    help="NAME=STATE[:a-b][,STATE:a-b] e.g. forklift=running:0-8 (opposite state elsewhere)")
    ap.add_argument("--source", choices=["auto", "offline", "api"], default="auto",
                    help="auto: committed detections JSON, then SAM cache/API; offline: JSON only")
    ap.add_argument("--max-gap", type=float, default=1.0, help="not-true seconds an episode tolerates")
    ap.add_argument("--timeline", action="store_true", help="print T/./? per second for each rule")
    ap.add_argument("--export-detections", action="store_true")
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS, help="for --export-detections")
    a = ap.parse_args(argv)
    src = DetectionSource(a.source)
    try:
        if a.export_detections:
            for clip in (CLIPS if a.clip == "all" else [resolve_clip(a.clip)]):
                p = export_detections(clip, a.prompts, src)
                print(f"{p}")
            print(f"paid calls this run: {src.calls}", file=sys.stderr)
            return 0
        if not a.rules:
            ap.error("--rules is required")
        machines = {}
        for m in a.machine:
            name, _, spec = m.partition("=")
            machines[name] = spec
        run = run_clip(a.clip, a.rules, a.zones, machines, source=src, max_gap_s=a.max_gap)
    except (ValueError, LookupError, FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"{run.clip}: {len(next(iter(run.timeline.values()), ''))} frames, frame {run.frame_size[0]}×{run.frame_size[1]}")
    for rid, why in run.rejected.items():
        print(f"  {rid}: rejected: {why}")
    for rid, why in run.not_evaluable.items():
        print(f"  {rid}: not evaluable: {why}")
    rows = [("rule", "start_s", "end_s", "since_s", "objects", "measurements")]
    for ev in sorted(run.events, key=lambda e: (e.rule_id, e.start_s)):
        rows.append((ev.rule_id, f"{ev.start_s:g}", f"{ev.end_s:g}", f"{ev.condition_since_s:g}",
                     _objects(ev), _measurements(ev)))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip())
    quiet = [rid for rid in run.timeline if not any(e.rule_id == rid for e in run.events)]
    for rid in quiet:
        print(f"{rid}: no events")
    if a.timeline:
        for rid, tl in run.timeline.items():
            print(f"{rid:12s} {tl}")
    print(f"paid calls this run: {run.calls}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Per-rule state machine idle -> candidate -> active -> closed.

Timing follows test-footage/expected.md: every frame "costs" frame_period_s
(1 s at 1 fps), so a condition true on k consecutive frames lasts k periods,
and the event is raised on the frame where that first reaches min_duration_s
(3 s rule true from s0 -> event start s2). One event per episode. The episode
ends when the condition has been not-true for longer than max_gap_s
(Engine default 1 s: one missing frame is tolerated; 0 = the first false frame closes it).
The gap applies to raised events only: a candidate needs min_duration_s of consecutive
true frames, any not-true frame resets it; end_s is the last true frame.
Not evaluable counts as not true.

Tracked objects (detections with track_id, SAM 3.1 video mode): the episode belongs to the
objects that make the rule true, e.g. person #2 in r02 or the pair forklift #0 / person #3
in r03 (ConditionResult.by_track). Each such object set runs its own idle -> candidate ->
active -> closed machine, so two people without a vest give two events and the duration is
per person. A person who leaves and comes back keeps the episode only if they were gone
for no longer than max_gap_s, as for a rule without ids. Without track ids there is one
machine per rule (key None), as before. With episodes="rule" and track ids the single
machine's event still names the tracked objects that were true when it was raised.
"""
from __future__ import annotations

import copy
from typing import Optional

from .types import ConditionResult, Event, Truth, track_label

EPS = 1e-6


def _fmt_time(t_s: float) -> str:
    t = int(round(t_s))
    return f"{t // 60:02d}:{t % 60:02d}"


def _person_like(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("person", "worker", "pedestrian", "operator", "people", "human"))


def explanation_inputs(event: Event, frame_period_s: float) -> dict:
    m = event.measurements
    counts = m.get("object_counts", {})
    duration = event.end_s - event.condition_since_s + frame_period_s
    objects = sorted({o["name"] for o in event.objects})
    out = {"subject": event.subject} if event.subject else {}
    return out | {
        "time": _fmt_time(event.condition_since_s),
        "duration_s": round(duration, 2),
        "person_count": m.get("max_person_count", sum(v for k, v in counts.items() if _person_like(k))),
        "missing_ppe": ", ".join(p.replace("_", " ") for p in m.get("missing_ppe", [])),
        "distance_m": m.get("min_distance_m", m.get("distance_m")),
        "count": m.get("max_count", m.get("count")),
        "height_m": m.get("height_m"),
        "object": ", ".join(o.replace("_", " ") for o in (m.get("objects_seen") or objects)),
    }


class _Track:
    """One episode machine: for the whole rule (key None) or for one tracked object set."""
    IDLE, CANDIDATE, ACTIVE = "idle", "candidate", "active"

    def __init__(self, rule: dict, frame_period_s: float = 1.0, max_gap_s: float = 0.0, key=None):
        self.key = key
        self.rule = rule
        self.rule_id = rule["rule_id"]
        self.min_duration_s = float(rule.get("min_duration_s") or 0)
        self.severity = rule.get("severity", "medium")
        self.frame_period_s = frame_period_s
        self.max_gap_s = max_gap_s
        self.phase = self.IDLE
        self.since_s: Optional[float] = None
        self.last_true_s: Optional[float] = None
        self.event: Optional[Event] = None
        self.closed: list[Event] = []
        self.last_result: Optional[ConditionResult] = None

    def _duration(self, t_s: float) -> float:
        return t_s - self.since_s + self.frame_period_s

    def _absorb(self, ev: Event, res: ConditionResult) -> None:
        ev.objects = [o.as_dict() for o in res.objects]
        m = ev.measurements
        for k, v in res.measurements.items():
            m[k] = v
        if "distance_m" in res.measurements:
            d = res.measurements["distance_m"]
            m["min_distance_m"] = d if "min_distance_m" not in m else min(m["min_distance_m"], d)
        if "count" in res.measurements:
            m["max_count"] = max(m.get("max_count", 0), res.measurements["count"])
        if "missing_ppe" in res.measurements:
            seen = m.setdefault("missing_ppe_seen", [])
            seen += [p for p in res.measurements["missing_ppe"] if p not in seen]
        persons = sum(v for k, v in res.measurements.get("object_counts", {}).items() if _person_like(k))
        m["max_person_count"] = max(m.get("max_person_count", 0), persons)
        objs_seen = m.setdefault("objects_seen", [])
        objs_seen += [o.name for o in res.objects if o.name not in objs_seen and not _person_like(o.name)]
        m["duration_s"] = round(self._duration(ev.end_s), 2)
        ev.explanation_inputs = explanation_inputs(ev, self.frame_period_s)

    def _close(self) -> Event:
        ev = self.event
        ev.status = "closed"
        ev.end_s = self.last_true_s
        self.closed.append(ev)
        self.event = None
        self.phase = self.IDLE
        self.since_s = None
        return ev

    def _step(self, t_s: float, res: ConditionResult, next_event_id) -> list[Event]:
        self.last_result = res
        out: list[Event] = []
        if res.truth is Truth.TRUE:
            if self.phase == self.IDLE:
                self.phase, self.since_s = self.CANDIDATE, t_s
            self.last_true_s = t_s
            if self.phase == self.CANDIDATE and self._duration(t_s) >= self.min_duration_s - EPS:
                self.phase = self.ACTIVE
                self.event = Event(self.rule_id, start_s=t_s, end_s=t_s, objects=[], measurements={},
                                   severity=self.severity, explanation_inputs={}, status="active",
                                   condition_since_s=self.since_s, event_id=next_event_id())
                key = self.key
                if not key and res.by_track:
                    # one episode per rule: name the tracked objects true at activation ("person #1")
                    key = tuple(sorted({o for k in res.by_track for o in k}))
                if key:
                    self.event.object_ids = [{"name": n, "track_id": t} for n, t in key]
                    self.event.subject = track_label(key)
                self._absorb(self.event, res)
                self.event.objects_at_start = copy.deepcopy(self.event.objects)
                out.append(self.event)
            elif self.phase == self.ACTIVE:
                self.event.end_s = t_s
                self._absorb(self.event, res)
                out.append(self.event)
            return out
        # false or not evaluable
        if self.phase == self.IDLE:
            return out
        if self.phase == self.CANDIDATE:
            # the window needs consecutive true frames (expected.md: alert "on the third frame in a row");
            # max_gap_s only keeps an event that is already raised open
            self.phase, self.since_s = self.IDLE, None
            return out
        if t_s - self.last_true_s > self.max_gap_s + EPS:
            out.append(self._close())
        return out

    def finish(self) -> list[Event]:
        if self.phase == self.ACTIVE:
            return [self._close()]
        self.phase, self.since_s = self.IDLE, None
        return []


class RuleState:
    """Episodes of one rule: one _Track per tracked object set, or a single one (key None)."""

    def __init__(self, rule: dict, frame_period_s: float = 1.0, max_gap_s: float = 0.0,
                 per_object: bool = True):
        self.rule = rule
        self.rule_id = rule["rule_id"]
        self.frame_period_s = frame_period_s
        self.max_gap_s = max_gap_s
        self.per_object = per_object
        self.tracks: dict = {}                       # key -> _Track
        self.last_result: Optional[ConditionResult] = None

    def _track(self, key) -> _Track:
        if key not in self.tracks:
            self.tracks[key] = _Track(self.rule, self.frame_period_s, self.max_gap_s, key)
        return self.tracks[key]

    @property
    def closed(self) -> list[Event]:
        return sorted((e for tr in self.tracks.values() for e in tr.closed), key=lambda e: e.event_id)

    @property
    def active(self) -> list[Event]:
        return sorted((tr.event for tr in self.tracks.values() if tr.event), key=lambda e: e.event_id)

    @property
    def event(self) -> Optional[Event]:
        """The active event (the first one if several tracked episodes are active)."""
        act = self.active
        return act[0] if act else None

    @property
    def phase(self) -> str:
        phases = {tr.phase for tr in self.tracks.values()}
        for p in (_Track.ACTIVE, _Track.CANDIDATE):
            if p in phases:
                return p
        return _Track.IDLE

    def step(self, t_s: float, res: ConditionResult, next_event_id) -> list[Event]:
        """Feed one frame; returns snapshots of events opened, updated or closed on it."""
        self.last_result = res
        if res.truth is Truth.TRUE:
            per_key = res.by_track if res.by_track is not None and self.per_object else {None: res}
        else:
            per_key = {}
        out: list[Event] = []
        for key in sorted(set(self.tracks) | set(per_key), key=lambda k: (k is not None, k or ())):
            r = per_key.get(key, res if res.truth is not Truth.TRUE else ConditionResult(Truth.FALSE))
            out += self._track(key)._step(t_s, r, next_event_id)
        # idle machines without an episode carry nothing worth keeping
        self.tracks = {k: tr for k, tr in self.tracks.items() if tr.phase != _Track.IDLE or tr.closed}
        return [copy.deepcopy(e) for e in sorted(out, key=lambda e: e.event_id)]

    def finish(self) -> list[Event]:
        out: list[Event] = []
        for tr in self.tracks.values():
            out += tr.finish()
        return sorted(out, key=lambda e: e.event_id)

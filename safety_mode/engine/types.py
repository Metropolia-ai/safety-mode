"""Data types of the condition engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

Box = Sequence[float]  # [x1, y1, x2, y2]


@dataclass
class Detection:
    """One object found by the detector.

    name: object name from rule.objects[].name (not the SAM prompt).
    box: [x1, y1, x2, y2] in frame pixels (or normalised 0..1 if the engine
         runs with box_units="norm").
    mask: optional 2-D boolean array covering the whole frame (any resolution).
    """

    name: str
    box: Box
    mask: Any = None
    score: Optional[float] = None
    track_id: Optional[int] = None


@dataclass
class Frame:
    """Detections of one frame.

    size: (width, height) in pixels; defaults to frame_size of the zones file.
    machine_states: optional per-frame override {machine object name: "running"|"stopped"}.
    """

    t_s: float
    detections: list[Detection] = field(default_factory=list)
    size: Optional[tuple[int, int]] = None
    machine_states: Optional[dict[str, str]] = None


class Truth(Enum):
    TRUE = "true"
    FALSE = "false"
    NOT_EVALUABLE = "not_evaluable"


@dataclass
class NotEvaluable:
    """Explicit result of a primitive the engine cannot compute (not an exception)."""

    reason: str


@dataclass
class MatchedObject:
    name: str
    index: int
    box: list[float]  # normalised 0..1
    track_id: Optional[int] = None
    score: Optional[float] = None

    def as_dict(self) -> dict:
        return {"name": self.name, "index": self.index, "box": self.box,
                "track_id": self.track_id, "score": self.score}


@dataclass
class ConditionResult:
    truth: Truth
    objects: list[MatchedObject] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)
    reason: Optional[str] = None  # set when truth is NOT_EVALUABLE
    # per tracked object set: {(("person", 2),): ConditionResult} when every detection of the
    # bound object names carries a track_id (SAM 3.1 video mode); None = no ids, one episode per rule
    by_track: Optional[dict] = None

    @property
    def is_true(self) -> bool:
        return self.truth is Truth.TRUE


@dataclass
class Event:
    rule_id: str
    start_s: float               # frame where the duration first reached min_duration_s
    end_s: float                 # last frame with the condition true
    objects: list[dict]
    measurements: dict
    severity: str
    explanation_inputs: dict
    status: str = "active"       # "active" | "closed"
    condition_since_s: float = 0.0  # first frame of the episode (candidate start)
    event_id: int = 0
    # tracked episodes (detections with track_id): the objects the episode is about,
    # [{"name": "person", "track_id": 2}], and the same as words, "person #2"; empty without ids
    object_ids: list = field(default_factory=list)
    subject: str = ""
    # objects (with boxes) at start_s, when the alert went up; `objects` follows the episode to its last
    # true frame. Explanations describe the alert time, so they use these (explain.build_facts).
    objects_at_start: list = field(default_factory=list)


TrackKey = tuple  # ((object name, track_id), ...) sorted


def track_label(key: TrackKey) -> str:
    """(("forklift", 0), ("person", 3)) -> "forklift #0 and person #3"."""
    parts = [f"{n.replace('_', ' ')} #{t}" for n, t in key]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]

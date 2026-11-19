"""Tracker contract: canonical detections in, tracked objects with persistent IDs out.

Zones and events depend only on these types, so the tracking algorithm (IoU today,
ByteTrack/OC-SORT-style adapters later) can change without touching them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sentinela_runtime.vision.inference.base import Detection


@dataclass(frozen=True)
class TrackedObject:
    """A confirmed track and the detection it was matched to most recently."""

    track_id: str
    detection: Detection
    hits: int  # frames in which the track was matched

    @property
    def class_name(self) -> str:
        return self.detection.class_name


@dataclass(frozen=True)
class TrackingResult:
    """Tracker output for one inferred frame.

    ``tracks``   confirmed tracks matched in this frame.
    ``started``  subset of ``tracks`` confirmed for the first time in this frame.
    ``ended``    confirmed tracks dropped in this frame (lost for too long), with their last state.
    """

    tracks: tuple[TrackedObject, ...] = ()
    started: tuple[TrackedObject, ...] = ()
    ended: tuple[TrackedObject, ...] = ()


class Tracker(Protocol):
    def update(self, detections: Sequence[Detection]) -> TrackingResult:
        """Associate one frame's detections with existing tracks."""
        ...

"""Deterministic event rules: tracked objects and zone transitions in, event candidates out.

Deduplication comes from tracking and zone debouncing rather than timers: a track produces
at most one ``object_detected`` (when it is confirmed) and one ``zone_entry`` per visit to a
zone, followed by one ``zone_exit``. Pure functions; no state is kept here.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from sentinela_runtime.events.schemas import EventType
from sentinela_runtime.vision.tracking.base import TrackedObject, TrackingResult
from sentinela_runtime.vision.zones.zone_evaluator import TransitionKind, Zone, ZoneTransition

TRANSITION_EVENT_TYPES = {
    TransitionKind.ENTRY: EventType.ZONE_ENTRY,
    TransitionKind.EXIT: EventType.ZONE_EXIT,
}


@dataclass(frozen=True)
class EventCandidate:
    event_type: EventType
    obj: TrackedObject
    zone: Zone | None = None


def select_events(
    tracking: TrackingResult,
    transitions: Sequence[ZoneTransition],
    detection_event_classes: Collection[str],
) -> tuple[EventCandidate, ...]:
    """New tracks of the configured classes, then zone transitions, in a stable order."""
    detected = (
        EventCandidate(EventType.OBJECT_DETECTED, obj)
        for obj in tracking.started
        if obj.class_name in detection_event_classes
    )
    zoned = (EventCandidate(TRANSITION_EVENT_TYPES[t.kind], t.obj, t.zone) for t in transitions)
    return (*detected, *zoned)

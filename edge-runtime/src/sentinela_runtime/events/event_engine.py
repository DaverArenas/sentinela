"""Event engine: turns tracking results and zone transitions into structured events.

For each event: build the schema-1.0 record with provenance, store evidence (snapshot)
when configured, then publish to every sink (SQLite store first, then the JSONL log).
A failing evidence recorder or sink is logged and never stops the pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from sentinela_runtime.core.config import EventsConfig
from sentinela_runtime.core.interfaces import EventSink, EvidenceRecorder
from sentinela_runtime.core.log import get_logger
from sentinela_runtime.events.rules import EventCandidate, select_events
from sentinela_runtime.events.schemas import (
    BoundingBox,
    Event,
    EvidenceInfo,
    ObjectInfo,
    ProcessingInfo,
    ZoneInfo,
)
from sentinela_runtime.vision.camera.video_source import CapturedFrame
from sentinela_runtime.vision.tracking.base import TrackingResult
from sentinela_runtime.vision.zones.zone_evaluator import ZoneTransition

log = get_logger("event_engine")


@dataclass(frozen=True)
class EventContext:
    """Static provenance attached to every event from one pipeline."""

    node_id: str
    camera_id: str
    model_id: str
    model_version: str
    model_sha256: str | None
    inference_backend: str
    config_hash: str


class EventEngine:
    def __init__(
        self,
        context: EventContext,
        config: EventsConfig,
        sinks: Sequence[EventSink],
        evidence: EvidenceRecorder | None = None,
    ) -> None:
        self._context = context
        self._config = config
        self._sinks = tuple(sinks)
        self._evidence = evidence

    def process(
        self, frame: CapturedFrame, tracking: TrackingResult, transitions: Sequence[ZoneTransition]
    ) -> tuple[Event, ...]:
        """Apply rules to one inferred frame; store evidence, publish and return the events."""
        candidates = select_events(tracking, transitions, self._config.detection_event_classes)
        events = tuple(self._attach_evidence(self._build_event(c, frame), frame) for c in candidates)
        for event in events:
            self._publish(event)
        return events

    def _build_event(self, candidate: EventCandidate, frame: CapturedFrame) -> Event:
        ctx, detection = self._context, candidate.obj.detection
        zone = candidate.zone
        return Event(
            event_type=candidate.event_type,
            timestamp=frame.captured_at,
            node_id=ctx.node_id,
            camera_id=ctx.camera_id,
            object=ObjectInfo(
                class_name=detection.class_name,
                class_id=detection.class_id,
                confidence=detection.confidence,
                bounding_box=BoundingBox(*detection.bbox_xyxy),
                track_id=candidate.obj.track_id,
            ),
            zone=ZoneInfo(zone.zone_id, zone.name) if zone else None,
            processing=ProcessingInfo(
                model_id=ctx.model_id,
                model_version=ctx.model_version,
                model_sha256=ctx.model_sha256,
                inference_backend=ctx.inference_backend,
                config_hash=ctx.config_hash,
                frame_index=frame.index,
            ),
        )

    def _attach_evidence(self, event: Event, frame: CapturedFrame) -> Event:
        if self._evidence is None:
            return event
        try:
            path = self._evidence.capture(event, frame.image)
        except Exception:
            log.exception("Evidence capture failed", extra={"event_id": event.event_id})
            return event
        return replace(event, evidence=EvidenceInfo(path)) if path else event

    def _publish(self, event: Event) -> None:
        """Deliver to every sink; one failing sink never blocks the others or the pipeline."""
        for sink in self._sinks:
            try:
                sink.publish(event)
            except Exception:
                log.exception(
                    "Event sink failed", extra={"sink": type(sink).__name__, "event_id": event.event_id}
                )

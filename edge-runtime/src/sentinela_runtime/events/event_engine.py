"""Event engine: turns canonical detections into structured events and dispatches them to sinks."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from sentinela_runtime.core.config import EventsConfig
from sentinela_runtime.core.interfaces import EventSink
from sentinela_runtime.core.log import get_logger
from sentinela_runtime.events.rules import select_detection_events
from sentinela_runtime.events.schemas import BoundingBox, Event, EventType, ObjectInfo, ProcessingInfo
from sentinela_runtime.vision.inference.base import Detection

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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._context = context
        self._config = config
        self._sinks = tuple(sinks)
        self._clock = clock
        self._last_emitted: Mapping[str, float] = {}

    def process(
        self, detections: Sequence[Detection], frame_index: int, captured_at: datetime
    ) -> tuple[Event, ...]:
        """Apply rules to one frame's detections; publish and return the resulting events."""
        due, self._last_emitted = select_detection_events(
            detections,
            self._config.target_classes,
            self._last_emitted,
            self._clock(),
            self._config.cooldown_s,
        )
        events = tuple(self._build_event(det, frame_index, captured_at) for det in due)
        for event in events:
            self._publish(event)
        return events

    def _build_event(self, detection: Detection, frame_index: int, captured_at: datetime) -> Event:
        ctx = self._context
        return Event(
            event_type=EventType.OBJECT_DETECTED,
            timestamp=captured_at,
            node_id=ctx.node_id,
            camera_id=ctx.camera_id,
            object=ObjectInfo(
                class_name=detection.class_name,
                class_id=detection.class_id,
                confidence=detection.confidence,
                bounding_box=BoundingBox(*detection.bbox_xyxy),
            ),
            processing=ProcessingInfo(
                model_id=ctx.model_id,
                model_version=ctx.model_version,
                model_sha256=ctx.model_sha256,
                inference_backend=ctx.inference_backend,
                config_hash=ctx.config_hash,
                frame_index=frame_index,
            ),
        )

    def _publish(self, event: Event) -> None:
        """Deliver to every sink; one failing sink never blocks the others or the pipeline."""
        for sink in self._sinks:
            try:
                sink.publish(event)
            except Exception:
                log.exception(
                    "Event sink failed", extra={"sink": type(sink).__name__, "event_id": event.event_id}
                )

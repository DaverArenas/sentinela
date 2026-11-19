"""Event schema, deduplication rules and the event engine."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta, timezone

import pytest

from sentinela_runtime.core.config import EventsConfig
from sentinela_runtime.events.event_engine import EventContext, EventEngine
from sentinela_runtime.events.rules import select_detection_events
from sentinela_runtime.events.schemas import (
    SCHEMA_VERSION,
    BoundingBox,
    Event,
    EventType,
    EvidenceInfo,
    ObjectInfo,
    ProcessingInfo,
    ZoneInfo,
    format_timestamp,
)
from sentinela_runtime.events.sinks import LoggingEventSink
from sentinela_runtime.vision.inference.base import Detection

from support import CAR, PERSON

CAPTURED_AT = datetime(2026, 11, 16, 20, 15, 30, 123456, tzinfo=UTC)
CONTEXT = EventContext(
    "edge-node-001", "camera-001", "yolox-s-coco", "0.1.1rc0", "a" * 64, "onnxruntime-cuda", "cfg123"
)


def make_event(**overrides: object) -> Event:
    values: dict[str, object] = dict(
        event_type=EventType.OBJECT_DETECTED,
        timestamp=CAPTURED_AT,
        node_id="edge-node-001",
        camera_id="camera-001",
        object=ObjectInfo("person", 0, 0.94123, BoundingBox(120.04, 80, 340, 460.96)),
        processing=ProcessingInfo("person-detector", "1.0", None, "onnxruntime-cuda", "cfg", 7),
    )
    return Event(**{**values, **overrides})  # type: ignore[arg-type]


# --- schema ---------------------------------------------------------------------------------


def test_event_serialises_to_documented_schema() -> None:
    event = make_event(zone=ZoneInfo("zone-001", "Restricted Area"), evidence=EvidenceInfo("events/x.jpg"))

    payload = json.loads(event.to_json())

    assert payload["schema_version"] == SCHEMA_VERSION == "1.0"
    assert payload["event_id"].startswith("evt-")
    assert payload["event_type"] == "object_detected"
    assert payload["timestamp"] == "2026-11-16T20:15:30.123Z"
    assert payload["object"] == {
        "class": "person",
        "class_id": 0,
        "track_id": None,
        "confidence": 0.9412,
        "bounding_box": {"x1": 120.0, "y1": 80, "x2": 340, "y2": 461.0},
    }
    assert payload["zone"] == {"zone_id": "zone-001", "zone_name": "Restricted Area"}
    assert payload["evidence"] == {"snapshot_path": "events/x.jpg"}
    assert payload["processing"]["frame_index"] == 7


def test_optional_sections_serialise_as_null() -> None:
    payload = make_event().to_dict()

    assert payload["zone"] is None and payload["evidence"] is None


def test_event_ids_are_unique_and_explicit_ids_are_preserved() -> None:
    assert make_event().event_id != make_event().event_id
    assert make_event(event_id="evt-fixed").event_id == "evt-fixed"


def test_timestamps_are_normalised_to_utc_and_must_be_aware() -> None:
    local = datetime(2026, 11, 16, 15, 15, 30, tzinfo=timezone(timedelta(hours=-5)))

    assert format_timestamp(local) == "2026-11-16T20:15:30.000Z"
    with pytest.raises(ValueError, match="timezone-aware"):
        format_timestamp(datetime(2026, 1, 1))


# --- rules -----------------------------------------------------------------------------------


def det(name: str, confidence: float) -> Detection:
    return Detection(0, name, confidence, (0.0, 0.0, 1.0, 1.0))


def test_rules_emit_best_detection_per_target_class() -> None:
    detections = [det("person", 0.6), det("person", 0.95), det("car", 0.99)]

    due, _ = select_detection_events(detections, {"person"}, {}, now=10.0, cooldown_s=5.0)

    assert due == (det("person", 0.95),)


def test_rules_apply_cooldown_without_mutating_state() -> None:
    state = {"person": 8.0}

    due_early, same = select_detection_events(
        [det("person", 0.9)], {"person"}, state, now=10.0, cooldown_s=5.0
    )
    due_late, updated = select_detection_events(
        [det("person", 0.9)], {"person"}, state, now=13.0, cooldown_s=5.0
    )

    assert due_early == () and same == {"person": 8.0}
    assert len(due_late) == 1 and updated == {"person": 13.0}
    assert state == {"person": 8.0}


# --- engine ----------------------------------------------------------------------------------


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)


class BrokenSink:
    def publish(self, event: Event) -> None:
        raise OSError("disk full")


def test_engine_builds_events_with_provenance_and_publishes() -> None:
    sink = RecordingSink()
    engine = EventEngine(CONTEXT, EventsConfig(("person", "car"), 5.0), [sink], clock=lambda: 100.0)

    events = engine.process([PERSON, CAR], frame_index=42, captured_at=CAPTURED_AT)

    assert sink.events == list(events)
    assert {e.object.class_name for e in events} == {"person", "car"}
    first = events[0]
    assert (first.node_id, first.camera_id, first.timestamp) == ("edge-node-001", "camera-001", CAPTURED_AT)
    assert first.processing == ProcessingInfo(
        "yolox-s-coco", "0.1.1rc0", "a" * 64, "onnxruntime-cuda", "cfg123", 42
    )


def test_engine_deduplicates_across_frames() -> None:
    now = [0.0]
    engine = EventEngine(CONTEXT, EventsConfig(("person",), 5.0), [], clock=lambda: now[0])

    counts = []
    for t in (0.0, 1.0, 4.9, 5.0):
        now[0] = t
        counts.append(len(engine.process([PERSON], 0, CAPTURED_AT)))

    assert counts == [1, 0, 0, 1]


def test_failing_sink_does_not_block_other_sinks(caplog: pytest.LogCaptureFixture) -> None:
    good = RecordingSink()
    engine = EventEngine(CONTEXT, EventsConfig(("person",), 0.0), [BrokenSink(), good])

    with caplog.at_level(logging.ERROR, logger="sentinela"):
        engine.process([PERSON], 0, CAPTURED_AT)

    assert len(good.events) == 1
    assert "Event sink failed" in caplog.text


def test_logging_sink_writes_event_json(caplog: pytest.LogCaptureFixture) -> None:
    event = make_event()

    with caplog.at_level(logging.INFO, logger="sentinela.events"):
        LoggingEventSink().publish(event)

    assert json.loads(caplog.records[-1].getMessage())["event_id"] == event.event_id

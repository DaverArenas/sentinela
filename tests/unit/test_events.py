"""Event schema, deduplication rules and the event engine."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from sentinela_runtime.core.config import EVENT_TYPE_NAMES, EventsConfig
from sentinela_runtime.events.event_engine import EventContext, EventEngine
from sentinela_runtime.events.rules import EventCandidate, select_events
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
from sentinela_runtime.vision.camera.video_source import CapturedFrame
from sentinela_runtime.vision.tracking.base import TrackedObject, TrackingResult
from sentinela_runtime.vision.zones.zone_evaluator import TransitionKind, Zone, ZoneTransition

from support import CAR, PERSON, RIGHT_HALF

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


def test_event_type_names_match_config_validation() -> None:
    assert {str(t) for t in EventType} == EVENT_TYPE_NAMES


def test_timestamps_are_normalised_to_utc_and_must_be_aware() -> None:
    local = datetime(2026, 11, 16, 15, 15, 30, tzinfo=timezone(timedelta(hours=-5)))

    assert format_timestamp(local) == "2026-11-16T20:15:30.000Z"
    with pytest.raises(ValueError, match="timezone-aware"):
        format_timestamp(datetime(2026, 1, 1))


# --- rules -----------------------------------------------------------------------------------

ZONE = Zone("zone-001", "Restricted Area", RIGHT_HALF, frozenset({"person"}), "bottom_center")
WALKER = TrackedObject("track-1", PERSON, hits=3)
DRIVER = TrackedObject("track-2", CAR, hits=3)
FRAME = CapturedFrame(index=42, captured_at=CAPTURED_AT, image=np.zeros((64, 96, 3), dtype=np.uint8))


def test_rules_emit_detection_events_only_for_new_tracks_of_configured_classes() -> None:
    tracking = TrackingResult(tracks=(WALKER, DRIVER), started=(WALKER, DRIVER))

    assert select_events(tracking, (), {"person"}) == (EventCandidate(EventType.OBJECT_DETECTED, WALKER),)
    assert select_events(TrackingResult(tracks=(WALKER,)), (), {"person"}) == ()  # not new
    assert select_events(tracking, (), ()) == ()  # disabled


def test_rules_map_zone_transitions_to_events_in_order() -> None:
    transitions = [
        ZoneTransition(TransitionKind.ENTRY, ZONE, WALKER),
        ZoneTransition(TransitionKind.EXIT, ZONE, DRIVER),
    ]

    candidates = select_events(TrackingResult(started=(WALKER,)), transitions, {"person"})

    assert [(c.event_type, c.obj.track_id, c.zone) for c in candidates] == [
        (EventType.OBJECT_DETECTED, "track-1", None),
        (EventType.ZONE_ENTRY, "track-1", ZONE),
        (EventType.ZONE_EXIT, "track-2", ZONE),
    ]


# --- engine ----------------------------------------------------------------------------------


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)


class BrokenSink:
    def publish(self, event: Event) -> None:
        raise OSError("disk full")


class FakeEvidence:
    def __init__(self, result: str | None | Exception = "snapshots/x.jpg") -> None:
        self.result = result
        self.captured: list[str] = []

    def capture(self, event: Event, image: np.ndarray) -> str | None:
        self.captured.append(event.event_id)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


ENTRY = ZoneTransition(TransitionKind.ENTRY, ZONE, WALKER)
EVENTS = EventsConfig(detection_event_classes=("person",))


def test_engine_builds_zone_event_with_track_zone_evidence_and_provenance() -> None:
    sink, evidence = RecordingSink(), FakeEvidence()
    engine = EventEngine(CONTEXT, EVENTS, [sink], evidence)

    (event,) = engine.process(FRAME, TrackingResult(tracks=(WALKER,)), [ENTRY])

    assert sink.events == [event]
    assert evidence.captured == [event.event_id]
    payload = event.to_dict()
    assert payload["event_type"] == "zone_entry"
    assert payload["object"]["track_id"] == "track-1"
    assert payload["zone"] == {"zone_id": "zone-001", "zone_name": "Restricted Area"}
    assert payload["evidence"] == {"snapshot_path": "snapshots/x.jpg"}
    assert (event.node_id, event.camera_id, event.timestamp) == ("edge-node-001", "camera-001", CAPTURED_AT)
    assert event.processing == ProcessingInfo(
        "yolox-s-coco", "0.1.1rc0", "a" * 64, "onnxruntime-cuda", "cfg123", 42
    )


def test_engine_emits_object_detected_for_new_tracks() -> None:
    engine = EventEngine(CONTEXT, EVENTS, [])

    events = engine.process(FRAME, TrackingResult(tracks=(WALKER,), started=(WALKER,)), [])

    assert [(str(e.event_type), e.object.track_id, e.zone) for e in events] == [
        ("object_detected", "track-1", None)
    ]


@pytest.mark.parametrize("result", [None, RuntimeError("camera frame unusable")])
def test_events_are_published_without_evidence_when_capture_yields_nothing(
    result: None | Exception, caplog: pytest.LogCaptureFixture
) -> None:
    sink = RecordingSink()
    engine = EventEngine(CONTEXT, EVENTS, [sink], FakeEvidence(result))

    with caplog.at_level(logging.ERROR, logger="sentinela"):
        engine.process(FRAME, TrackingResult(), [ENTRY])

    assert len(sink.events) == 1 and sink.events[0].evidence is None
    assert ("Evidence capture failed" in caplog.text) is (result is not None)


def test_failing_sink_does_not_block_other_sinks(caplog: pytest.LogCaptureFixture) -> None:
    good = RecordingSink()
    engine = EventEngine(CONTEXT, EVENTS, [BrokenSink(), good])

    with caplog.at_level(logging.ERROR, logger="sentinela"):
        engine.process(FRAME, TrackingResult(), [ENTRY])

    assert len(good.events) == 1
    assert "Event sink failed" in caplog.text


def test_logging_sink_writes_event_json(caplog: pytest.LogCaptureFixture) -> None:
    event = make_event()

    with caplog.at_level(logging.INFO, logger="sentinela.events"):
        LoggingEventSink().publish(event)

    assert json.loads(caplog.records[-1].getMessage())["event_id"] == event.event_id

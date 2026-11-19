"""Camera-to-event integration on recorded video: tracking, zones, events, storage, evidence."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinela_runtime.core.config import (
    CameraConfig,
    EventsConfig,
    PreviewConfig,
    SnapshotConfig,
    TrackingConfig,
)
from sentinela_runtime.events.event_engine import EventContext, EventEngine
from sentinela_runtime.events.schemas import Event
from sentinela_runtime.storage.snapshots import SnapshotStore
from sentinela_runtime.storage.sqlite_store import SqliteEventStore
from sentinela_runtime.vision import pipeline as pipeline_module
from sentinela_runtime.vision.camera.video_source import VideoSource
from sentinela_runtime.vision.inference.base import Detection, InferenceError
from sentinela_runtime.vision.pipeline import PipelineError, PipelineSettings, VisionPipeline
from sentinela_runtime.vision.tracking.iou_tracker import IouTracker
from sentinela_runtime.vision.zones.zone_evaluator import Zone, ZoneEvaluator

from support import PERSON, RIGHT_HALF, FakeBackend, walker, write_video

CONTEXT = EventContext("edge-node-001", "camera-001", "fake-model", "0.0.1", None, "fake-cpu", "cfg")
SETTINGS = PipelineSettings(inference_stride=1, preview=PreviewConfig(enabled=False), stats_interval_s=3600.0)
TRACKING = TrackingConfig(iou_threshold=0.3, min_hits=3, max_lost_frames=5)
EVENTS = EventsConfig(detection_event_classes=(), zone_entry_frames=3, zone_exit_frames=3)
ZONE = Zone("zone-001", "Restricted Area", RIGHT_HALF, frozenset({"person"}), "bottom_center")


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)


def build(
    video: Path,
    backend: FakeBackend,
    settings: PipelineSettings = SETTINGS,
    events: EventsConfig = EVENTS,
    sinks: Sequence[object] = (),
    evidence: SnapshotStore | None = None,
    **kwargs: object,
) -> tuple[VisionPipeline, RecordingSink]:
    sink = RecordingSink()
    engine = EventEngine(CONTEXT, events, [*sinks, sink], evidence)  # type: ignore[list-item]
    pipeline = VisionPipeline(
        source=VideoSource(CameraConfig(camera_id="camera-001", source=str(video))),
        backend=backend,
        tracker=IouTracker(TRACKING),
        zones=ZoneEvaluator([ZONE], events.zone_entry_frames, events.zone_exit_frames),
        engine=engine,
        settings=settings,
        **kwargs,  # type: ignore[arg-type]
    )
    return pipeline, sink


@pytest.fixture
def walk_video(tmp_path: Path) -> Path:
    return write_video(tmp_path / "walk.avi", frames=30)


def test_person_entering_and_leaving_a_zone_creates_one_entry_and_one_exit(
    walk_video: Path, tmp_path: Path
) -> None:
    """Acceptance scenario on recorded video: detect, track, cross the zone, store events + evidence.

    Walker centre x = 10 + 4n: inside the right half from frame 10, entry confirmed at frame 12
    (3 frames). It disappears at frame 20; the track ends after 5 unseen frames (frame 25),
    which produces the exit.
    """
    data_dir = tmp_path / "data"
    snapshots = SnapshotStore(data_dir, SnapshotConfig(min_free_disk_mb=0))
    with SqliteEventStore(data_dir / "sentinela.db") as store:
        pipeline, sink = build(walk_video, FakeBackend(walker()), sinks=[store], evidence=snapshots)

        stats = pipeline.run()

        stored = store.recent()

    entry, exit_ = sink.events
    assert [(str(e.event_type), e.object.track_id, e.processing.frame_index) for e in sink.events] == [
        ("zone_entry", "track-1", 12),
        ("zone_exit", "track-1", 25),
    ]
    assert entry.zone is not None and entry.zone.zone_id == "zone-001"
    assert stats.events == 2 and stats.frames_read == 30

    assert {e["event_id"] for e in stored} == {entry.event_id, exit_.event_id}
    assert entry.evidence is not None and entry.evidence.snapshot_path
    snapshot = cv2.imread(str(data_dir / entry.evidence.snapshot_path))
    assert snapshot is not None and snapshot.shape == (64, 96, 3)
    assert exit_.evidence is None  # snapshots configured for zone_entry only


def test_object_detected_fires_once_per_new_track(walk_video: Path) -> None:
    events = replace(EVENTS, detection_event_classes=("person",))
    pipeline, sink = build(walk_video, FakeBackend(walker()), events=events)

    pipeline.run()

    detected = [e for e in sink.events if str(e.event_type) == "object_detected"]
    assert [(e.object.track_id, e.processing.frame_index) for e in detected] == [("track-1", 2)]


def test_person_outside_zones_produces_no_events(walk_video: Path) -> None:
    pipeline, sink = build(walk_video, FakeBackend(lambda _n: (PERSON,)))  # stays at x 10..40

    assert pipeline.run().events == 0
    assert sink.events == []


def test_inference_stride_samples_frames(recorded_video: Path) -> None:
    backend = FakeBackend(lambda _n: (PERSON,))
    pipeline, _ = build(recorded_video, backend, replace(SETTINGS, inference_stride=4))

    stats = pipeline.run()

    assert backend.calls == 3
    assert (stats.frames_inferred, stats.frames_skipped) == (3, 9)


def test_transient_inference_errors_are_survived(walk_video: Path) -> None:
    script = walker()

    def flaky(n: int) -> Sequence[Detection] | Exception:
        return InferenceError("glitch") if n in (0, 1) else script(n)

    pipeline, sink = build(walk_video, FakeBackend(flaky))

    stats = pipeline.run()

    assert stats.inference_errors == 2
    assert [str(e.event_type) for e in sink.events] == ["zone_entry", "zone_exit"]


def test_persistent_inference_failure_aborts(recorded_video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "MAX_CONSECUTIVE_INFERENCE_ERRORS", 3)
    pipeline, _ = build(recorded_video, FakeBackend(lambda _n: InferenceError("dead")))

    with pytest.raises(PipelineError, match="3 times"):
        pipeline.run()


def test_stop_event_ends_the_run(recorded_video: Path) -> None:
    stop = threading.Event()
    stop.set()
    pipeline, _ = build(recorded_video, FakeBackend(), stop_event=stop)

    assert pipeline.run().frames_read == 0


def test_preview_shows_zones_and_tracks_until_quit(walk_video: Path) -> None:
    shown: list[np.ndarray] = []

    def display(_name: str, image: np.ndarray) -> tuple[bool, bool]:
        shown.append(image)
        return True, len(shown) == 14

    settings = replace(SETTINGS, preview=PreviewConfig(enabled=True))
    pipeline, _ = build(walk_video, FakeBackend(walker()), settings, display=display)

    assert pipeline.run().frames_read == 14
    zone_corner = shown[-1][63, 48]
    assert tuple(zone_corner) == (0, 0, 255)  # zone outline is red: the walker is inside


def test_preview_failure_continues_headless(recorded_video: Path) -> None:
    calls: list[int] = []

    def display(_name: str, _image: np.ndarray) -> tuple[bool, bool]:
        calls.append(1)
        return False, False

    settings = replace(SETTINGS, preview=PreviewConfig(enabled=True))
    pipeline, _ = build(recorded_video, FakeBackend(), settings, display=display)

    assert pipeline.run().frames_read == 12
    assert len(calls) == 1


def test_metrics_are_logged_periodically(recorded_video: Path, caplog: pytest.LogCaptureFixture) -> None:
    ticks = iter(float(i) for i in range(100))
    settings = replace(SETTINGS, stats_interval_s=5.0)
    pipeline, _ = build(recorded_video, FakeBackend(), settings, clock=lambda: next(ticks))

    with caplog.at_level("INFO", logger="sentinela"):
        pipeline.run()

    assert sum(r.getMessage() == "Pipeline metrics" for r in caplog.records) == 2

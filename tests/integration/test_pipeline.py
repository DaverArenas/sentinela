"""Camera-to-event integration tests on recorded video with a scripted inference backend."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sentinela_runtime.core.config import CameraConfig, EventsConfig, PreviewConfig
from sentinela_runtime.events.event_engine import EventContext, EventEngine
from sentinela_runtime.events.schemas import Event
from sentinela_runtime.vision import pipeline as pipeline_module
from sentinela_runtime.vision.camera.video_source import VideoSource
from sentinela_runtime.vision.inference.base import Detection, InferenceError
from sentinela_runtime.vision.pipeline import PipelineError, PipelineSettings, VisionPipeline

from support import CAR, PERSON, FakeBackend

CONTEXT = EventContext("edge-node-001", "camera-001", "fake-model", "0.0.1", None, "fake-cpu", "cfg")
SETTINGS = PipelineSettings(
    inference_stride=1,
    target_classes=("person", "car"),
    preview=PreviewConfig(enabled=False),
    stats_interval_s=3600.0,
)


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)


def build(
    video: Path,
    backend: FakeBackend,
    settings: PipelineSettings = SETTINGS,
    cooldown_s: float = 3600.0,
    **kwargs: object,
) -> tuple[VisionPipeline, RecordingSink]:
    sink = RecordingSink()
    engine = EventEngine(CONTEXT, EventsConfig(settings.target_classes, cooldown_s), [sink])
    source = VideoSource(CameraConfig(camera_id="camera-001", source=str(video)))
    return VisionPipeline(source, backend, engine, settings, **kwargs), sink  # type: ignore[arg-type]


def always(*detections: Detection) -> FakeBackend:
    return FakeBackend(lambda _n: detections)


def test_recorded_video_produces_one_event_per_class_with_cooldown(recorded_video: Path) -> None:
    pipeline, sink = build(recorded_video, always(PERSON, CAR))

    stats = pipeline.run()

    assert (stats.frames_read, stats.frames_inferred, stats.events) == (12, 12, 2)
    assert {e.object.class_name for e in sink.events} == {"person", "car"}
    assert all(e.processing.frame_index == 0 for e in sink.events)
    assert stats.latency_mean_ms is not None


def test_zero_cooldown_emits_on_every_inferred_frame(recorded_video: Path) -> None:
    pipeline, sink = build(recorded_video, always(PERSON), cooldown_s=0.0)

    stats = pipeline.run()

    assert stats.events == 12
    assert [e.processing.frame_index for e in sink.events] == list(range(12))


def test_inference_stride_samples_frames(recorded_video: Path) -> None:
    backend = always(PERSON)
    pipeline, _ = build(recorded_video, backend, replace(SETTINGS, inference_stride=4))

    stats = pipeline.run()

    assert backend.calls == 3
    assert (stats.frames_inferred, stats.frames_skipped) == (3, 9)


def test_transient_inference_errors_are_survived(recorded_video: Path) -> None:
    def script(n: int) -> Sequence[Detection] | Exception:
        return InferenceError("glitch") if n in (0, 1) else (PERSON,)

    pipeline, sink = build(recorded_video, FakeBackend(script))

    stats = pipeline.run()

    assert stats.inference_errors == 2
    assert len(sink.events) == 1 and sink.events[0].processing.frame_index == 2


def test_persistent_inference_failure_aborts(recorded_video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "MAX_CONSECUTIVE_INFERENCE_ERRORS", 3)
    pipeline, _ = build(recorded_video, FakeBackend(lambda _n: InferenceError("dead")))

    with pytest.raises(PipelineError, match="3 times"):
        pipeline.run()


def test_stop_event_ends_the_run(recorded_video: Path) -> None:
    stop = threading.Event()
    stop.set()
    pipeline, _ = build(recorded_video, always(PERSON), stop_event=stop)

    assert pipeline.run().frames_read == 0


def test_preview_shows_annotated_frames_until_quit(recorded_video: Path) -> None:
    shown: list[np.ndarray] = []

    def display(_name: str, image: np.ndarray) -> tuple[bool, bool]:
        shown.append(image)
        return True, len(shown) == 3

    settings = replace(SETTINGS, preview=PreviewConfig(enabled=True))
    pipeline, _ = build(recorded_video, always(PERSON), settings, display=display)

    assert pipeline.run().frames_read == 3
    assert shown[0].any()


def test_preview_failure_continues_headless(recorded_video: Path) -> None:
    calls: list[int] = []

    def display(_name: str, _image: np.ndarray) -> tuple[bool, bool]:
        calls.append(1)
        return False, False

    settings = replace(SETTINGS, preview=PreviewConfig(enabled=True))
    pipeline, _ = build(recorded_video, always(), settings, display=display)

    assert pipeline.run().frames_read == 12
    assert len(calls) == 1


def test_metrics_are_logged_periodically(recorded_video: Path, caplog: pytest.LogCaptureFixture) -> None:
    ticks = iter(float(i) for i in range(100))
    settings = replace(SETTINGS, stats_interval_s=5.0)
    pipeline, _ = build(recorded_video, always(), settings, clock=lambda: next(ticks))

    with caplog.at_level("INFO", logger="sentinela"):
        pipeline.run()

    assert sum(r.getMessage() == "Pipeline metrics" for r in caplog.records) == 2

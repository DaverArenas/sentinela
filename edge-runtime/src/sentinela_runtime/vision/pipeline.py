"""Sentinela Vision pipeline: frame source -> inference -> event engine (-> preview).

Every stage is injected through an interface, so the same pipeline runs against a live
camera, an RTSP stream or a recorded video fixture, with any inference backend.
Tracking and zone evaluation slot in between inference and the event engine.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from sentinela_runtime.core.config import PreviewConfig, RuntimeConfig
from sentinela_runtime.core.interfaces import FrameSource
from sentinela_runtime.core.log import get_logger
from sentinela_runtime.events.event_engine import EventEngine
from sentinela_runtime.monitoring.metrics import MetricsSnapshot, PipelineMetrics
from sentinela_runtime.vision.camera.video_source import CapturedFrame
from sentinela_runtime.vision.inference.base import Detection, InferenceBackend, InferenceError
from sentinela_runtime.vision.visualization import annotate_frame, close_windows, show_frame

log = get_logger("pipeline")

MAX_CONSECUTIVE_INFERENCE_ERRORS = 30
FPS_SMOOTHING = 0.9
Display = Callable[[str, np.ndarray], tuple[bool, bool]]


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot continue (e.g. inference keeps failing)."""


@dataclass(frozen=True)
class PipelineSettings:
    inference_stride: int
    target_classes: tuple[str, ...]
    preview: PreviewConfig
    stats_interval_s: float

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> PipelineSettings:
        return cls(
            inference_stride=config.vision.sampling.inference_stride,
            target_classes=config.vision.events.target_classes,
            preview=config.vision.preview,
            stats_interval_s=config.node.monitoring.stats_interval_s,
        )


class VisionPipeline:
    def __init__(
        self,
        source: FrameSource,
        backend: InferenceBackend,
        engine: EventEngine,
        settings: PipelineSettings,
        stop_event: threading.Event | None = None,
        display: Display = show_frame,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._backend = backend
        self._engine = engine
        self._settings = settings
        self._stop = stop_event or threading.Event()
        self._display = display
        self._clock = clock
        self._metrics = PipelineMetrics()
        self._preview_active = settings.preview.enabled
        self._last_detections: tuple[Detection, ...] = ()
        self._consecutive_errors = 0
        self._fps: float | None = None
        self._last_stats = clock()

    def run(self) -> MetricsSnapshot:
        """Process frames until the source ends, a stop is requested or the preview is closed."""
        try:
            with self._source as source:
                for frame in source.frames():
                    if self._stop.is_set() or not self._step(frame):
                        break
        finally:
            if self._settings.preview.enabled:
                close_windows()
        snapshot = self._snapshot()
        log.info("Pipeline stopped", extra=asdict(snapshot))
        return snapshot

    def _step(self, frame: CapturedFrame) -> bool:
        if frame.index % self._settings.inference_stride == 0:
            self._infer(frame)
        else:
            self._metrics.record_skip()
        self._maybe_log_stats()
        return self._render(frame)

    def _infer(self, frame: CapturedFrame) -> None:
        started = time.perf_counter()
        try:
            detections = self._backend.predict(frame.image)
        except InferenceError as exc:
            self._handle_inference_error(exc, frame.index)
            return
        latency_s = time.perf_counter() - started
        self._consecutive_errors = 0
        events = self._engine.process(detections, frame.index, frame.captured_at)
        self._metrics.record_inference(latency_s * 1000, len(events))
        self._last_detections = detections
        instant = 1.0 / max(latency_s, 1e-6)
        self._fps = (
            instant if self._fps is None else FPS_SMOOTHING * self._fps + (1 - FPS_SMOOTHING) * instant
        )

    def _handle_inference_error(self, exc: InferenceError, frame_index: int) -> None:
        self._metrics.record_error()
        self._consecutive_errors += 1
        log.error("Inference failed", extra={"frame_index": frame_index, "error": str(exc)})
        if self._consecutive_errors >= MAX_CONSECUTIVE_INFERENCE_ERRORS:
            raise PipelineError(f"Inference failed {self._consecutive_errors} times in a row") from exc

    def _render(self, frame: CapturedFrame) -> bool:
        if not self._preview_active:
            return True
        annotated = annotate_frame(
            frame.image, self._last_detections, self._settings.target_classes, self._fps
        )
        self._preview_active, quit_requested = self._display(self._settings.preview.window_name, annotated)
        if quit_requested:
            log.info("Quit requested from preview window")
        return not quit_requested

    def _maybe_log_stats(self) -> None:
        now = self._clock()
        if now - self._last_stats >= self._settings.stats_interval_s:
            self._last_stats = now
            log.info("Pipeline metrics", extra=asdict(self._snapshot()))

    def _snapshot(self) -> MetricsSnapshot:
        return self._metrics.snapshot(reconnects=self._source.reconnect_count)

"""Pipeline metrics and the local preview window."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sentinela_runtime.monitoring.metrics import PipelineMetrics, peak_rss_mib
from sentinela_runtime.vision import visualization as viz
from sentinela_runtime.vision.inference.base import Detection
from sentinela_runtime.vision.tracking.base import TrackedObject
from sentinela_runtime.vision.zones.zone_evaluator import Zone

from support import RIGHT_HALF


def test_metrics_snapshot_aggregates_counters_and_latency() -> None:
    now = [0.0]
    metrics = PipelineMetrics(clock=lambda: now[0])
    for latency in (10.0, 20.0, 30.0):
        metrics.record_inference(latency, events=1)
    metrics.record_skip()
    metrics.record_error()
    now[0] = 2.5

    snap = metrics.snapshot(reconnects=2)

    assert (snap.frames_read, snap.frames_inferred, snap.frames_skipped) == (5, 3, 1)
    assert (snap.inference_errors, snap.events, snap.reconnects) == (1, 3, 2)
    assert snap.throughput_fps == 2.0
    assert snap.latency_mean_ms == 20.0
    assert snap.latency_p95_ms == pytest.approx(29.0)
    assert snap.peak_rss_mib > 0


def test_metrics_without_inference_report_no_latency() -> None:
    snap = PipelineMetrics().snapshot()

    assert snap.latency_mean_ms is None and snap.latency_p95_ms is None
    assert peak_rss_mib() > 0


def test_annotate_frame_draws_zones_tracks_and_detections_on_a_copy() -> None:
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    zone = Zone("zone-001", "Restricted", RIGHT_HALF, frozenset({"person"}), "bottom_center")
    inside = TrackedObject("track-1", Detection(0, "person", 0.9, (60.0, 20.0, 80.0, 50.0)), 3)
    outside = TrackedObject("track-2", Detection(0, "person", 0.9, (5.0, 20.0, 25.0, 50.0)), 3)
    dog = Detection(16, "dog", 0.7, (30.0, 52.0, 40.0, 60.0))

    out = viz.annotate_frame(
        frame,
        detections=[dog],
        tracks=[inside, outside],
        zones=[zone],
        occupancy={"zone-001": frozenset({"track-1"})},
        fps=12.3,
    )

    assert not frame.any()
    assert tuple(out[50, 70]) == viz.COLOR_ALERT  # bottom edge of the track inside the zone
    assert tuple(out[50, 15]) == viz.COLOR_TRACK  # bottom edge of the track outside
    assert tuple(out[60, 35]) == viz.COLOR_OTHER  # raw detection
    assert tuple(out[63, 48]) == viz.COLOR_ALERT  # occupied zone outline (bottom-left corner)


def test_annotate_frame_without_overlays_returns_an_equal_copy() -> None:
    frame = np.full((8, 8, 3), 7, dtype=np.uint8)

    out = viz.annotate_frame(frame)

    assert out is not frame and np.array_equal(out, frame)


def test_show_frame_degrades_to_headless_without_gui(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_gui(*_: object) -> None:
        raise cv2.error("no display")

    monkeypatch.setattr(viz.cv2, "imshow", no_gui)
    monkeypatch.setattr(viz.cv2, "destroyAllWindows", no_gui)

    assert viz.show_frame("w", np.zeros((2, 2, 3), np.uint8)) == (False, False)
    viz.close_windows()  # must not raise


@pytest.mark.parametrize(
    ("key", "quit_requested"), [(ord("q"), True), (27, True), (ord("x"), False), (-1, False)]
)
def test_show_frame_reports_quit_keys(
    monkeypatch: pytest.MonkeyPatch, key: int, quit_requested: bool
) -> None:
    monkeypatch.setattr(viz.cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(viz.cv2, "waitKey", lambda *_: key)

    assert viz.show_frame("w", np.zeros((2, 2, 3), np.uint8)) == (True, quit_requested)

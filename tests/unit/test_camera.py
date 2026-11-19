"""Camera ingestion: source parsing, capture setup, EOF handling and reconnect with backoff."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from sentinela_runtime.core.config import CameraConfig
from sentinela_runtime.vision.camera import video_source as vs
from sentinela_runtime.vision.camera.video_source import CameraError, VideoSource, parse_source

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)
CAMERA = CameraConfig(camera_id="camera-001", source="0", reconnect_attempts=2, reconnect_delay_s=0.0)


@dataclass
class FakeCapture:
    frames: list[np.ndarray | None]
    opened: bool = True
    released: bool = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2 API
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        frame = self.frames.pop(0)
        return frame is not None, frame

    def release(self) -> None:
        self.released = True


@dataclass
class CaptureFactory:
    captures: list[FakeCapture]
    targets: list[int | str] = field(default_factory=list)

    def __call__(self, target: int | str, _config: CameraConfig) -> FakeCapture:
        self.targets.append(target)
        return self.captures.pop(0)


class RecordingStop(threading.Event):
    """Stop event that records backoff delays instead of sleeping."""

    def __init__(self, stop_after: int | None = None) -> None:
        super().__init__()
        self.delays: list[float] = []
        self._stop_after = stop_after

    def wait(self, timeout: float | None = None) -> bool:
        self.delays.append(timeout or 0.0)
        if self._stop_after is not None and len(self.delays) >= self._stop_after:
            self.set()
        return self.is_set()


def source(config: CameraConfig, factory: CaptureFactory, stop: threading.Event | None = None) -> VideoSource:
    return VideoSource(config, stop_event=stop, capture_factory=factory)


@pytest.mark.parametrize(
    ("raw", "expected"), [("0", 0), (" 2 ", 2), ("clip.mp4", "clip.mp4"), ("rtsp://c/s", "rtsp://c/s")]
)
def test_parse_source(raw: str, expected: int | str) -> None:
    assert parse_source(raw) == expected


def test_parse_source_rejects_empty() -> None:
    with pytest.raises(CameraError):
        parse_source(" ")


@pytest.mark.parametrize(
    ("raw", "live"), [("0", True), ("rtsp://cam/1", True), ("HTTP://cam/mjpg", True), ("videos/a.mp4", False)]
)
def test_live_source_detection(raw: str, live: bool) -> None:
    assert VideoSource(replace(CAMERA, source=raw)).is_live is live


def test_recorded_source_stops_at_eof_with_sequential_indices() -> None:
    capture = FakeCapture([FRAME, FRAME, FRAME])
    factory = CaptureFactory([capture])

    with source(replace(CAMERA, source="clip.mp4"), factory) as src:
        frames = list(src.frames())

    assert [f.index for f in frames] == [0, 1, 2]
    assert all(f.captured_at.tzinfo is not None for f in frames)
    assert factory.targets == ["clip.mp4"]
    assert capture.released


def test_live_source_reconnects_with_exponential_backoff() -> None:
    config = replace(CAMERA, reconnect_attempts=3, reconnect_delay_s=1.0, reconnect_max_delay_s=1.5)
    factory = CaptureFactory(
        [
            FakeCapture([FRAME, None]),
            FakeCapture([], opened=False),
            FakeCapture([], opened=False),
            FakeCapture([FRAME]),
        ]
    )
    stop = RecordingStop()

    with source(config, factory, stop) as src:
        frames = src.frames()
        received = [next(frames), next(frames)]

        assert [f.index for f in received] == [0, 1]
        assert stop.delays == [1.0, 1.5, 1.5]  # 1, 2 -> capped, 4 -> capped
        assert src.reconnect_count == 1


def test_live_source_raises_after_exhausting_attempts() -> None:
    factory = CaptureFactory([FakeCapture([]), FakeCapture([], opened=False), FakeCapture([], opened=False)])

    with (
        pytest.raises(CameraError, match="could not be recovered"),
        source(CAMERA, factory, RecordingStop()) as src,
    ):
        list(src.frames())


def test_unlimited_reconnect_ends_quietly_when_stop_is_requested() -> None:
    config = replace(CAMERA, reconnect_attempts=None)
    factory = CaptureFactory([FakeCapture([])] + [FakeCapture([], opened=False) for _ in range(5)])
    stop = RecordingStop(stop_after=4)

    with source(config, factory, stop) as src:
        assert list(src.frames()) == []

    assert len(stop.delays) == 4


def test_frames_stop_immediately_when_stop_already_set() -> None:
    stop = threading.Event()
    stop.set()

    with source(CAMERA, CaptureFactory([FakeCapture([FRAME])]), stop) as src:
        assert list(src.frames()) == []


def test_open_failure_masks_stream_credentials() -> None:
    config = replace(CAMERA, source="rtsp://admin:hunter2@10.0.0.5/stream")
    factory = CaptureFactory([FakeCapture([], opened=False)])

    with pytest.raises(CameraError) as excinfo:
        source(config, factory).__enter__()

    assert "hunter2" not in str(excinfo.value)
    assert "rtsp://***@10.0.0.5/stream" in str(excinfo.value)


def test_open_capture_applies_webcam_hints(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[Any] = []

    class StubCapture:
        def __init__(self, *args: Any) -> None:
            self.args, self.props = args, {}
            created.append(self)

        def set(self, prop: int, value: float) -> None:
            self.props[prop] = value

    monkeypatch.setattr(vs.cv2, "VideoCapture", StubCapture)

    vs.open_capture(0, replace(CAMERA, width=1280, height=None, fps=15))
    vs.open_capture("rtsp://cam/s", CAMERA)

    webcam, stream = created
    assert webcam.props == {cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FPS: 15, cv2.CAP_PROP_BUFFERSIZE: 1}
    assert stream.args[1] == cv2.CAP_FFMPEG
    assert stream.args[2] == [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000]


def test_missing_recorded_file_raises_camera_error(tmp_path: Path) -> None:
    with pytest.raises(CameraError, match="Unable to open"):
        VideoSource(replace(CAMERA, source=str(tmp_path / "absent.avi"))).__enter__()

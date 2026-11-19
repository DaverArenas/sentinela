"""Frame ingestion from USB webcams, RTSP/HTTP streams and recorded video files.

Live sources are reopened with exponential backoff when they fail; recorded files end at
EOF. Frames are pulled synchronously (no internal queue) so memory stays bounded.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Self

import cv2
import numpy as np

from sentinela_runtime.core.config import CameraConfig
from sentinela_runtime.core.log import get_logger

log = get_logger("camera")

CaptureFactory = Callable[[int | str, CameraConfig], Any]
STREAM_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://", "rtmp://", "udp://", "tcp://")
LIVE_BUFFER_FRAMES = 1  # keep only the newest frame in the driver buffer to avoid lag


class CameraError(RuntimeError):
    """Raised when a source cannot be opened or a live source cannot be recovered."""


@dataclass(frozen=True, eq=False)
class CapturedFrame:
    index: int
    captured_at: datetime  # UTC wall-clock capture time
    image: np.ndarray = field(repr=False)


def parse_source(source: str) -> int | str:
    """A bare integer string is a camera index; anything else is a URL or file path."""
    stripped = source.strip()
    if not stripped:
        raise CameraError("Camera source must not be empty")
    return int(stripped) if stripped.isdigit() else stripped


def is_stream_url(target: int | str) -> bool:
    return isinstance(target, str) and target.lower().startswith(STREAM_SCHEMES)


def open_capture(target: int | str, config: CameraConfig) -> Any:
    """Create a ``cv2.VideoCapture`` with timeouts (streams) and capture hints (webcams)."""
    if is_stream_url(target):
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.open_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, config.read_timeout_ms,
        ]  # fmt: skip
        capture = cv2.VideoCapture(target, cv2.CAP_FFMPEG, params)
    else:
        capture = cv2.VideoCapture(target)

    if isinstance(target, int):
        hints = (
            (cv2.CAP_PROP_FRAME_WIDTH, config.width),
            (cv2.CAP_PROP_FRAME_HEIGHT, config.height),
            (cv2.CAP_PROP_FPS, config.fps),
            (cv2.CAP_PROP_BUFFERSIZE, LIVE_BUFFER_FRAMES),
        )
        for prop, value in hints:
            if value is not None:
                capture.set(prop, value)
    return capture


def utc_now() -> datetime:
    return datetime.now(UTC)


class VideoSource:
    """Context-managed frame iterator implementing the ``FrameSource`` protocol."""

    def __init__(
        self,
        config: CameraConfig,
        stop_event: threading.Event | None = None,
        capture_factory: CaptureFactory = open_capture,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._config = config
        self._target = parse_source(config.source)
        self._stop = stop_event or threading.Event()
        self._capture_factory = capture_factory
        self._clock = clock
        self._capture: Any = None
        self._reconnects = 0

    @property
    def is_live(self) -> bool:
        return isinstance(self._target, int) or is_stream_url(self._target)

    @property
    def reconnect_count(self) -> int:
        return self._reconnects

    def __enter__(self) -> Self:
        self._open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def frames(self) -> Iterator[CapturedFrame]:
        index = 0
        while not self._stop.is_set():
            ok, image = self._capture.read() if self._capture is not None else (False, None)
            if ok and image is not None:
                yield CapturedFrame(index=index, captured_at=self._clock(), image=image)
                index += 1
                continue
            if not self.is_live:
                log.info("End of recorded source", extra={"camera_id": self._config.camera_id})
                return
            if not self._reconnect():
                if self._stop.is_set():
                    return
                raise CameraError(f"Camera {self._config.camera_id!r} lost and could not be recovered")

    def _open(self) -> None:
        capture = self._capture_factory(self._target, self._config)
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"Unable to open camera {self._config.camera_id!r} ({self._describe()})")
        self._capture = capture
        log.info("Camera opened", extra={"camera_id": self._config.camera_id, "live": self.is_live})

    def _reconnect(self) -> bool:
        self.close()
        attempts = self._config.reconnect_attempts
        attempt = 0
        while attempts is None or attempt < attempts:
            attempt += 1
            delay = min(
                self._config.reconnect_delay_s * 2 ** (attempt - 1), self._config.reconnect_max_delay_s
            )
            log.warning(
                "Camera lost; reconnecting",
                extra={"camera_id": self._config.camera_id, "attempt": attempt, "delay_s": delay},
            )
            if self._stop.wait(delay):
                return False
            try:
                self._open()
            except CameraError:
                continue
            self._reconnects += 1
            return True
        return False

    def _describe(self) -> str:
        """Source description safe for logs: never prints stream credentials."""
        if isinstance(self._target, str) and "@" in self._target and is_stream_url(self._target):
            scheme, rest = self._target.split("://", 1)
            return f"{scheme}://***@{rest.rsplit('@', 1)[1]}"
        return str(self._target)

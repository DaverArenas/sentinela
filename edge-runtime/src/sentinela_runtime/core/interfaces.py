"""Contracts between runtime components.

Components depend on these protocols rather than on each other's implementations, so a
camera, inference backend or event sink can be replaced (or faked in tests) without
touching the pipeline. The inference contract lives in ``vision.inference.base``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, Self

if TYPE_CHECKING:
    import numpy as np

    from sentinela_runtime.events.schemas import Event
    from sentinela_runtime.vision.camera.video_source import CapturedFrame


class FrameSource(Protocol):
    """A context-managed stream of captured frames (camera, stream or recorded file)."""

    @property
    def reconnect_count(self) -> int: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *exc_info: object) -> None: ...

    def frames(self) -> Iterator[CapturedFrame]: ...


class EventSink(Protocol):
    """Destination for structured events (SQLite store, JSONL log; MQTT in a later phase)."""

    def publish(self, event: Event) -> None: ...


class EvidenceRecorder(Protocol):
    """Stores evidence (e.g. a snapshot) for an event before it is published."""

    def capture(self, event: Event, image: np.ndarray) -> str | None:
        """Return the stored evidence path (relative to the data dir), or None if nothing was stored."""
        ...

"""Local evidence: JPEG snapshots of the frame that triggered an event (``EvidenceRecorder``).

Snapshots are written unmodified (no overlays) so the stored frame is faithful evidence; the
event record carries the bounding box and zone. Files go to
``<data_dir>/snapshots/<UTC date>/<event_id>.jpg`` via a temporary file, ``fsync`` and an atomic
rename, so a crash or power loss never leaves a half-written image. Footage stays on the node
(``data/`` is git-ignored).
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from datetime import UTC
from pathlib import Path

import cv2
import numpy as np

from sentinela_runtime.core.config import SnapshotConfig
from sentinela_runtime.core.log import get_logger
from sentinela_runtime.events.schemas import Event

log = get_logger("storage")

SNAPSHOT_DIR = "snapshots"
BYTES_PER_MIB = 1024 * 1024


def free_disk_mib(path: Path) -> float:
    return shutil.disk_usage(path).free / BYTES_PER_MIB


class SnapshotStore:
    def __init__(
        self,
        data_dir: Path,
        config: SnapshotConfig,
        free_space: Callable[[Path], float] = free_disk_mib,
    ) -> None:
        self._data_dir = data_dir
        self._config = config
        self._free_space = free_space

    def capture(self, event: Event, image: np.ndarray) -> str | None:
        """Save a snapshot when this event type is configured for one; return its relative path."""
        if not self._config.enabled or str(event.event_type) not in self._config.event_types:
            return None
        relative = (
            Path(SNAPSHOT_DIR)
            / event.timestamp.astimezone(UTC).strftime("%Y-%m-%d")
            / f"{event.event_id}.jpg"
        )
        target = self._data_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)

        free = self._free_space(target.parent)
        if free < self._config.min_free_disk_mb:
            log.warning(
                "Low disk space; snapshot skipped",
                extra={"event_id": event.event_id, "free_mib": round(free, 1)},
            )
            return None

        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self._config.jpeg_quality])
        if not ok:
            raise OSError(f"Could not encode snapshot {target}")
        write_durably(target, encoded.tobytes())
        return relative.as_posix()


def write_durably(target: Path, data: bytes) -> None:
    """Write via a temp file + fsync + atomic rename: the file is either complete or absent."""
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

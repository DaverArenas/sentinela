"""Sentinela structured event schema (version 1.0).

Serialised shape (see README "Event Architecture")::

    {"schema_version", "event_id", "event_type", "timestamp", "node_id", "camera_id",
     "object": {...}, "zone": {...} | null, "evidence": {...} | null, "processing": {...}}
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = "1.0"
COORD_DECIMALS = 1
CONFIDENCE_DECIMALS = 4


class EventType(StrEnum):
    OBJECT_DETECTED = "object_detected"


def new_event_id() -> str:
    return f"evt-{uuid.uuid4().hex}"


def format_timestamp(moment: datetime) -> str:
    """ISO-8601 UTC with millisecond precision and a ``Z`` suffix."""
    if moment.tzinfo is None:
        raise ValueError("Event timestamps must be timezone-aware")
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, COORD_DECIMALS) for k, v in vars(self).items()}


@dataclass(frozen=True)
class ObjectInfo:
    class_name: str
    class_id: int
    confidence: float
    bounding_box: BoundingBox
    track_id: str | None = None  # populated once tracking lands

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "track_id": self.track_id,
            "confidence": round(self.confidence, CONFIDENCE_DECIMALS),
            "bounding_box": self.bounding_box.to_dict(),
        }


@dataclass(frozen=True)
class ZoneInfo:
    zone_id: str
    zone_name: str

    def to_dict(self) -> dict[str, str]:
        return {"zone_id": self.zone_id, "zone_name": self.zone_name}


@dataclass(frozen=True)
class EvidenceInfo:
    snapshot_path: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {"snapshot_path": self.snapshot_path}


@dataclass(frozen=True)
class ProcessingInfo:
    """Provenance: which model, engine and configuration produced the event, from which frame."""

    model_id: str
    model_version: str
    model_sha256: str | None
    inference_backend: str
    config_hash: str
    frame_index: int

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


@dataclass(frozen=True)
class Event:
    event_type: EventType
    timestamp: datetime
    node_id: str
    camera_id: str
    object: ObjectInfo
    processing: ProcessingInfo
    zone: ZoneInfo | None = None
    evidence: EvidenceInfo | None = None
    event_id: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.event_id:
            object.__setattr__(self, "event_id", new_event_id())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": str(self.event_type),
            "timestamp": format_timestamp(self.timestamp),
            "node_id": self.node_id,
            "camera_id": self.camera_id,
            "object": self.object.to_dict(),
            "zone": self.zone.to_dict() if self.zone else None,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "processing": self.processing.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

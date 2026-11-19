"""Typed, validated configuration loaded from the repository ``config/`` directory.

Files (all YAML):
    node_config.yaml     node identity, camera source, storage, monitoring, polygon zones
    vision_config.yaml   model provenance, inference backend, sampling, tracking, event rules, preview
    logging_config.yaml  ``logging.config.dictConfig`` schema (consumed by ``core.log``)

Every config object is a frozen dataclass; derive changes with :func:`dataclasses.replace`.
Relative paths inside the files are resolved against ``RuntimeConfig.base_dir`` (the
parent of the config directory, i.e. the repository root).
"""

from __future__ import annotations

import hashlib
import json
import os
import typing
from collections.abc import Mapping
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

import yaml

from sentinela_runtime.vision.zones.geometry import ANCHORS, is_simple_polygon, polygon_area

CONFIG_DIR_ENV = "SENTINELA_CONFIG_DIR"
CAMERA_SOURCE_ENV = "SENTINELA_CAMERA_SOURCE"
NODE_CONFIG_FILE = "node_config.yaml"
VISION_CONFIG_FILE = "vision_config.yaml"
LOGGING_CONFIG_FILE = "logging_config.yaml"

VALID_DEVICES = frozenset({"auto", "cuda", "cpu"})
# Kept in sync with the tracker registry and EventType by unit tests.
VALID_TRACKERS = frozenset({"iou"})
EVENT_TYPE_NAMES = frozenset({"object_detected", "zone_entry", "zone_exit"})
CONFIG_HASH_LENGTH = 16
MIN_ZONE_AREA = 1e-4  # normalised units: 0.01% of the frame
MIN_POLYGON_VERTICES = 3
MAX_JPEG_QUALITY = 100

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


# --- node_config.yaml --------------------------------------------------------------


@dataclass(frozen=True)
class NodeIdentity:
    node_id: str


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    # Webcam index ("0"), rtsp:// / http:// URL, or recorded video file path.
    source: str
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    # None = retry forever (recommended for unattended live cameras).
    reconnect_attempts: int | None = None
    reconnect_delay_s: float = 2.0
    reconnect_max_delay_s: float = 30.0
    open_timeout_ms: int = 10_000
    read_timeout_ms: int = 5_000


@dataclass(frozen=True)
class SnapshotConfig:
    enabled: bool = True
    # Event types that store a JPEG of the triggering frame as evidence.
    event_types: tuple[str, ...] = ("zone_entry",)
    jpeg_quality: int = 85
    # Snapshots are skipped (event still stored) when free disk space drops below this.
    min_free_disk_mb: int = 500


@dataclass(frozen=True)
class StorageConfig:
    data_dir: str = "data"
    database_file: str = "sentinela.db"  # relative to data_dir
    snapshots: SnapshotConfig = field(default_factory=SnapshotConfig)


@dataclass(frozen=True)
class ZoneConfig:
    zone_id: str
    name: str
    # Ordered [x, y] vertices normalised to [0, 1] of frame width / height.
    polygon: tuple[tuple[float, float], ...]
    classes: tuple[str, ...] = ("person",)
    anchor: str = "bottom_center"  # bottom_center (feet) | center


@dataclass(frozen=True)
class MonitoringConfig:
    # Interval between pipeline metric log lines (latency, throughput, memory).
    stats_interval_s: float = 30.0


@dataclass(frozen=True)
class NodeConfig:
    node: NodeIdentity
    camera: CameraConfig
    storage: StorageConfig = field(default_factory=StorageConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    zones: tuple[ZoneConfig, ...] = ()


# --- vision_config.yaml ------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    family: str
    version: str
    format: str
    path: str
    sha256: str | None
    source: str
    license: str
    input_size: tuple[int, int]  # height, width


@dataclass(frozen=True)
class InferenceConfig:
    backend: str = "onnxruntime"
    device: str = "auto"
    confidence_threshold: float = 0.5
    nms_iou_threshold: float = 0.45


@dataclass(frozen=True)
class SamplingConfig:
    inference_stride: int = 1


@dataclass(frozen=True)
class TrackingConfig:
    tracker: str = "iou"
    iou_threshold: float = 0.3
    min_hits: int = 3  # consecutive matches before a track is confirmed and gets an ID
    max_lost_frames: int = 15  # inferred frames a confirmed track survives without a match


@dataclass(frozen=True)
class EventsConfig:
    # One object_detected event per newly confirmed track of these classes (empty = disabled).
    detection_event_classes: tuple[str, ...] = ()
    # Consecutive inferred frames inside / outside a zone before zone_entry / zone_exit fires.
    zone_entry_frames: int = 3
    zone_exit_frames: int = 5


@dataclass(frozen=True)
class PreviewConfig:
    enabled: bool = True
    window_name: str = "Sentinela Vision"


@dataclass(frozen=True)
class VisionConfig:
    model: ModelConfig
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)


# --- aggregate -----------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeConfig:
    base_dir: Path
    config_dir: Path
    node: NodeConfig
    vision: VisionConfig

    @property
    def logging_config_path(self) -> Path:
        return self.config_dir / LOGGING_CONFIG_FILE

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a config-relative path against the repository root."""
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else self.base_dir / candidate

    @property
    def data_dir(self) -> Path:
        return self.resolve_path(self.node.storage.data_dir)

    @property
    def database_path(self) -> Path:
        return self.data_dir / self.node.storage.database_file

    def fingerprint(self) -> str:
        """Short, stable hash of the effective configuration (event provenance)."""
        payload = {"node": asdict(self.node), "vision": asdict(self.vision)}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        return digest[:CONFIG_HASH_LENGTH]


# --- loading -------------------------------------------------------------------------


def default_config_dir() -> Path:
    """``$SENTINELA_CONFIG_DIR`` or ``<repo>/config`` (repo = 4 levels above this package)."""
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[4] / "config"


def load_runtime_config(config_dir: str | Path | None = None) -> RuntimeConfig:
    """Load, merge and validate ``node_config.yaml`` and ``vision_config.yaml``."""
    directory = Path(config_dir).expanduser().resolve() if config_dir else default_config_dir()
    if not directory.is_dir():
        raise ConfigError(f"Config directory not found: {directory}")

    node = build_config(NodeConfig, read_yaml(directory / NODE_CONFIG_FILE))
    camera_override = os.environ.get(CAMERA_SOURCE_ENV)
    if camera_override:
        node = replace(node, camera=replace(node.camera, source=camera_override))
    vision = build_config(VisionConfig, read_yaml(directory / VISION_CONFIG_FILE))

    config = RuntimeConfig(base_dir=directory.parent, config_dir=directory, node=node, vision=vision)
    return validate_config(config)


def read_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"Top level of {path} must be a mapping")
    return data


def build_config(cls: type[T], data: Mapping[str, Any], section: str = "") -> T:
    """Build a (nested) frozen dataclass from a mapping, rejecting unknown or missing keys."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"Section {section or '<root>'!r} must be a mapping")
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")

    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        where = f" in section {section!r}" if section else ""
        raise ConfigError(f"Unknown config key(s){where}: {unknown}")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        qualified = f"{section}.{f.name}" if section else f.name
        hint = hints[f.name]
        if is_dataclass(hint):
            kwargs[f.name] = build_config(hint, data.get(f.name) or {}, qualified)  # type: ignore[arg-type]
        elif f.name in data:
            kwargs[f.name] = _coerce(hint, data[f.name], qualified)
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ConfigError(f"Missing required config key: {qualified}")
    return cls(**kwargs)


def _coerce(hint: Any, value: Any, section: str) -> Any:
    """Turn YAML lists into (nested) tuples and mappings into dataclasses; validation checks the rest."""
    if isinstance(hint, type) and is_dataclass(hint):
        return build_config(hint, value, section)
    if typing.get_origin(hint) is not tuple or not isinstance(value, list):
        return value
    args = typing.get_args(hint)
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(_coerce(args[0], item, f"{section}[{i}]") for i, item in enumerate(value))
    if len(args) == len(value):
        pairs = enumerate(zip(args, value, strict=True))
        return tuple(_coerce(arg, item, f"{section}[{i}]") for i, (arg, item) in pairs)
    return tuple(value)  # wrong arity: reported by validation


def apply_overrides(
    config: RuntimeConfig,
    *,
    source: str | None = None,
    device: str | None = None,
    preview_enabled: bool | None = None,
) -> RuntimeConfig:
    """Return a new, re-validated config with CLI overrides layered on top."""
    node, vision = config.node, config.vision
    if source is not None:
        node = replace(node, camera=replace(node.camera, source=source))
    if device is not None:
        vision = replace(vision, inference=replace(vision.inference, device=device))
    if preview_enabled is not None:
        vision = replace(vision, preview=replace(vision.preview, enabled=preview_enabled))
    return validate_config(replace(config, node=node, vision=vision))


# --- validation ------------------------------------------------------------------------


def validate_config(config: RuntimeConfig) -> RuntimeConfig:
    """Return ``config`` unchanged or raise ``ConfigError`` listing every problem."""
    errors = [*_node_errors(config.node), *_vision_errors(config.vision)]
    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
    return config


def _node_errors(node: NodeConfig) -> list[str]:
    cam = node.camera
    checks = [
        (_non_empty(node.node.node_id), "node.node_id must be a non-empty string"),
        (_non_empty(cam.camera_id), "camera.camera_id must be a non-empty string"),
        (_non_empty(cam.source), "camera.source must be a non-empty string"),
        (_optional_positive_int(cam.width), "camera.width must be a positive integer or null"),
        (_optional_positive_int(cam.height), "camera.height must be a positive integer or null"),
        (cam.fps is None or _positive_number(cam.fps), "camera.fps must be a positive number or null"),
        (
            cam.reconnect_attempts is None
            or (_is_int(cam.reconnect_attempts) and cam.reconnect_attempts >= 0),
            "camera.reconnect_attempts must be an integer >= 0 or null (unlimited)",
        ),
        (_non_negative_number(cam.reconnect_delay_s), "camera.reconnect_delay_s must be >= 0"),
        (_non_negative_number(cam.reconnect_max_delay_s), "camera.reconnect_max_delay_s must be >= 0"),
        (_positive_int(cam.open_timeout_ms), "camera.open_timeout_ms must be a positive integer"),
        (_positive_int(cam.read_timeout_ms), "camera.read_timeout_ms must be a positive integer"),
        (_non_empty(node.storage.data_dir), "storage.data_dir must be a non-empty string"),
        (_non_empty(node.storage.database_file), "storage.database_file must be a non-empty string"),
        (_positive_number(node.monitoring.stats_interval_s), "monitoring.stats_interval_s must be > 0"),
    ]
    errors = [message for ok, message in checks if not ok]
    return [*errors, *_snapshot_errors(node.storage.snapshots), *_zone_errors(node.zones)]


def _snapshot_errors(snapshots: SnapshotConfig) -> list[str]:
    checks = [
        (isinstance(snapshots.enabled, bool), "storage.snapshots.enabled must be a boolean"),
        (
            _string_tuple(snapshots.event_types)
            and all(t in EVENT_TYPE_NAMES for t in snapshots.event_types),
            f"storage.snapshots.event_types must be a list drawn from {sorted(EVENT_TYPE_NAMES)}",
        ),
        (
            _positive_int(snapshots.jpeg_quality) and snapshots.jpeg_quality <= MAX_JPEG_QUALITY,
            "storage.snapshots.jpeg_quality must be an integer in [1, 100]",
        ),
        (
            _is_int(snapshots.min_free_disk_mb) and snapshots.min_free_disk_mb >= 0,
            "storage.snapshots.min_free_disk_mb must be an integer >= 0",
        ),
    ]
    return [message for ok, message in checks if not ok]


def _zone_errors(zones: Any) -> list[str]:
    if not isinstance(zones, tuple):
        return ["zones must be a list (use [] for no zones)"]
    errors: list[str] = []
    ids = [zone.zone_id for zone in zones]
    duplicates = sorted({zone_id for zone_id in ids if ids.count(zone_id) > 1})
    if duplicates:
        errors.append(f"zones: duplicate zone_id(s) {duplicates}")
    for index, zone in enumerate(zones):
        where = f"zones[{index}] ({zone.zone_id!r})"
        checks = [
            (_non_empty(zone.zone_id) and _non_empty(zone.name), f"{where}: zone_id and name are required"),
            (
                _string_tuple(zone.classes) and len(zone.classes) > 0,
                f"{where}: classes must be a non-empty list of class names",
            ),
            (zone.anchor in ANCHORS, f"{where}: anchor must be one of {sorted(ANCHORS)}"),
        ]
        errors.extend(message for ok, message in checks if not ok)
        errors.extend(f"{where}: {problem}" for problem in _polygon_problems(zone.polygon))
    return errors


def _polygon_problems(polygon: Any) -> list[str]:
    valid_vertices = isinstance(polygon, tuple) and all(
        isinstance(v, tuple) and len(v) == 2 and all(_is_number(c) and 0.0 <= c <= 1.0 for c in v)
        for v in polygon
    )
    if not valid_vertices:
        return ["polygon must be a list of [x, y] pairs normalised to [0, 1]"]
    if len(polygon) < MIN_POLYGON_VERTICES:
        return [f"polygon needs at least {MIN_POLYGON_VERTICES} vertices"]
    if any(polygon[i] == polygon[i - 1] for i in range(len(polygon))):
        return ["polygon repeats a vertex consecutively; remove the duplicate"]
    if not is_simple_polygon(polygon):
        return ["polygon edges cross each other; list vertices in boundary order"]
    if polygon_area(polygon) < MIN_ZONE_AREA:
        return ["polygon area is (nearly) zero"]
    return []


def _vision_errors(vision: VisionConfig) -> list[str]:
    model, inference = vision.model, vision.inference
    size = model.input_size
    checks = [
        (
            all(_non_empty(v) for v in (model.model_id, model.family, model.version, model.format)),
            "model.model_id, family, version and format must be non-empty strings",
        ),
        (_non_empty(model.path), "model.path must be a non-empty string"),
        (model.sha256 is None or _is_sha256(model.sha256), "model.sha256 must be 64 hex chars or null"),
        (
            _non_empty(model.source) and _non_empty(model.license),
            "model.source and model.license are required",
        ),
        (
            isinstance(size, tuple) and len(size) == 2 and all(_positive_int(v) for v in size),
            "model.input_size must be [height, width] positive integers",
        ),
        (_non_empty(inference.backend), "inference.backend must be a non-empty string"),
        (inference.device in VALID_DEVICES, f"inference.device must be one of {sorted(VALID_DEVICES)}"),
        (_unit_interval(inference.confidence_threshold), "inference.confidence_threshold must be in (0, 1]"),
        (_unit_interval(inference.nms_iou_threshold), "inference.nms_iou_threshold must be in (0, 1]"),
        (
            _positive_int(vision.sampling.inference_stride),
            "sampling.inference_stride must be an integer >= 1",
        ),
        (
            vision.tracking.tracker in VALID_TRACKERS,
            f"tracking.tracker must be one of {sorted(VALID_TRACKERS)}",
        ),
        (_unit_interval(vision.tracking.iou_threshold), "tracking.iou_threshold must be in (0, 1]"),
        (_positive_int(vision.tracking.min_hits), "tracking.min_hits must be an integer >= 1"),
        (
            _is_int(vision.tracking.max_lost_frames) and vision.tracking.max_lost_frames >= 0,
            "tracking.max_lost_frames must be an integer >= 0",
        ),
        (
            _string_tuple(vision.events.detection_event_classes),
            "events.detection_event_classes must be a list of class names ([] disables)",
        ),
        (_positive_int(vision.events.zone_entry_frames), "events.zone_entry_frames must be an integer >= 1"),
        (_positive_int(vision.events.zone_exit_frames), "events.zone_exit_frames must be an integer >= 1"),
        (isinstance(vision.preview.enabled, bool), "preview.enabled must be a boolean"),
    ]
    return [message for ok, message in checks if not ok]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive_int(value: Any) -> bool:
    return _is_int(value) and value > 0


def _optional_positive_int(value: Any) -> bool:
    return value is None or _positive_int(value)


def _positive_number(value: Any) -> bool:
    return _is_number(value) and value > 0


def _non_negative_number(value: Any) -> bool:
    return _is_number(value) and value >= 0


def _unit_interval(value: Any) -> bool:
    return _is_number(value) and 0.0 < value <= 1.0


def _non_empty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_tuple(value: Any) -> bool:
    return isinstance(value, tuple) and all(_non_empty(v) for v in value)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())

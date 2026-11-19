"""Typed, validated configuration loaded from the repository ``config/`` directory.

Files (all YAML):
    node_config.yaml     node identity, camera source, storage, monitoring
    vision_config.yaml   model provenance, inference backend, sampling, event rules, preview
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

CONFIG_DIR_ENV = "SENTINELA_CONFIG_DIR"
CAMERA_SOURCE_ENV = "SENTINELA_CAMERA_SOURCE"
NODE_CONFIG_FILE = "node_config.yaml"
VISION_CONFIG_FILE = "vision_config.yaml"
LOGGING_CONFIG_FILE = "logging_config.yaml"

VALID_DEVICES = frozenset({"auto", "cuda", "cpu"})
CONFIG_HASH_LENGTH = 16

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
class StorageConfig:
    data_dir: str = "data"


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
class EventsConfig:
    target_classes: tuple[str, ...] = ("person",)
    cooldown_s: float = 5.0


@dataclass(frozen=True)
class PreviewConfig:
    enabled: bool = True
    window_name: str = "Sentinela Vision"


@dataclass(frozen=True)
class VisionConfig:
    model: ModelConfig
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
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
            kwargs[f.name] = _coerce(hint, data[f.name])
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ConfigError(f"Missing required config key: {qualified}")
    return cls(**kwargs)


def _coerce(hint: Any, value: Any) -> Any:
    """Turn YAML lists into tuples for tuple-typed fields; leave everything else as-is."""
    if typing.get_origin(hint) is tuple and isinstance(value, list):
        return tuple(value)
    return value


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
        (_positive_number(node.monitoring.stats_interval_s), "monitoring.stats_interval_s must be > 0"),
    ]
    return [message for ok, message in checks if not ok]


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
            isinstance(vision.events.target_classes, tuple)
            and len(vision.events.target_classes) > 0
            and all(_non_empty(c) for c in vision.events.target_classes),
            "events.target_classes must be a non-empty list of class names",
        ),
        (_non_negative_number(vision.events.cooldown_s), "events.cooldown_s must be >= 0"),
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


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())

"""Configuration loading, validation and overrides."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sentinela_runtime.core import config as cfg
from sentinela_runtime.core.config import (
    ConfigError,
    NodeConfig,
    VisionConfig,
    apply_overrides,
    build_config,
    load_runtime_config,
)

from support import REPO_CONFIG_DIR, REPO_ROOT


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cfg.CONFIG_DIR_ENV, raising=False)
    monkeypatch.delenv(cfg.CAMERA_SOURCE_ENV, raising=False)


def test_repository_config_is_valid_and_matches_documented_defaults() -> None:
    config = load_runtime_config(REPO_CONFIG_DIR)

    assert config.base_dir == REPO_ROOT
    assert config.node.node.node_id == "edge-node-001"
    assert config.node.camera.source == "0"
    assert config.vision.model.family == "yolox"
    assert config.vision.model.input_size == (640, 640)
    assert config.vision.events.detection_event_classes == ()
    assert config.vision.tracking.tracker == "iou"
    assert [z.zone_id for z in config.node.zones] == ["zone-001"]
    assert config.node.zones[0].polygon == ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0))
    assert config.database_path == REPO_ROOT / "data" / "sentinela.db"
    assert config.node.storage.snapshots.event_types == ("zone_entry",)
    assert config.logging_config_path == REPO_CONFIG_DIR / "logging_config.yaml"


def test_default_config_dir_is_repository_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert cfg.default_config_dir() == REPO_CONFIG_DIR

    monkeypatch.setenv(cfg.CONFIG_DIR_ENV, str(tmp_path))
    assert cfg.default_config_dir() == tmp_path


def test_camera_source_env_var_keeps_credentials_out_of_yaml(
    monkeypatch: pytest.MonkeyPatch, make_config_dir: Callable[..., Path]
) -> None:
    monkeypatch.setenv(cfg.CAMERA_SOURCE_ENV, "rtsp://user:secret@cam/stream")

    config = load_runtime_config(make_config_dir())

    assert config.node.camera.source == "rtsp://user:secret@cam/stream"


def test_resolve_path_anchors_relative_paths_to_repository_root(make_config_dir: Callable[..., Path]) -> None:
    config = load_runtime_config(make_config_dir())

    assert config.resolve_path("models/x.onnx") == config.base_dir / "models/x.onnx"
    assert config.resolve_path("/abs/x.onnx") == Path("/abs/x.onnx")


def test_fingerprint_is_stable_and_tracks_changes(make_config_dir: Callable[..., Path]) -> None:
    config = load_runtime_config(make_config_dir())

    changed = apply_overrides(config, device="cpu")

    assert config.fingerprint() == load_runtime_config(make_config_dir()).fingerprint()
    assert len(config.fingerprint()) == cfg.CONFIG_HASH_LENGTH
    assert changed.fingerprint() != config.fingerprint()


def test_apply_overrides_returns_new_config(make_config_dir: Callable[..., Path]) -> None:
    config = load_runtime_config(make_config_dir())

    updated = apply_overrides(config, source="clip.mp4", device="cpu", preview_enabled=False)

    assert (updated.node.camera.source, updated.vision.inference.device) == ("clip.mp4", "cpu")
    assert updated.vision.preview.enabled is False
    assert config.node.camera.source == "0"
    assert apply_overrides(config) == config


def test_apply_overrides_revalidates() -> None:
    config = load_runtime_config(REPO_CONFIG_DIR)

    with pytest.raises(ConfigError, match="inference.device"):
        apply_overrides(config, device="tpu")


def test_config_objects_are_frozen() -> None:
    config = load_runtime_config(REPO_CONFIG_DIR)

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.node.camera.source = "1"  # type: ignore[misc]


# --- error handling -----------------------------------------------------------------------


def test_missing_config_dir_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="directory not found"):
        load_runtime_config(tmp_path / "absent")


def test_missing_config_file_is_rejected(make_config_dir: Callable[..., Path]) -> None:
    directory = make_config_dir()
    (directory / "vision_config.yaml").unlink()

    with pytest.raises(ConfigError, match="vision_config.yaml"):
        load_runtime_config(directory)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("node: [unclosed", "Invalid YAML"),
        ("- a\n- b\n", "must be a mapping"),
        ("", "Missing required config key: node.node_id"),
        ("node: {node_id: n}\ncamera: {camera_id: c, source: '0'}\nextra: 1\n", "Unknown config key"),
        ("node: {node_id: n}\ncamera: {camera_id: c, source: '0', fps_typo: 3}\n", "section 'camera'"),
        ("node: 5\ncamera: {camera_id: c, source: '0'}\n", "must be a mapping"),
    ],
)
def test_malformed_node_config_is_rejected(
    make_config_dir: Callable[..., Path], content: str, message: str
) -> None:
    directory = make_config_dir()
    (directory / "node_config.yaml").write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_runtime_config(directory)


def test_build_config_rejects_non_dataclass() -> None:
    with pytest.raises(TypeError):
        build_config(dict, {})


NODE_BASE: dict[str, Any] = {"node": {"node_id": "n"}, "camera": {"camera_id": "c", "source": "0"}}


@pytest.mark.parametrize(
    ("section", "values", "message"),
    [
        ("node", {"node_id": " "}, "node.node_id"),
        ("camera", {"source": ""}, "camera.source"),
        ("camera", {"width": 0}, "camera.width"),
        ("camera", {"height": "720"}, "camera.height"),
        ("camera", {"fps": -1}, "camera.fps"),
        ("camera", {"reconnect_attempts": -1}, "reconnect_attempts"),
        ("camera", {"reconnect_attempts": True}, "reconnect_attempts"),
        ("camera", {"reconnect_delay_s": -1}, "reconnect_delay_s"),
        ("camera", {"reconnect_max_delay_s": "x"}, "reconnect_max_delay_s"),
        ("camera", {"open_timeout_ms": 0}, "open_timeout_ms"),
        ("camera", {"read_timeout_ms": 1.5}, "read_timeout_ms"),
        ("storage", {"data_dir": ""}, "storage.data_dir"),
        ("storage", {"database_file": " "}, "storage.database_file"),
        ("storage", {"snapshots": {"enabled": "yes"}}, "snapshots.enabled"),
        ("storage", {"snapshots": {"event_types": ["intrusion"]}}, "snapshots.event_types"),
        ("storage", {"snapshots": {"jpeg_quality": 101}}, "snapshots.jpeg_quality"),
        ("storage", {"snapshots": {"min_free_disk_mb": -5}}, "snapshots.min_free_disk_mb"),
        ("monitoring", {"stats_interval_s": 0}, "monitoring.stats_interval_s"),
    ],
)
def test_invalid_node_values_fail_validation(section: str, values: dict[str, Any], message: str) -> None:
    node = build_config(NodeConfig, {**NODE_BASE, section: {**NODE_BASE.get(section, {}), **values}})
    vision = load_runtime_config(REPO_CONFIG_DIR).vision
    config = cfg.RuntimeConfig(base_dir=REPO_ROOT, config_dir=REPO_CONFIG_DIR, node=node, vision=vision)

    with pytest.raises(ConfigError, match=message):
        cfg.validate_config(config)


@pytest.mark.parametrize(
    ("section", "values", "message"),
    [
        ("model", {"version": ""}, "model.model_id, family, version"),
        ("model", {"path": ""}, "model.path"),
        ("model", {"sha256": "abc"}, "model.sha256"),
        ("model", {"license": ""}, "model.source and model.license"),
        ("model", {"input_size": [640]}, "model.input_size"),
        ("inference", {"backend": ""}, "inference.backend"),
        ("inference", {"device": "tpu"}, "inference.device"),
        ("inference", {"confidence_threshold": 0}, "confidence_threshold"),
        ("inference", {"nms_iou_threshold": 1.5}, "nms_iou_threshold"),
        ("sampling", {"inference_stride": 0}, "inference_stride"),
        ("tracking", {"tracker": "magic"}, "tracking.tracker"),
        ("tracking", {"iou_threshold": 0}, "tracking.iou_threshold"),
        ("tracking", {"min_hits": 0}, "tracking.min_hits"),
        ("tracking", {"max_lost_frames": -1}, "tracking.max_lost_frames"),
        ("events", {"detection_event_classes": [""]}, "detection_event_classes"),
        ("events", {"zone_entry_frames": 0}, "zone_entry_frames"),
        ("events", {"zone_exit_frames": 1.5}, "zone_exit_frames"),
        ("preview", {"enabled": "yes"}, "preview.enabled"),
    ],
)
def test_invalid_vision_values_fail_validation(
    make_config_dir: Callable[..., Path], section: str, values: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_runtime_config(make_config_dir(vision={section: values}))


def test_validation_lists_every_problem(make_config_dir: Callable[..., Path]) -> None:
    directory = make_config_dir(vision={"inference": {"device": "tpu", "confidence_threshold": 5}})

    with pytest.raises(ConfigError) as excinfo:
        load_runtime_config(directory)

    assert "inference.device" in str(excinfo.value)
    assert "confidence_threshold" in str(excinfo.value)


def test_unpinned_model_hash_is_allowed(make_config_dir: Callable[..., Path]) -> None:
    config = load_runtime_config(make_config_dir(vision={"model": {"sha256": None}}))

    assert config.vision.model.sha256 is None
    assert isinstance(config.vision, VisionConfig)


# --- zones ------------------------------------------------------------------------------------

SQUARE = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


def zone(**overrides: Any) -> dict[str, Any]:
    return {"zone_id": "zone-a", "name": "A", "polygon": SQUARE, **overrides}


def load_zones(make_config_dir: Callable[..., Path], zones: Any) -> cfg.RuntimeConfig:
    return load_runtime_config(make_config_dir(node={"zones": zones}))


def test_zones_load_with_defaults_and_tuple_coordinates(make_config_dir: Callable[..., Path]) -> None:
    config = load_zones(make_config_dir, [zone(), zone(zone_id="zone-b", anchor="center", classes=["car"])])

    first, second = config.node.zones
    assert first == cfg.ZoneConfig(
        "zone-a", "A", tuple(tuple(v) for v in SQUARE), ("person",), "bottom_center"
    )
    assert (second.anchor, second.classes) == ("center", ("car",))


def test_empty_zone_list_is_allowed(make_config_dir: Callable[..., Path]) -> None:
    assert load_zones(make_config_dir, []).node.zones == ()


@pytest.mark.parametrize(
    ("zones", "message"),
    [
        (None, "zones must be a list"),
        ([zone(), zone()], "duplicate zone_id"),
        ([zone(name="")], "zone_id and name are required"),
        ([zone(classes=[])], "classes must be a non-empty list"),
        ([zone(anchor="head")], "anchor must be one of"),
        ([zone(polygon=[[0.1, 0.1], [0.9, 0.1]])], "at least 3 vertices"),
        ([zone(polygon=[[0.1, 0.1], [1.5, 0.1], [0.9, 0.9]])], "normalised to \\[0, 1\\]"),
        ([zone(polygon=[[0.1, 0.1], [0.9], [0.9, 0.9]])], "list of \\[x, y\\] pairs"),
        ([zone(polygon="square")], "list of \\[x, y\\] pairs"),
        ([zone(polygon=[[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])], "area is"),
        ([zone(polygon=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.9, 0.9]])], "repeats a vertex"),
        ([zone(polygon=[[0.1, 0.1], [0.9, 0.9], [0.9, 0.1], [0.1, 0.9]])], "edges cross"),
    ],
)
def test_invalid_zones_fail_validation(
    make_config_dir: Callable[..., Path], zones: Any, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_zones(make_config_dir, zones)


def test_zone_entries_must_be_mappings_with_required_keys(make_config_dir: Callable[..., Path]) -> None:
    with pytest.raises(ConfigError, match=r"zones\[0\]' must be a mapping"):
        load_zones(make_config_dir, ["zone-a"])
    with pytest.raises(ConfigError, match=r"Missing required config key: zones\[0\].polygon"):
        load_zones(make_config_dir, [{"zone_id": "z", "name": "Z"}])

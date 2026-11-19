"""End-to-end runs of the ``sentinela-runtime`` entry point against recorded video."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sentinela_runtime import main as main_module
from sentinela_runtime.vision.inference.base import InferenceError
from sentinela_runtime.vision.pipeline import PipelineError

from support import PERSON, REPO_MODEL, FakeBackend


def headless(video: Path, **vision: Any) -> dict[str, Any]:
    return {
        "node": {"camera": {"source": str(video), "reconnect_attempts": 0}},
        "vision": {"preview": {"enabled": False}, **vision},
    }


def read_events(base_dir: Path) -> list[dict[str, Any]]:
    path = base_dir / "logs" / "events.jsonl"
    return (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    )


def test_end_to_end_with_scripted_backend_writes_structured_events(
    recorded_video: Path, make_config_dir: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = make_config_dir(**headless(recorded_video))
    backend = FakeBackend(lambda _n: (PERSON,))
    monkeypatch.setattr(main_module, "load_backend", lambda *_a: backend)

    exit_code = main_module.main(["--config-dir", str(config_dir)])

    events = read_events(config_dir.parent)
    assert exit_code == main_module.EXIT_OK
    assert backend.closed
    assert len(events) == 1
    assert events[0]["object"]["class"] == "person"
    assert events[0]["processing"]["inference_backend"] == "fake-cpu"
    assert events[0]["processing"]["model_id"] == "fake-model"


def test_invalid_configuration_exits_with_config_error(make_config_dir: Callable[..., Path]) -> None:
    config_dir = make_config_dir(vision={"inference": {"confidence_threshold": 7}})

    assert main_module.main(["--config-dir", str(config_dir)]) == main_module.EXIT_CONFIG_ERROR


def test_missing_model_exits_with_model_error(
    recorded_video: Path, make_config_dir: Callable[..., Path]
) -> None:
    config_dir = make_config_dir(**headless(recorded_video))  # models/ does not exist under tmp_path

    assert main_module.main(["--config-dir", str(config_dir)]) == main_module.EXIT_MODEL_ERROR


@pytest.mark.parametrize(
    ("source", "backend", "expected"),
    [
        ("missing.avi", FakeBackend(), main_module.EXIT_CAMERA_ERROR),
        (None, FakeBackend(lambda _n: InferenceError("dead")), main_module.EXIT_INFERENCE_ERROR),
    ],
)
def test_runtime_failures_map_to_exit_codes(
    recorded_video: Path,
    make_config_dir: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    source: str | None,
    backend: FakeBackend,
    expected: int,
) -> None:
    config_dir = make_config_dir(**headless(recorded_video))
    monkeypatch.setattr(main_module, "load_backend", lambda *_a: backend)
    monkeypatch.setattr("sentinela_runtime.vision.pipeline.MAX_CONSECUTIVE_INFERENCE_ERRORS", 2)
    args = ["--config-dir", str(config_dir), *(["--source", str(config_dir / source)] if source else [])]

    assert main_module.main(args) == expected
    assert backend.closed


def test_cli_flags_override_configuration(
    recorded_video: Path, make_config_dir: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(config: Any, _lifecycle: Any) -> int:
        captured["config"] = config
        return 0

    monkeypatch.setattr(main_module, "run_vision", fake_run)

    config_dir = str(make_config_dir())
    flags = ["--source", str(recorded_video), "--device", "cpu", "--no-preview", "--log-level", "DEBUG"]

    main_module.main(["--config-dir", config_dir, *flags])

    config = captured["config"]
    assert config.node.camera.source == str(recorded_video)
    assert config.vision.inference.device == "cpu"
    assert config.vision.preview.enabled is False


@pytest.mark.model
@pytest.mark.skipif(not REPO_MODEL.is_file(), reason="models/yolox_s.onnx not downloaded")
def test_real_yolox_model_runs_offline_on_recorded_video(
    recorded_video: Path, make_config_dir: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("onnxruntime")
    config_dir = make_config_dir(**headless(recorded_video, model={"path": str(REPO_MODEL)}))

    exit_code = main_module.main(["--config-dir", str(config_dir), "--device", "cpu"])

    assert exit_code == main_module.EXIT_OK
    assert read_events(config_dir.parent) == []  # synthetic frames contain no people


def test_pipeline_error_type_is_exported() -> None:
    assert issubclass(PipelineError, RuntimeError)

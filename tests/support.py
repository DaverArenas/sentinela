"""Test helpers: synthetic recorded-video fixtures, config merging and a scripted inference backend."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sentinela_runtime.vision.inference.base import (
    Detection,
    InferenceBackend,
    ModelMetadata,
    RuntimeInfo,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_CONFIG_DIR = REPO_ROOT / "config"
REPO_MODEL = REPO_ROOT / "models" / "yolox_s.onnx"
FIXTURE_SIZE = (96, 64)  # width, height

PERSON = Detection(0, "person", 0.91, (10.0, 5.0, 40.0, 60.0))
CAR = Detection(2, "car", 0.80, (50.0, 20.0, 90.0, 60.0))


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def write_video(path: Path, frames: int = 12, size: tuple[int, int] = FIXTURE_SIZE) -> Path:
    """Write a deterministic synthetic MJPG clip (no real footage is committed)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size)
    assert writer.isOpened(), "OpenCV build cannot write MJPG video"
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), (i * 20) % 255, dtype=np.uint8))
    writer.release()
    return path


class FakeBackend(InferenceBackend):
    """Scripted backend: returns ``script(frame_number)`` detections, or raises what it returns."""

    name = "fake"

    def __init__(self, script: Callable[[int], Sequence[Detection] | Exception] = lambda _n: ()) -> None:
        self._script = script
        self.calls = 0
        self.loaded_from: Path | None = None
        self.closed = False

    def load_model(self, model_path: str | Path) -> None:
        self.loaded_from = Path(model_path)

    def predict(self, frame: np.ndarray) -> tuple[Detection, ...]:
        result = self._script(self.calls)
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return tuple(result)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="fake-model",
            family="fake",
            version="0.0.1",
            format="none",
            source="tests",
            license="test-only",
            sha256="0" * 64,
            input_size=(64, 64),
            class_names=("person", "bicycle", "car"),
        )

    def get_runtime_info(self) -> RuntimeInfo:
        return RuntimeInfo(backend="fake", device="cpu", providers=("FakeProvider",))

    def close(self) -> None:
        self.closed = True

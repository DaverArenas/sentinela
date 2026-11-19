"""Common inference interface and Sentinela's canonical detection format.

Application code (pipeline, events, future tracking/zones) depends only on this module,
never on a specific inference engine or model output layout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import numpy as np


class InferenceError(RuntimeError):
    """Raised when a model cannot be loaded or executed."""


class InvalidInputError(InferenceError):
    """Raised when a frame does not satisfy the backend input contract."""


@dataclass(frozen=True)
class Detection:
    """Canonical detection: pixel coordinates in the original (un-resized) frame."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    family: str
    version: str
    format: str
    source: str
    license: str
    sha256: str | None
    input_size: tuple[int, int]
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeInfo:
    backend: str  # e.g. "onnxruntime"
    device: str  # resolved device actually in use: "cuda" | "cpu"
    providers: tuple[str, ...]

    @property
    def label(self) -> str:
        """Compact identifier recorded in event provenance, e.g. ``onnxruntime-cuda``."""
        return f"{self.backend}-{self.device}"


def validate_frame(frame: object) -> np.ndarray:
    """Ensure ``frame`` is an HxWx3 uint8 BGR image; return it typed."""
    if not isinstance(frame, np.ndarray):
        raise InvalidInputError(f"Frame must be a numpy array, got {type(frame).__name__}")
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise InvalidInputError(f"Frame must have shape HxWx3, got {frame.shape}")
    if frame.dtype != np.uint8:
        raise InvalidInputError(f"Frame must be uint8, got {frame.dtype}")
    return frame


class InferenceBackend(ABC):
    """Hardware-independent inference contract implemented by every backend adapter."""

    name: ClassVar[str]

    @abstractmethod
    def load_model(self, model_path: str | Path) -> None:
        """Load and initialise the model; must be called before ``predict``."""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> tuple[Detection, ...]:
        """Run inference on one BGR frame and return canonical detections."""

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Model provenance and I/O description."""

    @abstractmethod
    def get_runtime_info(self) -> RuntimeInfo:
        """Engine/device actually used (may differ from the requested device after fallback)."""

    @abstractmethod
    def close(self) -> None:
        """Release engine resources. Safe to call more than once."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

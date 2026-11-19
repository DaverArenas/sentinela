"""Model-family adapters: the model-specific pre/post-processing an engine backend delegates to.

Engines (ONNX Runtime today; TensorRT/HailoRT later) only move tensors. Adapters know a model
family's input convention and output layout and convert raw outputs into canonical detections.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Protocol

import numpy as np

from sentinela_runtime.vision.inference.backends.yolox import YoloxAdapter
from sentinela_runtime.vision.inference.base import Detection, InferenceError
from sentinela_runtime.vision.preprocessing.letterbox import LetterboxInfo


class ModelAdapter(Protocol):
    family: str

    @property
    def class_names(self) -> tuple[str, ...]: ...

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, LetterboxInfo]: ...

    def postprocess(
        self,
        output: np.ndarray,
        info: LetterboxInfo,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> tuple[Detection, ...]: ...


ADAPTERS: Mapping[str, Callable[[tuple[int, int]], ModelAdapter]] = MappingProxyType({"yolox": YoloxAdapter})


def create_adapter(family: str, input_hw: tuple[int, int]) -> ModelAdapter:
    factory = ADAPTERS.get(family)
    if factory is None:
        raise InferenceError(f"Unsupported model family {family!r}; available: {sorted(ADAPTERS)}")
    return factory(input_hw)

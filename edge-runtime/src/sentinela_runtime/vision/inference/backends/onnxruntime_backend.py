"""ONNX Runtime backend (CUDA execution provider on NVIDIA GPUs, CPU elsewhere)."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from sentinela_runtime.core.config import InferenceConfig, ModelConfig
from sentinela_runtime.core.log import get_logger
from sentinela_runtime.vision.inference.backends.adapters import ModelAdapter, create_adapter
from sentinela_runtime.vision.inference.base import (
    Detection,
    InferenceBackend,
    InferenceError,
    ModelMetadata,
    RuntimeInfo,
    validate_frame,
)

log = get_logger("inference.onnxruntime")

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
ORT_LOG_SEVERITY_ERROR = 3
INSTALL_HINT = "install an ONNX Runtime build: `uv sync --extra cuda` (NVIDIA) or `uv sync --extra cpu`"


def select_providers(device: str, available: tuple[str, ...]) -> tuple[str, ...]:
    """Map the configured device onto execution providers, degrading to CPU when CUDA is absent."""
    if device == "cpu":
        return (CPU_PROVIDER,)
    if CUDA_PROVIDER in available:
        return (CUDA_PROVIDER, CPU_PROVIDER)
    if device == "cuda":
        log.warning("CUDA requested but %s is unavailable; falling back to CPU", CUDA_PROVIDER)
    return (CPU_PROVIDER,)


def _import_onnxruntime() -> ModuleType:
    try:
        return importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise InferenceError(f"onnxruntime is not installed; {INSTALL_HINT}") from exc


class OnnxRuntimeBackend(InferenceBackend):
    name = "onnxruntime"

    def __init__(self, model: ModelConfig, inference: InferenceConfig, ort_module: Any = None) -> None:
        self._model_cfg = model
        self._inference_cfg = inference
        self._ort = ort_module
        self._session: Any = None
        self._input_name = ""
        self._adapter: ModelAdapter | None = None
        self._runtime: RuntimeInfo | None = None

    def load_model(self, model_path: str | Path) -> None:
        ort = self._ort or _import_onnxruntime()
        adapter = create_adapter(self._model_cfg.family, self._model_cfg.input_size)
        providers = select_providers(self._inference_cfg.device, tuple(ort.get_available_providers()))
        if CUDA_PROVIDER in providers and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # load pip-installed CUDA/cuDNN libraries before session creation

        options = ort.SessionOptions()
        options.log_severity_level = ORT_LOG_SEVERITY_ERROR
        try:
            session = ort.InferenceSession(str(model_path), sess_options=options, providers=list(providers))
        except Exception as exc:  # ORT raises engine-specific exception types
            raise InferenceError(f"Failed to load model {model_path}: {exc}") from exc

        active = tuple(session.get_providers())
        device = "cuda" if CUDA_PROVIDER in active else "cpu"
        if CUDA_PROVIDER in providers and device == "cpu":
            log.warning("CUDA execution provider failed to initialise; running on CPU")

        self._check_input(session)
        self._session, self._adapter = session, adapter
        self._input_name = session.get_inputs()[0].name
        self._runtime = RuntimeInfo(backend=self.name, device=device, providers=active)

    def _check_input(self, session: Any) -> None:
        shape = tuple(session.get_inputs()[0].shape)
        expected_hw = self._model_cfg.input_size
        static_hw = tuple(shape[2:4])
        if all(isinstance(v, int) for v in static_hw) and static_hw != expected_hw:
            raise InferenceError(f"Model input is {static_hw} but model.input_size is {expected_hw}")

    def predict(self, frame: np.ndarray) -> tuple[Detection, ...]:
        if self._session is None or self._adapter is None:
            raise InferenceError("Model not loaded; call load_model() first")
        image = validate_frame(frame)
        tensor, info = self._adapter.preprocess(image)
        try:
            outputs = self._session.run(None, {self._input_name: tensor})
        except Exception as exc:
            raise InferenceError(f"Inference failed: {exc}") from exc
        return self._adapter.postprocess(
            np.asarray(outputs[0]),
            info,
            self._inference_cfg.confidence_threshold,
            self._inference_cfg.nms_iou_threshold,
        )

    def get_metadata(self) -> ModelMetadata:
        cfg = self._model_cfg
        class_names = self._adapter.class_names if self._adapter else ()
        return ModelMetadata(
            model_id=cfg.model_id,
            family=cfg.family,
            version=cfg.version,
            format=cfg.format,
            source=cfg.source,
            license=cfg.license,
            sha256=cfg.sha256,
            input_size=cfg.input_size,
            class_names=class_names,
        )

    def get_runtime_info(self) -> RuntimeInfo:
        if self._runtime is None:
            raise InferenceError("Model not loaded; runtime info unavailable")
        return self._runtime

    def close(self) -> None:
        self._session = None
        self._adapter = None

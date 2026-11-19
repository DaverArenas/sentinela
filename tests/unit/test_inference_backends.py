"""Inference interface, YOLOX adapter, ONNX Runtime backend (with a fake engine) and model manager."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sentinela_runtime.core.config import load_runtime_config
from sentinela_runtime.vision.inference import model_manager
from sentinela_runtime.vision.inference.backends import onnxruntime_backend as ortb
from sentinela_runtime.vision.inference.backends.adapters import create_adapter
from sentinela_runtime.vision.inference.backends.yolox import YoloxAdapter, make_grids
from sentinela_runtime.vision.inference.base import (
    InferenceError,
    InvalidInputError,
    RuntimeInfo,
    validate_frame,
)
from sentinela_runtime.vision.preprocessing.letterbox import LetterboxInfo

from support import REPO_CONFIG_DIR, FakeBackend

CONFIG = load_runtime_config(REPO_CONFIG_DIR)
MODEL_CFG, INFERENCE_CFG = CONFIG.vision.model, CONFIG.vision.inference
SMALL_HW = (64, 64)  # 8x8 + 4x4 + 2x2 = 84 anchors
NUM_CLASSES = 80


# --- base ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frame",
    [
        [[1, 2, 3]],
        np.zeros((4, 4), np.uint8),
        np.zeros((4, 4, 4), np.uint8),
        np.zeros((0, 4, 3), np.uint8),
        np.zeros((4, 4, 3), np.float32),
    ],
)
def test_validate_frame_rejects_bad_input(frame: Any) -> None:
    with pytest.raises(InvalidInputError):
        validate_frame(frame)


def test_runtime_info_label_and_backend_context_manager() -> None:
    backend = FakeBackend()

    with backend as entered:
        assert entered is backend

    assert backend.closed
    assert RuntimeInfo("onnxruntime", "cuda", ()).label == "onnxruntime-cuda"


# --- YOLOX adapter ----------------------------------------------------------------------------


def raw_output(rows: dict[int, tuple[float, float, float, float, float, int, float]]) -> np.ndarray:
    """Build a 1xNx85 YOLOX output with given ``anchor -> (dx, dy, log_w, log_h, obj, cls, score)``."""
    output = np.zeros((1, len(make_grids(SMALL_HW)[0]), 5 + NUM_CLASSES), dtype=np.float32)
    for anchor, (dx, dy, lw, lh, obj, cls, score) in rows.items():
        output[0, anchor, :5] = (dx, dy, lw, lh, obj)
        output[0, anchor, 5 + cls] = score
    return output


IDENTITY = LetterboxInfo(scale=1.0, pad_x=0, pad_y=0, source_hw=SMALL_HW)


def test_make_grids_matches_yolox_anchor_layout() -> None:
    grids, strides = make_grids((640, 640))

    assert grids.shape == (8400, 2) and strides.shape == (8400, 1)
    assert grids[1].tolist() == [1, 0] and grids[80].tolist() == [0, 1]
    assert strides[0, 0] == 8 and strides[-1, 0] == 32


def test_yolox_postprocess_decodes_boxes_scores_and_classes() -> None:
    adapter = YoloxAdapter(SMALL_HW)
    # Anchor 64 is the first stride-16 cell (x=0, y=0); centre = (0.5 + 0) * 16 = 8, size = 2 * 16 = 32.
    output = raw_output({64: (0.5, 0.5, math.log(2), math.log(2), 0.9, 0, 0.8)})

    (det,) = adapter.postprocess(output, IDENTITY, 0.5, 0.45)

    assert (det.class_id, det.class_name) == (0, "person")
    assert det.confidence == pytest.approx(0.72)
    assert det.bbox_xyxy == pytest.approx((0, 0, 24, 24))  # (-8, -8, 24, 24) clipped to the frame


def test_yolox_postprocess_applies_threshold_and_nms() -> None:
    adapter = YoloxAdapter(SMALL_HW)
    output = raw_output(
        {
            10: (0, 0, 1, 1, 0.9, 2, 0.9),  # car
            11: (-1, 0, 1, 1, 0.9, 2, 0.6),  # same car, overlapping, lower score -> suppressed
            30: (0, 0, 1, 1, 0.3, 0, 0.3),  # below threshold
        }
    )

    detections = adapter.postprocess(output, IDENTITY, 0.5, 0.45)

    assert [(d.class_name, round(d.confidence, 2)) for d in detections] == [("car", 0.81)]


def test_yolox_postprocess_returns_empty_when_nothing_passes() -> None:
    assert YoloxAdapter(SMALL_HW).postprocess(raw_output({}), IDENTITY, 0.5, 0.45) == ()


def test_yolox_rejects_unexpected_output_shape_and_input_size() -> None:
    with pytest.raises(InferenceError, match="output shape"):
        YoloxAdapter(SMALL_HW).postprocess(np.zeros((1, 10, 85), np.float32), IDENTITY, 0.5, 0.45)
    with pytest.raises(InferenceError, match="multiple of 32"):
        YoloxAdapter((60, 64))


def test_yolox_preprocess_letterboxes_top_left_with_114() -> None:
    tensor, info = YoloxAdapter(SMALL_HW).preprocess(np.zeros((32, 64, 3), np.uint8))

    assert tensor.shape == (1, 3, 64, 64)
    assert (tensor[0, :, :32] == 0).all() and (tensor[0, :, 32:] == 114).all()
    assert info.scale == 1.0


def test_create_adapter_rejects_unknown_family() -> None:
    assert create_adapter("yolox", SMALL_HW).family == "yolox"
    with pytest.raises(InferenceError, match="Unsupported model family"):
        create_adapter("rt-detr", SMALL_HW)


# --- ONNX Runtime backend (fake engine) ----------------------------------------------------------


class FakeSession:
    def __init__(self, providers: list[str], input_shape: list[Any], output: Any) -> None:
        self._providers = providers
        self._input_shape = input_shape
        self._output = output
        self.feeds: list[dict[str, np.ndarray]] = []

    def get_providers(self) -> list[str]:
        return self._providers

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="images", shape=self._input_shape)]

    def run(self, _names: object, feeds: dict[str, np.ndarray]) -> list[Any]:
        self.feeds.append(feeds)
        if isinstance(self._output, Exception):
            raise self._output
        return [self._output]


def fake_ort(
    available: tuple[str, ...] = (ortb.CPU_PROVIDER,),
    active: list[str] | None = None,
    input_shape: list[Any] | None = None,
    output: Any = None,
    session_error: Exception | None = None,
) -> SimpleNamespace:
    created: list[FakeSession] = []

    def inference_session(_path: str, sess_options: Any, providers: list[str]) -> FakeSession:
        if session_error:
            raise session_error
        session = FakeSession(active or providers, input_shape or [1, 3, 64, 64], output)
        created.append(session)
        return session

    return SimpleNamespace(
        get_available_providers=lambda: list(available),
        SessionOptions=lambda: SimpleNamespace(log_severity_level=0),
        InferenceSession=inference_session,
        preload_dlls=lambda: None,
        created=created,
    )


SMALL_MODEL = replace(MODEL_CFG, input_size=SMALL_HW)


@pytest.mark.parametrize(
    ("device", "available", "expected"),
    [
        ("cpu", (ortb.CUDA_PROVIDER, ortb.CPU_PROVIDER), (ortb.CPU_PROVIDER,)),
        ("auto", (ortb.CUDA_PROVIDER, ortb.CPU_PROVIDER), (ortb.CUDA_PROVIDER, ortb.CPU_PROVIDER)),
        ("cuda", (ortb.CUDA_PROVIDER,), (ortb.CUDA_PROVIDER, ortb.CPU_PROVIDER)),
        ("auto", (ortb.CPU_PROVIDER,), (ortb.CPU_PROVIDER,)),
        ("cuda", (ortb.CPU_PROVIDER,), (ortb.CPU_PROVIDER,)),
    ],
)
def test_select_providers_degrades_to_cpu(
    device: str, available: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    assert ortb.select_providers(device, available) == expected


def test_onnxruntime_backend_runs_adapter_pipeline() -> None:
    ort = fake_ort(output=raw_output({64: (0.5, 0.5, 0, 0, 0.9, 0, 0.9)}))
    backend = ortb.OnnxRuntimeBackend(SMALL_MODEL, INFERENCE_CFG, ort_module=ort)
    backend.load_model("model.onnx")

    detections = backend.predict(np.zeros((64, 64, 3), np.uint8))

    assert [d.class_name for d in detections] == ["person"]
    assert ort.created[0].feeds[0]["images"].shape == (1, 3, 64, 64)
    assert backend.get_runtime_info() == RuntimeInfo("onnxruntime", "cpu", (ortb.CPU_PROVIDER,))
    assert backend.get_metadata().class_names[0] == "person"
    assert backend.get_metadata().model_id == MODEL_CFG.model_id


def test_onnxruntime_backend_reports_cuda_and_cpu_fallback() -> None:
    cuda = fake_ort(available=(ortb.CUDA_PROVIDER, ortb.CPU_PROVIDER))
    fallback = fake_ort(available=(ortb.CUDA_PROVIDER, ortb.CPU_PROVIDER), active=[ortb.CPU_PROVIDER])

    on_gpu = ortb.OnnxRuntimeBackend(SMALL_MODEL, INFERENCE_CFG, ort_module=cuda)
    on_gpu.load_model("m.onnx")
    degraded = ortb.OnnxRuntimeBackend(SMALL_MODEL, INFERENCE_CFG, ort_module=fallback)
    degraded.load_model("m.onnx")

    assert on_gpu.get_runtime_info().label == "onnxruntime-cuda"
    assert degraded.get_runtime_info().label == "onnxruntime-cpu"


@pytest.mark.parametrize(
    ("ort", "message"),
    [
        (fake_ort(session_error=RuntimeError("corrupt")), "Failed to load model"),
        (fake_ort(input_shape=[1, 3, 32, 32]), "Model input is"),
    ],
)
def test_onnxruntime_backend_load_errors(ort: SimpleNamespace, message: str) -> None:
    with pytest.raises(InferenceError, match=message):
        ortb.OnnxRuntimeBackend(SMALL_MODEL, INFERENCE_CFG, ort_module=ort).load_model("m.onnx")


def test_onnxruntime_backend_accepts_dynamic_input_shape() -> None:
    backend = ortb.OnnxRuntimeBackend(
        SMALL_MODEL, INFERENCE_CFG, ort_module=fake_ort(input_shape=[1, 3, "h", "w"])
    )

    backend.load_model("m.onnx")


def test_onnxruntime_backend_wraps_run_errors_and_guards_lifecycle() -> None:
    backend = ortb.OnnxRuntimeBackend(
        SMALL_MODEL, INFERENCE_CFG, ort_module=fake_ort(output=RuntimeError("x"))
    )
    frame = np.zeros((64, 64, 3), np.uint8)

    with pytest.raises(InferenceError, match="not loaded"):
        backend.predict(frame)
    with pytest.raises(InferenceError, match="runtime info"):
        backend.get_runtime_info()
    backend.load_model("m.onnx")
    with pytest.raises(InferenceError, match="Inference failed"):
        backend.predict(frame)
    backend.close()
    with pytest.raises(InferenceError, match="not loaded"):
        backend.predict(frame)


def test_missing_onnxruntime_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_module(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr(ortb.importlib, "import_module", no_module)

    with pytest.raises(InferenceError, match="uv sync --extra"):
        ortb.OnnxRuntimeBackend(SMALL_MODEL, INFERENCE_CFG).load_model("m.onnx")


# --- model manager ------------------------------------------------------------------------------


def write_model(tmp_path: Path, payload: bytes = b"weights") -> tuple[Path, str]:
    path = tmp_path / "model.onnx"
    path.write_bytes(payload)
    return path, model_manager.sha256_file(path)


def test_verify_model_file_checks_presence_and_hash(tmp_path: Path) -> None:
    path, digest = write_model(tmp_path)

    assert model_manager.verify_model_file(path, digest.upper(), "src") == digest
    assert model_manager.verify_model_file(path, None, "src") == digest
    with pytest.raises(model_manager.ModelError, match="checksum mismatch"):
        model_manager.verify_model_file(path, "0" * 64, "src")
    with pytest.raises(model_manager.ModelError, match="not found"):
        model_manager.verify_model_file(tmp_path / "absent.onnx", None, "https://example")


def test_load_backend_records_actual_hash_in_metadata(tmp_path: Path) -> None:
    path, digest = write_model(tmp_path)
    received: list[Any] = []

    def factory(model: Any, inference: Any) -> FakeBackend:
        received.append(model)
        return FakeBackend()

    backend = model_manager.load_backend(
        replace(MODEL_CFG, sha256=None), INFERENCE_CFG, path, backends={"onnxruntime": factory}
    )

    assert isinstance(backend, FakeBackend) and backend.loaded_from == path
    assert received[0].sha256 == digest


def test_load_backend_rejects_unknown_backend(tmp_path: Path) -> None:
    path, _ = write_model(tmp_path)

    with pytest.raises(model_manager.ModelError, match="Unknown inference backend"):
        model_manager.load_backend(MODEL_CFG, replace(INFERENCE_CFG, backend="hailort"), path)

"""Model resolution: backend selection, integrity verification and loading.

The SHA-256 check guarantees that the model producing events is exactly the one recorded in
``config/vision_config.yaml`` (and in ``models/README.md``); models are never swapped silently.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from sentinela_runtime.core.config import InferenceConfig, ModelConfig
from sentinela_runtime.core.log import get_logger
from sentinela_runtime.vision.inference.backends.onnxruntime_backend import OnnxRuntimeBackend
from sentinela_runtime.vision.inference.base import InferenceBackend, InferenceError

log = get_logger("inference.models")

HASH_CHUNK_BYTES = 1 << 20

BackendFactory = Callable[[ModelConfig, InferenceConfig], InferenceBackend]
BACKENDS: Mapping[str, BackendFactory] = MappingProxyType({OnnxRuntimeBackend.name: OnnxRuntimeBackend})


class ModelError(InferenceError):
    """Raised when a model is missing, unverified, or its backend is unknown."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_file(path: Path, expected_sha256: str | None, source: str) -> str:
    """Return the file's SHA-256, raising ``ModelError`` if it is missing or differs from the config."""
    if not path.is_file():
        raise ModelError(
            f"Model file not found: {path}. Download it once from {source} (see models/README.md)"
        )
    actual = sha256_file(path)
    if expected_sha256 is None:
        log.warning("model.sha256 is not pinned; using unverified model", extra={"sha256": actual})
    elif actual != expected_sha256.lower():
        raise ModelError(f"Model checksum mismatch for {path}: expected {expected_sha256}, got {actual}")
    return actual


def load_backend(
    model: ModelConfig,
    inference: InferenceConfig,
    model_path: Path,
    backends: Mapping[str, BackendFactory] = BACKENDS,
) -> InferenceBackend:
    """Verify the model file, construct the configured backend and load the model into it."""
    factory = backends.get(inference.backend)
    if factory is None:
        raise ModelError(f"Unknown inference backend {inference.backend!r}; available: {sorted(backends)}")

    actual_sha256 = verify_model_file(model_path, model.sha256, model.source)
    backend = factory(replace(model, sha256=actual_sha256), inference)
    backend.load_model(model_path)

    runtime = backend.get_runtime_info()
    log.info(
        "Model loaded",
        extra={
            "model_id": model.model_id,
            "model_version": model.version,
            "license": model.license,
            "sha256": actual_sha256,
            "inference_backend": runtime.label,
            "providers": list(runtime.providers),
        },
    )
    return backend

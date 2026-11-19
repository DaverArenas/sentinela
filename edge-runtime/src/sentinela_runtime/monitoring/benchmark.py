"""Inference benchmark through the common inference interface (Phase 1 deliverable: RTX 3060).

Measures the configured model on the configured backend/device, frame by frame:
``inference`` (preprocess + model + postprocess, i.e. ``backend.predict``) and the
``tracking + zones`` stage that follows it. Camera I/O and preview are excluded, so results
compare hardware and backends rather than webcams.

Examples (from ``edge-runtime/``)::

    uv run sentinela-benchmark --device cuda
    uv run sentinela-benchmark --device cpu --frames 100
    uv run sentinela-benchmark --source ../path/to/recording.mp4 --output ../logs/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import cycle, islice
from pathlib import Path

import cv2
import numpy as np

from sentinela_runtime.core.config import ConfigError, RuntimeConfig, apply_overrides, load_runtime_config
from sentinela_runtime.core.log import bootstrap_logging, get_logger
from sentinela_runtime.monitoring.metrics import peak_rss_mib
from sentinela_runtime.vision.inference.base import InferenceBackend, InferenceError
from sentinela_runtime.vision.inference.model_manager import load_backend
from sentinela_runtime.vision.tracking import create_tracker
from sentinela_runtime.vision.zones.zone_evaluator import ZoneEvaluator

log = get_logger("benchmark")

SYNTHETIC_HW = (720, 1280)
MAX_SOURCE_FRAMES = 120  # frames decoded from --source and replayed (bounded memory)
EXIT_OK, EXIT_CONFIG_ERROR, EXIT_MODEL_ERROR = 0, 2, 3


@dataclass(frozen=True)
class Stats:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    @classmethod
    def of(cls, samples_ms: Sequence[float]) -> Stats:
        values = np.asarray(samples_ms, dtype=np.float64)
        p50, p95, p99 = (float(np.percentile(values, q)) for q in (50, 95, 99))
        mean, peak = float(np.mean(values)), max(samples_ms)
        return cls(*(round(float(v), 2) for v in (mean, p50, p95, p99, peak)))


@dataclass(frozen=True)
class BenchmarkResult:
    model_id: str
    model_version: str
    inference_backend: str
    providers: tuple[str, ...]
    frame_hw: tuple[int, int]
    frames: int
    warmup: int
    inference: Stats
    tracking_zones: Stats
    throughput_fps: float  # frames / (inference + tracking + zones) time
    detections_per_frame: float
    peak_rss_mib: float


def synthetic_frames(frame_hw: tuple[int, int] = SYNTHETIC_HW, seed: int = 0) -> list[np.ndarray]:
    """Deterministic noise frame (reproducible across runs and machines)."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (*frame_hw, 3), dtype=np.uint8)]


def video_frames(path: Path, limit: int = MAX_SOURCE_FRAMES) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while len(frames) < limit:
            ok, image = capture.read()
            if not ok or image is None:
                break
            frames.append(image)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames could be read from {path}")
    return frames


def run_benchmark(
    config: RuntimeConfig,
    backend: InferenceBackend,
    frames: Sequence[np.ndarray],
    iterations: int,
    warmup: int,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkResult:
    for image in islice(cycle(frames), warmup):  # CUDA kernels, memory pools, caches
        backend.predict(image)

    tracker = create_tracker(config.vision.tracking)
    events = config.vision.events
    zones = ZoneEvaluator.from_config(config.node.zones, events.zone_entry_frames, events.zone_exit_frames)
    inference_ms: list[float] = []
    tracking_ms: list[float] = []
    detections = 0
    for image in islice(cycle(frames), iterations):
        started = clock()
        found = backend.predict(image)
        inferred = clock()
        zones.update(tracker.update(found), (image.shape[0], image.shape[1]))
        finished = clock()
        inference_ms.append((inferred - started) * 1000)
        tracking_ms.append((finished - inferred) * 1000)
        detections += len(found)

    metadata, runtime = backend.get_metadata(), backend.get_runtime_info()
    total_s = (sum(inference_ms) + sum(tracking_ms)) / 1000
    return BenchmarkResult(
        model_id=metadata.model_id,
        model_version=metadata.version,
        inference_backend=runtime.label,
        providers=runtime.providers,
        frame_hw=(frames[0].shape[0], frames[0].shape[1]),
        frames=iterations,
        warmup=warmup,
        inference=Stats.of(inference_ms),
        tracking_zones=Stats.of(tracking_ms),
        throughput_fps=round(iterations / max(total_s, 1e-9), 1),
        detections_per_frame=round(detections / iterations, 2),
        peak_rss_mib=round(peak_rss_mib(), 1),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sentinela-benchmark", description="Sentinela inference benchmark")
    parser.add_argument("--config-dir", help="Directory with node/vision YAML (default: <repo>/config)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], help="Override inference device")
    parser.add_argument(
        "--source", type=Path, help="Recorded video to benchmark on (default: synthetic 720p)"
    )
    parser.add_argument("--frames", type=_positive_int, default=300, help="Measured frames (default 300)")
    parser.add_argument("--warmup", type=_non_negative_int, default=30, help="Unmeasured warm-up frames")
    parser.add_argument("--output", type=Path, help="Also write the result as JSON to this file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    bootstrap_logging()
    args = parse_args(argv)
    try:
        config = apply_overrides(load_runtime_config(args.config_dir), device=args.device)
        frames = video_frames(args.source) if args.source else synthetic_frames()
    except (ConfigError, ValueError) as exc:
        log.error("%s", exc)
        return EXIT_CONFIG_ERROR
    try:
        backend = load_backend(
            config.vision.model, config.vision.inference, config.resolve_path(config.vision.model.path)
        )
    except InferenceError as exc:
        log.error("Model initialisation failed: %s", exc)
        return EXIT_MODEL_ERROR

    with backend:
        result = run_benchmark(config, backend, frames, args.frames, args.warmup)
    payload = asdict(result)
    log.info(
        "Benchmark %s on %s: inference mean %.2f ms / p95 %.2f ms, tracking+zones mean %.3f ms, %.1f FPS",
        result.model_id,
        result.inference_backend,
        result.inference.mean_ms,
        result.inference.p95_ms,
        result.tracking_zones.mean_ms,
        result.throughput_fps,
        extra={"result": payload},
    )
    log.info("Benchmark result: %s", json.dumps(payload))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return EXIT_OK


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number


if __name__ == "__main__":
    raise SystemExit(main())

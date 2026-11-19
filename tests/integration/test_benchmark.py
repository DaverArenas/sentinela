"""Benchmark command: measurement loop, frame sources and entry point."""

from __future__ import annotations

import json
from collections.abc import Callable
from itertools import count
from pathlib import Path

import pytest

from sentinela_runtime.core.config import load_runtime_config
from sentinela_runtime.monitoring import benchmark as bench
from sentinela_runtime.vision.inference.base import InferenceError

from support import REPO_CONFIG_DIR, REPO_MODEL, FakeBackend, walker


def test_run_benchmark_separates_inference_and_tracking_time() -> None:
    ticks = count()
    clock = lambda: next(ticks) * 0.001  # noqa: E731 - each clock read advances 1 ms
    backend = FakeBackend(walker())

    result = bench.run_benchmark(
        load_runtime_config(REPO_CONFIG_DIR), backend, bench.synthetic_frames((64, 96)), 10, 2, clock
    )

    assert backend.calls == 12  # 2 warm-up + 10 measured
    assert (result.frames, result.warmup, result.frame_hw) == (10, 2, (64, 96))
    assert result.inference.mean_ms == pytest.approx(1.0)
    assert result.tracking_zones.p95_ms == pytest.approx(1.0)
    assert result.throughput_fps == pytest.approx(500.0)
    assert result.detections_per_frame == 1.0
    assert result.inference_backend == "fake-cpu"


def test_frame_sources(recorded_video: Path, tmp_path: Path) -> None:
    assert bench.synthetic_frames()[0].shape == (*bench.SYNTHETIC_HW, 3)
    assert len(bench.video_frames(recorded_video, limit=5)) == 5
    assert len(bench.video_frames(recorded_video)) == 12
    with pytest.raises(ValueError, match="No frames"):
        bench.video_frames(tmp_path / "missing.avi")


def test_main_writes_json_result(
    recorded_video: Path,
    tmp_path: Path,
    make_config_dir: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bench, "load_backend", lambda *_a: FakeBackend())
    output = tmp_path / "out" / "bench.json"
    args = ["--config-dir", str(make_config_dir()), "--source", str(recorded_video)]

    exit_code = bench.main([*args, "--frames", "3", "--warmup", "0", "--output", str(output)])

    assert exit_code == bench.EXIT_OK
    assert json.loads(output.read_text())["frames"] == 3


@pytest.mark.parametrize(
    ("extra", "expected"),
    [(["--device", "cpu", "--source", "missing.avi"], bench.EXIT_CONFIG_ERROR), ([], bench.EXIT_MODEL_ERROR)],
)
def test_main_failures_map_to_exit_codes(
    make_config_dir: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, extra: list[str], expected: int
) -> None:
    def broken(*_a: object) -> FakeBackend:
        raise InferenceError("no model")

    monkeypatch.setattr(bench, "load_backend", broken)

    assert bench.main(["--config-dir", str(make_config_dir()), *extra]) == expected


@pytest.mark.parametrize("flag", [["--frames", "0"], ["--warmup", "-1"]])
def test_invalid_counts_are_rejected(flag: list[str]) -> None:
    with pytest.raises(SystemExit):
        bench.parse_args(flag)


@pytest.mark.model
@pytest.mark.skipif(not REPO_MODEL.is_file(), reason="models/yolox_s.onnx not downloaded")
def test_real_model_benchmark_on_cpu(make_config_dir: Callable[..., Path]) -> None:
    pytest.importorskip("onnxruntime")
    config_dir = make_config_dir(vision={"model": {"path": str(REPO_MODEL)}})

    assert (
        bench.main(["--config-dir", str(config_dir), "--device", "cpu", "--frames", "2", "--warmup", "1"])
        == 0
    )

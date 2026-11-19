"""Pipeline performance metrics: latency, throughput, skipped frames, errors and memory."""

from __future__ import annotations

import resource
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

LATENCY_WINDOW = 1_000  # bounded: keep only the most recent inference latencies
KIB_PER_MIB = 1024


@dataclass(frozen=True)
class MetricsSnapshot:
    elapsed_s: float
    frames_read: int
    frames_inferred: int
    frames_skipped: int
    inference_errors: int
    events: int
    reconnects: int
    throughput_fps: float
    latency_mean_ms: float | None
    latency_p95_ms: float | None
    peak_rss_mib: float


def peak_rss_mib() -> float:
    """Peak resident memory of this process (Linux reports ``ru_maxrss`` in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / KIB_PER_MIB


class PipelineMetrics:
    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._started = clock()
        self._latencies_ms: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._frames_read = 0
        self._frames_inferred = 0
        self._inference_errors = 0
        self._events = 0

    def record_inference(self, latency_ms: float, events: int) -> None:
        self._frames_read += 1
        self._frames_inferred += 1
        self._events += events
        self._latencies_ms.append(latency_ms)

    def record_skip(self) -> None:
        self._frames_read += 1

    def record_error(self) -> None:
        self._frames_read += 1
        self._inference_errors += 1

    def snapshot(self, reconnects: int = 0) -> MetricsSnapshot:
        elapsed = max(self._clock() - self._started, 1e-9)
        latencies = np.fromiter(self._latencies_ms, dtype=np.float64)
        has_latency = latencies.size > 0
        return MetricsSnapshot(
            elapsed_s=round(elapsed, 3),
            frames_read=self._frames_read,
            frames_inferred=self._frames_inferred,
            frames_skipped=self._frames_read - self._frames_inferred - self._inference_errors,
            inference_errors=self._inference_errors,
            events=self._events,
            reconnects=reconnects,
            throughput_fps=round(self._frames_read / elapsed, 2),
            latency_mean_ms=round(float(latencies.mean()), 2) if has_latency else None,
            latency_p95_ms=round(float(np.percentile(latencies, 95)), 2) if has_latency else None,
            peak_rss_mib=round(peak_rss_mib(), 1),
        )

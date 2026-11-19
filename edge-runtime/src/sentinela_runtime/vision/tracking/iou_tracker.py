"""IoU tracker: greedy class-aware IoU association with constant-velocity prediction.

Lifecycle of a track:
    tentative  created from an unmatched detection; dropped as soon as it misses a frame
    confirmed  after ``min_hits`` consecutive matches; receives a public ``track-<n>`` ID
    lost       confirmed but unmatched; kept (and its box extrapolated) for up to
               ``max_lost_frames`` inferred frames so short occlusions keep the same ID
    ended      lost for longer than ``max_lost_frames``; reported once in ``TrackingResult.ended``

All counters are in *inferred* frames, which keeps behaviour deterministic on recorded video.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import count

import numpy as np

from sentinela_runtime.core.config import TrackingConfig
from sentinela_runtime.vision.inference.base import Detection
from sentinela_runtime.vision.tracking.base import TrackedObject, TrackingResult

VELOCITY_SMOOTHING = 0.5  # weight of the previous velocity in the exponential moving average
BoxArray = np.ndarray
ZERO_VELOCITY = (0.0, 0.0, 0.0, 0.0)


def iou_matrix(boxes_a: BoxArray, boxes_b: BoxArray) -> np.ndarray:
    """Pairwise IoU between ``(N, 4)`` and ``(M, 4)`` xyxy boxes."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    a, b = boxes_a[:, None, :], boxes_b[None, :, :]
    inter_w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    inter_h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def greedy_match(scores: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Pair rows with columns by descending score; each row/column is used at most once."""
    rows, cols = np.nonzero(scores >= threshold)
    order = np.argsort(-scores[rows, cols], kind="stable")
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for idx in order:
        row, col = int(rows[idx]), int(cols[idx])
        if row not in used_rows and col not in used_cols:
            pairs.append((row, col))
            used_rows.add(row)
            used_cols.add(col)
    return pairs


@dataclass(frozen=True)
class _Track:
    detection: Detection
    velocity: tuple[float, float, float, float] = ZERO_VELOCITY
    hits: int = 1
    misses: int = 0
    track_id: str | None = None  # assigned on confirmation

    @property
    def confirmed(self) -> bool:
        return self.track_id is not None

    def predicted_box(self) -> tuple[float, ...]:
        steps = self.misses + 1
        return tuple(c + v * steps for c, v in zip(self.detection.bbox_xyxy, self.velocity, strict=True))

    def matched(self, detection: Detection) -> _Track:
        steps = self.misses + 1
        delta = (
            (new - old) / steps
            for new, old in zip(detection.bbox_xyxy, self.detection.bbox_xyxy, strict=True)
        )
        velocity = tuple(
            VELOCITY_SMOOTHING * v + (1 - VELOCITY_SMOOTHING) * d
            for v, d in zip(self.velocity, delta, strict=True)
        )
        return replace(self, detection=detection, velocity=velocity, hits=self.hits + 1, misses=0)  # type: ignore[arg-type]

    def public(self) -> TrackedObject:
        assert self.track_id is not None
        return TrackedObject(track_id=self.track_id, detection=self.detection, hits=self.hits)


class IouTracker:
    name = "iou"

    def __init__(self, config: TrackingConfig) -> None:
        self._config = config
        self._tracks: tuple[_Track, ...] = ()
        self._ids = count(1)

    @property
    def active_count(self) -> int:
        return sum(t.confirmed for t in self._tracks)

    def update(self, detections: Sequence[Detection]) -> TrackingResult:
        pairs = self._associate(detections)
        matched_tracks = {t for t, _ in pairs}
        matched_dets = {d for _, d in pairs}

        kept: list[_Track] = []
        current: list[TrackedObject] = []
        started: list[TrackedObject] = []
        ended: list[TrackedObject] = []

        for t_idx, d_idx in pairs:
            track, newly = self._confirm(self._tracks[t_idx].matched(detections[d_idx]))
            kept.append(track)
            if track.confirmed:
                current.append(track.public())
                if newly:
                    started.append(track.public())

        for t_idx, track in enumerate(self._tracks):
            if t_idx in matched_tracks or not track.confirmed:
                continue  # tentative tracks die on their first miss
            lost = replace(track, misses=track.misses + 1)
            if lost.misses > self._config.max_lost_frames:
                ended.append(lost.public())
            else:
                kept.append(lost)

        for d_idx, detection in enumerate(detections):
            if d_idx in matched_dets:
                continue
            track, newly = self._confirm(_Track(detection=detection))
            kept.append(track)
            if newly:
                current.append(track.public())
                started.append(track.public())

        self._tracks = tuple(kept)
        return TrackingResult(tracks=tuple(current), started=tuple(started), ended=tuple(ended))

    def _associate(self, detections: Sequence[Detection]) -> list[tuple[int, int]]:
        """Match confirmed tracks first, then tentative ones on the leftover detections, so a
        new tentative track can never take a detection away from an established ID."""
        confirmed = [i for i, t in enumerate(self._tracks) if t.confirmed]
        tentative = [i for i, t in enumerate(self._tracks) if not t.confirmed]
        first = self._match(confirmed, list(range(len(detections))), detections)
        used = {d for _, d in first}
        remaining = [d for d in range(len(detections)) if d not in used]
        return first + self._match(tentative, remaining, detections)

    def _match(
        self, track_ids: list[int], det_ids: list[int], detections: Sequence[Detection]
    ) -> list[tuple[int, int]]:
        if not track_ids or not det_ids:
            return []
        tracks = [self._tracks[i] for i in track_ids]
        dets = [detections[i] for i in det_ids]
        predicted = np.array([t.predicted_box() for t in tracks], dtype=np.float64)
        observed = np.array([d.bbox_xyxy for d in dets], dtype=np.float64)
        same_class = np.array([[t.detection.class_name == d.class_name for d in dets] for t in tracks])
        scores = np.where(same_class, iou_matrix(predicted, observed), 0.0)
        return [(track_ids[r], det_ids[c]) for r, c in greedy_match(scores, self._config.iou_threshold)]

    def _confirm(self, track: _Track) -> tuple[_Track, bool]:
        """Assign a public ID once a tentative track has enough hits. Returns (track, newly_confirmed)."""
        if track.confirmed or track.hits < self._config.min_hits:
            return track, False
        return replace(track, track_id=f"track-{next(self._ids)}"), True

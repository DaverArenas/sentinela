"""Deterministic event rules. Pure functions: state goes in, new state comes out."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from sentinela_runtime.vision.inference.base import Detection


def select_detection_events(
    detections: Sequence[Detection],
    target_classes: Collection[str],
    last_emitted: Mapping[str, float],
    now: float,
    cooldown_s: float,
) -> tuple[tuple[Detection, ...], Mapping[str, float]]:
    """Choose detections that should become ``object_detected`` events.

    At most one event per target class (its highest-confidence detection) and at most one
    per ``cooldown_s`` seconds, which prevents per-frame duplicates. Returns the detections
    to emit and a *new* cooldown map; ``last_emitted`` is never modified.
    """
    best: dict[str, Detection] = {}
    for detection in detections:
        if detection.class_name not in target_classes:
            continue
        current = best.get(detection.class_name)
        if current is None or detection.confidence > current.confidence:
            best[detection.class_name] = detection

    due = tuple(
        det
        for name, det in best.items()
        if name not in last_emitted or now - last_emitted[name] >= cooldown_s
    )
    updated = {**last_emitted, **{det.class_name: now for det in due}}
    return due, updated

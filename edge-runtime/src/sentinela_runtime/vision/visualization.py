"""Local preview: draw zones, raw detections and tracked objects on a frame and show it."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from sentinela_runtime.core.log import get_logger
from sentinela_runtime.vision.inference.base import Detection
from sentinela_runtime.vision.tracking.base import TrackedObject
from sentinela_runtime.vision.zones.geometry import to_pixels
from sentinela_runtime.vision.zones.zone_evaluator import Zone

log = get_logger("preview")

EXIT_KEYS = frozenset({ord("q"), 27})  # q or Esc
# BGR colours
COLOR_ALERT = (0, 0, 255)  # red: occupied zone / track inside a zone
COLOR_TRACK = (0, 200, 0)  # green: confirmed track outside zones
COLOR_ZONE = (0, 220, 255)  # yellow: empty zone
COLOR_OTHER = (180, 180, 180)  # grey: raw detection
COLOR_TEXT = (255, 255, 255)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5


def annotate_frame(
    image: np.ndarray,
    detections: Sequence[Detection] = (),
    tracks: Sequence[TrackedObject] = (),
    zones: Sequence[Zone] = (),
    occupancy: Mapping[str, frozenset[str]] | None = None,
    fps: float | None = None,
) -> np.ndarray:
    """Return an annotated copy of ``image``; the input is not modified."""
    canvas = image.copy()
    occupancy = occupancy or {}
    inside = frozenset().union(*occupancy.values()) if occupancy else frozenset()
    _draw_zones(canvas, zones, occupancy)
    for det in detections:
        _draw_box(canvas, det.bbox_xyxy, COLOR_OTHER, 1)
    for track in tracks:
        color = COLOR_ALERT if track.track_id in inside else COLOR_TRACK
        det = track.detection
        caption = f"{track.track_id} {det.class_name} {det.confidence:.2f}"
        _draw_box(canvas, det.bbox_xyxy, color, BOX_THICKNESS, caption)
    if fps is not None:
        cv2.putText(canvas, f"FPS {fps:.1f}", (10, 25), FONT, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)
    return canvas


def _draw_zones(canvas: np.ndarray, zones: Sequence[Zone], occupancy: Mapping[str, frozenset[str]]) -> None:
    frame_hw = (canvas.shape[0], canvas.shape[1])
    for zone in zones:
        points = np.array(to_pixels(zone.polygon, frame_hw), dtype=np.int32)
        color = COLOR_ALERT if occupancy.get(zone.zone_id) else COLOR_ZONE
        cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=BOX_THICKNESS)
        x, y = (int(v) for v in points.min(axis=0))
        cv2.putText(canvas, zone.name, (x + 5, y + 18), FONT, FONT_SCALE, color, 1, cv2.LINE_AA)


def _draw_box(
    canvas: np.ndarray,
    bbox: tuple[float, float, float, float],
    color: tuple[int, int, int],
    thickness: int,
    caption: str | None = None,
) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    if caption is None:
        return
    (text_w, text_h), baseline = cv2.getTextSize(caption, FONT, FONT_SCALE, 1)
    top = max(y1 - text_h - baseline, 0)
    cv2.rectangle(canvas, (x1, top), (x1 + text_w, top + text_h + baseline), color, -1)
    cv2.putText(canvas, caption, (x1, top + text_h), FONT, FONT_SCALE, COLOR_TEXT, 1, cv2.LINE_AA)


def show_frame(window_name: str, image: np.ndarray) -> tuple[bool, bool]:
    """Display a frame. Returns ``(preview_still_available, quit_requested)``."""
    try:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1) & 0xFF
    except cv2.error as exc:
        log.warning("Preview window unavailable (%s); continuing headless", exc)
        return False, False
    return True, key in EXIT_KEYS


def close_windows() -> None:
    with contextlib.suppress(cv2.error):  # no GUI backend: nothing to close
        cv2.destroyAllWindows()

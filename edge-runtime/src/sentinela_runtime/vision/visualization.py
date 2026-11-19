"""Local preview: draw canonical detections on a frame and show it in a window."""

from __future__ import annotations

import contextlib
from collections.abc import Collection, Sequence

import cv2
import numpy as np

from sentinela_runtime.core.log import get_logger
from sentinela_runtime.vision.inference.base import Detection

log = get_logger("preview")

EXIT_KEYS = frozenset({ord("q"), 27})  # q or Esc
COLOR_TARGET = (0, 0, 255)  # BGR red: classes that generate events
COLOR_OTHER = (180, 180, 180)  # BGR grey: everything else
COLOR_TEXT = (255, 255, 255)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5


def annotate_frame(
    image: np.ndarray,
    detections: Sequence[Detection],
    target_classes: Collection[str],
    fps: float | None = None,
) -> np.ndarray:
    """Return a copy of ``image`` with boxes, labels and FPS drawn; the input is not modified."""
    canvas = image.copy()
    for det in detections:
        color = COLOR_TARGET if det.class_name in target_classes else COLOR_OTHER
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        caption = f"{det.class_name} {det.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(caption, FONT, FONT_SCALE, 1)
        top = max(y1 - text_h - baseline, 0)
        cv2.rectangle(canvas, (x1, top), (x1 + text_w, top + text_h + baseline), color, -1)
        cv2.putText(canvas, caption, (x1, top + text_h), FONT, FONT_SCALE, COLOR_TEXT, 1, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(canvas, f"FPS {fps:.1f}", (10, 25), FONT, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)
    return canvas


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

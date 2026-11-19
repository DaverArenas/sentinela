"""Engine-independent non-maximum suppression used by backend postprocessing."""

from __future__ import annotations

import numpy as np


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS. Returns indices of kept boxes, highest score first."""
    if len(boxes_xyxy) == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        best, rest = order[0], order[1:]
        keep.append(int(best))
        inter_w = np.maximum(np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]), 0)
        inter_h = np.maximum(np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest]), 0)
        inter = inter_w * inter_h
        iou = inter / np.maximum(areas[best] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes_xyxy: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Class-aware NMS: boxes of different classes never suppress each other."""
    if len(boxes_xyxy) == 0:
        return np.empty(0, dtype=np.int64)
    offset = float(boxes_xyxy.max()) + 1.0
    shifted = boxes_xyxy + (class_ids.astype(np.float32) * offset)[:, None]
    return nms(shifted, scores, iou_threshold)

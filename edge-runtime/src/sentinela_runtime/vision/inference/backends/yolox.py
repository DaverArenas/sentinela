"""YOLOX model-family adapter (Megvii official ONNX exports, e.g. ``yolox_s.onnx``).

Input: 1x3xHxW float32, BGR, unnormalised 0-255, top-left letterbox padded with 114.
Output: 1xNx(5+C) rows ``[cx, cy, w, h, objectness, class scores...]`` where box terms are
grid-relative (decoded here) and objectness/class scores are already sigmoid-activated.
"""

from __future__ import annotations

import numpy as np

from sentinela_runtime.vision.inference.backends.coco import COCO80_CLASSES
from sentinela_runtime.vision.inference.base import Detection, InferenceError
from sentinela_runtime.vision.inference.nms import batched_nms
from sentinela_runtime.vision.preprocessing.letterbox import (
    LetterboxInfo,
    letterbox,
    to_nchw_float32,
    unletterbox_boxes,
)

STRIDES = (8, 16, 32)
PAD_VALUE = 114
BOX_FIELDS = 5  # cx, cy, w, h, objectness


def make_grids(
    input_hw: tuple[int, int], strides: tuple[int, ...] = STRIDES
) -> tuple[np.ndarray, np.ndarray]:
    """Anchor-point grid offsets ``(N, 2)`` and per-anchor strides ``(N, 1)``."""
    grids, expanded = [], []
    for stride in strides:
        rows, cols = input_hw[0] // stride, input_hw[1] // stride
        xv, yv = np.meshgrid(np.arange(cols), np.arange(rows))
        grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
        grids.append(grid)
        expanded.append(np.full((grid.shape[0], 1), stride))
    return np.concatenate(grids).astype(np.float32), np.concatenate(expanded).astype(np.float32)


class YoloxAdapter:
    family = "yolox"

    def __init__(self, input_hw: tuple[int, int], class_names: tuple[str, ...] = COCO80_CLASSES) -> None:
        if any(v % max(STRIDES) for v in input_hw):
            raise InferenceError(f"YOLOX input size must be a multiple of {max(STRIDES)}, got {input_hw}")
        self._input_hw = input_hw
        self._class_names = class_names
        self._grids, self._strides = make_grids(input_hw)

    @property
    def class_names(self) -> tuple[str, ...]:
        return self._class_names

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
        padded, info = letterbox(frame, self._input_hw, pad_value=PAD_VALUE, center=False)
        return to_nchw_float32(padded), info

    def postprocess(
        self,
        output: np.ndarray,
        info: LetterboxInfo,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> tuple[Detection, ...]:
        preds = output[0] if output.ndim == 3 else output
        expected = (len(self._grids), BOX_FIELDS + len(self._class_names))
        if preds.shape != expected:
            raise InferenceError(f"Unexpected YOLOX output shape {preds.shape}; expected {expected}")

        scores = preds[:, 4:5] * preds[:, BOX_FIELDS:]
        class_ids = scores.argmax(axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        keep_mask = confidences >= confidence_threshold
        if not keep_mask.any():
            return ()

        rows, strides, grids = preds[keep_mask], self._strides[keep_mask], self._grids[keep_mask]
        centers = (rows[:, :2] + grids) * strides
        sizes = np.exp(rows[:, 2:4]) * strides
        boxes = np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1)
        confidences, class_ids = confidences[keep_mask], class_ids[keep_mask]

        keep = batched_nms(boxes, confidences, class_ids, iou_threshold)
        boxes = unletterbox_boxes(boxes[keep], info)
        return tuple(
            Detection(
                class_id=int(cls),
                class_name=self._class_names[int(cls)],
                confidence=float(conf),
                bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )
            for box, conf, cls in zip(boxes, confidences[keep], class_ids[keep], strict=True)
        )

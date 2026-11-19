"""Aspect-preserving resize ("letterbox") and tensor layout helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_PAD_VALUE = 114


@dataclass(frozen=True)
class LetterboxInfo:
    """How a source frame was mapped into the model input; used to map boxes back."""

    scale: float
    pad_x: int
    pad_y: int
    source_hw: tuple[int, int]


def letterbox(
    image: np.ndarray,
    target_hw: tuple[int, int],
    pad_value: int = DEFAULT_PAD_VALUE,
    center: bool = True,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize ``image`` to fit ``target_hw`` without distortion and pad the remainder.

    ``center=False`` places the image at the top-left corner (YOLOX convention).
    Returns a new array; the input is never modified.
    """
    src_h, src_w = image.shape[:2]
    dst_h, dst_w = target_hw
    scale = min(dst_h / src_h, dst_w / src_w)
    new_h, new_w = max(int(round(src_h * scale)), 1), max(int(round(src_w * scale)), 1)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((dst_h, dst_w, image.shape[2]), pad_value, dtype=image.dtype)
    pad_x = (dst_w - new_w) // 2 if center else 0
    pad_y = (dst_h - new_h) // 2 if center else 0
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, LetterboxInfo(scale=scale, pad_x=pad_x, pad_y=pad_y, source_hw=(src_h, src_w))


def to_nchw_float32(image: np.ndarray, *, bgr_to_rgb: bool = False, scale: float = 1.0) -> np.ndarray:
    """Convert an HxWx3 uint8 image into a contiguous 1x3xHxW float32 tensor."""
    pixels = image[..., ::-1] if bgr_to_rgb else image
    tensor = pixels.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    if scale != 1.0:
        tensor *= scale
    return np.ascontiguousarray(tensor)


def unletterbox_boxes(boxes_xyxy: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Map model-space ``(N, 4)`` xyxy boxes back to source-frame pixels, clipped to the frame."""
    boxes = (
        boxes_xyxy - np.array([info.pad_x, info.pad_y, info.pad_x, info.pad_y], dtype=np.float32)
    ) / info.scale
    src_h, src_w = info.source_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, src_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, src_h)
    return boxes

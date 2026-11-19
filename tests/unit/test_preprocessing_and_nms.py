"""Letterbox geometry and non-maximum suppression."""

from __future__ import annotations

import numpy as np
import pytest

from sentinela_runtime.vision.inference.nms import batched_nms, nms
from sentinela_runtime.vision.preprocessing.letterbox import (
    LetterboxInfo,
    letterbox,
    to_nchw_float32,
    unletterbox_boxes,
)


def test_letterbox_centers_and_preserves_aspect_ratio() -> None:
    image = np.full((100, 200, 3), 7, dtype=np.uint8)

    padded, info = letterbox(image, (64, 64))

    assert padded.shape == (64, 64, 3)
    assert info == LetterboxInfo(scale=0.32, pad_x=0, pad_y=16, source_hw=(100, 200))
    assert (padded[:16] == 114).all() and (padded[16:48] == 7).all() and (padded[48:] == 114).all()
    assert (image == 7).all(), "input must not be modified"


def test_letterbox_top_left_alignment() -> None:
    padded, info = letterbox(np.zeros((50, 100, 3), np.uint8), (64, 64), pad_value=1, center=False)

    assert (info.pad_x, info.pad_y) == (0, 0)
    assert (padded[:32] == 0).all() and (padded[32:] == 1).all()


def test_to_nchw_float32_layout_scaling_and_channel_order() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[..., 0] = 10  # blue channel in BGR

    tensor = to_nchw_float32(image, bgr_to_rgb=True, scale=0.5)

    assert tensor.shape == (1, 3, 2, 3)
    assert tensor.dtype == np.float32 and tensor.flags.c_contiguous
    assert (tensor[0, 2] == 5).all() and (tensor[0, 0] == 0).all()
    assert (to_nchw_float32(image)[0, 0] == 10).all()


def test_unletterbox_round_trip_and_clipping() -> None:
    info = LetterboxInfo(scale=0.5, pad_x=0, pad_y=10, source_hw=(100, 200))
    model_space = np.array([[10, 20, 50, 60], [-5, 0, 120, 70]], dtype=np.float32)

    boxes = unletterbox_boxes(model_space, info)

    np.testing.assert_allclose(boxes[0], [20, 20, 100, 100])
    np.testing.assert_allclose(boxes[1], [0, 0, 200, 100])


def test_nms_suppresses_overlaps_and_keeps_disjoint_boxes() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.6, 0.9, 0.5], dtype=np.float32)

    assert nms(boxes, scores, 0.5).tolist() == [1, 2]
    assert nms(boxes, scores, 0.95).tolist() == [1, 0, 2]


def test_batched_nms_never_suppresses_across_classes() -> None:
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)

    assert sorted(batched_nms(boxes, scores, np.array([0, 1]), 0.5).tolist()) == [0, 1]
    assert batched_nms(boxes, scores, np.array([0, 0]), 0.5).tolist() == [0]


@pytest.mark.parametrize("fn", [nms, lambda b, s, t: batched_nms(b, s, np.empty(0), t)])
def test_nms_handles_empty_input(fn: object) -> None:
    empty = np.empty((0, 4), dtype=np.float32)

    assert len(fn(empty, np.empty(0), 0.5)) == 0  # type: ignore[operator]

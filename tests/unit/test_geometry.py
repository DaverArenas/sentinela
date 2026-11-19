"""Polygon zone geometry."""

from __future__ import annotations

import pytest

from sentinela_runtime.vision.zones import geometry as geo

SQUARE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
BOWTIE = ((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0))
# Concave "U": the notch (0.4..0.6, 0.0..0.6) is outside the polygon.
U_SHAPE = ((0.0, 0.0), (0.4, 0.0), (0.4, 0.6), (0.6, 0.6), (0.6, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def test_polygon_area_uses_shoelace_formula_regardless_of_winding() -> None:
    assert geo.polygon_area(SQUARE) == 1.0
    assert geo.polygon_area(tuple(reversed(SQUARE))) == 1.0
    assert geo.polygon_area(((0, 0), (1, 0), (2, 0))) == 0.0


@pytest.mark.parametrize(
    ("point", "inside"),
    [((0.5, 0.5), True), ((1.5, 0.5), False), ((-0.1, 0.5), False), ((0.5, 1.2), False)],
)
def test_point_in_convex_polygon(point: geo.Point, inside: bool) -> None:
    assert geo.point_in_polygon(point, SQUARE) is inside


@pytest.mark.parametrize(
    ("point", "inside"), [((0.5, 0.3), False), ((0.5, 0.8), True), ((0.2, 0.3), True), ((0.8, 0.3), True)]
)
def test_point_in_concave_polygon(point: geo.Point, inside: bool) -> None:
    assert geo.point_in_polygon(point, U_SHAPE) is inside


def test_segments_intersect_detects_crossing_touching_and_disjoint() -> None:
    assert geo.segments_intersect((0, 0), (1, 1), (0, 1), (1, 0))
    assert geo.segments_intersect((0, 0), (1, 0), (1, 0), (2, 5))  # shared endpoint
    assert geo.segments_intersect((0, 0), (2, 0), (1, 0), (3, 0))  # collinear overlap
    assert not geo.segments_intersect((0, 0), (1, 0), (0, 1), (1, 1))
    assert not geo.segments_intersect((0, 0), (1, 0), (2, 0), (3, 0))  # collinear, apart


def test_simple_polygon_check_rejects_self_intersection() -> None:
    assert geo.is_simple_polygon(SQUARE)
    assert geo.is_simple_polygon(U_SHAPE)
    assert not geo.is_simple_polygon(BOWTIE)


def test_anchor_points() -> None:
    bbox = (10.0, 20.0, 30.0, 60.0)

    assert geo.anchor_point(bbox, geo.ANCHOR_BOTTOM_CENTER) == (20.0, 60.0)
    assert geo.anchor_point(bbox, geo.ANCHOR_CENTER) == (20.0, 40.0)
    with pytest.raises(ValueError, match="Unknown anchor"):
        geo.anchor_point(bbox, "top")


def test_normalisation_round_trip_with_pixels() -> None:
    frame_hw = (480, 640)

    assert geo.normalise_point((320.0, 120.0), frame_hw) == (0.5, 0.25)
    assert geo.to_pixels(((0.5, 0.25), (1.0, 1.0)), frame_hw) == [(320, 120), (640, 480)]

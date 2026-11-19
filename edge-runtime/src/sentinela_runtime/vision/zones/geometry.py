"""Pure 2D geometry for polygon zones (no external dependency needed for Phase 1).

Coordinates are ``(x, y)`` tuples. Zone polygons are stored normalised to ``[0, 1]`` of the
frame width/height, so one zone definition survives camera resolution changes.
"""

from __future__ import annotations

from collections.abc import Sequence

Point = tuple[float, float]
Polygon = tuple[Point, ...]
BBox = tuple[float, float, float, float]

ANCHOR_BOTTOM_CENTER = "bottom_center"  # feet position: right choice for floor-level zones
ANCHOR_CENTER = "center"
ANCHORS = frozenset({ANCHOR_BOTTOM_CENTER, ANCHOR_CENTER})


def polygon_area(polygon: Sequence[Point]) -> float:
    """Unsigned area (shoelace formula)."""
    twice_area = sum(
        x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(polygon, (*polygon[1:], polygon[0]), strict=True)
    )
    return abs(twice_area) / 2.0


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Even-odd ray casting test. Points exactly on an edge may fall on either side."""
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True when segments p1-p2 and q1-q2 cross or touch."""
    d1, d2 = _orientation(q1, q2, p1), _orientation(q1, q2, p2)
    d3, d4 = _orientation(p1, p2, q1), _orientation(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and 0 not in (d1, d2, d3, d4):
        return True
    return (
        (d1 == 0 and _on_segment(q1, q2, p1))
        or (d2 == 0 and _on_segment(q1, q2, p2))
        or (d3 == 0 and _on_segment(p1, p2, q1))
        or (d4 == 0 and _on_segment(p1, p2, q2))
    )


def is_simple_polygon(polygon: Sequence[Point]) -> bool:
    """True when no two non-adjacent edges intersect (no bow-ties or self-crossings)."""
    count = len(polygon)
    edges = [(polygon[i], polygon[(i + 1) % count]) for i in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            adjacent = j == i + 1 or (i == 0 and j == count - 1)
            if not adjacent and segments_intersect(*edges[i], *edges[j]):
                return False
    return True


def anchor_point(bbox: BBox, anchor: str) -> Point:
    """Reference point of a bounding box used for zone membership."""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    if anchor == ANCHOR_BOTTOM_CENTER:
        return center_x, y2
    if anchor == ANCHOR_CENTER:
        return center_x, (y1 + y2) / 2.0
    raise ValueError(f"Unknown anchor {anchor!r}; expected one of {sorted(ANCHORS)}")


def normalise_point(point: Point, frame_hw: tuple[int, int]) -> Point:
    height, width = frame_hw
    return point[0] / width, point[1] / height


def to_pixels(polygon: Sequence[Point], frame_hw: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = frame_hw
    return [(round(x * width), round(y * height)) for x, y in polygon]

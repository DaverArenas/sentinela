"""Zone membership, debouncing and entry/exit transitions."""

from __future__ import annotations

import pytest

from sentinela_runtime.core.config import ZoneConfig
from sentinela_runtime.vision.tracking.base import TrackedObject, TrackingResult
from sentinela_runtime.vision.zones.zone_evaluator import (
    OUTSIDE,
    Membership,
    TransitionKind,
    Zone,
    ZoneEvaluator,
    step_membership,
)

from support import CAR, RIGHT_HALF, person_at

FRAME_HW = (64, 96)  # height, width
ZONE = Zone("zone-001", "Restricted Area", RIGHT_HALF, frozenset({"person"}), "bottom_center")
INSIDE_X, OUTSIDE_X = 60.0, 10.0  # box left edge; 20 px wide -> centre 70 (inside) / 20 (outside)


def tracked(x1: float, track_id: str = "track-1") -> TrackedObject:
    return TrackedObject(track_id, person_at(x1), hits=3)


def seen(*objects: TrackedObject) -> TrackingResult:
    return TrackingResult(tracks=objects)


def kinds(evaluator: ZoneEvaluator, xs: list[float]) -> list[list[str]]:
    return [[str(t.kind) for t in evaluator.update(seen(tracked(x)), FRAME_HW)] for x in xs]


# --- pure debounce step -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "observed", "expected"),
    [
        (OUTSIDE, False, (OUTSIDE, None)),
        (OUTSIDE, True, (Membership(False, 1), None)),
        (Membership(False, 1), True, (Membership(True, 0), TransitionKind.ENTRY)),
        (Membership(False, 1), False, (OUTSIDE, None)),  # streak broken
        (Membership(True, 0), False, (Membership(True, 1), None)),
        (Membership(True, 2), False, (Membership(False, 0), TransitionKind.EXIT)),
    ],
)
def test_step_membership(
    state: Membership, observed: bool, expected: tuple[Membership, TransitionKind | None]
) -> None:
    assert step_membership(state, observed, entry_frames=2, exit_frames=3) == expected


# --- evaluator ----------------------------------------------------------------------------


def test_zone_uses_the_configured_anchor() -> None:
    straddling = TrackedObject("t", person_at(30.0, width=30.0), 1)  # x 30..60, centre 45

    assert not ZONE.contains(straddling, FRAME_HW)
    assert Zone("z", "z", ((0.4, 0.0), (1.0, 0.0), (1.0, 1.0), (0.4, 1.0)), ZONE.classes, "center").contains(
        straddling, FRAME_HW
    )


def test_entry_then_exit_fire_once_each_after_debounce() -> None:
    evaluator = ZoneEvaluator([ZONE], entry_frames=2, exit_frames=2)

    result = kinds(evaluator, [OUTSIDE_X, INSIDE_X, INSIDE_X, INSIDE_X, OUTSIDE_X, OUTSIDE_X, OUTSIDE_X])

    assert result == [[], [], ["entry"], [], [], ["exit"], []]


def test_edge_flicker_is_debounced() -> None:
    evaluator = ZoneEvaluator([ZONE], entry_frames=3, exit_frames=3)

    result = kinds(evaluator, [INSIDE_X, OUTSIDE_X, INSIDE_X, INSIDE_X, OUTSIDE_X, INSIDE_X])

    assert result == [[]] * 6


def test_occupancy_reports_tracks_inside_each_zone() -> None:
    evaluator = ZoneEvaluator([ZONE], entry_frames=1, exit_frames=1)

    evaluator.update(seen(tracked(INSIDE_X, "track-1"), tracked(OUTSIDE_X, "track-2")), FRAME_HW)

    assert evaluator.occupancy() == {"zone-001": frozenset({"track-1"})}


def test_ended_track_inside_a_zone_produces_a_final_exit() -> None:
    evaluator = ZoneEvaluator([ZONE], entry_frames=1, exit_frames=5)
    evaluator.update(seen(tracked(INSIDE_X)), FRAME_HW)

    transitions = evaluator.update(TrackingResult(ended=(tracked(INSIDE_X),)), FRAME_HW)
    again = evaluator.update(TrackingResult(ended=(tracked(INSIDE_X),)), FRAME_HW)

    assert [(t.kind, t.obj.track_id) for t in transitions] == [(TransitionKind.EXIT, "track-1")]
    assert again == ()
    assert evaluator.occupancy() == {"zone-001": frozenset()}


def test_unseen_track_keeps_its_membership_until_it_ends() -> None:
    evaluator = ZoneEvaluator([ZONE], entry_frames=1, exit_frames=1)
    evaluator.update(seen(tracked(INSIDE_X)), FRAME_HW)

    assert evaluator.update(TrackingResult(), FRAME_HW) == ()  # occluded: no exit
    assert evaluator.occupancy()["zone-001"] == {"track-1"}


def test_zone_ignores_other_classes() -> None:
    evaluator = ZoneEvaluator([ZONE], entry_frames=1, exit_frames=1)
    car = TrackedObject("track-9", CAR, 3)  # x 50..90: inside the right half

    assert evaluator.update(seen(car), FRAME_HW) == ()


def test_from_config_builds_zones() -> None:
    config = ZoneConfig("zone-002", "Dock", RIGHT_HALF, ("person", "car"), "center")

    evaluator = ZoneEvaluator.from_config([config], entry_frames=1, exit_frames=1)

    assert evaluator.zones == (Zone("zone-002", "Dock", RIGHT_HALF, frozenset({"person", "car"}), "center"),)

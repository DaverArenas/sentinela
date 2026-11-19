"""IoU tracker: association, lifecycle, lost-track handling and the tracker registry."""

from __future__ import annotations

import numpy as np
import pytest

from sentinela_runtime.core.config import VALID_TRACKERS, TrackingConfig
from sentinela_runtime.vision.inference.base import Detection
from sentinela_runtime.vision.tracking import TRACKERS, create_tracker
from sentinela_runtime.vision.tracking.iou_tracker import IouTracker, greedy_match, iou_matrix

from support import CAR, person_at

CONFIG = TrackingConfig(tracker="iou", iou_threshold=0.3, min_hits=3, max_lost_frames=2)


def run(tracker: IouTracker, frames: list[tuple[Detection, ...]]) -> list[list[str]]:
    return [[t.track_id for t in tracker.update(dets).tracks] for dets in frames]


def test_iou_matrix_values() -> None:
    a = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=float)

    iou = iou_matrix(a, b)

    assert iou.shape == (2, 2)
    assert iou[0, 0] == 1.0
    assert iou[0, 1] == pytest.approx(50 / 150)
    assert iou[1].tolist() == [0.0, 0.0]
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 2)


def test_greedy_match_prefers_highest_score_and_respects_threshold() -> None:
    scores = np.array([[0.9, 0.8], [0.85, 0.1]])

    assert greedy_match(scores, 0.3) == [(0, 0)]  # row 1's only option (col 0) is taken; 0.1 < 0.3
    assert greedy_match(np.array([[0.9, 0.8], [0.85, 0.4]]), 0.3) == [(0, 0), (1, 1)]


def test_track_is_confirmed_after_min_hits_with_a_persistent_id() -> None:
    tracker = IouTracker(CONFIG)

    results = [tracker.update((person_at(2.0 * n),)) for n in range(5)]

    assert [len(r.tracks) for r in results] == [0, 0, 1, 1, 1]
    assert [len(r.started) for r in results] == [0, 0, 1, 0, 0]
    assert {t.track_id for r in results for t in r.tracks} == {"track-1"}
    assert results[-1].tracks[0].hits == 5


def test_min_hits_one_confirms_immediately() -> None:
    tracker = IouTracker(TrackingConfig(min_hits=1))

    result = tracker.update((person_at(0.0),))

    assert [t.track_id for t in result.started] == ["track-1"]


def test_tentative_track_is_dropped_on_first_miss() -> None:
    tracker = IouTracker(CONFIG)

    ids = run(
        tracker, [(person_at(0),), (person_at(0),), (), (person_at(0),), (person_at(0),), (person_at(0),)]
    )

    assert ids == [[], [], [], [], [], ["track-1"]]


def test_short_occlusion_keeps_the_id_and_long_one_ends_the_track() -> None:
    tracker = IouTracker(CONFIG)
    for _ in range(3):
        tracker.update((person_at(0),))

    occluded = [tracker.update(()) for _ in range(2)]
    back = tracker.update((person_at(0),))
    assert all(not r.ended for r in occluded)
    assert [t.track_id for t in back.tracks] == ["track-1"]

    gone = [tracker.update(()) for _ in range(3)]
    assert [len(r.ended) for r in gone] == [0, 0, 1]
    assert gone[-1].ended[0].track_id == "track-1"
    assert tracker.active_count == 0


def test_two_objects_get_distinct_ids_and_classes_never_mix() -> None:
    tracker = IouTracker(TrackingConfig(min_hits=1))
    car_on_person = Detection(2, "car", 0.8, person_at(0).bbox_xyxy)

    first = tracker.update((person_at(0), CAR))
    second = tracker.update((car_on_person,))

    assert {t.track_id for t in first.tracks} == {"track-1", "track-2"}
    assert [t.track_id for t in second.tracks] == ["track-3"]  # same box, other class: new track


def test_velocity_prediction_bridges_an_occlusion_of_a_moving_person() -> None:
    """After 2 unseen frames the walker reappears 12 px further on: raw IoU with its last box is
    0.25 (< 0.3), so only the constant-velocity prediction can keep the same ID."""
    tracker = IouTracker(TrackingConfig(iou_threshold=0.3, min_hits=1, max_lost_frames=2))
    walking = [(person_at(4.0 * n),) for n in range(5)]  # x = 0 .. 16

    ids = run(tracker, [*walking, (), (), (person_at(28.0),)])

    assert ids[0] == ["track-1"]
    assert ids[-1] == ["track-1"]


def test_confirmed_tracks_have_priority_over_tentative_ones() -> None:
    """The detection overlaps the tentative track more (IoU 0.9) than the confirmed one (0.6);
    it must still extend the confirmed track rather than steal its identity."""
    tracker = IouTracker(TrackingConfig(iou_threshold=0.3, min_hits=2, max_lost_frames=5))
    tracker.update((person_at(0.0),))
    tracker.update((person_at(0.0), person_at(6.0)))  # track-1 confirmed; x=6 starts a tentative track

    result = tracker.update((person_at(5.0),))

    assert [t.track_id for t in result.tracks] == ["track-1"]


def test_registry_matches_config_validation_and_creates_trackers() -> None:
    assert set(TRACKERS) == VALID_TRACKERS
    assert isinstance(create_tracker(CONFIG), IouTracker)
    with pytest.raises(ValueError, match="Unknown tracker"):
        create_tracker(TrackingConfig(tracker="bytetrack"))

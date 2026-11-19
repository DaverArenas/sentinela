"""Zone membership for tracked objects, with entry/exit debouncing.

A track enters a zone after its anchor point is inside the polygon for ``entry_frames``
consecutive inferred frames, and exits after ``exit_frames`` consecutive frames outside.
Debouncing stops a person standing on a zone edge from producing entry/exit storms.
A confirmed track that ends while inside a zone produces a final exit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from sentinela_runtime.core.config import ZoneConfig
from sentinela_runtime.vision.tracking.base import TrackedObject, TrackingResult
from sentinela_runtime.vision.zones.geometry import Polygon, anchor_point, normalise_point, point_in_polygon


class TransitionKind(StrEnum):
    ENTRY = "entry"
    EXIT = "exit"


@dataclass(frozen=True)
class Zone:
    zone_id: str
    name: str
    polygon: Polygon  # normalised [0, 1] vertices
    classes: frozenset[str]
    anchor: str

    @classmethod
    def from_config(cls, config: ZoneConfig) -> Zone:
        return cls(config.zone_id, config.name, config.polygon, frozenset(config.classes), config.anchor)

    def applies_to(self, obj: TrackedObject) -> bool:
        return obj.class_name in self.classes

    def contains(self, obj: TrackedObject, frame_hw: tuple[int, int]) -> bool:
        point = normalise_point(anchor_point(obj.detection.bbox_xyxy, self.anchor), frame_hw)
        return point_in_polygon(point, self.polygon)


@dataclass(frozen=True)
class ZoneTransition:
    kind: TransitionKind
    zone: Zone
    obj: TrackedObject


@dataclass(frozen=True)
class Membership:
    inside: bool = False
    streak: int = 0  # consecutive observations contradicting ``inside``


OUTSIDE = Membership()


def step_membership(
    state: Membership, observed_inside: bool, entry_frames: int, exit_frames: int
) -> tuple[Membership, TransitionKind | None]:
    """Pure debounce step: returns the new state and the transition it triggers, if any."""
    if observed_inside == state.inside:
        return Membership(state.inside, 0), None
    streak = state.streak + 1
    if streak >= (entry_frames if observed_inside else exit_frames):
        return Membership(
            observed_inside, 0
        ), TransitionKind.ENTRY if observed_inside else TransitionKind.EXIT
    return Membership(state.inside, streak), None


class ZoneEvaluator:
    def __init__(self, zones: Sequence[Zone], entry_frames: int, exit_frames: int) -> None:
        self._zones = tuple(zones)
        self._entry_frames = entry_frames
        self._exit_frames = exit_frames
        self._memberships: Mapping[tuple[str, str], Membership] = {}

    @classmethod
    def from_config(cls, zones: Sequence[ZoneConfig], entry_frames: int, exit_frames: int) -> ZoneEvaluator:
        return cls([Zone.from_config(z) for z in zones], entry_frames, exit_frames)

    @property
    def zones(self) -> tuple[Zone, ...]:
        return self._zones

    def occupancy(self) -> dict[str, frozenset[str]]:
        """Zone ID -> IDs of the tracks currently inside it."""
        return {
            zone.zone_id: frozenset(
                track_id
                for (track_id, zone_id), state in self._memberships.items()
                if zone_id == zone.zone_id and state.inside
            )
            for zone in self._zones
        }

    def update(self, tracking: TrackingResult, frame_hw: tuple[int, int]) -> tuple[ZoneTransition, ...]:
        memberships = dict(self._memberships)
        transitions: list[ZoneTransition] = []

        for obj in tracking.tracks:
            for zone in self._zones:
                if not zone.applies_to(obj):
                    continue
                key = (obj.track_id, zone.zone_id)
                state, kind = step_membership(
                    memberships.get(key, OUTSIDE),
                    zone.contains(obj, frame_hw),
                    self._entry_frames,
                    self._exit_frames,
                )
                if state == OUTSIDE:
                    memberships.pop(key, None)  # keep state bounded to tracks near/inside zones
                else:
                    memberships[key] = state
                if kind is not None:
                    transitions.append(ZoneTransition(kind, zone, obj))

        for obj in tracking.ended:
            for zone in self._zones:
                state = memberships.pop((obj.track_id, zone.zone_id), OUTSIDE)
                if state.inside:
                    transitions.append(ZoneTransition(TransitionKind.EXIT, zone, obj))

        self._memberships = memberships
        return tuple(transitions)

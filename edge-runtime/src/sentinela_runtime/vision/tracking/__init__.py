"""Object tracking behind a tracker adapter interface (``base.Tracker``).

``create_tracker`` selects the algorithm named in ``vision_config.yaml`` -> ``tracking.tracker``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from sentinela_runtime.core.config import TrackingConfig
from sentinela_runtime.vision.tracking.base import Tracker
from sentinela_runtime.vision.tracking.iou_tracker import IouTracker

TRACKERS: Mapping[str, Callable[[TrackingConfig], Tracker]] = MappingProxyType({IouTracker.name: IouTracker})


def create_tracker(config: TrackingConfig) -> Tracker:
    factory = TRACKERS.get(config.tracker)
    if factory is None:
        raise ValueError(f"Unknown tracker {config.tracker!r}; available: {sorted(TRACKERS)}")
    return factory(config)

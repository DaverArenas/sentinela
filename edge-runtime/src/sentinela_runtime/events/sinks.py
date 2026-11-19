"""Log-stream event sink.

``LoggingEventSink`` writes each event as one JSON line to the ``sentinela.events`` logger,
which ``config/logging_config.yaml`` routes to ``logs/events.jsonl`` (a convenient live tail).
The durable record is the SQLite store (``storage.sqlite_store``), which implements the
same ``EventSink`` protocol.
"""

from __future__ import annotations

import logging

from sentinela_runtime.core.log import EVENTS_LOGGER
from sentinela_runtime.events.schemas import Event


class LoggingEventSink:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(EVENTS_LOGGER)

    def publish(self, event: Event) -> None:
        self._logger.info(event.to_json())

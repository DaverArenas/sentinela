"""Process lifecycle: a shared stop signal and graceful handling of SIGINT/SIGTERM."""

from __future__ import annotations

import signal
import threading
from types import FrameType
from typing import Any, Self

from sentinela_runtime.core.log import get_logger

log = get_logger("lifecycle")

HANDLED_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class Lifecycle:
    """Owns the stop event every long-running component polls.

    The first SIGINT/SIGTERM requests a graceful stop; a second SIGINT forces exit.
    Use as a context manager so previous signal handlers are always restored.
    """

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self._previous: dict[int, Any] = {}

    @property
    def stopping(self) -> bool:
        return self.stop_event.is_set()

    def request_stop(self, reason: str) -> None:
        if not self.stop_event.is_set():
            log.info("Stop requested: %s", reason)
        self.stop_event.set()

    def __enter__(self) -> Self:
        for sig in HANDLED_SIGNALS:
            self._previous[sig] = signal.signal(sig, self._handle_signal)
        return self

    def __exit__(self, *exc_info: object) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous = {}

    def _handle_signal(self, signum: int, _frame: FrameType | None) -> None:
        if self.stopping and signum == signal.SIGINT:
            raise KeyboardInterrupt
        self.request_stop(signal.Signals(signum).name)

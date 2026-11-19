"""Structured logging and process lifecycle."""

from __future__ import annotations

import json
import logging
import os
import signal
from pathlib import Path

import pytest

from sentinela_runtime.core.lifecycle import Lifecycle
from sentinela_runtime.core.log import JsonFormatter, get_logger, setup_logging

from support import REPO_CONFIG_DIR


def make_record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord("sentinela.test", logging.INFO, __file__, 1, "hello %s", ("edge",), None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_includes_extra_fields() -> None:
    payload = json.loads(JsonFormatter().format(make_record(camera_id="camera-001", attempt=2)))

    assert payload["msg"] == "hello edge"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "sentinela.test"
    assert payload["camera_id"] == "camera-001"
    assert payload["attempt"] == 2
    assert payload["ts"].endswith("+00:00")


def test_json_formatter_serialises_exceptions() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())

    assert "ValueError: boom" in json.loads(JsonFormatter().format(record))["exc"]


def test_setup_logging_writes_runtime_and_event_files_under_base_dir(tmp_path: Path) -> None:
    setup_logging(REPO_CONFIG_DIR / "logging_config.yaml", tmp_path, level="DEBUG")
    get_logger("test").debug("diagnostic", extra={"k": 1})
    logging.getLogger("sentinela.events").info('{"event_id":"evt-1"}')
    for handler in logging.getLogger("sentinela").handlers + logging.getLogger("sentinela.events").handlers:
        handler.flush()

    runtime_lines = (tmp_path / "logs/runtime.jsonl").read_text(encoding="utf-8").splitlines()
    assert {"msg": "diagnostic", "k": 1}.items() <= json.loads(runtime_lines[0]).items()
    assert (tmp_path / "logs/events.jsonl").read_text(encoding="utf-8") == '{"event_id":"evt-1"}\n'


@pytest.mark.parametrize("content", ["not: [valid", "- a list\n", "version: 99\n"])
def test_unusable_logging_config_degrades_to_stderr(
    tmp_path: Path, content: str, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "logging_config.yaml"
    path.write_text(content, encoding="utf-8")

    setup_logging(path, tmp_path)

    assert not (tmp_path / "logs").exists()


def test_missing_logging_config_degrades_to_stderr(tmp_path: Path) -> None:
    setup_logging(tmp_path / "absent.yaml", tmp_path, level="WARNING")

    assert logging.getLogger("sentinela").level == logging.WARNING


# --- lifecycle --------------------------------------------------------------------------------


def test_sigterm_requests_graceful_stop_and_handlers_are_restored() -> None:
    original = signal.getsignal(signal.SIGTERM)

    with Lifecycle() as lifecycle:
        os.kill(os.getpid(), signal.SIGTERM)
        assert lifecycle.stopping

    assert signal.getsignal(signal.SIGTERM) is original


def test_second_sigint_forces_exit() -> None:
    with Lifecycle() as lifecycle:
        lifecycle.request_stop("test")
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)


def test_request_stop_is_idempotent() -> None:
    lifecycle = Lifecycle()

    lifecycle.request_stop("a")
    lifecycle.request_stop("b")

    assert lifecycle.stopping

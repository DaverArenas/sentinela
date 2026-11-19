"""Structured logging setup driven by ``config/logging_config.yaml``."""

from __future__ import annotations

import copy
import json
import logging
import logging.config
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT_LOGGER = "sentinela"
EVENTS_LOGGER = "sentinela.events"
VALID_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
BOOTSTRAP_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Attributes present on every LogRecord; anything else came from ``extra=`` and is kept.
_STANDARD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Render each record as one JSON object, including any ``extra=`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def bootstrap_logging() -> None:
    """Minimal stderr logging used until the full configuration has been loaded."""
    logging.basicConfig(level=logging.INFO, format=BOOTSTRAP_FORMAT)


def setup_logging(config_path: Path, base_dir: Path, level: str | None = None) -> None:
    """Apply the dictConfig in ``config_path``; degrade to stderr-only logging if it is unusable.

    Relative handler ``filename`` values are resolved against ``base_dir`` and their
    parent directories are created.
    """
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("logging config must be a mapping")
        logging.config.dictConfig(_resolve_handler_paths(raw, base_dir))
    except (OSError, ValueError, TypeError, AttributeError, ImportError, yaml.YAMLError) as exc:
        bootstrap_logging()
        get_logger("log").warning("Logging config %s unusable (%s); using stderr only", config_path, exc)

    if level:
        logging.getLogger(ROOT_LOGGER).setLevel(level.upper())


def _resolve_handler_paths(raw: Mapping[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(raw))
    for handler in (resolved.get("handlers") or {}).values():
        filename = handler.get("filename") if isinstance(handler, dict) else None
        if not filename:
            continue
        path = Path(filename).expanduser()
        path = path if path.is_absolute() else base_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        handler["filename"] = str(path)
    return resolved


def get_logger(name: str) -> logging.Logger:
    """Return ``sentinela.<name>``."""
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")

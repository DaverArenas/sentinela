"""Shared pytest fixtures (helpers live in ``support.py``)."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from support import REPO_CONFIG_DIR, deep_merge, write_video


@pytest.fixture(autouse=True)
def _reset_runtime_logging() -> Iterator[None]:
    """``main()`` applies dictConfig globally; undo it so later tests see default propagation."""
    yield
    for name in ("sentinela", "sentinela.events"):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


@pytest.fixture
def recorded_video(tmp_path: Path) -> Path:
    return write_video(tmp_path / "clip.avi")


@pytest.fixture
def make_config_dir(tmp_path: Path) -> Callable[..., Path]:
    """Copy the repository config into ``tmp_path/config`` with optional per-test overrides."""

    def make(node: Mapping[str, Any] | None = None, vision: Mapping[str, Any] | None = None) -> Path:
        target = tmp_path / "config"
        target.mkdir(exist_ok=True)
        for name, override in (("node_config.yaml", node), ("vision_config.yaml", vision)):
            data = yaml.safe_load((REPO_CONFIG_DIR / name).read_text(encoding="utf-8"))
            (target / name).write_text(yaml.safe_dump(deep_merge(data, override or {})), encoding="utf-8")
        shutil.copy(REPO_CONFIG_DIR / "logging_config.yaml", target / "logging_config.yaml")
        return target

    return make

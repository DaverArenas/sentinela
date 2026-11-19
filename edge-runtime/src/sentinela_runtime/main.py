"""Sentinela Edge Runtime entry point (Sentinela Vision, Phase 1).

Examples (from ``edge-runtime/``)::

    uv run sentinela-runtime                              # camera from config/node_config.yaml
    uv run sentinela-runtime --source path/to/clip.mp4    # recorded video
    uv run sentinela-runtime --source rtsp://cam/stream --no-preview
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from sentinela_runtime.core.config import ConfigError, RuntimeConfig, apply_overrides, load_runtime_config
from sentinela_runtime.core.lifecycle import Lifecycle
from sentinela_runtime.core.log import VALID_LEVELS, bootstrap_logging, get_logger, setup_logging
from sentinela_runtime.events.event_engine import EventContext, EventEngine
from sentinela_runtime.events.sinks import LoggingEventSink
from sentinela_runtime.storage.snapshots import SnapshotStore
from sentinela_runtime.storage.sqlite_store import SqliteEventStore, StorageError
from sentinela_runtime.vision.camera.video_source import CameraError, VideoSource
from sentinela_runtime.vision.inference.base import InferenceBackend, InferenceError
from sentinela_runtime.vision.inference.model_manager import load_backend
from sentinela_runtime.vision.pipeline import PipelineError, PipelineSettings, VisionPipeline
from sentinela_runtime.vision.tracking import create_tracker
from sentinela_runtime.vision.zones.zone_evaluator import ZoneEvaluator

log = get_logger("main")

EXIT_OK = 0
EXIT_CAMERA_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_MODEL_ERROR = 3
EXIT_INFERENCE_ERROR = 4
EXIT_STORAGE_ERROR = 5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sentinela-runtime", description="Sentinela Edge Runtime")
    parser.add_argument(
        "--config-dir", help="Directory with node/vision/logging YAML (default: <repo>/config)"
    )
    parser.add_argument("--source", help="Override camera source: webcam index, rtsp:// URL or video file")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], help="Override inference device")
    parser.add_argument("--no-preview", action="store_true", help="Disable the local preview window")
    parser.add_argument("--log-level", choices=sorted(VALID_LEVELS), help="Override the sentinela log level")
    return parser.parse_args(argv)


def build_event_engine(
    config: RuntimeConfig, backend: InferenceBackend, store: SqliteEventStore
) -> EventEngine:
    metadata, runtime = backend.get_metadata(), backend.get_runtime_info()
    context = EventContext(
        node_id=config.node.node.node_id,
        camera_id=config.node.camera.camera_id,
        model_id=metadata.model_id,
        model_version=metadata.version,
        model_sha256=metadata.sha256,
        inference_backend=runtime.label,
        config_hash=config.fingerprint(),
    )
    return EventEngine(
        context,
        config.vision.events,
        sinks=[store, LoggingEventSink()],  # persist locally first
        evidence=SnapshotStore(config.data_dir, config.node.storage.snapshots),
    )


def build_zone_evaluator(config: RuntimeConfig) -> ZoneEvaluator:
    events = config.vision.events
    evaluator = ZoneEvaluator.from_config(
        config.node.zones, events.zone_entry_frames, events.zone_exit_frames
    )
    if evaluator.zones:
        log.info("Zones loaded", extra={"zones": [z.zone_id for z in evaluator.zones]})
    else:
        log.warning("No zones configured; zone_entry/zone_exit events are disabled")
    return evaluator


def run_vision(config: RuntimeConfig, lifecycle: Lifecycle) -> int:
    """Open local storage, load the model, then run the pipeline until the source ends or a stop."""
    try:
        store = SqliteEventStore(config.database_path)
    except StorageError as exc:
        log.error("Event store initialisation failed: %s", exc)
        return EXIT_STORAGE_ERROR

    with store:
        model_path = config.resolve_path(config.vision.model.path)
        try:
            backend = load_backend(config.vision.model, config.vision.inference, model_path)
        except InferenceError as exc:
            log.error("Model initialisation failed: %s", exc)
            return EXIT_MODEL_ERROR

        with backend:
            pipeline = VisionPipeline(
                source=VideoSource(config.node.camera, stop_event=lifecycle.stop_event),
                backend=backend,
                tracker=create_tracker(config.vision.tracking),
                zones=build_zone_evaluator(config),
                engine=build_event_engine(config, backend, store),
                settings=PipelineSettings.from_config(config),
                stop_event=lifecycle.stop_event,
            )
            return _run_pipeline(pipeline)


def _run_pipeline(pipeline: VisionPipeline) -> int:
    try:
        pipeline.run()
    except CameraError as exc:
        log.error("Camera failure: %s", exc)
        return EXIT_CAMERA_ERROR
    except PipelineError as exc:
        log.error("Pipeline aborted: %s", exc)
        return EXIT_INFERENCE_ERROR
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    bootstrap_logging()
    args = parse_args(argv)
    try:
        config = apply_overrides(
            load_runtime_config(args.config_dir),
            source=args.source,
            device=args.device,
            preview_enabled=False if args.no_preview else None,
        )
    except ConfigError as exc:
        log.error("%s", exc)
        return EXIT_CONFIG_ERROR

    setup_logging(config.logging_config_path, config.base_dir, args.log_level)
    log.info(
        "Sentinela runtime starting",
        extra={
            "node_id": config.node.node.node_id,
            "camera_id": config.node.camera.camera_id,
            "config_hash": config.fingerprint(),
            "event_store": str(config.database_path),
        },
    )
    with Lifecycle() as lifecycle:
        return run_vision(config, lifecycle)


if __name__ == "__main__":
    raise SystemExit(main())

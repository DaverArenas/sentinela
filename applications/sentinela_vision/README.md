# Sentinela Vision

The first application built on the Sentinela Platform: local camera ingestion, person detection,
tracking, polygon zone evaluation and structured events, all on the edge node.

## Where things live (Phase 1)

| Concern | Location |
|---|---|
| Vision pipeline code | `edge-runtime/src/sentinela_runtime/vision/` |
| Tracking (tracker adapter interface + IoU tracker) | `edge-runtime/src/sentinela_runtime/vision/tracking/` |
| Polygon zones and entry/exit evaluation | `edge-runtime/src/sentinela_runtime/vision/zones/` |
| Events (schema, rules, engine) | `edge-runtime/src/sentinela_runtime/events/` |
| SQLite event store and snapshot evidence | `edge-runtime/src/sentinela_runtime/storage/` |
| Node, camera, storage and zone configuration | `config/node_config.yaml` |
| Model, inference, sampling, tracking and event rules | `config/vision_config.yaml` |
| Model provenance | `models/README.md` |

For Phase 1 the runtime runs this single application directly. No application or plugin
framework exists yet (see CLAUDE.md §7.1); `config/` here is reserved for application-specific
configuration once the runtime hosts more than one application.

## How a zone event is produced

1. **Detection**: the inference backend returns canonical detections for each sampled frame.
2. **Tracking**: the IoU tracker associates detections across frames (class-aware, with
   constant-velocity prediction). A track is confirmed after `tracking.min_hits` matches and
   receives an ID such as `track-1`. It survives `tracking.max_lost_frames` unseen frames, so
   short occlusions keep the same ID.
3. **Zones**: each track's anchor point (feet by default) is tested against every polygon zone
   for its classes. Entry needs `events.zone_entry_frames` consecutive frames inside; exit needs
   `events.zone_exit_frames` frames outside, or the track ending while inside.
4. **Events**: each transition becomes one `zone_entry` / `zone_exit` event (schema 1.0) with
   track ID, zone, confidence, bounding box and provenance (model, SHA-256, backend, config hash,
   frame index).
5. **Persistence**: the triggering frame is saved as a JPEG snapshot (for `zone_entry` by default),
   the event is committed to SQLite, then logged to `logs/events.jsonl`.

## Phase 1 status

| Deliverable | Status |
|---|---|
| USB webcam / RTSP / recorded-video ingestion with reconnect | Done (webcam verified on the RTX 3060 node; RTSP not yet tested on a real camera) |
| Frame preprocessing | Done |
| Common inference interface + ONNX Runtime (CUDA/CPU) backend | Done |
| Pretrained detector (YOLOX-S, commercial review pending) | Done |
| Canonical detections + event schema 1.0 | Done |
| Object tracking (tracker adapter + IoU tracker) | Done |
| Polygon zones + person zone entry/exit events | Done |
| Structured logging + pipeline metrics | Done |
| SQLite event store + snapshot evidence | Done |
| Recorded-video test suite | Done (synthetic clips + scripted detector; real consented acceptance footage pending) |
| RTX 3060 benchmark | Done (see below) |

## Benchmark: RTX 3060 Laptop GPU

Measured with `uv run sentinela-benchmark` on 2026-11-19. YOLOX-S `0.1.1rc0` ONNX, 640×640 input,
synthetic 1280×720 frames. Inference covers preprocessing, the model and postprocessing; camera I/O is excluded.
Environment: driver 595.99, ONNX Runtime GPU 1.32.0.

| Backend | Frames | Inference mean | Inference p95 | Tracking + zones mean | Throughput |
|---|---|---|---|---|---|
| `onnxruntime-cuda` | 500 | 14.29 ms | 16.59 ms | 0.02 ms | 69.9 FPS |
| `onnxruntime-cpu` | 100 | 89.70 ms | 114.84 ms | 0.02 ms | 11.1 FPS |

Live webcam run on the same node (pipeline metrics, 640×480, CUDA): 17.3 ms mean / 18.4 ms p95
inference, about 1.2 GiB peak RSS. Throughput was limited by the camera (about 11 FPS), not by inference.

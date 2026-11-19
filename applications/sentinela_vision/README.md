# Sentinela Vision

The first application built on the Sentinela Platform: local camera ingestion, person detection,
tracking, polygon zone evaluation and structured events, all on the edge node.

## Where things live (Phase 1)

| Concern | Location |
|---|---|
| Vision pipeline code | `edge-runtime/src/sentinela_runtime/vision/` |
| Events (schema, rules, engine) | `edge-runtime/src/sentinela_runtime/events/` |
| Node / camera configuration | `config/node_config.yaml` |
| Model, inference, sampling, event rules | `config/vision_config.yaml` |
| Model provenance | `models/README.md` |

For Phase 1 the runtime runs this single application directly. No application or plugin
framework exists yet (see CLAUDE.md §7.1); `config/` here is reserved for application-specific
configuration once the runtime hosts more than one application.

## Phase 1 status

| Deliverable | Status |
|---|---|
| USB webcam / RTSP / recorded-video ingestion with reconnect | Done |
| Frame preprocessing | Done |
| Common inference interface + ONNX Runtime (CUDA/CPU) backend | Done |
| Pretrained detector (YOLOX-S, commercial review pending) | Done |
| Canonical detections + event schema 1.0 | Done |
| Structured logging + pipeline metrics | Done |
| Recorded-video test suite | Done (synthetic clips; real acceptance footage pending) |
| Object tracking | Next |
| Polygon zones + zone entry/exit events | Next |
| SQLite event store + evidence snapshots | Next |
| RTX 3060 benchmark | Blocked on NVIDIA driver (CUDA device not detected) |

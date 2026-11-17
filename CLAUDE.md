# Project Sentinela — Engineering Context & Architectural Guardrails

## 1. Product Identity & Mission
**Sentinela** is a dual-tech Edge AI and AIoT platform providing autonomous physical intelligence for security, industrial operations, smart agriculture, and defense. 
- **Core Value Proposition:** Zero-cloud-dependency perception and actuation. The edge node processes video and microcontroller sensor streams locally in real time, acting immediately offline and syncing only lightweight metadata upstream.

## 2. Active Development State
- **Current Phase:** Phase 0 Complete -> Transitioning to **Phase 1 (Local AI Vision Core)**.
- **Primary Dev Environment:** ASUS TUF A15 (NVIDIA GeForce RTX 3060, CUDA enabled).
- **Package Manager:** `uv` (Astral). Always use `uv` commands (`uv init`, `uv add`, `uv run`) and `pyproject.toml` rather than legacy `pip` or raw `requirements.txt`.

## 3. Strict Architectural Rules (Do Not Violate)
1. **Local-First Autonomy:** Never place a cloud API call inside the critical path of video inference, sensor evaluation, or local actuator triggering.
2. **Bandwidth Discipline:** Never stream raw video frames upstream by default. Output structured JSON event payloads (event type, confidence, bounding box, zone ID, sensor state, ISO-8601 timestamp).
3. **Configuration Over Hardcoding:** All thresholds, camera sources (webcam index vs. RTSP URL), MQTT topics, and restricted zone coordinates must be loaded from `config/node_config.yaml`.
4. **Hardware Portability:** Keep hardware-specific inference backends abstracted behind a clean interface so the same application logic runs on x86 CUDA (RTX 3060), ARM64 TensorRT (Jetson), and HailoRT (Raspberry Pi AI HAT+).
5. **Clean Decoupling:** Vision processing, MQTT sensor ingestion, and local SQLite persistence must run as modular, loosely coupled components within `edge-runtime/src/`.

## 4. Coding Standards
- Python 3.11+, strict type hints (`typing`), structured logging (`logging` module, not bare `print` statements in production modules), and clean docstrings.
- Before writing code for a new phase, confirm alignment with the folder structure defined in `README.md`.


## 5. Engineering Principles

1. Architecture Before Implementation
   - Read README.md and this file before modifying code.
   - Inspect the existing repository structure before creating files.
   - Do not introduce new architectural layers without explaining why.
   - Prefer small, testable modules over large monolithic files.

2. Vision Pipeline Separation
   - Separate camera ingestion, preprocessing, inference,
     tracking, zone evaluation, and event generation.
   - Never couple business rules directly to a particular AI model.
   - Vision logic must work with recorded video fixtures as well
     as live cameras.
   - Support configurable frame sampling and inference rates.

3. Model and Hardware Abstraction
   - Use a common inference interface.
   - Keep model-specific code inside backend adapters.
   - Record model name, version, format, source, and license.
   - Do not assume every model runs on every hardware accelerator.
   - Never silently change model versions.

4. Event-Driven Architecture
   - Vision and sensor modules produce structured events.
   - Event processing must not depend on cloud connectivity.
   - Events must include unique IDs, timestamps, source IDs,
     confidence, and schema versions.
   - Preserve the original detection data and event provenance.

5. Reliability and Safety
   - A camera failure must not crash the entire runtime.
   - Implement reconnect and timeout handling.
   - Avoid unbounded video-frame queues.
   - Critical local actions must have explicit safety limits.
   - AI detections must not be treated as safety-certified signals.
   - Actuators must have safe defaults and manual overrides.

6. Testing
   - Add unit tests for geometry, event rules, and configuration.
   - Add integration tests for camera-to-event processing.
   - Use recorded video for repeatable vision tests.
   - Test internet disconnection and service restarts.
   - Measure latency, throughput, memory, and dropped frames.

7. Development Workflow
   - Use uv for Python project and dependency management.
   - Use type hints and structured logging.
   - Add tests before considering a feature complete.
   - Update README.md when architecture or commands change.
   - Never add dependencies without explaining their purpose.
   - Never commit secrets, API keys, credentials, or private footage.

## 6. Phase 1 Scope Restrictions

Phase 1 is exclusively the local computer-vision core.

Do not implement:
- Cloud services
- Multi-device orchestration
- Kubernetes
- ESP32 firmware
- Sensor fusion
- Face recognition
- Autonomous high-impact actuation

Phase 1 must demonstrate:
- Local camera ingestion
- Object detection
- Object tracking
- Configurable polygon zones
- Structured event generation
- Local evidence storage
- Stable operation without internet

## 7. Runtime and Control Plane Evolution

The Sentinela architecture must evolve incrementally. Future platform capabilities must not become premature dependencies of Phase 1.

### 7.1 Runtime and Application Separation

- Keep reusable runtime infrastructure separate from application-specific functionality.
- Sentinela Vision is the first application built on the platform.
- Do not introduce a generalized plugin or application-management framework until a concrete requirement exists.
- Keep camera ingestion, inference execution, tracking, zones, event processing, and persistence behind well-defined interfaces.

### 7.2 API Independence

- FastAPI may be introduced in future phases to expose local Runtime functionality and central Control Plane functionality.
- The critical inference and event-processing paths must not depend on HTTP API availability.
- Local processing must continue if the API, dashboard, or Control Plane is unavailable.
- Keep local Runtime APIs conceptually separate from central fleet-management APIs.

### 7.3 Containerization

- Docker and Docker Compose are future deployment technologies, not Phase 1 requirements.
- Do not containerize the Phase 1 application unless explicitly requested or technically justified.
- Keep deployment configuration reproducible and avoid unnecessary dependencies on container-specific behavior.
- GPU access, drivers, accelerator runtimes, persistent storage, and device permissions must be considered when designing future container deployments.
- Do not assume a single container image is portable across all accelerator architectures.

### 7.4 Local-First Event Persistence

- Persist important events and evidence locally before optional upstream synchronization.
- Local event generation and persistence must not require internet connectivity.
- Design event processing to tolerate temporary service failures and restarts.
- Preserve event IDs, timestamps, schema versions, and provenance across retries and synchronization.

## 8. Phase 1 Implementation Boundaries

Phase 1 must remain a focused local computer-vision application.

Required:
- Python 3.11+ and uv.
- Ubuntu Linux development environment.
- USB webcam and RTSP camera support.
- Local person detection using the selected inference backend.
- Canonical detection output.
- Object tracking.
- Configurable polygon-based restricted zones.
- Structured event generation.
- SQLite persistence and local evidence storage.
- Repeatable tests using recorded video.
- Operation without internet connectivity.

Not required:
- Docker or Docker Compose.
- FastAPI or a web dashboard.
- MQTT sensor ingestion.
- Cloud synchronization.
- Central device management.
- Multi-device orchestration.
- Kubernetes or K3s.
- Face recognition.
- Autonomous high-impact actuation.

Any proposed addition to Phase 1 must include a clear technical justification and must not compromise its scope.
# Project Sentinela
## Autonomous Physical Intelligence at the Edge

**Sentinela is a local-first Edge AI platform for deploying, operating, and managing intelligent applications on distributed physical devices.**

It combines computer vision, sensor integration, local inference, event processing, and edge automation. Applications run close to cameras, sensors, and machines, without requiring cloud connectivity for their core operations.

Our long-term vision is to provide reusable edge infrastructure for security, industrial operations, smart agriculture, environmental monitoring, and other physical-world applications.

---

# 1. Product Vision and Identity

## 1.1 The Platform

**Sentinela Platform** is the reusable technology infrastructure for deploying and operating AI applications at the edge.

It consists of three principal capabilities:

| Platform component | Responsibility |
|---|---|
| Sentinela Runtime | Executes and manages applications, AI inference, device integrations, events, and local data |
| Sentinela Control Plane | Manages edge devices, application deployments, model versions, configurations, and fleet health |
| Intelligence Layer | Correlates events from multiple cameras, sensors, and devices to generate operational intelligence |

The platform is designed to be hardware-agnostic, modular, and local-first.

## 1.2 Applications

Applications are independently deployable solutions built on top of the Sentinela Platform.

**Initial application: Sentinela Vision**

A local computer-vision application that processes camera streams, detects and tracks objects, evaluates configurable spatial zones, and generates structured events.

Future applications may include:

- Sentinela Sentinel — industrial perimeter and site monitoring.
- Sentinela AIoT — environmental monitoring and intelligent automation.
- Sentinela Inspect — infrastructure and asset inspection.
- Sentinela Intelligence — multi-camera and multi-sensor event correlation.

These are product directions, not Phase 1 implementation requirements.

## 1.3 Core Value Proposition

- Local-first AI inference and event processing.
- No cloud dependency in the critical path.
- Hardware-independent application logic.
- Integration with existing cameras and sensors.
- Structured events rather than continuous upstream video streaming.
- Offline operation and eventual synchronization.
- Centralized management of distributed edge devices.

---

# 2. Core Architectural Principles

### 2.1 Local-First Autonomy

All critical video processing, inference, event evaluation, and authorized local automation must operate without requiring internet access.

Cloud connectivity is optional and is used for centralized management, synchronization, monitoring, and remote updates.

### 2.2 Hardware Abstraction

Application logic must not depend directly on a particular AI framework or hardware accelerator.

Inference engines must be accessed through a common interface.

Hardware-specific execution is implemented through backend adapters.

### 2.3 Separation of Responsibilities

Camera ingestion, preprocessing, inference, tracking, spatial evaluation, event generation, storage, and messaging must be independent modules.

A change to the AI model must not require rewriting application rules or event processing.

### 2.4 Bandwidth Discipline

Raw video remains local by default.

The system produces structured metadata and event notifications. Images, clips, and other evidence are retained locally unless an authorized policy explicitly permits their transfer.

### 2.5 Configuration-Driven Operation

Camera sources, inference thresholds, model paths, tracking settings, zones, MQTT topics, and application rules must be configurable.

Avoid hardcoding operational parameters.

### 2.6 Security and Reliability

- Encrypted communication between authorized components.
- Unique device identities.
- Secure credential management.
- Bounded queues and storage.
- Reconnection and recovery mechanisms.
- Auditable configuration and deployment changes.
- Safe defaults and manual overrides for physical actuators.

### 2.7 Cascaded Wake-on-Event Inference

Some applications use a cascade architecture. Tier-2 TinyML microcontrollers act as always-on, ultra-low-power sentinels: acoustic keyword or sound spotting, vibration/IMU anomaly detection, and mmWave presence. Running heavy video inference 24/7 uses a lot of compute. In cascaded mode the Peripheral Mesh instead wakes the Tier-1 Vision Runtime over MQTT only when a physical anomaly occurs. This greatly reduces power consumption, thermal load, and false positives.

| Tier | Role | Hardware | Duty cycle |
|---|---|---|---|
| Tier-1 — Vision Runtime | Heavy perception: detection, tracking, zones, events, evidence | Edge computer (RTX 3060, Jetson, Raspberry Pi + Hailo) | On demand when cascaded, continuous otherwise |
| Tier-2 — Peripheral Mesh | Always-on anomaly sentinels using TinyML | ESP32 / TinyML microcontrollers, mmWave and IMU sensors | Always on, ultra-low power |

Guardrails:

- **Opt-in per application.** Continuous inference remains the default. Cascaded mode is enabled in `config/node_config.yaml` only for applications that benefit from it.
- **Local wake path.** Wake signals travel over the local MQTT broker. A cloud service must never sit between a Tier-2 trigger and Tier-1 activation.
- **Silence is not "all clear".** If the mesh or broker becomes unreachable, or a sentinel misses its heartbeat, the Vision Runtime falls back to continuous or duty-cycled inference. A missing wake signal must never be treated as the absence of a threat.
- **Bounded wake windows.** Wake duration, cooldown, and re-trigger limits are configurable. This prevents wake storms and runaway power use.
- **Wake latency is budgeted.** The time from a Tier-2 trigger to the first Tier-1 inference must be measured. The idle state is configurable (for example camera streaming with inference paused, or a warm model with the camera idle) to trade power against responsiveness.
- **Provenance is preserved.** Wake triggers are structured events. Vision events generated during a wake window reference the triggering wake event ID.
- **Not safety-certified.** As with vision detections, TinyML triggers are probabilistic signals, not safety-certified inputs.

---

# 3. System Architecture

Sentinela follows a distributed architecture with three principal platform components and an optional peripheral-device layer.

```text
+---------------------------------------------------------------+
|                 SENTINELA CONTROL PLANE                       |
|                                                               |
| Device Registry | Application Registry | Model Registry       |
| Deployment Manager | Configuration | Monitoring               |
| Central API | Fleet Dashboard | OTA Management                |
+-----------------------------+---------------------------------+
                              |
                   Secure MQTT / HTTPS
                    Metadata and Config
                              |
+-----------------------------v---------------------------------+
|                    SENTINELA EDGE RUNTIME                     |
|                                                               |
| +----------------------+  +-------------------------------+   |
| | Application Manager  |  | Local Device Connectors       |   |
| +----------------------+  +-------------------------------+   |
|                                                               |
| +----------------------+  +-------------------------------+   |
| | Vision Pipeline      |  | Sensor Ingestion              |   |
| |                      |  |                               |   |
| | Camera Ingestion     |  | MQTT / Serial / GPIO          |   |
| | Preprocessing        |  |                               |   |
| | Inference Interface  |  +-------------------------------+   |
| | Tracking             |                                      |
| | Zone Evaluation      |  +-------------------------------+   |
| +----------------------+  | Event Engine                  |   |
|                           +-------------------------------+   |
|                                                               |
| +----------------------+  +-------------------------------+   |
| | Local SQLite Store   |  | Synchronization Agent         |   |
| +----------------------+  +-------------------------------+   |
|                                                               |
+-----------------------------+---------------------------------+
                              |
                   Local network / MQTT
                              |
+-----------------------------v---------------------------------+
|                  PERIPHERAL DEVICE LAYER                      |
|                                                               |
| ESP32 Nodes | ESP-Mosaico | TinyML Devices | Sensors          |
|                                                               |
| Temperature | Humidity | Motion | Vibration | Other Inputs    |
| Relays | Pumps | Fans | Other Authorized Outputs              |
+---------------------------------------------------------------+
```

## 3.1 Sentinela Runtime

Installed on an edge computer running a supported operating system.

Responsibilities:

- Application lifecycle.
- Camera and sensor connectivity.
- Local AI inference.
- Event generation and processing.
- Local storage.
- Local messaging.
- Synchronization.
- Health monitoring.

## 3.2 Sentinela Control Plane

A central management service deployed in cloud, on-premises, or hybrid infrastructure.

Responsibilities:

- Device registration and identity.
- Application and model registries.
- Configuration management.
- Software and model deployment.
- Device health and telemetry.
- Remote updates.
- Fleet monitoring.

The Control Plane must not be required for ongoing local inference.

## 3.3 Intelligence Layer

Transforms individual detections and sensor observations into meaningful operational information.

Future capabilities:

- Multi-camera event correlation.
- Multi-sensor fusion.
- Incident management.
- Site-level operational intelligence.
- Geospatial event representation.

The initial implementation will use deterministic rules and event processing rather than attempting to build a generalized reasoning engine.

## 3.4 Peripheral Device Layer

Provides connectivity to external sensors, microcontrollers, and actuators.

Supported and planned technologies:

- ESP32 development boards.
- ESP-Mosaico modular hardware.
- ESP-IDF and ESPHome.
- TinyML development kits.
- MQTT, Wi-Fi, serial, and supported hardware interfaces.

Peripheral devices are not necessarily edge computers. They can collect measurements, perform lightweight local inference, and execute authorized physical actions.

In cascaded deployments (see [2.7](#27-cascaded-wake-on-event-inference)) this layer forms the **Tier-2 Peripheral Mesh**. Its TinyML sentinels publish wake events over local MQTT to activate the Tier-1 Vision Runtime on demand.

---

# 4. Vision Architecture: Common Inference Interface

A fundamental architectural requirement is to separate model execution from computer-vision application logic.

Sentinela will implement a common inference abstraction inspired by the design philosophy of MLflow PyFunc.

The purpose is to allow different models and hardware backends to be accessed through a consistent application interface.

## 4.1 Architectural Pattern

```text
              SENTINELA VISION APPLICATION
                            |
                            |
                  Common Inference API
                            |
             +--------------+--------------+
             |              |              |
          ONNX Runtime   TensorRT        HailoRT
             |              |              |
             +--------------+--------------+
                            |
                      Hardware Layer
                            |
           NVIDIA GPU / CPU / Hailo Accelerator
```

## 4.2 Initial Interface Responsibilities

The inference abstraction must support:

- Model loading and initialization.
- Input validation.
- Model execution.
- Standardized output.
- Model metadata.
- Runtime information.
- Resource cleanup.

The initial conceptual interface:

```python
class InferenceBackend:
    def load_model(self, model_path: str) -> None:
        ...

    def predict(self, frame):
        ...

    def get_metadata(self):
        ...

    def close(self) -> None:
        ...
```

The interface will evolve into typed request and response objects.

All backends must normalize model-specific outputs into Sentinela's canonical detection format.

Example:

```python
@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
```

The application must not contain backend-specific inference logic.

## 4.3 Planned Inference Backends

| Backend | Target hardware | Development priority |
|---|---|---|
| ONNX Runtime CPU | x86 / ARM CPU | Initial portability |
| ONNX Runtime CUDA | NVIDIA GPU | Initial development |
| ONNX Runtime TensorRT EP | NVIDIA GPU | Optimization |
| Native TensorRT | NVIDIA Jetson / GPU | Future optimization |
| HailoRT | Hailo accelerator | Future hardware port |

The initial implementation will prioritize ONNX Runtime with CUDA on the development workstation.

Other backends are architectural extension points, not Phase 1 dependencies.

---

# 5. Vision Pipeline Architecture

The computer-vision application must separate the following responsibilities.

```text
Camera Source
     |
     v
Video Ingestion
     |
     v
Frame Preprocessing
     |
     v
Inference Backend
     |
     v
Canonical Detection Results
     |
     v
Object Tracking
     |
     v
Spatial / Zone Evaluation
     |
     v
Event Engine
     |
     +------> SQLite Persistence
     |
     +------> Local API / Dashboard
     |
     +------> MQTT / Future Control Plane
```

## 5.1 Video Ingestion

Responsibilities:

- USB webcam support.
- IP camera RTSP support.
- Frame capture.
- Connection monitoring.
- Automatic reconnect.
- Frame-rate and resolution configuration.
- Wake-on-event activation for cascaded deployments (Phase 3+). Ingestion and inference can be paused or resumed by a local wake controller without changing the downstream pipeline stages.

## 5.2 Preprocessing

Responsibilities:

- Image resizing.
- Color-space conversion.
- Normalization.
- Tensor preparation.
- Model-specific input formatting.

## 5.3 Inference

Responsibilities:

- Model loading.
- Hardware backend selection.
- Model execution.
- Model-specific postprocessing.
- Standardized detections.

## 5.4 Object Tracking

Responsibilities:

- Persistent track IDs.
- Object association across frames.
- Track lifecycle.
- Lost-track handling.
- Duplicate event prevention.

## 5.5 Spatial Evaluation

Responsibilities:

- Configurable polygon zones.
- Object-in-zone evaluation.
- Zone entry and exit detection.
- Class-specific rules.
- Spatial configuration.

## 5.6 Event Engine

Responsibilities:

- Convert detections and tracking states into application events.
- Apply business rules.
- Deduplicate repeated detections.
- Generate structured event payloads.
- Preserve event provenance.

Vision processing must not depend directly on the MQTT implementation or database.

---

# 6. Technology Stack

The stack below distinguishes initial dependencies from future options.

## 6.1 Phase 1 Core

| Area | Technology |
|---|---|
| Language | Python 3.11+ |
| Package management | uv |
| Configuration | YAML |
| Video ingestion | OpenCV |
| Video streaming extension | GStreamer |
| Inference interface | Custom Sentinela abstraction |
| Initial inference engine | ONNX Runtime GPU |
| Model candidates | RT-DETR / YOLOX |
| Tracking | Modular tracker adapter |
| Geometry | Shapely or equivalent |
| Local storage | SQLite |
| Logging | Python logging |
| Testing | pytest |
| Type checking | mypy or pyright |

## 6.2 Peripheral and AIoT Layer

| Area | Technology |
|---|---|
| Microcontrollers | ESP32 |
| Firmware | ESP-IDF / ESPHome |
| Modular hardware | ESP-Mosaico |
| TinyML | Existing TinyML development kit |
| Messaging | Eclipse Mosquitto / MQTT |
| Integration | Python MQTT client |

## 6.3 Future Runtime and Control Plane

| Area | Technology |
|---|---|
| Containers | Docker |
| Local multi-container deployment | Docker Compose |
| Backend API | FastAPI |
| Central database | PostgreSQL |
| Fleet messaging | MQTT / HTTPS |
| Monitoring | Prometheus / Grafana |
| Multi-node orchestration | K3s or other suitable fleet infrastructure |
| Edge fleet management | Evaluate Open Horizon, balena, and other alternatives |

Future technology options are not required dependencies for Phase 1.

---

# 7. AI Model Selection and Commercial Licensing

Sentinela prioritizes OS models and dependencies with clear commercial-use rights.

Initial model evaluation candidates:

- RT-DETR.
- YOLOX.
- Other models with suitable commercial licensing and deployment characteristics.

Model selection must consider:

- Detection quality.
- Person-detection recall and precision.
- Inference latency.
- Throughput.
- Hardware compatibility.
- Model size.
- Export compatibility.
- Commercial licensing.

Every model must have a documented provenance record containing:

- Model name and version.
- Original repository.
- Model artifact source.
- Code license.
- Weight license.
- Training dataset provenance and relevant restrictions.
- Required attribution.
- Modification history.

A permissive code license does not automatically establish that pretrained weights or datasets are commercially unrestricted.

No model should be treated as commercially approved until its relevant licenses have been reviewed.

---

# 8. Initial Development Environment

Primary development workstation:

**ASUS TUF A15**
- NVIDIA GeForce RTX 3060.
- Ubuntu Linux installed directly.
- CUDA-enabled development environment.

Development standards:

- Python 3.11+.
- uv for project and dependency management.
- pyproject.toml for dependency declarations.
- Strict type hints.
- Structured logging.
- Modular source code.
- Unit and integration tests.

Use:

```bash
uv init
uv add
uv run
uv sync
```

Avoid unmanaged global Python dependencies.

## 8.1 Running the Edge Runtime (Phase 1)

All commands run from `edge-runtime/`. Configuration is read from the repository `config/` directory.

```bash
cd edge-runtime

# Install (pick exactly one ONNX Runtime build)
uv sync --extra cuda        # NVIDIA GPU: onnxruntime-gpu, CUDA 13 (driver >= 580)
uv sync --extra cpu         # CPU-only / portability

# One-time model download (see models/README.md for provenance and checksum)
curl -L -o ../models/yolox_s.onnx \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx

# Run Sentinela Vision
uv run sentinela-runtime                                   # camera from config/node_config.yaml
uv run sentinela-runtime --source ../path/to/recording.mp4 # recorded video
uv run sentinela-runtime --source rtsp://camera/stream --no-preview --device cuda

# Inspect stored events (Python's built-in SQLite shell)
uv run python -m sqlite3 ../data/sentinela.db \
  "SELECT timestamp, event_type, track_id, zone_id, snapshot_path FROM events ORDER BY timestamp"

# Inference benchmark (model + backend + device from config; camera I/O excluded)
uv run sentinela-benchmark --device cuda --output ../logs/benchmark-cuda.json

# Quality gates
uv run pytest               # unit + integration tests, coverage gate >= 80%
uv run ruff check src ../tests && uv run ruff format --check src ../tests
uv run mypy                 # strict type checking
```

- Zones are polygons in `config/node_config.yaml` (`zones:`), with `[x, y]` vertices normalised to 0..1 of the frame. The preview draws them (yellow when empty, red when occupied) together with track IDs.
- Tracking (`tracking:`) and zone entry/exit debouncing (`events:`) are tuned in `config/vision_config.yaml`.
- Every event is stored in SQLite at `data/sentinela.db` first, then appended to `logs/events.jsonl`. `zone_entry` events also save the triggering frame to `data/snapshots/<date>/<event_id>.jpg`. `data/` and `logs/` stay on the node and are git-ignored.
- RTSP credentials belong in `SENTINELA_CAMERA_SOURCE`, not in YAML.
- If CUDA is unavailable, the runtime falls back to CPU and records `onnxruntime-cpu` in each event's provenance.
- Exit codes: `0` ok, `1` camera, `2` configuration, `3` model, `4` inference, `5` event storage.

---

# 9. Repository Structure

```text
sentinela/
│
├── .claude.md
├── README.md
├── .gitignore
│
├── config/
│   ├── node_config.yaml
│   ├── vision_config.yaml
│   └── logging_config.yaml
│
├── edge-runtime/
│   ├── pyproject.toml
│   ├── Dockerfile
│   └── src/
│       └── sentinela_runtime/
│           ├── main.py
│           │
│           ├── core/
│           │   ├── config.py
│           │   ├── lifecycle.py
│           │   └── interfaces.py
│           │
│           ├── vision/
│           │   ├── camera/
│           │   ├── preprocessing/
│           │   ├── inference/
│           │   │   ├── base.py
│           │   │   ├── model_manager.py
│           │   │   └── backends/
│           │   ├── tracking/
│           │   ├── zones/
│           │   └── pipeline.py
│           │
│           ├── events/
│           │   ├── schemas.py
│           │   ├── rules.py
│           │   └── event_engine.py
│           │
│           ├── messaging/
│           ├── storage/
│           └── monitoring/
│
├── applications/
│   └── sentinela_vision/
│       ├── README.md
│       └── config/
│
├── sensor-mesh/
│   ├── esp32_node/
│   ├── esp_mosaico/
│   └── tinyml/
│
├── control-plane/
│   └── backend/
│
├── dashboard/
│
├── models/
│   └── README.md
│
├── infra/
│   ├── docker-compose.yml
│   └── mosquitto/
│
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

The Control Plane, dashboard, and sensor mesh directories are reserved for future implementation. They are not Phase 1 requirements.

---

# 10. Event Architecture

All applications must use a common structured event format.

Example:

```json
{
  "schema_version": "1.0",
  "event_id": "evt-001",
  "event_type": "zone_entry",
  "timestamp": "2026-11-16T20:15:30Z",
  "node_id": "edge-node-001",
  "camera_id": "camera-001",
  "object": {
    "class": "person",
    "track_id": "track-42",
    "confidence": 0.94,
    "bounding_box": {
      "x1": 120,
      "y1": 80,
      "x2": 340,
      "y2": 460
    }
  },
  "zone": {
    "zone_id": "zone-001",
    "zone_name": "Restricted Area"
  },
  "evidence": {
    "snapshot_path": "events/evt-001.jpg"
  },
  "processing": {
    "model_id": "person-detector",
    "model_version": "1.0",
    "inference_backend": "onnxruntime-cuda"
  }
}
```

The event architecture must support:

- Unique event identifiers.
- Schema versioning.
- ISO-8601 timestamps in UTC.
- Source identification.
- Model and configuration provenance.
- Local persistence.
- Event deduplication.
- Optional sensor observations.
- Evidence references.
- Wake-event provenance for cascaded deployments. Events produced during a wake window carry the ID of the Tier-2 trigger that activated the Vision Runtime.

---

# 11. Development Roadmap

## Phase 0 — Architecture and Foundation

Status: Defined.

- Product identity.
- Platform/application distinction.
- Hardware abstraction strategy.
- Development environment.
- Repository architecture.
- Engineering guardrails.

## Phase 1 — Sentinela Vision Core

Current priority.

Objective: Build a reliable local computer-vision engine.

Deliverables:

1. USB webcam and RTSP camera ingestion.
2. Frame preprocessing.
3. Common inference interface.
4. Initial ONNX Runtime CUDA backend.
5. Commercially suitable pretrained detection model.
6. Canonical detection schema.
7. Object tracking.
8. Configurable polygon zones.
9. Person zone-entry and zone-exit events.
10. Structured logging.
11. Local event persistence.
12. Recorded-video test suite.
13. Performance benchmark on RTX 3060.

Acceptance demonstration:

A person enters a configured zone. Sentinela detects the person, tracks the object, identifies the zone crossing, creates one structured event, and stores the evidence locally.

## Phase 2 — Local Application and Event Services

Objective: Make Sentinela Vision usable as a standalone local product.

Deliverables:

- Local event API.
- Basic dashboard.
- SQLite event repository.
- Snapshot and evidence management.
- Application health monitoring.
- Service recovery.
- Configurable event rules.
- Offline operation testing.

## Phase 3 — AIoT Peripheral Integration

Objective: Integrate physical sensors and microcontrollers.

Deliverables:

- Local MQTT broker.
- ESP32 integration.
- ESP-Mosaico development.
- TinyML experiments.
- Sensor telemetry ingestion.
- Sensor state management.
- Event correlation.
- Controlled actuator integration.
- Cascaded wake-on-event inference: TinyML sentinels (acoustic, vibration/IMU, mmWave) waking the Vision Runtime over local MQTT, with heartbeat-based fallback to continuous inference.
- Power, thermal, wake-latency, and false-positive measurements comparing cascaded and continuous modes.

## Phase 4 — Runtime and Containerization

Objective: Make Sentinela deployable and maintainable on dedicated edge computers.

Deliverables:

- Docker packaging.
- Docker Compose deployment.
- Runtime lifecycle management.
- Hardware-specific inference adapters.
- Local configuration management.
- Store-and-forward synchronization.
- Recovery and monitoring.

## Phase 5 — Control Plane

Objective: Manage multiple edge computers from a central management platform.

Deliverables:

- Device registry.
- Device authentication.
- Application registry.
- Model registry.
- Remote deployment.
- Configuration distribution.
- Fleet monitoring.
- Remote updates.
- Central telemetry API.

## Phase 6 — Hardware Expansion and Commercial Pilots

Objective: Validate the platform on different edge hardware and in real operating environments.

Potential targets:

- NVIDIA Jetson.
- Industrial x86 computers.
- Raspberry Pi.
- Hailo accelerators.
- Distributed sensor networks.

Validate deployment reliability, model performance, hardware utilization, connectivity resilience, security, and commercial value.

---

# 12. Phase 1 Success Criteria

Phase 1 is complete when Sentinela can:

- Connect to a USB webcam.
- Connect to an RTSP camera.
- Load a model through the common inference interface.
- Run local person detection on the RTX 3060.
- Generate canonical detections.
- Track people across frames.
- Evaluate configurable polygon zones.
- Generate zone-entry and zone-exit events.
- Save event records and selected snapshots locally.
- Recover from camera disconnection.
- Operate without internet connectivity.
- Run repeatable tests against recorded footage.

Performance targets will be established after measuring the first working implementation rather than assumed in advance.

---

# 13. Long-Term Product Vision

Sentinela aims to become a reusable Edge AI platform for intelligent physical environments.

The long-term direction includes:

- Multi-camera intelligence.
- Sensor and vision fusion.
- Distributed event correlation.
- Industrial monitoring.
- Smart agriculture.
- Infrastructure inspection.
- Edge fleet management.
- Local autonomous operation.
- Integration with external operational and situational-awareness platforms.

Advanced technologies such as biometric recognition, specialized orchestration, and additional AI models may be evaluated in future product phases. They are not required for the initial MVP.

---

**Sentinela: Intelligence at the Edge. Autonomous by Design.
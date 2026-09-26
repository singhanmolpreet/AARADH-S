# Team Architecture & Project Plan

## 1. High-Level Architecture

The Rotax 914 Digital Twin integrates real-time thermodynamics, CAN bus communication, time-series storage, machine learning fault diagnostics, and an interactive 3D web cockpit.

```mermaid
graph TD
    A[Physics Core / Simulink Models] -->|Telemetry Frames| B(SocketCAN / vcan0)
    B -->|can_publisher/bridge.py| C(Eclipse Mosquitto MQTT)
    C -->|mqtt_client.py| D[FastAPI Backend]
    D -->|TimescaleDB| E[(PostgreSQL / TimescaleDB)]
    D -->|WebSocket /ws/live/{id}| F[Frontend Cockpit]
    G[ML Inference Service] -->|Subscribes Telemetry| C
    G -->|Publishes Health & Faults| C
    F -->|Inject Fault POST| D
    D -->|Publish Inject Command| C
    C -->|Command Listener| A
```

---

## 2. Subsystem Ownership & Deliverables

### P1: 3D Engine Visualization (`frontend/src/3d/`)
- **Deliverables**:
  - GLTF/GLB Rotax 914 asset rendering with PBR metallic-roughness materials.
  - Custom GLSL gradient shader mapping component health to dynamic surface heatmaps.
  - Mesh hierarchy mapping (cylinders 1–4, oil cooler, turbocharger/injectors, exhaust manifold).
  - Component raycasting and interactive selection.
  - Exploded-view interaction (smooth translation along mesh normals).

### P2: Operator Cockpit & Dashboard (`frontend/src/dashboard/`)
- **Deliverables**:
  - Real-time and historical multi-channel telemetry trend charts (RPM, CHT, EGT, oil pressure, boost).
  - Operator Diagnosis Card format for active faults (Severity badge, confidence, explanation, actionable emergency checklist).
  - Mission replay scrubber with playback speed controls (1x, 2x, 5x, pause/scrub).
  - Mission advisory panel visualizing RUL vs. flight plan and alternative diversion strategies.

### P3: Engine Physics Simulator & CAN Layer (`simulator/`)
- **Deliverables**:
  - Mean-value physics engine simulator implementing Otto cycle energy balance, manifold air dynamics, and friction models (`rotax_914_simulator.py`).
  - Labeled fault injection module (`rotax_914_fault_injection.py`) for the 5 certified failure modes with configurable ramp durations and severities.
  - CAN 2.0B serialization and deserialization layer (`simulator/can_publisher/codec.py`).
  - SocketCAN to MQTT ingestion bridge (`simulator/can_publisher/bridge.py`).
  - MATLAB Simulink subsystem validation models (`simulator/matlab/`).

### P4: Machine Learning & Predictive Maintenance (`ml/`)
- **Deliverables**:
  - Feature engineering pipeline (`features.py`): 30s rolling statistics, 5s lag derivatives, one-hot mission phase encoding.
  - Unsupervised Anomaly Detection (`anomaly_detector.py`): Isolation Forest trained strictly on healthy flight regimes with per-phase normalization.
  - Multi-class Fault Classifier (`fault_classifier.py`): Random Forest / XGBoost mapping anomalous telemetry to the 5 known failure classes.
  - Remaining Useful Life (RUL) Estimator (`rul_estimator.py`): Degradation trend regression estimating minutes remaining before critical threshold breach.
  - Explainability Engine (`explainability.py`): SHAP / feature attribution identifying the top 2 telemetry drivers for operator review.
  - Live Inference Service (`inference_service.py`): Async MQTT subscriber consuming telemetry and publishing HealthIndex, FaultEvent, and MissionAdvisory.

### Core Backend & Infrastructure (`backend/`, `docker-compose.yml`)
- **Deliverables**:
  - FastAPI application handling REST queries and live WebSocket fan-out (`/ws/live/{engine_id}`).
  - TimescaleDB time-series hypertables and indexed storage (`backend/db/001_init_timescale.sql`).
  - Paho MQTT background client with connection retry and thread-safe asyncio loop hand-off.
  - Multi-container Docker Compose orchestrating database, broker, backend, and frontend.

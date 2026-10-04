# AARADH-S

### AI-Assisted Aero-Engine Reliability Analysis & Digital Health Simulator

AARADH-S is a digital-twin and AI-assisted engine health monitoring platform designed for a **Rotax 914 aero piston engine** used in UAV applications.

The system combines simulated/recorded engine telemetry, CAN communication, MQTT messaging, time-series storage, machine-learning inference, a FastAPI backend, and a React-based 3D visualization dashboard.

The objective is to provide a unified view of:

- Engine telemetry
- Component-level health
- Fault classification
- Fault severity
- Anomaly detection
- Mission advisories
- Historical telemetry and health data
- Real-time engine state
- 3D engine visualization

> **Hackathon:** Smart India Hackathon 2026 (SIH 2026)

---

## Overview

AARADH-S is designed around the following telemetry-to-visualization pipeline:

```text
┌─────────────────────┐
│ Telemetry Source    │
│                     │
│ Simulator /         │
│ CSV Replayer        │
└──────────┬──────────┘
           │
           │ CAN / Telemetry
           ▼
┌─────────────────────┐
│ CAN / MQTT Layer    │
└──────────┬──────────┘
           │
           │ MQTT
           ▼
┌─────────────────────┐
│ FastAPI Backend     │
│                     │
│ REST + WebSocket    │
└───────┬─────┬───────┘
        │     │
        │     │
        │     ▼
        │  ┌─────────────────────┐
        │  │ ML Inference        │
        │  │                     │
        │  │ Fault / Health /    │
        │  │ Mission Advisory    │
        │  └──────────┬──────────┘
        │             │
        │             │ MQTT
        │             ▼
        │       ┌─────────────┐
        │       │ MQTT Broker │
        │       └─────────────┘
        │
        ▼
┌─────────────────────┐
│ TimescaleDB         │
│                     │
│ Telemetry / Health  │
│ / Fault History     │
└─────────────────────┘

           │
           │ REST / WebSocket
           ▼

┌──────────────────────────────┐
│ React / Vite Dashboard       │
│                              │
│ • Live telemetry             │
│ • Engine health              │
│ • Fault information          │
│ • 3D engine visualization    │
│ • Component inspection       │
│ • Mission information        │
└──────────────────────────────┘
```

The exact runtime topology is defined by the repository's current `docker-compose.yml` and the canonical project documentation.

---

# Key Capabilities

## Real-Time Engine Monitoring

The backend receives engine telemetry and makes current engine state available to the dashboard through REST APIs and WebSockets.

The dashboard is designed to provide a live representation of engine condition rather than requiring repeated manual refreshes.

---

## AI-Based Fault Classification

The project contains trained machine-learning artifacts for engine fault classification.

The documented fault classes include:

| Code | Fault |
|---:|---|
| `0` | Healthy |
| `1` | Misfire |
| `2` | Lubrication issue |
| `3` | Sensor drift |
| `4` | Overheating |
| `5` | Wastegate fault |
| `6` | Injector fault |
| `7` | Alternator fault |
| `8` | Air-filter blockage |

The inference service uses the existing feature-engineering pipeline and trained model artifacts rather than generating synthetic predictions.

---

## Fault Severity

The system also includes a fault-severity prediction component.

Severity is represented in the project's documented range:

```text
0 ─────────────────────────────── 1
Healthy                         Severe
```

The current implementation maps the predicted severity to a health score:

```text
health_score = 100 × (1 - severity)
```

The exact implementation and model artifacts are maintained under the project's ML/model directories.

---

## Anomaly Detection

AARADH-S also contains anomaly-detection functionality for identifying telemetry observations that differ from the learned/reference data distribution.

The anomaly-detection pipeline operates on engineered telemetry features rather than treating an individual raw sensor value in isolation.

---

## 3D Digital Twin

The frontend includes a 3D engine visualization based on the project's GLB engine model.

Engine components can be associated with health information so that the visualization provides a spatial representation of the engine's condition.

The dashboard supports **click-to-inspect** behavior for components.

Clicking a component can expose information such as:

- Component name
- Health
- Error/fault state
- Associated diagnostic information

The application does **not** use click-to-explode as the component inspection mechanism.

---

# Technology Stack

## Backend

- Python
- FastAPI
- MQTT
- WebSockets
- PostgreSQL / TimescaleDB
- CAN / SocketCAN integration where applicable

## Machine Learning

- Python
- XGBoost
- Existing project feature-engineering pipeline
- Trained model artifacts stored within the repository/project structure

## Frontend

- React
- TypeScript
- Vite
- 3D engine visualization
- WebSocket-based live updates

## Infrastructure

- Docker
- Docker Compose
- Eclipse Mosquitto
- TimescaleDB

---

# Repository Structure

The repository contains multiple layers of the system.

A simplified view is:

```text
AARADH-S/
│
├── backend/
│   ├── Dockerfile
│   └── ...
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   └── ...
│
├── simulator/
│   ├── ...
│   └── Dockerfile
│
├── inference/
│   └── ...
│
├── project/
│   ├── backend/
│   │   └── canbus/
│   └── model/
│
├── ml/
│   └── ...
│
├── data/
│   └── processed/
│       └── rotax_combined_clean.csv
│
├── shared/
│   └── component_map.json
│
├── docs/
│   ├── schemas.md
│   └── implementation_context.md
│
├── docker-compose.yml
└── README.md
```

> The repository layout above is a conceptual overview. The actual repository contents are authoritative; existing top-level directories should not be renamed because other parts of the system may depend on their paths.

---

# Canonical Project Documentation

Several files are particularly important when modifying AARADH-S.

### `docs/schemas.md`

This is the canonical source of truth for the project's:

- telemetry schemas
- message structures
- data contracts
- relevant protocol definitions

Changes to data contracts should be made consistently with this document.

### `docs/implementation_context.md`

This document maintains persistent implementation context between development sessions and agents.

It should contain:

- architectural decisions
- implementation status
- integration details
- known limitations
- deployment information
- outstanding work

### `shared/component_map.json`

Contains the shared component mapping used to connect engine components with the visualization/diagnostic system.

---

# Machine Learning

The repository contains the model-training and inference components used by AARADH-S.

The current fault-classification design uses the documented fixed class mapping rather than dynamically assigning class IDs.

The model pipeline includes:

```text
Raw telemetry
      │
      ▼
Feature engineering
      │
      ▼
Fixed feature representation
      │
      ▼
ML model
      │
      ├───────────────┐
      ▼               ▼
Fault class       Severity
      │               │
      └───────┬───────┘
              ▼
        Health / Diagnostic
            output
```

The existing implementation also includes run-level dataset splitting and validation logic intended to prevent leakage between training and held-out runs.

---

# RUL Disclaimer

The current RUL functionality should be interpreted carefully.

The project's current RUL value is a **proxy**, not a validated time-to-failure prediction.

It should therefore not be interpreted as:

> "The engine will fail in exactly N minutes."

Instead, it is an application-level indicator derived from the current implementation.

Any future production RUL system would require appropriate time-to-event/failure data and validation against real engine degradation/failure histories.

---

# Mission Advisory Disclaimer

The application contains mission-advisory logic.

Some advisory calculations currently depend on explicit engineering/application assumptions, including parameters such as:

```text
EXTENSION_FACTOR = 1.5
RPM_REDUCTION = 400
ALTITUDE_REDUCTION_M = 600
```

These values must be treated as **assumptions used by the current application**, not experimentally validated guarantees of additional flight time or engine performance.

---

# Running Locally

## Prerequisites

Install:

- Git
- Docker
- Docker Compose

For ML/model development outside containers, a suitable Python environment may also be required.

---

## Clone the Repository

```bash
git clone https://github.com/singhanmolpreet/AARADH-S.git
cd AARADH-S
```

---

## Environment Configuration

Create the environment file expected by the current Compose configuration.

For example:

```bash
cp .env.example .env
```

Then configure the required values.

Typical configuration includes:

```text
POSTGRES_USER
POSTGRES_PASSWORD
POSTGRES_DB

MQTT_BROKER_PORT

FAKE_HEALTH_ENGINE_IDS
FAKE_HEALTH_INTERVAL_SECONDS
```

The exact required variables are determined by the current `docker-compose.yml` and service implementations.

**Never commit production credentials to Git.**

---

# Run with Docker Compose

Build the services:

```bash
docker compose build
```

Start the stack:

```bash
docker compose up
```

For detached mode:

```bash
docker compose up -d
```

Check service status:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs
```

Follow logs for a specific service:

```bash
docker compose logs -f backend
```

Replace `backend` with another service name when needed.

---

# Stopping the Application

Stop containers while preserving persistent database volumes:

```bash
docker compose down
```

Do **not** routinely use:

```bash
docker compose down -v
```

because removing volumes can delete persistent TimescaleDB data.

---

# Development Workflow

A typical development workflow is:

```text
1. Modify source
       ↓
2. Run local tests
       ↓
3. Build affected Docker service
       ↓
4. Start Compose
       ↓
5. Verify service health
       ↓
6. Verify telemetry flow
       ↓
7. Verify inference
       ↓
8. Verify dashboard
```

For changes involving schemas or message contracts, check:

```text
docs/schemas.md
```

before modifying producers or consumers.

---

# Data Flow

A typical telemetry cycle follows this general path:

```text
Telemetry
   │
   ▼
Simulator / CSV Replayer
   │
   ▼
CAN / Telemetry Transport
   │
   ▼
MQTT
   │
   ├───────────────► Backend
   │                    │
   │                    ├──► TimescaleDB
   │                    │
   │                    └──► WebSocket
   │
   └───────────────► ML Inference
                        │
                        ├──► Health
                        ├──► Fault
                        └──► Mission Advisory
                                  │
                                  ▼
                                MQTT
                                  │
                                  ▼
                               Backend
                                  │
                                  ▼
                              Frontend
```

The exact topic names and payload structures are defined by `docs/schemas.md` and the implementation.

---

# CAN Integration

The project includes CAN-related code under:

```text
project/backend/canbus/
```

and simulator-side CAN functionality.

Existing CAN definitions and codecs should be treated as part of the project's established protocol.

Do not modify CAN IDs or payload layouts without updating the corresponding canonical documentation and all producers/consumers.

---

# Frontend

The frontend is located under:

```text
frontend/
```

It is a React/Vite application.

The dashboard includes:

- live engine telemetry;
- health state;
- fault information;
- component-level information;
- 3D engine visualization;
- WebSocket-based updates.

The frontend may also support a mock-data mode for development.

If live WebSocket communication fails and mock fallback is available, the UI should clearly identify that it is displaying mock data rather than silently presenting it as live telemetry.

---

# Component Visualization

The 3D engine model uses component mappings to associate model nodes with logical engine components.

The shared mapping is maintained in:

```text
shared/component_map.json
```

The visualization supports component inspection.

The intended interaction model is:

```text
Click component
      ↓
Identify engine component
      ↓
Find corresponding health state
      ↓
Display diagnostic information
```

It is not intended to use click-to-explode as the primary inspection mechanism.

---

# Database

AARADH-S uses TimescaleDB for time-series telemetry and related engine-health data.

The Compose deployment provides persistent storage for the database.

Database schema initialization/migrations are maintained under the backend database directory.

When changing database structures:

1. inspect the existing schema;
2. check the canonical documentation;
3. preserve existing data where possible;
4. use appropriate migrations;
5. do not casually modify immutable initialization migrations.

---

# Deployment

The application is designed to support deployment as a Docker Compose stack on a single VM.

A suitable deployment architecture is:

```text
                 Internet
                    │
                    ▼
             ┌─────────────┐
             │ Azure VM    │
             │             │
             │ Docker      │
             │ Compose     │
             └──────┬──────┘
                    │
       ┌────────────┼─────────────┐
       │            │             │
       ▼            ▼             ▼
   Frontend      Backend       Inference
       │            │             │
       │            └──────┬──────┘
       │                   │
       │              MQTT / DB
       │                   │
       └───────────────────┘
```

Only services that must be publicly reachable should be exposed externally.

In particular, database and MQTT services should normally remain internal to the Docker network.

---

# Redeployment

The deployment is intended to support repeatable updates.

A typical update workflow is:

```bash
git pull

docker compose build

docker compose up -d
```

Then verify:

```bash
docker compose ps
```

and inspect logs:

```bash
docker compose logs -f
```

The database volume should be preserved during normal redeployment.

Avoid destructive commands such as:

```bash
docker compose down -v
```

unless intentionally resetting the database.

---

# Project Status

AARADH-S is an active SIH 2026 project.

The repository contains the major building blocks for:

- telemetry simulation/replay;
- CAN integration;
- MQTT communication;
- FastAPI backend;
- TimescaleDB storage;
- ML-based inference;
- React dashboard;
- 3D engine visualization.

Some components may still be under active integration and deployment development.

For the authoritative current implementation status, see:

```text
docs/implementation_context.md
```

---

# Engineering Principles

The project follows several important principles:

### 1. Existing contracts are authoritative

Do not invent new schemas when an existing documented contract already exists.

### 2. Reuse existing implementations

Prefer importing/reusing existing CAN, MQTT, feature-engineering, and ML functionality rather than duplicating it.

### 3. No fake success

A container starting successfully does not mean the application works.

End-to-end data flow should be verified where possible.

### 4. Preserve data integrity

Telemetry and database history should survive normal service/container redeployments.

### 5. Keep claims honest

RUL, mission-advisory calculations, and ML outputs should not be presented as more validated than they actually are.

### 6. Keep the deployment repeatable

The project should be deployable and redeployable through reproducible Docker/Compose commands rather than manual container modifications.

---

# Troubleshooting

## Check all services

```bash
docker compose ps
```

## View all logs

```bash
docker compose logs
```

## View backend logs

```bash
docker compose logs -f backend
```

## View inference logs

```bash
docker compose logs -f inference
```

## View frontend logs

```bash
docker compose logs -f dashboard
```

## Restart the stack

```bash
docker compose restart
```

## Rebuild after source changes

```bash
docker compose build
docker compose up -d
```

## Check running containers

```bash
docker ps
```

---

# Important Files

| File / Directory | Purpose |
|---|---|
| `docker-compose.yml` | Multi-service runtime definition |
| `backend/` | FastAPI backend |
| `frontend/` | React/Vite dashboard |
| `simulator/` | Telemetry simulation/runtime tools |
| `inference/` | ML inference service |
| `project/model/` | Model training/features/artifacts |
| `project/backend/canbus/` | CAN integration |
| `data/processed/` | Processed telemetry dataset |
| `shared/component_map.json` | Shared engine-component mapping |
| `docs/schemas.md` | Canonical data/schema documentation |
| `docs/implementation_context.md` | Persistent implementation/deployment context |

---


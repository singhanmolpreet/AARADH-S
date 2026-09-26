# Build and Execution Sequence

This guide explains how to start, test, and run each layer of the Rotax 914 Digital Twin.

---

## 1. Prerequisites
- **Python**: 3.10+ (with `venv` recommended)
- **Node.js**: 18+ and `npm`
- **Docker & Docker Compose**: (for containerized stack)
- *(Optional for native CAN on Linux/WSL2)*: `can-utils`, `iproute2`

---

## 2. Environment Configuration
Check `.env` at the project root. Defaults are pre-configured:
```bash
POSTGRES_USER=postgres
POSTGRES_PASSWORD=anmolpreet
POSTGRES_DB=engines
MQTT_BROKER_PORT=1883
FAKE_HEALTH_ENGINE_IDS=ENG01
FAKE_HEALTH_INTERVAL_SECONDS=10
CAN_INTERFACE=vcan0
CAN_ENGINE_ID=ENG01
```

---

## 3. Quickstart: Full Docker Stack
To run the entire system with TimescaleDB, Mosquitto MQTT, FastAPI backend, and Frontend:
```bash
docker compose up --build
```
- **Backend API**: `http://localhost:8000/docs`
- **Frontend Dashboard**: `http://localhost:3000`
- **TimescaleDB**: `localhost:5432`
- **Mosquitto**: `localhost:1883`

---

## 4. Local Development Setup

### A. Frontend (3D Cockpit & Dashboard)
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`.
- By default, `frontend/src/shared/data/engineDataSource.ts` runs in `'mock'` mode so the 3D model, shaders, and UI can be developed without spinning up the backend.
- To connect to the live backend, set `ACTIVE_MODE = 'ws'` and specify `WS_HOST = 'localhost'`.

### B. Backend (FastAPI & TimescaleDB)
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### C. SocketCAN Layer & Tests (Linux / WSL2)
Setup virtual CAN interface:
```bash
sudo bash simulator/can_publisher/setup_vcan.sh
```
Run standalone codec test:
```bash
cd simulator/can_publisher
python test_codec.py
```
Start publishing fake telemetry to `vcan0`:
```bash
python fake_publisher.py
```
Bridge `vcan0` frames to MQTT:
```bash
python bridge.py
```

### D. Physics Simulator & Dataset Generation
To generate healthy and fault telemetry datasets:
```bash
cd simulator
python rotax_914_fault_injection.py
```
This produces raw CSVs in `data/raw/`.

### E. Machine Learning Pipeline
To train and test the anomaly detector:
```bash
cd ml
python -c "import anomaly_detector; print('Anomaly Detector loaded')"
```
Interactive analysis notebook available at `ml/notebooks/exploration.ipynb`.

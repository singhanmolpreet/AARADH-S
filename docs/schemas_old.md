# Canonical Message Schemas

This document defines the canonical message contracts and schemas across the Rotax 914 Digital Twin architecture. All components (CAN bridge, MQTT broker, TimescaleDB, Backend REST/WebSocket API, ML models, and Frontend) adhere strictly to these schemas.

---

## 1. Engine Telemetry (`uav/engine/{engine_id}/telemetry`)

Raw 1 Hz sampled engine sensor data transmitted over CAN bus and bridged to MQTT.

```json
{
  "timestamp": "2026-09-26T22:00:00.000Z",
  "engine_id": "ENG01",
  "mission_phase": "cruise",
  "rpm": 5400.0,
  "cht_c": 132.5,
  "egt_c": 720.0,
  "oil_pressure_bar": 3.4,
  "oil_temp_c": 98.2,
  "fuel_flow_lph": 19.5,
  "vibration_rms_g": 0.35,
  "battery_voltage_v": 27.8,
  "alternator_current_a": 14.2,
  "injection_timing_deg": 12.0,
  "map_kpa": 89.5,
  "boost_pressure_bar": 0.25
}
```

### Fields Specification

| Field | Type | Unit | Description | Certified / Normal Limits |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | ISO-8601 string | UTC | Sampling timestamp | Required |
| `engine_id` | string | - | Unique engine identifier | e.g. "ENG01" |
| `mission_phase` | string enum | - | Flight mission phase | `ground`, `takeoff`, `climb`, `cruise`, `high_altitude_cruise`, `loiter`, `descent` |
| `rpm` | float | RPM | Engine rotational speed | Idle ~1400, Max Continuous 5500, Max Takeoff 5800 |
| `cht_c` | float | °C | Cylinder Head Temperature | Max 135 °C (conventional head) |
| `egt_c` | float | °C | Exhaust Gas Temperature | Max 950 °C |
| `oil_pressure_bar` | float | bar | Engine oil lubrication pressure | 2.0–5.0 bar (>3500 RPM), min 0.8 bar (idle), max 7.0 bar |
| `oil_temp_c` | float | °C | Engine oil temperature | 90–110 °C normal, max 130 °C |
| `fuel_flow_lph` | float | L/h | Fuel consumption rate | Typical 15–28 L/h |
| `vibration_rms_g` | float | g (RMS) | Structural vibration intensity | Baseline < 0.5 g |
| `battery_voltage_v` | float | V | Electrical bus voltage | 26.0–28.5 V |
| `alternator_current_a` | float | A | Electrical alternator load | 5–25 A |
| `injection_timing_deg` | float | °BTDC | Electronic injection timing | 10–18 °BTDC |
| `map_kpa` | float | kPa | Manifold Absolute Pressure | 80–115 kPa |
| `boost_pressure_bar` | float | bar | Turbocharger gauge boost | +0.25 bar nominal (+0.15 to +0.35 bar) |

---

## 2. Health Index (`uav/engine/{engine_id}/health`)

Computed health score assessing subsystem degradation and anomaly status.

```json
{
  "timestamp": "2026-09-26T22:00:00.000Z",
  "engine_id": "ENG01",
  "overall_health_score": 88.5,
  "overall_status": "green",
  "active_faults": [],
  "components": [
    {
      "component": "cylinder_1",
      "health_score": 92.0,
      "status": "green",
      "rul_hours": 85.0,
      "top_features": ["cht_trend", "vibration_rms_g_roll_var"]
    },
    {
      "component": "oil_system",
      "health_score": 78.0,
      "status": "yellow",
      "rul_hours": 42.5,
      "top_features": ["oil_pressure_bar_diff", "oil_temp_c_roll_slope"]
    }
  ]
}
```

### Health Status Bands
- **Green**: 80–100 (Nominal, healthy)
- **Yellow**: 55–79 (Degraded, watch condition)
- **Orange**: 30–54 (Advisory warning)
- **Red**: < 30 (Critical fault, immediate action required)

---

## 3. Fault Event (`uav/engine/{engine_id}/fault`)

Generated when the anomaly detection or fault classifier identifies an active or developing fault condition.

```json
{
  "timestamp": "2026-09-26T22:00:00.000Z",
  "engine_id": "ENG01",
  "fault_type": "lubrication_issue",
  "component": "oil_system",
  "confidence": 0.94,
  "severity": "warning",
  "title": "Gradual Oil Pressure Decay",
  "evidence": "Oil pressure dropped by 0.8 bar over 60s while oil temperature increased by 14°C during level cruise.",
  "recommended_action": "Reduce continuous power setting to 65% MCP and divert to secondary airfield."
}
```

### Valid Fault Types
- `misfire`
- `lubrication_issue`
- `sensor_drift`
- `overheating`
- `wastegate_fault`

---

## 4. Mission Replanning Advisory (`uav/engine/{engine_id}/mission_advisory`)

Advisory computed based on RUL estimation and active mission duration.

```json
{
  "timestamp": "2026-09-26T22:00:00.000Z",
  "engine_id": "ENG01",
  "rul_minutes": 35.0,
  "mission_time_remaining_minutes": 55.0,
  "recommended_strategy": "Divert to nearest alternate recovery base",
  "estimated_new_rul_minutes": 68.0
}
```

---

## 5. Fault Injection Request (`POST /api/simulate/inject_fault`)

HTTP request body payload to command the simulation framework to inject a specific failure mode:

```json
{
  "fault_type": "misfire",
  "engine_id": "ENG01"
}
```

---

## 6. CAN Bus Frame Mapping (Classic CAN 2.0B / 5 Frames)

Transmitted at 1 Hz by `simulator/can_publisher/fake_publisher.py` or the physics simulator and decoded by `simulator/can_publisher/codec.py`.

- **`0x100` (RPM & Core Temps)**:
  - `bytes 0-1`: `rpm` (uint16)
  - `bytes 2-3`: `cht_c * 10` (int16)
  - `bytes 4-5`: `egt_c * 10` (int16)
  - `bytes 6-7`: `oil_temp_c * 10` (int16)
- **`0x101` (Pressures & Flow)**:
  - `bytes 0-1`: `oil_pressure_bar * 100` (int16)
  - `bytes 2-3`: `fuel_flow_lph * 10` (uint16)
  - `bytes 4-5`: `vibration_rms_g * 1000` (uint16)
  - `bytes 6-7`: `map_kpa * 10` (uint16)
- **`0x102` (Electrical, Timing & Boost)**:
  - `bytes 0-1`: `battery_voltage_v * 100` (uint16)
  - `bytes 2-3`: `alternator_current_a * 100` (int16)
  - `bytes 4-5`: `injection_timing_deg * 100` (int16)
  - `bytes 6-7`: `boost_pressure_bar * 1000` (int16)
- **`0x103` (Timestamp)**:
  - `bytes 0-7`: Unix epoch microseconds UTC (int64)
- **`0x104` (Mission Phase)**:
  - `bytes 0-7`: ASCII string right-padded with 0x00 (up to 8 characters)

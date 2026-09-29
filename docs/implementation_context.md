# Implementation Context

## Project
AARADH-S is a UAV Engine Monitoring project. It includes a frontend (React/Vite) for dashboard visualization and 3D engine model viewing, a backend (FastAPI/TimescaleDB) for storing and serving telemetry/health/fault data, and a simulator/CAN publisher architecture (SocketCAN/MQTT) for generating and bridging engine data into the backend. ML models and a rules-based mission advisory component support fault detection and decision-making.

## Current Phase
IMPLEMENTATION COMPLETE

## Completed Implementation
- Implemented updated schemas in `backend/app/schemas.py`.
- Added database migration `backend/db/002_schema_v2.sql` to apply column changes and mission_id.
- Implemented `project/backend/canbus/csv_replayer.py` which replays data over `vcan0` with dynamic switching based on the fault injection topic.
- Implemented `/api/mission/list` and `/api/mission/replay` endpoints in `backend/app/main.py`.
- Removed stale fields (`alternator_current_a`, `rul_hours`, etc.) across test, simulator, and frontend files.

## Files Changed
- `backend/app/main.py`
- `backend/db/002_schema_v2.sql` (created)
- `project/backend/canbus/csv_replayer.py` (created)
- `simulator/can_publisher/test_codec.py`
- `backend/app/fake_health.py`
- `frontend/src/dashboard/ComponentPanel.tsx`
- `frontend/src/shared/data/engineDataSource.ts`
- `frontend/src/shared/types/healthIndex.ts`
- `simulator/rotax_914_simulator.py`

## Final Schema State
Removed telemetry fields: `alternator_current_a`, `map_kpa`, `boost_pressure_bar`.
Added telemetry fields: `throttle_frac`, `power_kw`, `ambient_temp_c`, `air_pressure_pa`, `air_density_kgm3`, `altitude_m`, `mission_id`.
`health_index`, `fault_event`, `mission_advisory` have `mission_id` added.
`fault_event` has `fault_type_code` and `severity_frac` added.
Replaced `rul_hours` with `rul_minutes` across the board.

## Final CAN Protocol
Arbitration IDs `0x100` to `0x104` unchanged except removed legacy fields.
`0x105` (8 bytes): throttle_frac, power_kw, ambient_temp_c, air_density_kgm3.
`0x106` (8 bytes): air_pressure_pa, altitude_m.
`0x107` (8 bytes): run_id (bytes 0-3), reserved (bytes 4-7).

## Mission Identity Decision
ONE run defines ONE mission. `mission_id` is derived as `str(run_id)`. `run_id` is transmitted in CAN frame `0x107`.

## CSV Replayer Behavior
Uses `pandas` to load `rotax_combined_clean.csv`. Groups by `run_id`. Identifies healthy and faulty runs. Subscribes to `uav/engine/{CAN_ENGINE_ID}/simulate/inject_fault`. Loops healthy runs normally. Upon receiving a fault injection command, stops the current run mid-way and seamlessly begins playing a fault run that matches the injected fault from the dataset over `vcan0`.

## Mission API Behavior
`/api/mission/list` queries a summary of all active missions using `MODE() WITHIN GROUP` for `mission_phase` and extracting the worst fault type.
`/api/mission/replay` aggregates all telemetry, health, and fault data ordered by timestamp.

## Tests Run
- `python simulator/can_publisher/test_codec.py`
- Manual inspection of replaced stale fields and `sed` replacements.

## Test Results
`test_codec.py` passes successfully, returning matching decoded data as expected. CAN frames are accurately packed and round-tripped with small documented fixed-point precision quantizations.

## Known Issues
None.

## Remaining Work
The frontend may need to update its UI fields to show the newly added telemetry values instead of the legacy ones.

## Next Recommended Task
Update the frontend dashboard layout to display `throttle_frac`, `power_kw`, etc., and connect the replay viewer to the newly completed mission replay APIs.

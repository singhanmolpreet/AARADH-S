Telemetry (uav/engine/{id}/telemetry):
{timestamp, engine_id, mission_id, mission_phase, rpm, cht_c, egt_c,
 oil_pressure_bar, oil_temp_c, fuel_flow_lph, vibration_rms_g,
 battery_voltage_v, injection_timing_deg, throttle_frac, power_kw,
 ambient_temp_c, air_pressure_pa, air_density_kgm3, altitude_m}
mission_phase: endurance_low_throttle | steady_cruise | high_power_climb | rapid_throttle_transitions

Health (uav/engine/{id}/health):
{timestamp, engine_id, mission_id,
 components: [{component, health_score(0-100), status(green>=80/yellow>=55/orange>=30/red), rul_minutes, top_features:[]}],
 overall_health_score, overall_status, active_faults:[]}

Fault (uav/engine/{id}/fault):
{timestamp, engine_id, mission_id, fault_type, fault_type_code, component,
 confidence, severity(watch<0.5/warning<=0.8/critical>0.8), severity_frac,
 title, evidence, recommended_action}

Mission advisory (uav/engine/{id}/mission_advisory):
{timestamp, engine_id, mission_id, rul_minutes, mission_time_remaining_minutes,
 recommended_strategy, estimated_new_rul_minutes}

Fault classes: 0 healthy, 1 misfire, 2 lubrication_issue, 3 sensor_drift,
4 overheating, 5 wastegate_fault, 6 injector_fault, 7 alternator_fault, 8 air_filter_blockage

### Mission Identity Decision
- For the AARADH-S dataset, ONE run represents ONE mission.
- `mission_id` is derived losslessly as `str(run_id)`.
- The source datasets contain `run_id` (used as the replay/grouping identifier).

### Final CAN Contract
- All multi-byte integers are big-endian (`>`). 
- Floating-point fields are scaled by their factor and rounded to the nearest integer (`int(round(v))`) before transmission.
- `0x100`: `rpm` (uint16 x1), `cht_c` (int16 x10), `egt_c` (int16 x10), `oil_temp_c` (int16 x10)
- `0x101`: `oil_pressure_bar` (int16 x100), `fuel_flow_lph` (uint16 x10), `vibration_rms_g` (uint16 x1000)
- `0x102`: `battery_voltage_v` (uint16 x100), `injection_timing_deg` (int16 x100)
- `0x103`: Timestamp (int64 microseconds)
- `0x104`: Mission phase (8-byte ASCII)
- `0x105` (New):
  - bytes 0-1: `throttle_frac` (uint16 x10000, range 0-10000)
  - bytes 2-3: `power_kw` (uint16 x1000, range 0-33011)
  - bytes 4-5: `ambient_temp_c` (int16 x100, range -4522 to 3646)
  - bytes 6-7: `air_density_kgm3` (uint16 x10000, range 4228 to 12314)
- `0x106` (New):
  - bytes 0-3: `air_pressure_pa` (uint32 x1, range 30288 to 101325)
  - bytes 4-7: `altitude_m` (uint32 x10, range 0 to 90997)
- `0x107` (New):
  - bytes 0-3: `run_id` (uint32 x1, min 1, max 10768)
  - bytes 4-7: reserved (uint8[4])
-- Add mission_id and new telemetry fields
ALTER TABLE telemetry ADD COLUMN mission_id TEXT;
ALTER TABLE telemetry ADD COLUMN throttle_frac DOUBLE PRECISION;
ALTER TABLE telemetry ADD COLUMN power_kw DOUBLE PRECISION;
ALTER TABLE telemetry ADD COLUMN ambient_temp_c DOUBLE PRECISION;
ALTER TABLE telemetry ADD COLUMN air_pressure_pa DOUBLE PRECISION;
ALTER TABLE telemetry ADD COLUMN air_density_kgm3 DOUBLE PRECISION;
ALTER TABLE telemetry ADD COLUMN altitude_m DOUBLE PRECISION;

-- Drop removed telemetry fields
ALTER TABLE telemetry DROP COLUMN alternator_current_a;
ALTER TABLE telemetry DROP COLUMN map_kpa;
ALTER TABLE telemetry DROP COLUMN boost_pressure_bar;

-- Add mission_id to health tables
ALTER TABLE health_index ADD COLUMN mission_id TEXT;

-- Add missing fields to fault_event
ALTER TABLE fault_event ADD COLUMN mission_id TEXT;
ALTER TABLE fault_event ADD COLUMN fault_type_code INT;
ALTER TABLE fault_event ADD COLUMN severity_frac DOUBLE PRECISION;

-- Add mission_id to mission_advisory
ALTER TABLE mission_advisory ADD COLUMN mission_id TEXT;

-- Create mission index to optimize queries
CREATE INDEX IF NOT EXISTS idx_telemetry_mission_id ON telemetry (mission_id);
CREATE INDEX IF NOT EXISTS idx_fault_mission_id ON fault_event (mission_id);

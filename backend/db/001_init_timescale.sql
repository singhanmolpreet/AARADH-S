-- Requires the TimescaleDB extension available in the target Postgres instance.
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Telemetry: uav/engine/{engine_id}/telemetry
CREATE TABLE IF NOT EXISTS telemetry (
    time                    TIMESTAMPTZ NOT NULL,
    engine_id               TEXT NOT NULL,
    mission_phase           TEXT NOT NULL,
    rpm                     DOUBLE PRECISION NOT NULL,
    cht_c                   DOUBLE PRECISION NOT NULL,
    egt_c                   DOUBLE PRECISION NOT NULL,
    oil_pressure_bar        DOUBLE PRECISION NOT NULL,
    oil_temp_c              DOUBLE PRECISION NOT NULL,
    fuel_flow_lph           DOUBLE PRECISION NOT NULL,
    vibration_rms_g         DOUBLE PRECISION NOT NULL,
    battery_voltage_v       DOUBLE PRECISION NOT NULL,
    alternator_current_a    DOUBLE PRECISION NOT NULL,
    injection_timing_deg    DOUBLE PRECISION NOT NULL,
    map_kpa                 DOUBLE PRECISION NOT NULL,
    boost_pressure_bar      DOUBLE PRECISION NOT NULL
);
SELECT create_hypertable('telemetry', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_telemetry_engine_time ON telemetry (engine_id, time DESC);

-- Health index: uav/engine/{engine_id}/health
CREATE TABLE IF NOT EXISTS health_index (
    time                    TIMESTAMPTZ NOT NULL,
    engine_id               TEXT NOT NULL,
    components              JSONB NOT NULL,
    overall_health_score    DOUBLE PRECISION NOT NULL,
    overall_status          TEXT NOT NULL,
    active_faults           JSONB NOT NULL
);
SELECT create_hypertable('health_index', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_health_engine_time ON health_index (engine_id, time DESC);

-- Fault event: uav/engine/{engine_id}/fault
CREATE TABLE IF NOT EXISTS fault_event (
    time                    TIMESTAMPTZ NOT NULL,
    engine_id               TEXT NOT NULL,
    fault_type              TEXT NOT NULL,
    component               TEXT NOT NULL,
    confidence              DOUBLE PRECISION NOT NULL,
    severity                TEXT NOT NULL,
    title                   TEXT NOT NULL,
    evidence                TEXT NOT NULL,
    recommended_action      TEXT NOT NULL
);
SELECT create_hypertable('fault_event', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_fault_engine_time ON fault_event (engine_id, time DESC);

-- Mission replanning advisory: uav/engine/{engine_id}/mission_advisory
CREATE TABLE IF NOT EXISTS mission_advisory (
    time                            TIMESTAMPTZ NOT NULL,
    engine_id                       TEXT NOT NULL,
    rul_minutes                     DOUBLE PRECISION NOT NULL,
    mission_time_remaining_minutes  DOUBLE PRECISION NOT NULL,
    recommended_strategy            TEXT NOT NULL,
    estimated_new_rul_minutes       DOUBLE PRECISION NOT NULL
);
SELECT create_hypertable('mission_advisory', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_mission_advisory_engine_time ON mission_advisory (engine_id, time DESC);

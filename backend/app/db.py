import asyncio
import json
import logging
from datetime import datetime
from typing import List, Optional
import asyncpg

from . import config

logger = logging.getLogger("db")

_pool: Optional[asyncpg.Pool] = None


def _parse_ts(ts: str) -> datetime:
    """
    asyncpg needs a real datetime object for timestamptz columns -- it
    won't accept a raw ISO string even with a ::timestamptz cast in the
    query. Handles a trailing 'Z' since Python's fromisoformat doesn't
    accept that before 3.11 and some producers use it regardless of version.
    """
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


async def init_pool(retries: int = 10, delay_seconds: float = 2.0):
    """
    Retries pool creation instead of crashing on the first connection
    attempt. Even with a compose healthcheck on timescaledb, "accepting
    TCP connections" and "ready for asyncpg" aren't always the same
    instant, so a short retry loop makes `docker compose up --build` from
    a cold, empty volume reliable instead of occasionally racy.
    """
    global _pool
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            _pool = await asyncpg.create_pool(dsn=config.POSTGRES_DSN, min_size=1, max_size=10)
            return _pool
        except (asyncpg.PostgresError, OSError) as e:
            last_error = e
            logger.warning("DB pool init attempt %d/%d failed: %s", attempt, retries, e)
            await asyncio.sleep(delay_seconds)
    raise RuntimeError(f"Could not connect to Postgres after {retries} attempts") from last_error


async def close_pool():
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("DB pool not initialized. Call init_pool() on startup.")
    return _pool


async def insert_telemetry(msg: dict):
    pool = get_pool()
    await pool.execute(
        """
        INSERT INTO telemetry (
            time, engine_id, mission_id, mission_phase, rpm, cht_c, egt_c,
            oil_pressure_bar, oil_temp_c, fuel_flow_lph, vibration_rms_g,
            battery_voltage_v, injection_timing_deg,
            throttle_frac, power_kw, ambient_temp_c,
            air_pressure_pa, air_density_kgm3, altitude_m
        ) VALUES (
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19
        )
        """,
        _parse_ts(msg["timestamp"]),
        msg["engine_id"],
        msg.get("mission_id"),
        msg["mission_phase"],
        msg["rpm"],
        msg["cht_c"],
        msg["egt_c"],
        msg["oil_pressure_bar"],
        msg["oil_temp_c"],
        msg["fuel_flow_lph"],
        msg["vibration_rms_g"],
        msg["battery_voltage_v"],
        msg["injection_timing_deg"],
        msg["throttle_frac"],
        msg["power_kw"],
        msg["ambient_temp_c"],
        msg["air_pressure_pa"],
        msg["air_density_kgm3"],
        msg["altitude_m"],
    )


async def get_latest_telemetry(engine_id: str):
    pool = get_pool()
    row = await pool.fetchrow(
        """
        SELECT * FROM telemetry
        WHERE engine_id = $1
        ORDER BY time DESC
        LIMIT 1
        """,
        engine_id,
    )
    return dict(row) if row else None


async def get_telemetry_history(engine_id: str, start: str, end: str):
    pool = get_pool()
    rows = await pool.fetch(
        """
        SELECT * FROM telemetry
        WHERE engine_id = $1 AND time BETWEEN $2 AND $3
        ORDER BY time ASC
        """,
        engine_id,
        start,
        end,
    )
    return [dict(r) for r in rows]


async def insert_health(msg: dict):
    pool = get_pool()
    await pool.execute(
        """
        INSERT INTO health_index (
            time, engine_id, mission_id, components, overall_health_score,
            overall_status, active_faults
        ) VALUES ($1,$2,$3,$4,$5,$6,$7)
        """,
        _parse_ts(msg["timestamp"]),
        msg["engine_id"],
        msg.get("mission_id"),
        json.dumps(msg["components"]),
        msg["overall_health_score"],
        msg["overall_status"],
        json.dumps(msg["active_faults"]),
    )


async def get_latest_health(engine_id: str):
    pool = get_pool()
    row = await pool.fetchrow(
        """
        SELECT * FROM health_index
        WHERE engine_id = $1
        ORDER BY time DESC
        LIMIT 1
        """,
        engine_id,
    )
    if not row:
        return None
    d = dict(row)
    d["components"] = json.loads(d["components"])
    d["active_faults"] = json.loads(d["active_faults"])
    return d


async def get_health_history(engine_id: str, start: str, end: str):
    pool = get_pool()
    rows = await pool.fetch(
        """
        SELECT * FROM health_index
        WHERE engine_id = $1 AND time BETWEEN $2 AND $3
        ORDER BY time ASC
        """,
        engine_id,
        start,
        end,
    )
    out = []
    for r in rows:
        d = dict(r)
        d["components"] = json.loads(d["components"])
        d["active_faults"] = json.loads(d["active_faults"])
        out.append(d)
    return out


async def insert_fault(msg: dict):
    pool = get_pool()
    await pool.execute(
        """
        INSERT INTO fault_event (
            time, engine_id, mission_id, fault_type, fault_type_code, component,
            confidence, severity, severity_frac, title, evidence, recommended_action
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
        """,
        _parse_ts(msg["timestamp"]),
        msg["engine_id"],
        msg.get("mission_id"),
        msg["fault_type"],
        msg.get("fault_type_code"),
        msg["component"],
        msg["confidence"],
        msg["severity"],
        msg.get("severity_frac"),
        msg["title"],
        msg["evidence"],
        msg["recommended_action"],
    )


async def get_active_faults(engine_id: str):
    pool = get_pool()
    rows = await pool.fetch(
        """
        SELECT * FROM fault_event
        WHERE engine_id = $1
        ORDER BY time DESC
        """,
        engine_id,
    )
    return [dict(r) for r in rows]


async def insert_mission_advisory(msg: dict):
    pool = get_pool()
    await pool.execute(
        """
        INSERT INTO mission_advisory (
            time, engine_id, mission_id, rul_minutes,
            mission_time_remaining_minutes,
            recommended_strategy, estimated_new_rul_minutes
        ) VALUES ($1,$2,$3,$4,$5,$6,$7)
        """,
        _parse_ts(msg["timestamp"]),
        msg["engine_id"],
        msg.get("mission_id"),
        msg["rul_minutes"],
        msg["mission_time_remaining_minutes"],
        msg["recommended_strategy"],
        msg["estimated_new_rul_minutes"],
    )


async def get_latest_mission_advisory(engine_id: str):
    pool = get_pool()
    row = await pool.fetchrow(
        """
        SELECT * FROM mission_advisory
        WHERE engine_id = $1
        ORDER BY time DESC
        LIMIT 1
        """,
        engine_id,
    )
    return dict(row) if row else None


async def list_missions(engine_id: Optional[str] = None) -> List[dict]:
    """
    Returns one summary row per mission_id: first/last timestamp,
    sample count, dominant mission_phase, worst fault type and max
    severity_frac seen during that mission.
    """
    pool = get_pool()
    if engine_id:
        rows = await pool.fetch(
            """
            SELECT
                t.mission_id,
                t.engine_id,
                MIN(t.time)::text          AS start_ts,
                MAX(t.time)::text          AS end_ts,
                COUNT(*)::int              AS n_samples,
                MODE() WITHIN GROUP (ORDER BY t.mission_phase) AS mission_phase,
                f.worst_fault_type,
                f.max_severity_frac
            FROM telemetry t
            LEFT JOIN LATERAL (
                SELECT fault_type AS worst_fault_type,
                       MAX(severity_frac) AS max_severity_frac
                FROM fault_event
                WHERE mission_id = t.mission_id
                GROUP BY fault_type
                ORDER BY MAX(severity_frac) DESC
                LIMIT 1
            ) f ON TRUE
            WHERE t.engine_id = $1 AND t.mission_id IS NOT NULL
            GROUP BY t.mission_id, t.engine_id, f.worst_fault_type, f.max_severity_frac
            ORDER BY MIN(t.time) DESC
            """,
            engine_id,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT
                t.mission_id,
                t.engine_id,
                MIN(t.time)::text          AS start_ts,
                MAX(t.time)::text          AS end_ts,
                COUNT(*)::int              AS n_samples,
                MODE() WITHIN GROUP (ORDER BY t.mission_phase) AS mission_phase,
                f.worst_fault_type,
                f.max_severity_frac
            FROM telemetry t
            LEFT JOIN LATERAL (
                SELECT fault_type AS worst_fault_type,
                       MAX(severity_frac) AS max_severity_frac
                FROM fault_event
                WHERE mission_id = t.mission_id
                GROUP BY fault_type
                ORDER BY MAX(severity_frac) DESC
                LIMIT 1
            ) f ON TRUE
            WHERE t.mission_id IS NOT NULL
            GROUP BY t.mission_id, t.engine_id, f.worst_fault_type, f.max_severity_frac
            ORDER BY MIN(t.time) DESC
            """,
        )
    return [dict(r) for r in rows]


async def get_mission_replay(mission_id: str) -> List[dict]:
    """
    Returns all telemetry, health and fault rows for a mission_id,
    merged and time-ordered, each row shaped as {ts, telemetry, health, fault}.
    """
    pool = get_pool()
    tel_rows = await pool.fetch(
        "SELECT * FROM telemetry WHERE mission_id = $1 ORDER BY time ASC",
        mission_id,
    )
    health_rows = await pool.fetch(
        "SELECT * FROM health_index WHERE mission_id = $1 ORDER BY time ASC",
        mission_id,
    )
    fault_rows = await pool.fetch(
        "SELECT * FROM fault_event WHERE mission_id = $1 ORDER BY time ASC",
        mission_id,
    )

    timeline: dict = {}
    for r in tel_rows:
        ts = str(r["time"])
        timeline.setdefault(ts, {"ts": ts, "telemetry": None, "health": None, "fault": None})
        timeline[ts]["telemetry"] = dict(r)
    for r in health_rows:
        ts = str(r["time"])
        entry = timeline.setdefault(ts, {"ts": ts, "telemetry": None, "health": None, "fault": None})
        d = dict(r)
        if isinstance(d.get("components"), str):
            d["components"] = json.loads(d["components"])
        if isinstance(d.get("active_faults"), str):
            d["active_faults"] = json.loads(d["active_faults"])
        entry["health"] = d
    for r in fault_rows:
        ts = str(r["time"])
        entry = timeline.setdefault(ts, {"ts": ts, "telemetry": None, "health": None, "fault": None})
        entry["fault"] = dict(r)

    return sorted(timeline.values(), key=lambda x: x["ts"])
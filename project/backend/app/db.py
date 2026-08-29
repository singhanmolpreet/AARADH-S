import asyncio
import json
import logging
from datetime import datetime
from typing import Optional
import asyncpg

from ....backend.app import config

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
            time, engine_id, mission_phase, rpm, cht_c, egt_c,
            oil_pressure_bar, oil_temp_c, fuel_flow_lph, vibration_rms_g,
            battery_voltage_v, alternator_current_a, injection_timing_deg,
            map_kpa, boost_pressure_bar
        ) VALUES (
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15
        )
        """,
        _parse_ts(msg["timestamp"]),
        msg["engine_id"],
        msg["mission_phase"],
        msg["rpm"],
        msg["cht_c"],
        msg["egt_c"],
        msg["oil_pressure_bar"],
        msg["oil_temp_c"],
        msg["fuel_flow_lph"],
        msg["vibration_rms_g"],
        msg["battery_voltage_v"],
        msg["alternator_current_a"],
        msg["injection_timing_deg"],
        msg["map_kpa"],
        msg["boost_pressure_bar"],
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
            time, engine_id, components, overall_health_score,
            overall_status, active_faults
        ) VALUES ($1,$2,$3,$4,$5,$6)
        """,
        _parse_ts(msg["timestamp"]),
        msg["engine_id"],
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
            time, engine_id, fault_type, component, confidence,
            severity, title, evidence, recommended_action
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
        """,
        _parse_ts(msg["timestamp"]),
        msg["engine_id"],
        msg["fault_type"],
        msg["component"],
        msg["confidence"],
        msg["severity"],
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
            time, engine_id, rul_minutes, mission_time_remaining_minutes,
            recommended_strategy, estimated_new_rul_minutes
        ) VALUES ($1,$2,$3,$4,$5,$6)
        """,
        _parse_ts(msg["timestamp"]),
        msg["engine_id"],
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
"""
Publishes fake-but-correctly-shaped health messages on a timer, matching the
health schema exactly, so downstream consumers (dashboard) aren't blocked
while the real ML/health service isn't wired in yet.

This does NOT invent new fields -- it only fills the fields defined in the
HealthIndex schema with placeholder values, clearly randomized/static rather
than pretending to be real inference output.
"""
import asyncio
import logging
import random
from datetime import datetime, timezone

from . import db
from .schemas import HealthIndex
from .websocket_manager import ws_manager

logger = logging.getLogger("fake_health")

# Engine IDs to generate fake health for. Not specified in the source
# material beyond the example "ENG01" -- add more via FAKE_HEALTH_ENGINE_IDS
# env var if your fleet has more engines.
import os

_engine_ids_env = os.environ.get("FAKE_HEALTH_ENGINE_IDS", "ENG01")
ENGINE_IDS = [e.strip() for e in _engine_ids_env.split(",") if e.strip()]

INTERVAL_SECONDS = int(os.environ.get("FAKE_HEALTH_INTERVAL_SECONDS", "10"))


def _generate_fake_health(engine_id: str) -> dict:
    score = round(random.uniform(60, 100), 1)
    status = "green" if score >= 80 else ("yellow" if score >= 60 else "red")
    msg = HealthIndex(
        timestamp=datetime.now(timezone.utc).isoformat(),
        engine_id=engine_id,
        components=[
            {
                "component": "cylinder_1",
                "health_score": score,
                "status": status,
                "rul_hours": round(random.uniform(10, 100), 1),
                "top_features": ["cht_trend", "egt_variance"],
            }
        ],
        overall_health_score=score,
        overall_status=status,
        active_faults=[],
    )
    return msg.model_dump()


async def run():
    logger.info(
        "Fake health publisher started for engines=%s interval=%ss",
        ENGINE_IDS,
        INTERVAL_SECONDS,
    )
    while True:
        for engine_id in ENGINE_IDS:
            payload = _generate_fake_health(engine_id)
            try:
                await db.insert_health(payload)
                await ws_manager.broadcast(engine_id, {"type": "health", "data": payload})
            except Exception as e:
                logger.warning("Failed to publish fake health for %s: %s", engine_id, e)
        await asyncio.sleep(INTERVAL_SECONDS)

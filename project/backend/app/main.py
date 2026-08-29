import asyncio
import json
import logging
import os

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
import paho.mqtt.client as mqtt

from ....backend.app import db, mqtt_client
from ....backend.app import config
from ....backend.app import fake_health
from .schemas import InjectFaultRequest
from .websocket_manager import ws_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="UAV Engine Monitoring Backend")

_fake_health_task: asyncio.Task = None

# NOTE: no MQTT topic for "inject a fault into the simulator" was given in the
# source material -- only the HTTP request body {"fault_type","engine_id"}.
# This publishes to a command topic I picked by extending the existing
# uav/engine/{engine_id}/... convention. Confirm/change SIMULATE_COMMAND_TOPIC
# if P3's simulator expects something different.
SIMULATE_COMMAND_TOPIC_TEMPLATE = "uav/engine/{engine_id}/simulate/inject_fault"


@app.on_event("startup")
async def on_startup():
    config.require_config()
    await db.init_pool()
    loop = asyncio.get_event_loop()
    mqtt_client.start(loop)
    global _fake_health_task
    _fake_health_task = asyncio.create_task(fake_health.run())
    logger.info("Startup complete")


@app.on_event("shutdown")
async def on_shutdown():
    if _fake_health_task:
        _fake_health_task.cancel()
    mqtt_client.stop()
    await db.close_pool()


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

@app.get("/api/telemetry/latest")
async def telemetry_latest(engine_id: str = Query(...)):
    row = await db.get_latest_telemetry(engine_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No telemetry found for engine_id={engine_id}")
    return row


@app.get("/api/telemetry/history")
async def telemetry_history(
    engine_id: str = Query(...),
    start: str = Query(...),
    end: str = Query(...),
):
    return await db.get_telemetry_history(engine_id, start, end)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health/current")
async def health_current(engine_id: str = Query(...)):
    row = await db.get_latest_health(engine_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No health data found for engine_id={engine_id}")
    return row


@app.get("/api/health/history")
async def health_history(
    engine_id: str = Query(...),
    start: str = Query(...),
    end: str = Query(...),
):
    return await db.get_health_history(engine_id, start, end)


# ---------------------------------------------------------------------------
# Faults
# ---------------------------------------------------------------------------

@app.get("/api/faults/active")
async def faults_active(engine_id: str = Query(...)):
    return await db.get_active_faults(engine_id)


# ---------------------------------------------------------------------------
# Mission
# ---------------------------------------------------------------------------
# No schema was given for a "mission" object or for mission/replay's response
# shape -- only mission_advisory has a defined schema. These two endpoints
# are left as explicit not-implemented stubs rather than guessing a shape.

@app.get("/api/mission/list")
async def mission_list():
    raise HTTPException(
        status_code=501,
        detail="Not implemented: no schema for a 'mission' object was provided. "
        "Tell me the fields a mission record should have and I'll wire this up.",
    )


@app.get("/api/mission/replay")
async def mission_replay(mission_id: str = Query(...), speed: str = Query("1x")):
    raise HTTPException(
        status_code=501,
        detail="Not implemented: no schema for mission replay data was provided. "
        "Tell me what a replay response should contain and I'll wire this up.",
    )


@app.get("/api/mission/advisory")
async def mission_advisory(engine_id: str = Query(...)):
    row = await db.get_latest_mission_advisory(engine_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No mission advisory found for engine_id={engine_id}")
    return row


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------

@app.post("/api/simulate/inject_fault")
async def simulate_inject_fault(body: InjectFaultRequest):
    topic = SIMULATE_COMMAND_TOPIC_TEMPLATE.format(engine_id=body.engine_id)
    payload = json.dumps({"fault_type": body.fault_type, "engine_id": body.engine_id})
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    try:
        client.connect(config.MQTT_BROKER_HOST, config.MQTT_BROKER_PORT, keepalive=10)
        client.publish(topic, payload)
        client.disconnect()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to publish fault injection command: {e}")
    return {"status": "published", "topic": topic, "fault_type": body.fault_type, "engine_id": body.engine_id}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/live/{engine_id}")
async def ws_live(websocket: WebSocket, engine_id: str):
    await ws_manager.connect(engine_id, websocket)
    try:
        while True:
            # Client isn't expected to send anything; just keep the socket open.
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(engine_id, websocket)

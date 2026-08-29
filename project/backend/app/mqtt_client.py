"""
Background MQTT subscriber.

Uses paho-mqtt's threaded client (paho runs its own network loop thread)
and hands each received message back onto the FastAPI asyncio event loop
via loop.call_soon_threadsafe, since asyncpg calls must run on that loop.
"""
import asyncio
import json
import logging

import paho.mqtt.client as mqtt

from . import db
from . import config
from .schemas import Telemetry, HealthIndex, FaultEvent, MissionAdvisory
from .websocket_manager import ws_manager

logger = logging.getLogger("mqtt_client")

_client: mqtt.Client = None
_loop: asyncio.AbstractEventLoop = None


def _topic_engine_id(topic: str) -> str:
    # topic shape: uav/engine/{engine_id}/<kind>
    parts = topic.split("/")
    return parts[2] if len(parts) >= 3 else "unknown"


def _on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info("Connected to MQTT broker, subscribing to topics")
        client.subscribe(config.MQTT_TELEMETRY_TOPIC)
        client.subscribe(config.MQTT_HEALTH_TOPIC)
        client.subscribe(config.MQTT_FAULT_TOPIC)
        client.subscribe(config.MQTT_MISSION_ADVISORY_TOPIC)
    else:
        logger.error("MQTT connect failed with rc=%s", rc)


def _on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning("Dropping unparseable MQTT message on %s: %s", msg.topic, e)
        return

    if _loop is None:
        return

    if msg.topic.endswith("/telemetry"):
        asyncio.run_coroutine_threadsafe(_handle_telemetry(payload), _loop)
    elif msg.topic.endswith("/health"):
        asyncio.run_coroutine_threadsafe(_handle_health(payload), _loop)
    elif msg.topic.endswith("/fault"):
        asyncio.run_coroutine_threadsafe(_handle_fault(payload), _loop)
    elif msg.topic.endswith("/mission_advisory"):
        asyncio.run_coroutine_threadsafe(_handle_mission_advisory(payload), _loop)
    else:
        logger.warning("Unrecognized topic: %s", msg.topic)


async def _handle_telemetry(payload: dict):
    try:
        validated = Telemetry(**payload)
    except Exception as e:
        logger.warning("Invalid telemetry payload, dropping: %s", e)
        return
    await db.insert_telemetry(validated.model_dump())
    await ws_manager.broadcast(validated.engine_id, {"type": "telemetry", "data": validated.model_dump()})


async def _handle_health(payload: dict):
    try:
        validated = HealthIndex(**payload)
    except Exception as e:
        logger.warning("Invalid health payload, dropping: %s", e)
        return
    await db.insert_health(validated.model_dump())
    await ws_manager.broadcast(validated.engine_id, {"type": "health", "data": validated.model_dump()})


async def _handle_fault(payload: dict):
    try:
        validated = FaultEvent(**payload)
    except Exception as e:
        logger.warning("Invalid fault payload, dropping: %s", e)
        return
    await db.insert_fault(validated.model_dump())
    await ws_manager.broadcast(validated.engine_id, {"type": "fault", "data": validated.model_dump()})


async def _handle_mission_advisory(payload: dict):
    try:
        validated = MissionAdvisory(**payload)
    except Exception as e:
        logger.warning("Invalid mission advisory payload, dropping: %s", e)
        return
    await db.insert_mission_advisory(validated.model_dump())
    await ws_manager.broadcast(validated.engine_id, {"type": "mission_advisory", "data": validated.model_dump()})


def start(loop: asyncio.AbstractEventLoop):
    global _client, _loop
    _loop = loop
    _client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    _client.on_connect = _on_connect
    _client.on_message = _on_message
    # connect_async + loop_start (rather than a blocking connect()) hands
    # connection retries to paho's own reconnect logic instead of crashing
    # the app if the broker isn't accepting connections in the first instant
    # `docker compose up` starts this container.
    _client.reconnect_delay_set(min_delay=1, max_delay=30)
    _client.connect_async(config.MQTT_BROKER_HOST, config.MQTT_BROKER_PORT, keepalive=60)
    _client.loop_start()
    logger.info("MQTT client loop started")


def stop():
    global _client
    if _client is not None:
        _client.loop_stop()
        _client.disconnect()
        _client = None
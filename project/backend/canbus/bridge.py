"""
Bridges a SocketCAN interface (e.g. vcan0) to MQTT.

Sits between the simulator and the existing MQTT ingestion in
app/mqtt_client.py: the simulator writes CAN frames to vcan0 using
canbus/codec.py's encode_telemetry(); this script reads those frames,
decodes them back into the telemetry dict, and publishes that dict as JSON
to uav/engine/{engine_id}/telemetry -- exactly the topic/schema the
existing backend already ingests. The backend doesn't change at all.

Requires environment variables (same names/shapes as the backend's own
config -- see .env.example):
  MQTT_BROKER_HOST, MQTT_BROKER_PORT

Plus two new ones specific to this bridge, since the prompt didn't specify
them and I'm not going to guess:
  CAN_INTERFACE       e.g. "vcan0"
  CAN_ENGINE_ID        which engine this specific bus belongs to, e.g. "ENG01"
                       (see the one-engine-per-bus design note in codec.py)

Run with:  python -m canbus.bridge
"""
import json
import logging
import os
import time

import can
import paho.mqtt.client as mqtt

from .codec import CANTelemetryDecoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("can_mqtt_bridge")

MQTT_BROKER_HOST = os.environ.get("MQTT_BROKER_HOST")
MQTT_BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", "0") or 0)
CAN_INTERFACE = os.environ.get("CAN_INTERFACE", "vcan0")
CAN_ENGINE_ID = os.environ.get("CAN_ENGINE_ID")


def require_config():
    missing = [
        name
        for name, val in [
            ("MQTT_BROKER_HOST", MQTT_BROKER_HOST),
            ("MQTT_BROKER_PORT", MQTT_BROKER_PORT),
            ("CAN_ENGINE_ID", CAN_ENGINE_ID),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables for the CAN bridge: {', '.join(missing)}"
        )


def main():
    require_config()

    topic = f"uav/engine/{CAN_ENGINE_ID}/telemetry"
    decoder = CANTelemetryDecoder(engine_id=CAN_ENGINE_ID)

    mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)
    mqtt_client.connect_async(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
    mqtt_client.loop_start()

    logger.info("Opening CAN interface %s for engine_id=%s", CAN_INTERFACE, CAN_ENGINE_ID)
    bus = can.interface.Bus(channel=CAN_INTERFACE, interface="socketcan")

    logger.info("Listening for CAN frames, publishing decoded telemetry to %s", topic)
    try:
        for msg in bus:
            telemetry = decoder.feed(msg.arbitration_id, bytes(msg.data))
            if telemetry is None:
                continue  # still waiting on the rest of this sample's frames
            payload = json.dumps(telemetry)
            mqtt_client.publish(topic, payload)
            logger.debug("Published telemetry sample: %s", payload)
    except KeyboardInterrupt:
        pass
    finally:
        bus.shutdown()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()


if __name__ == "__main__":
    main()

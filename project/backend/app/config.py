"""
All connection details come from environment variables.
None of these values were specified in the source prompt, so nothing here
is hardcoded as a real default -- these must be set via .env / docker-compose.
See .env.example for the variable names.
"""
import os

MQTT_BROKER_HOST = os.environ.get("MQTT_BROKER_HOST")
MQTT_BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", "0") or 0)

POSTGRES_DSN = os.environ.get("POSTGRES_DSN")

# uav/engine/{engine_id}/telemetry etc. -- topic pattern given in the prompt
MQTT_TELEMETRY_TOPIC = "uav/engine/+/telemetry"
MQTT_HEALTH_TOPIC = "uav/engine/+/health"
MQTT_FAULT_TOPIC = "uav/engine/+/fault"
MQTT_MISSION_ADVISORY_TOPIC = "uav/engine/+/mission_advisory"


def require_config():
    missing = []
    if not MQTT_BROKER_HOST:
        missing.append("MQTT_BROKER_HOST")
    if not MQTT_BROKER_PORT:
        missing.append("MQTT_BROKER_PORT")
    if not POSTGRES_DSN:
        missing.append("POSTGRES_DSN")
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Set them in your .env file (see .env.example) before starting the app."
        )

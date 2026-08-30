"""
PLACEHOLDER -- this is P3's service to fill in with the real simulator.

Its job, per the architecture, will be to write engine telemetry onto
vcan0 (see backend/canbus/codec.py for the encode function it should use),
so backend/canbus/bridge.py can pick it up and publish to MQTT.

Right now it does nothing but log a heartbeat, so `docker compose up`
succeeds end-to-end on Day 0 without this service blocking anyone.
"""
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simulator")

if __name__ == "__main__":
    logger.info("Simulator placeholder running -- P3 hasn't wired in the real simulator yet.")
    while True:
        logger.info("heartbeat: waiting for real simulator implementation")
        time.sleep(30)

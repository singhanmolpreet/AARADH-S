"""
Writes fake-but-correctly-shaped telemetry to a SocketCAN interface on a
timer, using canbus/codec.py's encode_telemetry(). This stands in for P3's
real simulator so you can demo/test the full vcan0 -> bridge -> MQTT ->
backend path (and literally run `candump vcan0` to see it) before the real
simulator exists.

Values here are placeholder/randomized, NOT specified anywhere in the
prompt -- this is a test fixture, not real telemetry.

Requires: CAN_INTERFACE (default vcan0), CAN_ENGINE_ID, FAKE_CAN_INTERVAL_SECONDS (default 2)

Run with:  python -m canbus.fake_publisher
"""
import logging
import os
import random
import time
from datetime import datetime, timezone

import can

from .codec import encode_telemetry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fake_can_publisher")

CAN_INTERFACE = os.environ.get("CAN_INTERFACE", "vcan0")
CAN_ENGINE_ID = os.environ.get("CAN_ENGINE_ID")
INTERVAL_SECONDS = float(os.environ.get("FAKE_CAN_INTERVAL_SECONDS", "2"))


def _fake_telemetry() -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine_id": CAN_ENGINE_ID,
        "mission_phase": "cruise",
        "rpm": random.uniform(4800, 5600),
        "cht_c": random.uniform(120, 160),
        "egt_c": random.uniform(650, 750),
        "oil_pressure_bar": random.uniform(2.8, 3.6),
        "oil_temp_c": random.uniform(85, 105),
        "fuel_flow_lph": random.uniform(15, 22),
        "vibration_rms_g": random.uniform(0.1, 0.6),
        "battery_voltage_v": random.uniform(26.5, 28.5),
        "alternator_current_a": random.uniform(8, 16),
        "injection_timing_deg": random.uniform(10, 15),
        "map_kpa": random.uniform(80, 95),
        "boost_pressure_bar": random.uniform(0.1, 0.4),
    }


def main():
    if not CAN_ENGINE_ID:
        raise RuntimeError("CAN_ENGINE_ID environment variable is required")

    bus = can.interface.Bus(channel=CAN_INTERFACE, interface="socketcan")
    logger.info("Publishing fake telemetry for %s onto %s every %ss", CAN_ENGINE_ID, CAN_INTERFACE, INTERVAL_SECONDS)
    try:
        while True:
            telemetry = _fake_telemetry()
            for frame in encode_telemetry(telemetry):
                bus.send(can.Message(arbitration_id=frame.arbitration_id, data=frame.data, is_extended_id=False))
            time.sleep(INTERVAL_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        bus.shutdown()


if __name__ == "__main__":
    main()

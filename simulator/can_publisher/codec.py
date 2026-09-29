"""
Encodes/decodes the telemetry schema to/from CAN frames.
Updated for AARADH-S new telemetry contract including mission identity over CAN.
"""
import struct
from datetime import datetime, timezone
from typing import Dict, List, NamedTuple, Optional

FRAME_RPM_TEMPS = 0x100
FRAME_PRESSURES_FLOW = 0x101
FRAME_ELECTRICAL = 0x102
FRAME_TIMESTAMP = 0x103
FRAME_MISSION_PHASE = 0x104
FRAME_NEW_TELEMETRY_1 = 0x105
FRAME_NEW_TELEMETRY_2 = 0x106
FRAME_MISSION_ID = 0x107

ALL_FRAME_IDS = {
    FRAME_RPM_TEMPS,
    FRAME_PRESSURES_FLOW,
    FRAME_ELECTRICAL,
    FRAME_TIMESTAMP,
    FRAME_MISSION_PHASE,
    FRAME_NEW_TELEMETRY_1,
    FRAME_NEW_TELEMETRY_2,
    FRAME_MISSION_ID,
}


class CANFrame(NamedTuple):
    arbitration_id: int
    data: bytes


def _clamp_i16(v: float) -> int:
    return max(-32768, min(32767, int(round(v))))


def _clamp_u16(v: float) -> int:
    return max(0, min(65535, int(round(v))))


def _clamp_u32(v: float) -> int:
    return max(0, min(4294967295, int(round(v))))


def encode_telemetry(telemetry: Dict) -> List[CANFrame]:
    frames = []

    frames.append(CANFrame(
        FRAME_RPM_TEMPS,
        struct.pack(
            ">Hhhh",
            _clamp_u16(telemetry["rpm"]),
            _clamp_i16(telemetry["cht_c"] * 10),
            _clamp_i16(telemetry["egt_c"] * 10),
            _clamp_i16(telemetry["oil_temp_c"] * 10),
        ),
    ))

    # Note: legacy fields (map_kpa etc) are no longer encoded. 
    # To keep the frame size, we will just send 0 or change struct.
    # The requirement says "Preserve all existing working frames."
    # We'll put 0 for map_kpa since it was at the end of 0x101.
    frames.append(CANFrame(
        FRAME_PRESSURES_FLOW,
        struct.pack(
            ">hHHH",
            _clamp_i16(telemetry["oil_pressure_bar"] * 100),
            _clamp_u16(telemetry["fuel_flow_lph"] * 10),
            _clamp_u16(telemetry["vibration_rms_g"] * 1000),
            0, # removed map_kpa
        ),
    ))

    # 0x102 removed alternator_current_a, boost_pressure_bar
    frames.append(CANFrame(
        FRAME_ELECTRICAL,
        struct.pack(
            ">Hhhh",
            _clamp_u16(telemetry["battery_voltage_v"] * 100),
            0, # removed alternator_current_a
            _clamp_i16(telemetry["injection_timing_deg"] * 100),
            0, # removed boost_pressure_bar
        ),
    ))

    ts = telemetry["timestamp"]
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    micros = int(dt.timestamp() * 1_000_000)
    frames.append(CANFrame(FRAME_TIMESTAMP, struct.pack(">q", micros)))

    phase_bytes = telemetry["mission_phase"].encode("ascii")[:8]
    phase_bytes = phase_bytes.ljust(8, b"\x00")
    frames.append(CANFrame(FRAME_MISSION_PHASE, phase_bytes))

    frames.append(CANFrame(
        FRAME_NEW_TELEMETRY_1,
        struct.pack(
            ">HHhH",
            _clamp_u16(telemetry["throttle_frac"] * 10000),
            _clamp_u16(telemetry["power_kw"] * 1000),
            _clamp_i16(telemetry["ambient_temp_c"] * 100),
            _clamp_u16(telemetry["air_density_kgm3"] * 10000),
        ),
    ))

    frames.append(CANFrame(
        FRAME_NEW_TELEMETRY_2,
        struct.pack(
            ">II",
            _clamp_u32(telemetry["air_pressure_pa"] * 1),
            _clamp_u32(telemetry["altitude_m"] * 10),
        ),
    ))

    frames.append(CANFrame(
        FRAME_MISSION_ID,
        struct.pack(
            ">I4x",
            _clamp_u32(int(telemetry["mission_id"])),
        ),
    ))

    return frames


class CANTelemetryDecoder:
    def __init__(self, engine_id: str):
        self.engine_id = engine_id
        self._buffer: Dict[int, bytes] = {}

    def feed(self, arbitration_id: int, data: bytes) -> Optional[Dict]:
        if arbitration_id not in ALL_FRAME_IDS:
            return None

        self._buffer[arbitration_id] = data

        if not ALL_FRAME_IDS.issubset(self._buffer.keys()):
            return None

        result = self._decode_buffer()
        self._buffer = {}
        return result

    def _decode_buffer(self) -> Dict:
        rpm, cht10, egt10, oil_temp10 = struct.unpack(">Hhhh", self._buffer[FRAME_RPM_TEMPS])
        oil_p100, fuel10, vib1000, _ = struct.unpack(">hHHH", self._buffer[FRAME_PRESSURES_FLOW])
        batt100, _, inj100, _ = struct.unpack(">Hhhh", self._buffer[FRAME_ELECTRICAL])
        (micros,) = struct.unpack(">q", self._buffer[FRAME_TIMESTAMP])
        phase = self._buffer[FRAME_MISSION_PHASE].rstrip(b"\x00").decode("ascii")
        throttle10000, power1000, amb_temp100, dens10000 = struct.unpack(">HHhH", self._buffer[FRAME_NEW_TELEMETRY_1])
        press1, alt10 = struct.unpack(">II", self._buffer[FRAME_NEW_TELEMETRY_2])
        run_id, = struct.unpack(">I", self._buffer[FRAME_MISSION_ID][:4])

        dt = datetime.fromtimestamp(micros / 1_000_000, tz=timezone.utc)

        return {
            "timestamp": dt.isoformat(),
            "engine_id": self.engine_id,
            "mission_id": str(run_id),
            "mission_phase": phase,
            "rpm": rpm,
            "cht_c": cht10 / 10,
            "egt_c": egt10 / 10,
            "oil_pressure_bar": oil_p100 / 100,
            "oil_temp_c": oil_temp10 / 10,
            "fuel_flow_lph": fuel10 / 10,
            "vibration_rms_g": vib1000 / 1000,
            "battery_voltage_v": batt100 / 100,
            "injection_timing_deg": inj100 / 100,
            "throttle_frac": throttle10000 / 10000,
            "power_kw": power1000 / 1000,
            "ambient_temp_c": amb_temp100 / 100,
            "air_pressure_pa": press1 / 1,
            "air_density_kgm3": dens10000 / 10000,
            "altitude_m": alt10 / 10,
        }


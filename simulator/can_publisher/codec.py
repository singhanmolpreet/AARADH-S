"""
Encodes/decodes the telemetry schema (given schema, unchanged) to/from
classic CAN frames (8-byte payloads) for a SocketCAN vcan0 bus.

DESIGN NOTE -- this framing was NOT given in the prompt, so flagging it
clearly as MY design choice, not given data:

The prompt's own example named only two arbitration IDs (0x100, 0x101) as
illustrative ("one per sensor group is fine"). The telemetry schema has 13
numeric fields plus mission_phase (string) and timestamp (string) -- more
than fits in two 8-byte classic CAN frames. I split it into 5 frames.

ASSUMPTION: one engine per vcan bus. engine_id is NOT encoded on the wire --
it's a property of which bus/interface you're listening to (you pass it in
when constructing the decoder/bridge), matching the fact that MQTT topics
are already scoped per engine_id upstream. If you actually need multiple
engines sharing one physical/virtual CAN bus, tell me and I'll fold an
engine index into the arbitration IDs instead -- that's a real design
change, not a tweak.

Frame layout (all multi-byte integers big-endian / '>' struct format):

  0x100  RPM / core temps
    bytes 0-1  rpm            uint16            raw rpm (0..65535)
    bytes 2-3  cht_c          int16   (x10)      0.1 C resolution
    bytes 4-5  egt_c          int16   (x10)      0.1 C resolution
    bytes 6-7  oil_temp_c     int16   (x10)      0.1 C resolution

  0x101  Pressures / flow
    bytes 0-1  oil_pressure_bar   int16  (x100)  0.01 bar resolution
    bytes 2-3  fuel_flow_lph      uint16 (x10)   0.1 L/h resolution
    bytes 4-5  vibration_rms_g    uint16 (x1000) 0.001 g resolution
    bytes 6-7  map_kpa            uint16 (x10)   0.1 kPa resolution

  0x102  Electrical / timing / boost
    bytes 0-1  battery_voltage_v      uint16 (x100)  0.01 V resolution
    bytes 2-3  alternator_current_a   int16  (x100)  0.01 A resolution
    bytes 4-5  injection_timing_deg   int16  (x100)  0.01 deg resolution
    bytes 6-7  boost_pressure_bar     int16  (x1000) 0.001 bar resolution

  0x103  Timestamp
    bytes 0-7  int64  microseconds since Unix epoch (UTC)

  0x104  Mission phase
    bytes 0-7  ASCII string, right-padded with 0x00, TRUNCATED TO 8 CHARS.
               "preflight" (9 chars) would truncate to "prefligh" -- if you
               have a fixed, known set of mission phases, tell me and I'll
               switch this to a 1-byte enum instead, which is both more
               CAN-idiomatic and removes the truncation risk entirely.

All scaling factors above are a resolution/range tradeoff I picked for
16-bit fields -- reasonable for engine telemetry, but arbitrary. Change the
SCALE constants below if you need different precision or range.
"""
import struct
from datetime import datetime, timezone
from typing import Dict, List, NamedTuple, Optional

FRAME_RPM_TEMPS = 0x100
FRAME_PRESSURES_FLOW = 0x101
FRAME_ELECTRICAL = 0x102
FRAME_TIMESTAMP = 0x103
FRAME_MISSION_PHASE = 0x104

ALL_FRAME_IDS = {
    FRAME_RPM_TEMPS,
    FRAME_PRESSURES_FLOW,
    FRAME_ELECTRICAL,
    FRAME_TIMESTAMP,
    FRAME_MISSION_PHASE,
}


class CANFrame(NamedTuple):
    arbitration_id: int
    data: bytes


def _clamp_i16(v: float) -> int:
    return max(-32768, min(32767, int(round(v))))


def _clamp_u16(v: float) -> int:
    return max(0, min(65535, int(round(v))))


def encode_telemetry(telemetry: Dict) -> List[CANFrame]:
    """
    Takes a telemetry dict matching the given schema exactly and returns
    the list of CAN frames to send. Raises KeyError if a required field
    is missing -- this deliberately does not fill in defaults for missing
    data.
    """
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

    frames.append(CANFrame(
        FRAME_PRESSURES_FLOW,
        struct.pack(
            ">hHHH",
            _clamp_i16(telemetry["oil_pressure_bar"] * 100),
            _clamp_u16(telemetry["fuel_flow_lph"] * 10),
            _clamp_u16(telemetry["vibration_rms_g"] * 1000),
            _clamp_u16(telemetry["map_kpa"] * 10),
        ),
    ))

    frames.append(CANFrame(
        FRAME_ELECTRICAL,
        struct.pack(
            ">Hhhh",
            _clamp_u16(telemetry["battery_voltage_v"] * 100),
            _clamp_i16(telemetry["alternator_current_a"] * 100),
            _clamp_i16(telemetry["injection_timing_deg"] * 100),
            _clamp_i16(telemetry["boost_pressure_bar"] * 1000),
        ),
    ))

    ts = telemetry["timestamp"]
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    micros = int(dt.timestamp() * 1_000_000)
    frames.append(CANFrame(FRAME_TIMESTAMP, struct.pack(">q", micros)))

    phase_bytes = telemetry["mission_phase"].encode("ascii")[:8]
    phase_bytes = phase_bytes.ljust(8, b"\x00")
    frames.append(CANFrame(FRAME_MISSION_PHASE, phase_bytes))

    return frames


class CANTelemetryDecoder:
    """
    Stateful decoder for one engine's vcan bus. Feed it every received
    frame; once all 5 frame types for a given telemetry sample have
    arrived, feed() returns the reassembled dict (matching the schema
    exactly, with engine_id filled in from what you constructed this with)
    and resets its internal buffer for the next sample.

    Because CAN doesn't guarantee frames from one "sample" arrive in any
    particular order, this buffers by frame type rather than by sequence.
    """

    def __init__(self, engine_id: str):
        self.engine_id = engine_id
        self._buffer: Dict[int, bytes] = {}

    def feed(self, arbitration_id: int, data: bytes) -> Optional[Dict]:
        if arbitration_id not in ALL_FRAME_IDS:
            return None  # not one of ours; ignore

        self._buffer[arbitration_id] = data

        if not ALL_FRAME_IDS.issubset(self._buffer.keys()):
            return None  # still waiting on other frame types

        result = self._decode_buffer()
        self._buffer = {}
        return result

    def _decode_buffer(self) -> Dict:
        rpm, cht10, egt10, oil_temp10 = struct.unpack(">Hhhh", self._buffer[FRAME_RPM_TEMPS])
        oil_p100, fuel10, vib1000, map10 = struct.unpack(">hHHH", self._buffer[FRAME_PRESSURES_FLOW])
        batt100, alt100, inj100, boost1000 = struct.unpack(">Hhhh", self._buffer[FRAME_ELECTRICAL])
        (micros,) = struct.unpack(">q", self._buffer[FRAME_TIMESTAMP])
        phase = self._buffer[FRAME_MISSION_PHASE].rstrip(b"\x00").decode("ascii")

        dt = datetime.fromtimestamp(micros / 1_000_000, tz=timezone.utc)

        return {
            "timestamp": dt.isoformat(),
            "engine_id": self.engine_id,
            "mission_phase": phase,
            "rpm": rpm,
            "cht_c": cht10 / 10,
            "egt_c": egt10 / 10,
            "oil_pressure_bar": oil_p100 / 100,
            "oil_temp_c": oil_temp10 / 10,
            "fuel_flow_lph": fuel10 / 10,
            "vibration_rms_g": vib1000 / 1000,
            "battery_voltage_v": batt100 / 100,
            "alternator_current_a": alt100 / 100,
            "injection_timing_deg": inj100 / 100,
            "map_kpa": map10 / 10,
            "boost_pressure_bar": boost1000 / 1000,
        }

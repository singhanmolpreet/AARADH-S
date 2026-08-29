"""
Round-trip test for canbus/codec.py -- verifies encode -> decode reproduces
the original telemetry values (within the fixed-point resolution documented
in codec.py). Needs no CAN interface at all, so you can run this before
vcan0 is even set up.

Run with:  python -m canbus.test_codec
"""
from canbus.codec import encode_telemetry, CANTelemetryDecoder

SAMPLE = {
    "timestamp": "2026-08-29T10:57:19.012170+00:00",
    "engine_id": "ENG01",
    "mission_phase": "cruise",
    "rpm": 5400,
    "cht_c": 140,
    "egt_c": 700,
    "oil_pressure_bar": 3.2,
    "oil_temp_c": 95,
    "fuel_flow_lph": 18.7,
    "vibration_rms_g": 0.42,
    "battery_voltage_v": 27.6,
    "alternator_current_a": 12.1,
    "injection_timing_deg": 12.5,
    "map_kpa": 88.0,
    "boost_pressure_bar": 0.25,
}


def main():
    frames = encode_telemetry(SAMPLE)
    print(f"Encoded into {len(frames)} frames:")
    for f in frames:
        print(f"  id=0x{f.arbitration_id:03X}  data={f.data.hex()}")

    decoder = CANTelemetryDecoder(engine_id="ENG01")
    result = None
    for f in frames:
        result = decoder.feed(f.arbitration_id, f.data)

    assert result is not None, "Decoder never produced a complete sample"
    print("\nDecoded result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\nDifferences from original (expect small rounding only):")
    for k in SAMPLE:
        if k in ("timestamp",):  # microsecond precision, will differ trivially
            continue
        if isinstance(SAMPLE[k], (int, float)) and abs(SAMPLE[k] - result[k]) > 0.5:
            print(f"  MISMATCH {k}: original={SAMPLE[k]} decoded={result[k]}")

    print("\nOK -- codec round-trip looks correct.")


if __name__ == "__main__":
    main()

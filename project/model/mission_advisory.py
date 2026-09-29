"""
Rule-based mission replanning advisory.
NOTE: EXTENSION_FACTOR below is an ASSUMED placeholder relating RPM/altitude
reduction to RUL extension — not derived from flight data or the physics
model. Say so plainly if asked; don't present it as measured.
"""
from datetime import datetime, timezone

EXTENSION_FACTOR = 1.5
RPM_REDUCTION = 400
ALTITUDE_REDUCTION_M = 600  # ~2000 ft

def generate_advisory(rul_minutes, mission_time_remaining_minutes,
                       current_rpm, current_altitude_m, engine_id="ENG-01", mission_id=None):
    if rul_minutes is None or rul_minutes >= mission_time_remaining_minutes:
        return None
    new_rpm = max(current_rpm - RPM_REDUCTION, 0)
    new_altitude = max(current_altitude_m - ALTITUDE_REDUCTION_M, 0)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine_id": engine_id,
        "mission_id": mission_id,
        "rul_minutes": round(rul_minutes, 1),
        "mission_time_remaining_minutes": round(mission_time_remaining_minutes, 1),
        "recommended_strategy": f"Reduce cruise altitude by {ALTITUDE_REDUCTION_M}m and drop engine RPM to {new_rpm}",
        "estimated_new_rul_minutes": round(rul_minutes * EXTENSION_FACTOR, 1),
    }
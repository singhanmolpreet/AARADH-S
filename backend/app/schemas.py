"""
Pydantic models matching EXACTLY the message schemas provided.
No fields are added, removed, or renamed beyond what was specified.
"""
from typing import List, Optional
from pydantic import BaseModel


class Telemetry(BaseModel):
    timestamp: str
    engine_id: str
    mission_id: str
    mission_phase: str
    rpm: float
    cht_c: float
    egt_c: float
    oil_pressure_bar: float
    oil_temp_c: float
    fuel_flow_lph: float
    vibration_rms_g: float
    battery_voltage_v: float
    injection_timing_deg: float
    throttle_frac: float
    power_kw: float
    ambient_temp_c: float
    air_pressure_pa: float
    air_density_kgm3: float
    altitude_m: float


class ComponentHealth(BaseModel):
    component: str
    health_score: float
    status: str
    rul_minutes: float
    top_features: List[str]


class HealthIndex(BaseModel):
    timestamp: str
    engine_id: str
    mission_id: str
    components: List[ComponentHealth]
    overall_health_score: float
    overall_status: str
    active_faults: List[str]


class FaultEvent(BaseModel):
    timestamp: str
    engine_id: str
    mission_id: str
    fault_type: str
    fault_type_code: int
    component: str
    confidence: float
    severity: str
    severity_frac: float
    title: str
    evidence: str
    recommended_action: str


class MissionAdvisory(BaseModel):
    timestamp: str
    engine_id: str
    mission_id: str
    rul_minutes: float
    mission_time_remaining_minutes: float
    recommended_strategy: str
    estimated_new_rul_minutes: float


class MissionSummary(BaseModel):
    mission_id: str
    engine_id: str
    start_ts: str
    end_ts: str
    n_samples: int
    mission_phase: str
    worst_fault_type: Optional[str] = None
    max_severity_frac: Optional[float] = None


class ReplayFrame(BaseModel):
    ts: str
    telemetry: Optional[dict] = None
    health: Optional[dict] = None
    fault: Optional[dict] = None


class InjectFaultRequest(BaseModel):
    fault_type: str
    engine_id: str

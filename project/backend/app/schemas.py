"""
Pydantic models matching EXACTLY the message schemas provided.
No fields are added, removed, or renamed beyond what was specified.
"""
from typing import List, Optional
from pydantic import BaseModel


class Telemetry(BaseModel):
    timestamp: str
    engine_id: str
    mission_phase: str
    rpm: float
    cht_c: float
    egt_c: float
    oil_pressure_bar: float
    oil_temp_c: float
    fuel_flow_lph: float
    vibration_rms_g: float
    battery_voltage_v: float
    alternator_current_a: float
    injection_timing_deg: float
    map_kpa: float
    boost_pressure_bar: float


class ComponentHealth(BaseModel):
    component: str
    health_score: float
    status: str
    rul_hours: float
    top_features: List[str]


class HealthIndex(BaseModel):
    timestamp: str
    engine_id: str
    components: List[ComponentHealth]
    overall_health_score: float
    overall_status: str
    active_faults: List[str]


class FaultEvent(BaseModel):
    timestamp: str
    engine_id: str
    fault_type: str
    component: str
    confidence: float
    severity: str
    title: str
    evidence: str
    recommended_action: str


class MissionAdvisory(BaseModel):
    timestamp: str
    engine_id: str
    rul_minutes: float
    mission_time_remaining_minutes: float
    recommended_strategy: str
    estimated_new_rul_minutes: float


class InjectFaultRequest(BaseModel):
    fault_type: str
    engine_id: str

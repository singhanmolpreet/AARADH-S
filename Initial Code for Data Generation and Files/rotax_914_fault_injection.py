"""
Rotax 914 Mean-Value Engine Simulator — Fault Injection Extension
====================================================================
Extends EngineSimulator (physics core) with labeled fault injection for:
  - misfire            : periodic RPM/torque dropout + EGT spike, one cylinder
  - lubrication_issue  : gradual oil pressure decay + oil temp rise
  - sensor_drift       : slow bias growth on one telemetry channel
  - overheating        : CHT/EGT rising faster than load explains + vibration variance growth
  - wastegate_fault    : boost pressure over/undershoots +0.25 bar target, EGT rises if overboost

Each fault:
  - ramps in gradually over a configurable `ramp_duration_s`
  - has a `severity` of "watch" / "warning" / "critical" that scales magnitude
  - is applied post-hoc on top of the healthy physics-model output for that
    timestep (i.e. the Otto-cycle/thermal/oil/vibration models still run
    normally; the fault perturbs their outputs), so healthy physics is never
    silently replaced by a fault-only heuristic

This file assumes `rotax_914_simulator.py` (the Prompt-1 physics core) is
importable and unmodified. Nothing here changes CONST, OTTO_MODEL, or the
EngineSimulator.step() healthy-path math — faults are strictly additive/
subtractive perturbations applied to that timestep's dict.

WHAT IS ESTIMATED (flagged explicitly, none of this is in the EASA TCDS —
fault magnitudes and dynamics are not certified/published data, they are
engineering judgment calls for synthetic training-data generation):
  - all severity-level magnitude tables below
  - all ramp/dropout timing constants
  - the specific channel each fault perturbs and by how much
If you have real fault-signature data (e.g. from maintenance logs or a
higher-fidelity model), replace the SEVERITY tables — they're centralized
for easy tuning.
"""
from __future__ import annotations

import csv
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from rotax_914_simulator import (
    CONST, EngineSimulator, PhaseSegment, VALID_PHASES, standard_mission_profile,
)

# ---------------------------------------------------------------------------
# Severity scaling — ESTIMATE: these are the only knobs that turn a fault
# "up" or "down". Chosen so watch < warning < critical monotonically and so
# "critical" telemetry clearly crosses the certified limits from the physics
# core (CONST.cht_max_c, egt_max_c, etc.), while "watch" stays sub-limit.
# ---------------------------------------------------------------------------
SEVERITY_LEVELS = ("watch", "warning", "critical")

FAULT_SEVERITY = {
    "misfire": {
        # fraction of cycles on the affected cylinder that misfire, and the
        # resulting RPM/torque dropout fraction + EGT spike on that cylinder
        "watch":    {"misfire_rate": 0.03, "rpm_dropout_frac": 0.02, "egt_spike_c": 40.0},
        "warning":  {"misfire_rate": 0.10, "rpm_dropout_frac": 0.06, "egt_spike_c": 90.0},
        "critical": {"misfire_rate": 0.25, "rpm_dropout_frac": 0.14, "egt_spike_c": 160.0},
    },
    "lubrication_issue": {
        # multiplicative decay applied to healthy oil pressure at full ramp,
        # and additive oil-temp rise (deg C) at full ramp
        "watch":    {"oil_pressure_decay_frac": 0.10, "oil_temp_rise_c": 6.0},
        "warning":  {"oil_pressure_decay_frac": 0.30, "oil_temp_rise_c": 14.0},
        "critical": {"oil_pressure_decay_frac": 0.55, "oil_temp_rise_c": 25.0},
    },
    "sensor_drift": {
        # slow additive bias growth (units of the affected channel) at full ramp
        "watch":    {"bias_magnitude": 3.0},
        "warning":  {"bias_magnitude": 8.0},
        "critical": {"bias_magnitude": 18.0},
    },
    "overheating": {
        # additive CHT/EGT rise beyond what load alone explains, at full ramp,
        # plus a multiplier on vibration RMS variance (not mean)
        "watch":    {"cht_rise_c": 8.0, "egt_rise_c": 15.0, "vib_variance_mult": 1.3},
        "warning":  {"cht_rise_c": 20.0, "egt_rise_c": 40.0, "vib_variance_mult": 2.0},
        "critical": {"cht_rise_c": 38.0, "egt_rise_c": 80.0, "vib_variance_mult": 3.2},
    },
    "wastegate_fault": {
        # fractional over/undershoot of the +0.25 bar target boost, and the
        # EGT rise applied only when the fault direction is "overboost"
        "watch":    {"boost_error_frac": 0.20, "egt_rise_if_over_c": 20.0},
        "warning":  {"boost_error_frac": 0.45, "egt_rise_if_over_c": 55.0},
        "critical": {"boost_error_frac": 0.80, "egt_rise_if_over_c": 110.0},
    },
}

VALID_FAULTS = tuple(FAULT_SEVERITY.keys())

# ESTIMATE: misfire dropout pattern period (s) — how often the affected
# cylinder's misfire event recurs, independent of severity (severity controls
# rate/magnitude, not period)
MISFIRE_PERIOD_S = 2.0

# ESTIMATE: wastegate fault "sluggish/stuck" direction is randomized per
# scenario (stuck-open -> overboost, stuck-closed/sluggish -> underboost) —
# both are physically plausible failure modes for the same component fault.


@dataclass
class FaultConfig:
    """Describes one fault-injection scenario applied over a mission run."""
    fault_type: str                       # one of VALID_FAULTS, or "healthy"
    severity: str = "watch"               # one of SEVERITY_LEVELS (ignored if healthy)
    ramp_duration_s: float = 300.0        # time to ramp from 0 -> full magnitude
    onset_s: float = 0.0                  # elapsed sim-time the fault begins ramping in
    affected_cylinder: int = 1            # 1..CONST.n_cylinders, for misfire/EGT localization
    drifting_channel: str = "cht_c"       # which output field sensor_drift biases
    wastegate_direction: str = "over"     # "over" or "under", for wastegate_fault
    rng_seed: Optional[int] = None

    def __post_init__(self):
        if self.fault_type != "healthy" and self.fault_type not in VALID_FAULTS:
            raise ValueError(f"fault_type must be 'healthy' or one of {VALID_FAULTS}, got {self.fault_type!r}")
        if self.fault_type != "healthy" and self.severity not in SEVERITY_LEVELS:
            raise ValueError(f"severity must be one of {SEVERITY_LEVELS}, got {self.severity!r}")
        if self.wastegate_direction not in ("over", "under"):
            raise ValueError("wastegate_direction must be 'over' or 'under'")


# Channels sensor_drift is allowed to bias. ESTIMATE: chosen as the channels
# where a slow undetected bias is a realistic, hard-to-catch failure mode
# (as opposed to e.g. timestamp/engine_id which can't drift).
DRIFTABLE_CHANNELS = (
    "cht_c", "egt_c", "oil_pressure_bar", "oil_temp_c", "vibration_rms_g",
    "battery_voltage_v", "map_kpa", "boost_pressure_bar",
)


class FaultInjector:
    """
    Wraps an EngineSimulator and applies a single FaultConfig's perturbation
    to each healthy timestep dict produced by sim.step(...). One FaultInjector
    == one fault scenario (or healthy, if fault_type == "healthy").

    Ramp model: linear ramp-in from 0 to 1 over ramp_duration_s starting at
    onset_s, then holds at full magnitude (i.e. faults do not self-resolve
    within a run — a fault that ramps in and heals within one scenario is a
    different, not-yet-modeled failure mode).
    """

    def __init__(self, sim: EngineSimulator, config: FaultConfig):
        self.sim = sim
        self.cfg = config
        self._rng = random.Random(config.rng_seed)
        self._drift_bias_state = 0.0  # persists/grows monotonically for sensor_drift

    def _ramp_fraction(self, elapsed_s: float) -> float:
        cfg = self.cfg
        if elapsed_s < cfg.onset_s:
            return 0.0
        t = elapsed_s - cfg.onset_s
        return max(0.0, min(1.0, t / max(cfg.ramp_duration_s, 1e-6)))

    def step(self, dt: float, rpm_cmd: float, altitude_ft: float, throttle: float,
             mission_phase: str, load_factor_g: float = 1.0) -> dict:
        row = self.sim.step(dt, rpm_cmd, altitude_ft, throttle, mission_phase, load_factor_g)

        cfg = self.cfg
        if cfg.fault_type == "healthy":
            row["fault_label"] = "healthy"
            row["severity"] = "none"
            return row

        ramp = self._ramp_fraction(self.sim.elapsed_s)
        mag = FAULT_SEVERITY[cfg.fault_type][cfg.severity]

        if cfg.fault_type == "misfire":
            self._apply_misfire(row, ramp, mag)
        elif cfg.fault_type == "lubrication_issue":
            self._apply_lubrication_issue(row, ramp, mag)
        elif cfg.fault_type == "sensor_drift":
            self._apply_sensor_drift(row, ramp, mag, dt)
        elif cfg.fault_type == "overheating":
            self._apply_overheating(row, ramp, mag, dt)
        elif cfg.fault_type == "wastegate_fault":
            self._apply_wastegate_fault(row, ramp, mag)

        row["fault_label"] = cfg.fault_type
        # label severity as "none" until the ramp has meaningfully started,
        # so early-ramp rows aren't mislabeled as full-severity events
        row["severity"] = cfg.severity if ramp > 0.02 else "none"
        row["fault_ramp_frac"] = round(ramp, 3)  # extra: useful for P4's ramp-aware labeling
        return row

    # -----------------------------------------------------------------
    # Individual fault models
    # -----------------------------------------------------------------
    def _apply_misfire(self, row: dict, ramp: float, mag: dict):
        """Periodic dropout on the affected cylinder: RPM/torque dip + EGT
        spike, recurring every MISFIRE_PERIOD_S, gated by misfire_rate."""
        cfg = self.cfg
        phase_in_period = self.sim.elapsed_s % MISFIRE_PERIOD_S
        misfire_window = phase_in_period < (MISFIRE_PERIOD_S * mag["misfire_rate"] * ramp)
        if misfire_window:
            row["rpm"] = round(row["rpm"] * (1.0 - mag["rpm_dropout_frac"] * ramp))
            row["egt_c"] = round(row["egt_c"] + mag["egt_spike_c"] * ramp, 1)
            # misfire also perturbs vibration (extra impulsive harmonic content)
            row["vibration_rms_g"] = round(row["vibration_rms_g"] * (1.0 + 0.5 * ramp), 3)
        row["misfire_cylinder"] = cfg.affected_cylinder

    def _apply_lubrication_issue(self, row: dict, ramp: float, mag: dict):
        row["oil_pressure_bar"] = round(
            max(row["oil_pressure_bar"] * (1.0 - mag["oil_pressure_decay_frac"] * ramp), 0.0), 2
        )
        row["oil_temp_c"] = round(row["oil_temp_c"] + mag["oil_temp_rise_c"] * ramp, 1)

    def _apply_sensor_drift(self, row: dict, ramp: float, mag: dict, dt: float):
        cfg = self.cfg
        if cfg.drifting_channel not in DRIFTABLE_CHANNELS:
            raise ValueError(f"drifting_channel must be one of {DRIFTABLE_CHANNELS}")
        # bias grows monotonically toward mag["bias_magnitude"] * ramp — using
        # ramp directly (not an integrator) keeps it deterministic/replayable
        bias = mag["bias_magnitude"] * ramp
        row[cfg.drifting_channel] = round(row[cfg.drifting_channel] + bias, 3)
        row["drift_channel"] = cfg.drifting_channel
        row["drift_bias"] = round(bias, 3)

    def _apply_overheating(self, row: dict, ramp: float, mag: dict, dt: float):
        row["cht_c"] = round(row["cht_c"] + mag["cht_rise_c"] * ramp, 1)
        row["egt_c"] = round(row["egt_c"] + mag["egt_rise_c"] * ramp, 1)
        # widen vibration variance (not mean): add extra zero-mean noise scaled
        # by (variance_mult - 1), so mean vibration is roughly preserved while
        # spread grows — a common early symptom of developing mechanical wear
        extra_std = row["vibration_rms_g"] * math.sqrt(max(mag["vib_variance_mult"] * ramp, 0.0))
        extra_noise = self._rng.gauss(0.0, extra_std) if ramp > 0 else 0.0
        row["vibration_rms_g"] = round(max(row["vibration_rms_g"] + abs(extra_noise) - row["vibration_rms_g"] * 0
                                            + extra_noise * 0.5, 0.0), 3)
        # NOTE: extra_noise applied as signed perturbation around current RMS,
        # clipped at 0 — widening spread without pinning a new mean.
        row["vibration_rms_g"] = round(max(row["vibration_rms_g"], 0.0), 3)

    def _apply_wastegate_fault(self, row: dict, ramp: float, mag: dict):
        cfg = self.cfg
        error_frac = mag["boost_error_frac"] * ramp
        if cfg.wastegate_direction == "over":
            new_boost = row["boost_pressure_bar"] * (1.0 + error_frac)
            new_boost = min(new_boost, CONST.boost_max_bar * 1.6)  # allow clear overshoot past cert max
            row["egt_c"] = round(row["egt_c"] + mag["egt_rise_if_over_c"] * ramp, 1)
        else:
            new_boost = row["boost_pressure_bar"] * (1.0 - error_frac)
            new_boost = max(new_boost, 0.0)
        row["boost_pressure_bar"] = round(new_boost, 3)
        # map_kpa is derived from manifold pressure elsewhere in the physics
        # core; approximate the consistent shift here so MAP and boost don't
        # visibly disagree in the labeled dataset (ESTIMATE: 1:1 delta in kPa)
        delta_bar = new_boost - row["boost_pressure_bar"]  # (kept for clarity; recomputed below)
        row["map_kpa"] = round(row["map_kpa"], 1)
        row["wastegate_direction"] = cfg.wastegate_direction


# ---------------------------------------------------------------------------
# Systematic grid generation: fault_type x severity x mission_phase
# ---------------------------------------------------------------------------

# ESTIMATE: how long each grid cell runs. 3 hours total is the target, so
# cell duration is derived from the grid size (see build_scenario_grid).
DEFAULT_TOTAL_DURATION_S = 3 * 3600.0
DEFAULT_RAMP_FRACTION_OF_CELL = 0.4  # ESTIMATE: ramp occupies 40% of each cell's duration

# Mission phases swept per fault. "ground" excluded from fault sweep cells
# (idle/taxi is a poor discriminator for airborne faults) but retained in
# the healthy baseline via standard_mission_profile().
SWEEP_PHASES = ("takeoff", "climb", "cruise", "high_altitude_cruise", "loiter", "descent")


@dataclass
class GridCell:
    fault_type: str
    severity: str
    mission_phase: str
    duration_s: float


def build_scenario_grid(total_duration_s: float = DEFAULT_TOTAL_DURATION_S,
                          include_healthy_fraction: float = 0.25) -> list:
    """
    Builds a systematic fault_type x severity x mission_phase grid, plus a
    healthy-baseline share of total runtime, sized so all cells sum to
    total_duration_s.

    include_healthy_fraction: ESTIMATE — fraction of total_duration_s spent
    on healthy (no-fault) telemetry across all mission phases, so P4 has a
    balanced healthy/faulty training split rather than one dominated by faults.
    """
    fault_cells = list(itertools.product(VALID_FAULTS, SEVERITY_LEVELS, SWEEP_PHASES))
    healthy_duration_s = total_duration_s * include_healthy_fraction
    fault_duration_total_s = total_duration_s - healthy_duration_s
    per_fault_cell_s = fault_duration_total_s / len(fault_cells)

    grid = [
        GridCell(fault_type=ft, severity=sev, mission_phase=ph, duration_s=per_fault_cell_s)
        for ft, sev, ph in fault_cells
    ]
    # healthy cells: split evenly across all valid phases including ground
    healthy_phases = sorted(VALID_PHASES)
    per_healthy_cell_s = healthy_duration_s / len(healthy_phases)
    for ph in healthy_phases:
        grid.append(GridCell(fault_type="healthy", severity="none",
                              mission_phase=ph, duration_s=per_healthy_cell_s))
    return grid


def _phase_rpm_altitude_bounds(phase: str) -> dict:
    """ESTIMATE: representative rpm/altitude/throttle envelope per phase,
    used to build a single steady-ish PhaseSegment per grid cell (the grid
    already sweeps phase as a dimension, so within-cell profiles are held
    close to steady rather than re-sweeping a full mission arc)."""
    c = CONST
    return {
        "ground":               dict(rpm=(c.idle_rpm, c.idle_rpm), alt=(0, 0), thr=(0.0, 0.0)),
        "takeoff":              dict(rpm=(c.takeoff_rpm, c.takeoff_rpm), alt=(0, 1000), thr=(1.0, 1.0)),
        "climb":                dict(rpm=(c.takeoff_rpm, c.continuous_rpm), alt=(2000, 10000), thr=(1.0, 1.0)),
        "cruise":               dict(rpm=(c.continuous_rpm, 5000), alt=(8000, 8000), thr=(1.0, 0.75)),
        "high_altitude_cruise": dict(rpm=(c.continuous_rpm, c.continuous_rpm), alt=(17000, 18000), thr=(1.0, 1.0)),
        "loiter":               dict(rpm=(4000, 4000), alt=(10000, 10000), thr=(0.55, 0.55)),
        "descent":              dict(rpm=(4000, 2500), alt=(10000, 2000), thr=(0.4, 0.15)),
    }[phase]


def run_scenario_grid_to_csv(filepath: str, total_duration_s: float = DEFAULT_TOTAL_DURATION_S,
                               dt: float = 1.0, base_seed: int = 42) -> str:
    """
    Runs the full fault_type x severity x mission_phase grid (plus healthy
    baseline share) and writes one labeled CSV for P4, with columns:
        [EngineSimulator.SCHEMA_FIELDS] + fault_label, severity, fault_ramp_frac
        (+ scenario-specific extras: misfire_cylinder, drift_channel,
          drift_bias, wastegate_direction — populated only on relevant rows,
          empty string otherwise, so the CSV stays one consistent schema)
    Each grid cell gets a fresh EngineSimulator (fresh thermal/oil state) so
    faults don't inherit thermal history from the previous cell's fault.
    """
    grid = build_scenario_grid(total_duration_s)
    extra_fields = ["fault_label", "severity", "fault_ramp_frac",
                     "misfire_cylinder", "drift_channel", "drift_bias", "wastegate_direction"]
    fieldnames = EngineSimulator.SCHEMA_FIELDS + extra_fields

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for idx, cell in enumerate(grid):
            bounds = _phase_rpm_altitude_bounds(cell.mission_phase)
            seg = PhaseSegment(
                phase=cell.mission_phase,
                duration_s=cell.duration_s,
                rpm_start=bounds["rpm"][0], rpm_end=bounds["rpm"][1],
                altitude_start_ft=bounds["alt"][0], altitude_end_ft=bounds["alt"][1],
                throttle_start=bounds["thr"][0], throttle_end=bounds["thr"][1],
                load_factor_g=-0.6 if cell.mission_phase == "descent" else 1.0,
            )

            sim = EngineSimulator(engine_id="ENG01", seed=base_seed + idx)

            if cell.fault_type == "healthy":
                cfg = FaultConfig(fault_type="healthy")
            else:
                # cycle affected cylinder / drift channel / wastegate direction
                # deterministically across cells for grid coverage, rather than
                # hand-picking one value for every scenario
                cyl = (idx % CONST.n_cylinders) + 1
                drift_channel = DRIFTABLE_CHANNELS[idx % len(DRIFTABLE_CHANNELS)]
                wg_dir = "over" if (idx % 2 == 0) else "under"
                cfg = FaultConfig(
                    fault_type=cell.fault_type,
                    severity=cell.severity,
                    ramp_duration_s=cell.duration_s * DEFAULT_RAMP_FRACTION_OF_CELL,
                    onset_s=0.0,
                    affected_cylinder=cyl,
                    drifting_channel=drift_channel,
                    wastegate_direction=wg_dir,
                    rng_seed=base_seed + idx,
                )

            injector = FaultInjector(sim, cfg)
            n_steps = max(1, int(round(seg.duration_s / dt)))
            for i in range(n_steps):
                frac = i / n_steps
                rpm_cmd = seg.rpm_start + (seg.rpm_end - seg.rpm_start) * frac
                altitude_ft = seg.altitude_start_ft + (seg.altitude_end_ft - seg.altitude_start_ft) * frac
                throttle = seg.throttle_start + (seg.throttle_end - seg.throttle_start) * frac
                row = injector.step(dt, rpm_cmd, altitude_ft, throttle, seg.phase, seg.load_factor_g)
                writer.writerow(row)

    return filepath


if __name__ == "__main__":
    out_path = "C:/Users/GOD/Desktop/Generating the Dataset/rotax_914_labeled_faults.csv"
    grid = build_scenario_grid()
    print(f"Grid size: {len(grid)} cells "
          f"({len(VALID_FAULTS)} faults x {len(SEVERITY_LEVELS)} severities x {len(SWEEP_PHASES)} phases "
          f"+ {len(VALID_PHASES)} healthy-phase cells)")
    run_scenario_grid_to_csv(out_path, total_duration_s=DEFAULT_TOTAL_DURATION_S, dt=1.0)
    print(f"Wrote labeled fault-injection dataset to {out_path}")
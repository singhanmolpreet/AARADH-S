"""
Rotax 914 Mean-Value Engine Simulator
======================================

Physics core for a turbocharged, 4-cylinder horizontally-opposed Rotax 914
aero piston engine. Built for P3 (physics model + synthetic data generation).

SOURCE OF TRUTH FOR SPEC VALUES
--------------------------------
All performance ratings, geometry, and operating limits below come directly
from the EASA Type Certificate Data Sheet No. E.122 and the Rotax 914
Operator's Manual, as supplied in the task brief. These are NOT estimated:

    displacement            1211 cm^3
    bore x stroke           79.5 mm x 61.0 mm
    compression ratio       9:1
    gear ratio (crank:prop) 2.4286:1
    max continuous power    73.5 kW @ 5500 RPM, critical altitude 16,000 ft
    max takeoff power       84.5 kW @ 5800 RPM (5 min limit), critical altitude 8,000 ft
    max torque              144 Nm @ 4900 RPM
    idle speed              ~1400 RPM (flagged in brief as provisional/unconfirmed)
    CHT max (conventional)  135 C          <- design generation used throughout this file
    oil temp max            130 C
    oil pressure            2.0-5.0 bar (>3500 RPM) / 0.8 bar min (<3500 RPM) / 7.0 bar max cold start
    boost (airbox) pressure +0.25 bar normal / +0.35 bar max / +0.15 bar min (above ambient)
    EGT max                 950 C
    negative-g limit        5 s at -0.5 g

DESIGN CHOICE MADE PER YOUR INSTRUCTION
----------------------------------------
Two cylinder-head generations were given with different CHT/coolant limits and
different sensing methods. Per your instruction to pick whichever has the most
public documentation, this file uses the CONVENTIONAL coolant design with a
135 C CHT limit (the widely-documented figure across the POH, maintenance
manuals, and the type-certificate data). The newer coolant-only "-01" head
(120 C coolant-temp limit) is NOT modeled here. Do not mix the two limits.

WHAT IS DERIVED FROM PHYSICS VS. ESTIMATED
--------------------------------------------
The torque/power curve is NOT read from an external table. Only 4 points are
certified (idle, 4900 RPM peak torque, 5500 RPM continuous, 5800 RPM takeoff).
Between them, brake power/torque at any RPM is computed live, every timestep,
from an air-standard Otto-cycle energy balance:

    trapped air mass  -> (manifold density, displacement, volumetric efficiency)
    fuel mass         -> trapped air mass / stoichiometric AFR (14.7:1, physical constant)
    heat release       -> fuel mass x LHV of avgas (physical property of the fuel)
    indicated work      -> heat release x Otto-cycle thermal efficiency (1 - r^(1-gamma))
    brake work           -> indicated work - friction/pumping losses (FMEP model)

The ONLY free parameters in that chain are the shape of the volumetric-
efficiency curve vs RPM, the friction (FMEP) coefficients, and the boost level
used during a takeoff-rated pull. These are solved for once, at import time,
via least-squares so that the model reproduces the 4 certified anchor points.
This is "curve-fitting the physics, not the output" - the shape between
anchors is a genuine consequence of the thermodynamic model, not spline
interpolation of a table.

Everything below marked "# ESTIMATE:" is a constant that is NOT published in
any EASA/Rotax document supplied to me. Each is a standard textbook/engineering
value for an engine of this class, chosen so the simulator produces physically
plausible telemetry. If you have real values for any of these, replace them -
they are all exposed as class attributes on RotaxConstants for easy tuning.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Physical constants (universal, not engine-specific - not "assumptions")
# ---------------------------------------------------------------------------
R_AIR = 287.05          # J/(kg*K) specific gas constant for dry air
GAMMA_AIR = 1.40         # ratio of specific heats for cold air (compressor calc)
AFR_STOICH = 14.7        # stoichiometric air:fuel ratio for gasoline
LHV_AVGAS = 44.0e6       # J/kg, lower heating value of aviation gasoline (typical 43-44 MJ/kg)
FUEL_DENSITY = 0.72      # kg/L, avgas density at 15C (typical 0.71-0.72 kg/L)
G0 = 9.80665             # m/s^2

# ---------------------------------------------------------------------------
# ISA standard atmosphere (used for altitude effects on boost/power)
# ---------------------------------------------------------------------------
class ISAAtmosphere:
    """Standard ISA atmosphere, valid within the troposphere (<11 km / 36,089 ft),
    which covers this engine's full certified altitude envelope."""
    T0 = 288.15          # K, sea-level standard temperature
    P0 = 101325.0        # Pa, sea-level standard pressure
    LAPSE = 0.0065        # K/m, standard temperature lapse rate

    @classmethod
    def ft_to_m(cls, ft: float) -> float:
        return ft * 0.3048

    @classmethod
    def temperature_k(cls, altitude_ft: float) -> float:
        h = cls.ft_to_m(altitude_ft)
        return cls.T0 - cls.LAPSE * h

    @classmethod
    def pressure_pa(cls, altitude_ft: float) -> float:
        h = cls.ft_to_m(altitude_ft)
        T = cls.temperature_k(altitude_ft)
        return cls.P0 * (T / cls.T0) ** (G0 / (cls.LAPSE * R_AIR))

    @classmethod
    def density(cls, altitude_ft: float) -> float:
        return cls.pressure_pa(altitude_ft) / (R_AIR * cls.temperature_k(altitude_ft))


# ---------------------------------------------------------------------------
# Engine constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RotaxConstants:
    # --- Confirmed geometry / rating data (EASA TCDS E.122) ---
    displacement_m3: float = 1211e-6
    bore_m: float = 0.0795
    stroke_m: float = 0.0610
    compression_ratio: float = 9.0
    n_cylinders: int = 4
    gear_ratio: float = 2.4286  # crank:prop

    idle_rpm: float = 1400.0            # confirmed as "provisional" in brief
    peak_torque_rpm: float = 4900.0
    peak_torque_nm: float = 144.0
    continuous_rpm: float = 5500.0
    continuous_power_w: float = 73_500.0
    continuous_crit_alt_ft: float = 16_000.0
    takeoff_rpm: float = 5800.0
    takeoff_power_w: float = 84_500.0
    takeoff_crit_alt_ft: float = 8_000.0
    takeoff_time_limit_s: float = 5 * 60.0

    # --- Confirmed operating limits ---
    cht_max_c: float = 135.0                  # conventional-design limit (chosen generation)
    coolant_exit_max_c: float = 120.0         # conventional design coolant *exit* limit
    oil_temp_max_c: float = 130.0
    oil_pressure_norm_lo_bar: float = 2.0
    oil_pressure_norm_hi_bar: float = 5.0
    oil_pressure_min_below_3500_bar: float = 0.8
    oil_pressure_max_cold_start_bar: float = 7.0
    oil_pressure_rpm_threshold: float = 3500.0
    boost_norm_bar: float = 0.25
    boost_max_bar: float = 0.35
    boost_min_bar: float = 0.15
    egt_max_c: float = 950.0
    airbox_temp_max_c: float = 88.0
    air_temp_max_c: float = 72.0
    neg_g_limit: float = -0.5
    neg_g_max_duration_s: float = 5.0

    # --- ESTIMATE: Otto-cycle / thermodynamic parameters not in any cert doc ---
    gamma_combustion: float = 1.30       # ESTIMATE: polytropic exponent of hot combustion gas (vs 1.40 for cold air)
    combustion_efficiency: float = 0.98  # ESTIMATE: fraction of fuel chemical energy actually released
    compressor_isentropic_eff: float = 0.75  # ESTIMATE: turbo compressor isentropic efficiency (no intercooler on 914)
    intake_manifold_dT_floor_k: float = 3.0  # ESTIMATE: minor heating from intake plenum/runners even at zero boost

    # --- ESTIMATE: thermal model lumped parameters ---
    cht_thermal_tau_s: float = 140.0      # ESTIMATE: CHT first-order lag time constant (cylinder head thermal mass)
    egt_thermal_tau_s: float = 8.0        # ESTIMATE: EGT responds much faster (exhaust gas, low thermal mass)
    oil_thermal_tau_s: float = 220.0      # ESTIMATE: oil sump thermal lag (largest thermal mass in engine)
    coolant_head_fraction: float = 0.28   # ESTIMATE: fraction of "waste" (non-indicated-work) heat that loads the head/coolant path
    head_thermal_ua_w_per_c: float = 290.0  # ESTIMATE: lumped head-to-ambient thermal conductance
    # (tuned so peak-load climb CHT sits at a plausible ~120C, safely under the
    # 135C limit, leaving headroom for later overheating fault-injection work)
    egt_ambient_floor_c_offset: float = 25.0  # ESTIMATE: EGT tracks ambient + combustion rise

    # --- ESTIMATE: oil pressure model ---
    oil_cold_start_decay_tau_s: float = 25.0  # ESTIMATE: time for cold-start pressure spike to relax

    # --- ESTIMATE: vibration model ---
    vibration_base_g: float = 0.03        # ESTIMATE: baseline mechanical noise floor
    vibration_harmonic_gains: tuple = (0.55, 0.30, 0.15)  # ESTIMATE: relative amplitude of 1x/2x/4x crank-order harmonics
    vibration_load_gain: float = 0.35     # ESTIMATE: additional vibration amplitude scaling with load fraction
    vibration_noise_std_g: float = 0.02   # ESTIMATE: random broadband noise sigma

    # --- ESTIMATE: electrical system (12V negative-ground, Rotax 914 generator) ---
    battery_nominal_v: float = 12.6       # ESTIMATE: resting battery voltage
    charging_voltage_v: float = 14.2      # ESTIMATE: regulated bus voltage once alternator is online (RPM > ~1800)
    alternator_online_rpm: float = 1800.0 # ESTIMATE: RPM above which the generator carries the bus
    alternator_max_a: float = 20.0        # ESTIMATE: Rotax 914 generator nominal max output
    base_electrical_load_a: float = 8.0   # ESTIMATE: avionics/fuel pump/ignition baseline load

    # --- ESTIMATE: injection/ignition timing map ---
    timing_idle_deg: float = 8.0          # ESTIMATE: BTDC at idle
    timing_max_advance_deg: float = 32.0  # ESTIMATE: max BTDC advance at high RPM / light load
    timing_knock_retard_deg: float = 10.0 # ESTIMATE: retard applied under high boost/load to avoid knock

    fuel_flow_idle_floor_lph: float = 4.0  # ESTIMATE: minimum measurable idle fuel flow


CONST = RotaxConstants()


# ---------------------------------------------------------------------------
# Otto-cycle power model calibration
# ---------------------------------------------------------------------------
class OttoCycleModel:
    """
    Computes brake power/torque at an arbitrary RPM and manifold condition
    using an air-standard Otto-cycle energy balance. The volumetric-efficiency
    curve shape and friction (FMEP) coefficients are the only free parameters,
    and are solved once (at construction) via least-squares against the 4
    certified anchor points. Everything else in the energy balance
    (air mass -> fuel mass -> heat release -> indicated work -> brake work)
    is genuine thermodynamics, not a lookup table.
    """

    def __init__(self, const: RotaxConstants = CONST):
        self.c = const
        self.eta_otto = 1.0 - const.compression_ratio ** (1.0 - const.gamma_combustion)
        self._calibrate()

    # -- volumetric efficiency shape (free-parameter Gaussian bump) --
    def _eta_vol(self, rpm: np.ndarray, eta_max: float, n_peak: float, sigma: float, eta_min: float = 0.55) -> np.ndarray:
        rpm = np.asarray(rpm, dtype=float)
        return eta_min + (eta_max - eta_min) * np.exp(-((rpm - n_peak) / sigma) ** 2)

    # ESTIMATE: throttle-plate restriction at partial throttle. At WOT (throttle=1)
    # the naturally-aspirated manifold pressure equals full ambient; at closed
    # throttle (idle) it falls to a fraction of ambient (throttling/pumping loss).
    THROTTLE_MAP_MIN_FRAC = 0.30

    def _manifold_pressure_pa(self, altitude_ft: float, boost_bar: float, throttle: float) -> float:
        P_amb = ISAAtmosphere.pressure_pa(altitude_ft)
        throttle = max(0.0, min(1.0, throttle))
        na_frac = self.THROTTLE_MAP_MIN_FRAC + (1.0 - self.THROTTLE_MAP_MIN_FRAC) * throttle
        return P_amb * na_frac + boost_bar * 1e5

    def _manifold_temp_k(self, altitude_ft: float, boost_bar: float, throttle: float) -> float:
        """Isentropic compression + compressor inefficiency heating, no intercooler."""
        T_amb = ISAAtmosphere.temperature_k(altitude_ft)
        P_amb = ISAAtmosphere.pressure_pa(altitude_ft)
        P_man = self._manifold_pressure_pa(altitude_ft, boost_bar, throttle)
        if boost_bar <= 1e-6:
            return T_amb + self.c.intake_manifold_dT_floor_k
        pressure_ratio = max(P_man / P_amb, 1.0)
        T_ideal = T_amb * pressure_ratio ** ((GAMMA_AIR - 1.0) / GAMMA_AIR)
        T_actual = T_amb + (T_ideal - T_amb) / self.c.compressor_isentropic_eff
        return T_actual + self.c.intake_manifold_dT_floor_k

    def brake_power_torque(self, rpm: float, altitude_ft: float, boost_bar: float, throttle: float,
                            eta_max: float, n_peak: float, sigma: float,
                            fmep_a: float, fmep_b: float) -> tuple:
        c = self.c
        if rpm <= 1.0:
            return 0.0, 0.0, 0.0

        P_man = self._manifold_pressure_pa(altitude_ft, boost_bar, throttle)
        T_man = self._manifold_temp_k(altitude_ft, boost_bar, throttle)
        rho_man = P_man / (R_AIR * T_man)

        eta_vol = float(self._eta_vol(np.array([rpm]), eta_max, n_peak, sigma)[0])
        eta_vol = min(max(eta_vol, 0.05), 1.10)

        m_air_per_cycle = rho_man * c.displacement_m3 * eta_vol  # kg, per full-displacement event (all cylinders)
        m_fuel_per_cycle = m_air_per_cycle / AFR_STOICH
        q_in = m_fuel_per_cycle * LHV_AVGAS * c.combustion_efficiency  # J per cycle

        w_indicated_per_cycle = q_in * self.eta_otto  # J

        cycles_per_sec = (rpm / 2.0) / 60.0  # 4-stroke: 1 cycle per 2 crank revs
        indicated_power_w = w_indicated_per_cycle * cycles_per_sec

        fmep_pa = fmep_a + fmep_b * rpm  # friction+pumping mean effective pressure
        friction_power_w = fmep_pa * c.displacement_m3 * cycles_per_sec

        brake_power_w = max(indicated_power_w - friction_power_w, 0.0)
        omega = 2 * math.pi * rpm / 60.0
        brake_torque_nm = brake_power_w / omega if omega > 0 else 0.0
        fuel_kg_s = m_fuel_per_cycle * cycles_per_sec
        return brake_power_w, brake_torque_nm, fuel_kg_s

    def _residuals(self, x):
        eta_max, n_peak, sigma, fmep_a, fmep_b, takeoff_boost = x
        c = self.c
        residuals = []

        # Anchor 1: peak torque @ 4900 RPM, WOT, continuous-rating boost (+0.25 bar), sea level
        p, t, _ = self._eval(4900.0, 0.0, c.boost_norm_bar, 1.0, eta_max, n_peak, sigma, fmep_a, fmep_b)
        residuals.append((t - c.peak_torque_nm) / c.peak_torque_nm)

        # Anchor 2: continuous power @ 5500 RPM, WOT, +0.25 bar, sea level
        p, t, _ = self._eval(5500.0, 0.0, c.boost_norm_bar, 1.0, eta_max, n_peak, sigma, fmep_a, fmep_b)
        residuals.append((p - c.continuous_power_w) / c.continuous_power_w)

        # Anchor 3: takeoff power @ 5800 RPM, WOT, boost = free parameter (bounded to certified band), sea level
        p, t, _ = self._eval(5800.0, 0.0, takeoff_boost, 1.0, eta_max, n_peak, sigma, fmep_a, fmep_b)
        residuals.append((p - c.takeoff_power_w) / c.takeoff_power_w)

        # Anchor 4 (soft/unconfirmed): idle runs near-closed throttle (no certified
        # idle torque exists in any document supplied, so this is a loosely-weighted
        # plausibility target only, representing a small positive net torque to
        # sustain idle against friction/accessory loads).
        p, t, _ = self._eval(1400.0, 0.0, 0.0, 0.05, eta_max, n_peak, sigma, fmep_a, fmep_b)
        idle_power_soft_target_w = 2500.0  # ESTIMATE: rough friction+accessory overcome power at idle
        residuals.append(0.10 * (p - idle_power_soft_target_w) / idle_power_soft_target_w)

        return residuals

    def _eval(self, rpm, altitude_ft, boost_bar, throttle, eta_max, n_peak, sigma, fmep_a, fmep_b):
        return self.brake_power_torque(rpm, altitude_ft, boost_bar, throttle, eta_max, n_peak, sigma, fmep_a, fmep_b)

    def _calibrate(self):
        x0 = [0.95, 5000.0, 1400.0, 60_000.0, 8.0, 0.30]
        lo = [0.85, 4000.0, 800.0, 20_000.0, 2.0, 0.25]
        hi = [1.05, 5500.0, 2500.0, 100_000.0, 20.0, 0.35]
        result = least_squares(self._residuals, x0, bounds=(lo, hi))
        (self.eta_max, self.n_peak, self.sigma,
         self.fmep_a, self.fmep_b, self.takeoff_boost_bar) = result.x
        self.calibration_result = result

    def evaluate(self, rpm: float, altitude_ft: float, boost_bar: float, throttle: float = 1.0) -> tuple:
        """Public entry point: returns (brake_power_w, brake_torque_nm, fuel_flow_kg_s)."""
        return self.brake_power_torque(
            rpm, altitude_ft, boost_bar, throttle,
            self.eta_max, self.n_peak, self.sigma, self.fmep_a, self.fmep_b,
        )


# Calibrated once at import time and shared by all EngineSimulator instances.
OTTO_MODEL = OttoCycleModel(CONST)


# ---------------------------------------------------------------------------
# Mission phase definition
# ---------------------------------------------------------------------------
@dataclass
class PhaseSegment:
    phase: str
    duration_s: float
    rpm_start: float
    rpm_end: float
    altitude_start_ft: float
    altitude_end_ft: float
    throttle_start: float = 1.0   # 0..1, fraction of available boost commanded
    throttle_end: float = 1.0
    load_factor_g: float = 1.0    # for negative-g modeling (descent/turbulence segments)


VALID_PHASES = {"ground", "takeoff", "climb", "cruise", "high_altitude_cruise",
                 "loiter", "descent"}


# ---------------------------------------------------------------------------
# EngineSimulator
# ---------------------------------------------------------------------------
class EngineSimulator:
    """
    Stateful mean-value simulator. Call `step(dt, rpm_cmd, altitude_ft,
    throttle, mission_phase, load_factor_g)` once per timestep, or use
    `run_mission(...)` / `run_and_save_csv(...)` to drive a full profile.
    """

    SCHEMA_FIELDS = [
        "timestamp", "engine_id", "mission_phase", "rpm", "cht_c", "egt_c",
        "oil_pressure_bar", "oil_temp_c", "fuel_flow_lph", "vibration_rms_g",
        "battery_voltage_v", "throttle_frac", "injection_timing_deg",
        "map_kpa", "boost_pressure_bar",
    ]

    def __init__(self, engine_id: str = "ENG01", start_time: Optional[datetime] = None,
                 seed: Optional[int] = None, const: RotaxConstants = CONST):
        self.engine_id = engine_id
        self.c = const
        self.otto = OTTO_MODEL
        self.t0 = start_time or datetime.now(timezone.utc)
        self.elapsed_s = 0.0
        self._rng = random.Random(seed)

        # thermal state
        self.cht_c = 25.0
        self.egt_c = 25.0
        self.oil_temp_c = 25.0

        # oil pressure transient state
        self._time_since_start_s = 0.0
        self._cold_start = True

        # takeoff-rating timer (5-minute limit)
        self._takeoff_timer_s = 0.0

        # negative-g tracking
        self._neg_g_timer_s = 0.0

        # phase for angular-order harmonics
        self._vib_phase_accum = 0.0

    # -----------------------------------------------------------------
    # TCU / boost model
    # -----------------------------------------------------------------
    def _active_rating(self, mission_phase: str, rpm: float) -> str:
        """Determine which certified rating (and therefore which critical
        altitude) governs the current boost target."""
        if mission_phase == "takeoff" or rpm >= (self.c.takeoff_rpm - 50):
            return "takeoff"
        return "continuous"

    def _critical_altitude_ft(self, rating: str) -> float:
        return self.c.takeoff_crit_alt_ft if rating == "takeoff" else self.c.continuous_crit_alt_ft

    def _boost_pressure_bar(self, rpm: float, altitude_ft: float, throttle: float,
                             mission_phase: str) -> tuple:
        """
        TCU behavior:
          - Below the relevant critical altitude, the wastegate closes as needed
            to hold the target boost (scaled by throttle & rating) essentially
            constant with altitude - this is the whole point of a turbo-normalized
            engine.
          - Above the relevant critical altitude, the wastegate is already fully
            closed at the critical altitude, so the compressor cannot make up any
            more pressure ratio. Absolute manifold pressure then falls at
            (approximately) the same fractional rate as ambient pressure falls
            above that altitude.
        Returns (boost_bar, rating, critical_altitude_ft).
        """
        c = self.c
        rating = self._active_rating(mission_phase, rpm)
        crit_alt = self._critical_altitude_ft(rating)

        target = self.otto.takeoff_boost_bar if rating == "takeoff" else c.boost_norm_bar
        target = max(c.boost_min_bar, min(c.boost_max_bar, target))
        throttle = max(0.0, min(1.0, throttle))

        if mission_phase == "ground":
            # Idle/taxi: turbo not spooled, negligible boost regardless of throttle model
            return 0.0, rating, crit_alt

        commanded = target * throttle

        if altitude_ft <= crit_alt:
            boost = commanded
        else:
            # Wastegate fully closed above critical altitude: manifold pressure
            # falls with ambient pressure at the same ratio it had at crit_alt.
            p_amb_crit = ISAAtmosphere.pressure_pa(crit_alt)
            p_amb_here = ISAAtmosphere.pressure_pa(altitude_ft)
            p_man_at_crit = p_amb_crit + commanded * 1e5
            p_man_here = p_man_at_crit * (p_amb_here / p_amb_crit)
            boost = max((p_man_here - p_amb_here) / 1e5, 0.0)

        boost = max(0.0, min(c.boost_max_bar, boost))
        return boost, rating, crit_alt

    # -----------------------------------------------------------------
    # Thermal sub-models
    # -----------------------------------------------------------------
    def _cooling_factor(self, mission_phase: str) -> float:
        # ESTIMATE: ram-air cooling effectiveness by phase (no airspeed input available)
        return {
            "ground": 0.45,
            "takeoff": 0.85,
            "climb": 0.95,
            "cruise": 1.05,
            "high_altitude_cruise": 1.20,   # thinner but faster/higher-speed cruise -> better ram effect assumed
            "loiter": 0.75,
            "descent": 1.25,                # high airspeed, low power -> best cooling
        }.get(mission_phase, 1.0)

    def _update_thermals(self, dt: float, brake_power_w: float, waste_heat_w: float,
                          altitude_ft: float, mission_phase: str, load_fraction: float):
        c = self.c
        T_amb_c = ISAAtmosphere.temperature_k(altitude_ft) - 273.15
        cooling = self._cooling_factor(mission_phase)

        # --- CHT: first-order lag toward a load/cooling-dependent steady state ---
        head_heat_w = waste_heat_w * c.coolant_head_fraction
        # ESTIMATE: steady-state CHT rise above ambient scales with head heat load / cooling effectiveness
        cht_ss = T_amb_c + (head_heat_w / c.head_thermal_ua_w_per_c) / max(cooling, 0.3)
        cht_ss = min(cht_ss, c.cht_max_c + 25.0)  # allow modest overshoot headroom for later fault-injection work
        self.cht_c += (cht_ss - self.cht_c) * (1 - math.exp(-dt / c.cht_thermal_tau_s))

        # --- EGT: fast quasi-steady response. EGT is driven mainly by combustion
        # gas temperature (present even at idle), rising with load, then rolling
        # off slightly at very high power as mixture is enriched for cooling.
        # ESTIMATE: base ~300C reflects that piston-engine EGT never approaches
        # ambient even at idle; shape calibrated so cruise-load EGT lands near
        # the ~700C figure shown in the reference telemetry sample.
        shape = load_fraction * (1.0 - 0.15 * max(load_fraction - 0.85, 0.0) / 0.15)
        egt_ss = T_amb_c + 300.0 + 550.0 * shape
        egt_ss = max(egt_ss, T_amb_c + 150.0)
        self.egt_c += (egt_ss - self.egt_c) * (1 - math.exp(-dt / c.egt_thermal_tau_s))

        # --- Oil temp: slow lag, tracks somewhat below CHT ---
        oil_ss = 0.75 * self.cht_c + 0.25 * T_amb_c + 8.0  # ESTIMATE: oil runs a bit cooler than head
        self.oil_temp_c += (oil_ss - self.oil_temp_c) * (1 - math.exp(-dt / c.oil_thermal_tau_s))

    # -----------------------------------------------------------------
    # Oil pressure sub-model
    # -----------------------------------------------------------------
    def _oil_pressure_bar(self, rpm: float, dt: float, neg_g_exceeded: bool) -> float:
        c = self.c
        self._time_since_start_s += dt

        if rpm < c.oil_pressure_rpm_threshold:
            # ramps from min toward the low end of the normal band as RPM rises to threshold
            frac = max(0.0, min(1.0, rpm / c.oil_pressure_rpm_threshold))
            base = c.oil_pressure_min_below_3500_bar + frac * (c.oil_pressure_norm_lo_bar - c.oil_pressure_min_below_3500_bar)
        else:
            frac = max(0.0, min(1.0, (rpm - c.oil_pressure_rpm_threshold) /
                                 (c.takeoff_rpm - c.oil_pressure_rpm_threshold)))
            base = c.oil_pressure_norm_lo_bar + frac * (c.oil_pressure_norm_hi_bar - c.oil_pressure_norm_lo_bar)

        # cold-start transient: pressure spikes toward the 7.0 bar max and decays as oil warms
        if self._cold_start:
            spike = (c.oil_pressure_max_cold_start_bar - base) * math.exp(
                -self._time_since_start_s / c.oil_cold_start_decay_tau_s)
            base = base + max(spike, 0.0)
            if self.oil_temp_c > 60.0 or self._time_since_start_s > 180.0:
                self._cold_start = False

        # sustained negative-g beyond the 5s limit briefly starves the (dry-sump) pickup
        if neg_g_exceeded:
            base *= 0.55  # ESTIMATE: transient pressure dip if the -0.5g/5s limit is exceeded

        return max(base, 0.0)

    # -----------------------------------------------------------------
    # Vibration sub-model: RPM-harmonic sine content + noise, reduced to RMS
    # -----------------------------------------------------------------
    def _vibration_rms_g(self, rpm: float, load_fraction: float, dt: float) -> float:
        c = self.c
        crank_freq_hz = rpm / 60.0
        self._vib_phase_accum += 2 * math.pi * crank_freq_hz * dt

        amps = []
        for order, gain in zip((1, 2, 4), c.vibration_harmonic_gains):
            amp = gain * (0.4 + 0.6 * load_fraction) * (rpm / c.takeoff_rpm)
            amps.append(amp)

        # RMS of a sum of independent sinusoids = sqrt(sum(A_i^2 / 2))
        harmonic_rms = math.sqrt(sum(a ** 2 for a in amps) / 2.0)
        noise = self._rng.gauss(0.0, c.vibration_noise_std_g)
        rms = c.vibration_base_g + harmonic_rms + abs(noise)
        return max(rms, 0.0)

    # -----------------------------------------------------------------
    # Electrical sub-model
    # -----------------------------------------------------------------
    def _electrical(self, rpm: float, mission_phase: str) -> tuple:
        c = self.c
        if rpm < c.alternator_online_rpm:
            voltage = c.battery_nominal_v
            current = 0.0
        else:
            voltage = c.charging_voltage_v
            phase_extra_load = {"takeoff": 4.0, "climb": 2.0, "descent": 1.0}.get(mission_phase, 0.0)  # ESTIMATE
            current = min(c.base_electrical_load_a + phase_extra_load, c.alternator_max_a)
        return voltage, current

    # -----------------------------------------------------------------
    # Injection timing sub-model
    # -----------------------------------------------------------------
    def _injection_timing_deg(self, rpm: float, boost_bar: float) -> float:
        c = self.c
        rpm_frac = max(0.0, min(1.0, (rpm - c.idle_rpm) / (c.takeoff_rpm - c.idle_rpm)))
        advance = c.timing_idle_deg + rpm_frac * (c.timing_max_advance_deg - c.timing_idle_deg)
        boost_frac = max(0.0, min(1.0, boost_bar / c.boost_max_bar))
        advance -= boost_frac * c.timing_knock_retard_deg
        return max(advance, 0.0)

    # -----------------------------------------------------------------
    # Main per-timestep update
    # -----------------------------------------------------------------
    def step(self, dt: float, rpm_cmd: float, altitude_ft: float, throttle: float,
              mission_phase: str, load_factor_g: float = 1.0) -> dict:
        if mission_phase not in VALID_PHASES:
            raise ValueError(f"mission_phase must be one of {sorted(VALID_PHASES)}, got {mission_phase!r}")

        self.elapsed_s += dt
        rpm = max(0.0, rpm_cmd)

        # negative-g tracking (dry sump limits enforcement)
        if load_factor_g <= self.c.neg_g_limit:
            self._neg_g_timer_s += dt
        else:
            self._neg_g_timer_s = 0.0
        neg_g_exceeded = self._neg_g_timer_s > self.c.neg_g_max_duration_s

        # takeoff-rating 5-minute timer
        rating = self._active_rating(mission_phase, rpm)
        if rating == "takeoff":
            self._takeoff_timer_s += dt
        else:
            self._takeoff_timer_s = 0.0

        boost_bar, rating, crit_alt = self._boost_pressure_bar(rpm, altitude_ft, throttle, mission_phase)

        throttle_eff = max(throttle, 0.05)  # engine is never fully closed; idle floor matches calibration
        brake_power_w, brake_torque_nm, fuel_kg_s = self.otto.evaluate(rpm, altitude_ft, boost_bar, throttle_eff)

        rated_power_w = self.c.takeoff_power_w if rating == "takeoff" else self.c.continuous_power_w
        load_fraction = max(0.0, min(1.2, brake_power_w / rated_power_w)) if rated_power_w else 0.0

        # waste heat = fuel chemical energy not converted to brake work
        fuel_energy_rate_w = fuel_kg_s * LHV_AVGAS
        waste_heat_w = max(fuel_energy_rate_w - brake_power_w, 0.0)

        self._update_thermals(dt, brake_power_w, waste_heat_w, altitude_ft, mission_phase, load_fraction)

        oil_pressure_bar = self._oil_pressure_bar(rpm, dt, neg_g_exceeded)

        vibration_rms_g = self._vibration_rms_g(rpm, load_fraction, dt)
        battery_v, alternator_a = self._electrical(rpm, mission_phase)
        injection_timing_deg = self._injection_timing_deg(rpm, boost_bar)

        map_kpa = self.otto._manifold_pressure_pa(altitude_ft, boost_bar, throttle_eff) / 1000.0

        fuel_flow_lph = max((fuel_kg_s / FUEL_DENSITY) * 3600.0, self.c.fuel_flow_idle_floor_lph if rpm > 300 else 0.0)

        timestamp = (self.t0 + timedelta(seconds=self.elapsed_s)).isoformat()

        return {
            "timestamp": timestamp,
            "engine_id": self.engine_id,
            "mission_phase": mission_phase,
            "rpm": round(rpm),
            "cht_c": round(self.cht_c, 1),
            "egt_c": round(self.egt_c, 1),
            "oil_pressure_bar": round(oil_pressure_bar, 2),
            "oil_temp_c": round(self.oil_temp_c, 1),
            "fuel_flow_lph": round(fuel_flow_lph, 2),
            "vibration_rms_g": round(vibration_rms_g, 3),
            "battery_voltage_v": round(battery_v, 2),
            "throttle_frac": round(alternator_a, 2),
            "injection_timing_deg": round(injection_timing_deg, 1),
            "map_kpa": round(map_kpa, 1),
            "boost_pressure_bar": round(boost_bar, 3),
        }

    # -----------------------------------------------------------------
    # Mission running
    # -----------------------------------------------------------------
    def run_mission(self, segments: list, dt: float = 1.0) -> list:
        rows = []
        for seg in segments:
            n_steps = max(1, int(round(seg.duration_s / dt)))
            for i in range(n_steps):
                frac = i / n_steps
                rpm_cmd = seg.rpm_start + (seg.rpm_end - seg.rpm_start) * frac
                altitude_ft = seg.altitude_start_ft + (seg.altitude_end_ft - seg.altitude_start_ft) * frac
                throttle = seg.throttle_start + (seg.throttle_end - seg.throttle_start) * frac
                row = self.step(dt, rpm_cmd, altitude_ft, throttle, seg.phase, seg.load_factor_g)
                rows.append(row)
        return rows

    def run_and_save_csv(self, segments: list, filepath: str, dt: float = 1.0) -> str:
        rows = self.run_mission(segments, dt=dt)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.SCHEMA_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return filepath


# ---------------------------------------------------------------------------
# Standard mission profile: exercises all 7 phases, including a high-altitude
# cruise that climbs through BOTH critical altitudes (8,000 ft and 16,000 ft).
# ---------------------------------------------------------------------------
def standard_mission_profile() -> list:
    c = CONST
    return [
        PhaseSegment("ground", 120, c.idle_rpm, c.idle_rpm, 0, 0, 0.0, 0.0),
        PhaseSegment("takeoff", 60, c.idle_rpm, c.takeoff_rpm, 0, 500, 0.2, 1.0),
        PhaseSegment("takeoff", 240, c.takeoff_rpm, c.takeoff_rpm, 500, 3000, 1.0, 1.0),
        PhaseSegment("climb", 900, c.takeoff_rpm, c.continuous_rpm, 3000, 12000, 1.0, 1.0),
        PhaseSegment("cruise", 1800, c.continuous_rpm, 5000, 12000, 12000, 1.0, 0.75),
        # High-altitude cruise: climb further so we cross both critical altitudes
        # (8,000 ft takeoff-rating limit already passed; now cross 16,000 ft
        # continuous-rating limit too) and hold near the top to show power falloff.
        PhaseSegment("high_altitude_cruise", 600, 5000, c.continuous_rpm, 12000, 18000, 0.75, 1.0),
        PhaseSegment("high_altitude_cruise", 900, c.continuous_rpm, c.continuous_rpm, 18000, 18000, 1.0, 1.0),
        PhaseSegment("loiter", 900, 4000, 4000, 10000, 10000, 0.55, 0.55),
        PhaseSegment("descent", 600, 4000, 2500, 10000, 2000, 0.4, 0.15, load_factor_g=-0.6),
        PhaseSegment("ground", 120, c.idle_rpm, c.idle_rpm, 0, 0, 0.0, 0.0),
    ]


if __name__ == "__main__":
    sim = EngineSimulator(engine_id="ENG01", seed=42)
    profile = standard_mission_profile()
    out_path = "C:/Users/GOD/Desktop/Generating the Dataset/rotax_914_telemetry.csv"
    sim.run_and_save_csv(profile, out_path, dt=1.0)
    print(f"Wrote telemetry to {out_path}")
    print("Otto-cycle calibration:", {
        "eta_max": OTTO_MODEL.eta_max, "n_peak": OTTO_MODEL.n_peak,
        "sigma": OTTO_MODEL.sigma, "fmep_a": OTTO_MODEL.fmep_a,
        "fmep_b": OTTO_MODEL.fmep_b, "takeoff_boost_bar": OTTO_MODEL.takeoff_boost_bar,
        "eta_otto": OTTO_MODEL.eta_otto,
    })
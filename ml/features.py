"""
ml/features.py
==============
Turns cleaned Rotax 914 telemetry rows into a model-ready feature matrix.

Pure function, pandas/numpy only, no I/O, no side effects.

Design decisions
----------------
- Rolling window = 30 samples (30 s at the confirmed 1 Hz sample rate).
- Rate-of-change uses diff(1) — one-step finite difference — which keeps
  the RoC feature at the same temporal resolution as mean/variance without
  an extra diff_lag parameter to tune.  A diff over the full window would
  alias with the rolling-mean gradient; diff(1) is independent.
- Windows are computed strictly WITHIN each run_id (rows are first sorted
  by run_id then time_s).  This guarantees that no rolling stat ever spans
  two different simulated runs or two different fault episodes.
- The first (window-1) rows of every run where min_periods is not yet met
  produce NaN rolling stats; those rows are DROPPED, not backfilled.
- mission_phase is one-hot encoded against a FIXED vocabulary (4 values
  from the actual dataset).  Even if a batch is missing one phase, the
  dummy column is still emitted (all zeros), so training and inference
  always produce identical column layouts.
- Columns that must NEVER appear as model inputs (labels, ids, timing):
      fault_type_code, fault_type, fault_severity, run_id, time_s
  An AssertionError is raised if any of them leak into the feature matrix.

What this module does NOT handle
---------------------------------
- Normalisation / standardisation  -- caller's responsibility.
- Train/val/test splitting          -- caller's responsibility.
- Handling NaN in raw sensor cols   -- NaN propagates into rolling stats;
  caller should impute or filter upstream if needed.
- Fault-severity-weighted sampling  -- not done here.
- Any feature beyond the 5 requested rolling sensors + mission_phase OHE.
  Raw pass-through columns (battery_voltage_v, fuel_flow_lph, etc.) ARE
  included in the returned DataFrame but are NOT listed in FEATURE_COLS;
  use FEATURE_COLS to select the model input matrix.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sensors to roll
ROLL_SENSORS: list[str] = [
    "cht_c",
    "egt_c",
    "oil_pressure_bar",
    "rpm",
    "vibration_rms_g",
]

# Fixed mission-phase vocabulary — order determines dummy column order.
MISSION_PHASES: list[str] = [
    "endurance_low_throttle",
    "steady_cruise",
    "high_power_climb",
    "rapid_throttle_transitions",
]

# Default rolling window (samples)
DEFAULT_WINDOW: int = 30

# Columns that are LABELS / IDs and must never appear in the feature matrix.
_LABEL_COLS: frozenset[str] = frozenset(
    ["fault_type_code", "fault_type", "fault_severity", "run_id", "time_s"]
)

# ---------------------------------------------------------------------------
# Public API: explicit list of feature columns produced by compute_features()
# ---------------------------------------------------------------------------

def feature_column_names(
    roll_sensors: list[str] = ROLL_SENSORS,
    mission_phases: list[str] = MISSION_PHASES,
) -> list[str]:
    """Return the exact ordered list of feature columns that compute_features()
    adds to the output DataFrame.  Downstream code should use this to select
    the model input matrix X rather than guessing column names.

    Generated columns:
      - <sensor>_roll_mean  : rolling mean over window
      - <sensor>_roll_var   : rolling variance over window (ddof=1)
      - <sensor>_roc        : rate-of-change (diff(1), i.e. sample-to-sample)
      - mission_phase_<tag> : one-hot dummy for each phase in vocabulary
    """
    cols: list[str] = []
    for s in roll_sensors:
        cols += [f"{s}_roll_mean", f"{s}_roll_var", f"{s}_roc"]
    cols += [f"mission_phase_{p}" for p in mission_phases]
    return cols


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def compute_features(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    roll_sensors: list[str] = ROLL_SENSORS,
    mission_phases: list[str] = MISSION_PHASES,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Compute rolling-window features from cleaned Rotax 914 telemetry.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``run_id``, ``time_s``, ``mission_phase``, and all
        columns listed in ``roll_sensors``.  Label columns
        (``fault_type_code``, ``fault_type``, ``fault_severity``) may be
        present — they are passed through in the output but are never used
        as inputs, and an AssertionError is raised if any of them appear
        in the computed feature set.
    window : int
        Rolling window size in samples.  Rows where fewer than ``window``
        prior rows exist within the same ``run_id`` produce NaN stats and
        are dropped when ``drop_incomplete=True``.
    roll_sensors : list[str]
        Sensor columns to roll.  Defaults to the 5 specified sensors.
    mission_phases : list[str]
        Fixed vocabulary for one-hot encoding of ``mission_phase``.
    drop_incomplete : bool
        Drop rows where any rolling stat is NaN (i.e. the first
        ``window - 1`` rows per run).  Default True.

    Returns
    -------
    pd.DataFrame
        All original columns (minus nothing — labels pass through) PLUS
        rolling and OHE columns.  Index is reset.  Use
        ``feature_column_names()`` to select the model input sub-matrix.

    Raises
    ------
    ValueError
        If required columns are missing from ``df``.
    AssertionError
        If any label / id column leaks into the computed feature columns.
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    required = list(roll_sensors) + ["run_id", "time_s", "mission_phase"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"compute_features(): missing required columns: {missing}"
        )

    # ------------------------------------------------------------------
    # 2. Sort within runs so rolling is temporally correct
    # ------------------------------------------------------------------
    out = (
        df.sort_values(["run_id", "time_s"])
        .reset_index(drop=True)
        .copy()
    )

    # ------------------------------------------------------------------
    # 3. Rolling features — computed per run_id so windows never bleed
    #    across run boundaries.
    # ------------------------------------------------------------------
    grouped = out.groupby("run_id", sort=False)

    for sensor in roll_sensors:
        series = grouped[sensor]

        out[f"{sensor}_roll_mean"] = series.transform(
            lambda s, w=window: s.rolling(w, min_periods=w).mean()
        )
        out[f"{sensor}_roll_var"] = series.transform(
            lambda s, w=window: s.rolling(w, min_periods=w).var(ddof=1)
        )
        # Rate-of-change: diff(1) within each run
        out[f"{sensor}_roc"] = series.transform(
            lambda s: s.diff(1)
        )

    # ------------------------------------------------------------------
    # 4. Drop incomplete-window rows (NaN rolling stats)
    # ------------------------------------------------------------------
    if drop_incomplete:
        roll_cols = [c for c in out.columns if c.endswith(("_roll_mean", "_roll_var"))]
        complete_mask = ~out[roll_cols].isna().any(axis=1)
        n_dropped = (~complete_mask).sum()
        if n_dropped:
            # Informational — not suppressed, visible in scripts that call this.
            print(
                f"  [compute_features] dropped {n_dropped:,} incomplete-window rows "
                f"({n_dropped / len(out) * 100:.2f}% of input)"
            )
        out = out.loc[complete_mask].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. One-hot encode mission_phase against fixed vocabulary
    # ------------------------------------------------------------------
    phase_cat = pd.Categorical(out["mission_phase"], categories=mission_phases)
    dummies = pd.get_dummies(phase_cat, prefix="mission_phase")

    # Guarantee all vocabulary columns exist even if a phase is absent
    for phase in mission_phases:
        col = f"mission_phase_{phase}"
        if col not in dummies.columns:
            dummies[col] = 0

    # Enforce column order and integer dtype (0/1 not bool)
    ohe_cols = [f"mission_phase_{p}" for p in mission_phases]
    dummies = dummies[ohe_cols].astype(np.int8)
    dummies.index = out.index

    out = pd.concat([out, dummies], axis=1)

    # ------------------------------------------------------------------
    # 6. Leak guard — hard assert that no label/id column ended up in the
    #    computed feature set.
    # ------------------------------------------------------------------
    computed_features = set(feature_column_names(roll_sensors, mission_phases))
    leaking = computed_features & _LABEL_COLS
    assert not leaking, (
        f"[BUG] Label/ID columns leaked into feature matrix: {leaking}\n"
        "This should never happen — file a bug against compute_features()."
    )

    # Also assert that the feature columns we advertise actually exist in out.
    missing_features = computed_features - set(out.columns)
    assert not missing_features, (
        f"[BUG] Advertised feature columns not present in output: {missing_features}"
    )

    return out


# ---------------------------------------------------------------------------
# Convenience: select just the feature matrix from a compute_features() result
# ---------------------------------------------------------------------------

def to_feature_matrix(
    df: pd.DataFrame,
    roll_sensors: list[str] = ROLL_SENSORS,
    mission_phases: list[str] = MISSION_PHASES,
) -> pd.DataFrame:
    """Return only the model-input columns from a compute_features() result.

    Equivalent to ``df[feature_column_names()]``.  The returned DataFrame
    contains NO label, ID, or raw-sensor pass-through columns.
    """
    feat_cols = feature_column_names(roll_sensors, mission_phases)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "to_feature_matrix(): these feature columns are missing — "
            f"did you call compute_features() first? Missing: {missing}"
        )
    return df[feat_cols].copy()


# ---------------------------------------------------------------------------
# Quick smoke-test  (python -m ml.features  or  python ml/features.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib, sys, time

    csv_path = pathlib.Path("data/processed/rotax_combined_clean.csv")
    if not csv_path.exists():
        print(f"[smoke-test] CSV not found at {csv_path} — skipping.")
        sys.exit(0)

    print(f"[smoke-test] Loading {csv_path} ...")
    t0 = time.perf_counter()

    raw = pd.read_csv(csv_path, low_memory=False)
    print(f"  Loaded {len(raw):,} rows in {time.perf_counter() - t0:.1f}s")

    # Run on a 3-run sample first for speed
    sample_runs = raw["run_id"].unique()[:3]
    sample = raw[raw["run_id"].isin(sample_runs)].copy()
    print(f"  Smoke-test on {len(sample):,} rows (runs {sample_runs}) ...")

    t1 = time.perf_counter()
    feat_df = compute_features(sample)
    elapsed = time.perf_counter() - t1

    feat_cols = feature_column_names()
    X = to_feature_matrix(feat_df)

    print(f"\n  compute_features() -> {len(feat_df):,} rows in {elapsed:.2f}s")
    print(f"  Feature matrix shape : {X.shape}")
    print(f"  Feature columns ({len(feat_cols)}):")
    for c in feat_cols:
        print(f"    {c:<45s}  min={X[c].min():>9.4f}  max={X[c].max():>9.4f}")

    # Verify no NaN in feature matrix
    nan_count = X.isna().sum().sum()
    assert nan_count == 0, f"[FAIL] NaN in feature matrix: {nan_count}"
    print(f"\n  NaN check : PASS (0 NaN in feature matrix)")

    # Verify leak guard
    label_in_X = set(X.columns) & _LABEL_COLS
    assert not label_in_X, f"[FAIL] Label leak: {label_in_X}"
    print(f"  Leak guard: PASS (no label columns in X)")

    # Run on full dataset
    print(f"\n  Running on full {len(raw):,} rows ...")
    t2 = time.perf_counter()
    full_feat = compute_features(raw)
    print(f"  Full dataset: {len(full_feat):,} rows in {time.perf_counter()-t2:.1f}s")
    print(f"  Output shape : {full_feat.shape}")

    print("\n[smoke-test] All checks passed.")

"""
project/model/features.py
=========================

Canonical feature engineering pipeline for Rotax 914 telemetry.

Calculates temporal features independently per run_id with:
- Grouping by run_id, sorting by time_s
- Strict run_id validation
- Required, single non-null fault_type_code per run
- 30-sample rolling mean
- 30-sample rolling variance (ddof=0)
- 5-sample difference: value(t) - value(t-5)
- 1 Hz sampling-rate enforcement
- Warm-up truncation: dropping the first 29 rows of every run_id
- Context features passed through unchanged
- Fixed-vocabulary one-hot encoding for mission_phase
- Strict exclusion and loud assertions against label/environment leakage

Canonical input:
    data/processed/rotax_combined_clean.csv

Expected full-dataset result:
    Input rows:  1,046,474
    Runs:        1,248
    Warm-up:     36,192 rows
    Output rows: 1,010,282
    Features:    30
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants & Specifications
# ---------------------------------------------------------------------------

SENSOR_COLUMNS: List[str] = [
    "rpm",
    "cht_c",
    "egt_c",
    "oil_temp_c",
    "oil_pressure_bar",
    "battery_voltage_v",
    "vibration_rms_g",
    "fuel_flow_lph",
]

CONTEXT_COLUMNS: List[str] = [
    "throttle_frac",
    "power_kw",
]

# FIXED vocabulary.
# Do not dynamically infer or modify these values.
MISSION_PHASES: List[str] = [
    "endurance_low_throttle",
    "steady_cruise",
    "high_power_climb",
    "rapid_throttle_transitions",
]

DEFAULT_WINDOW: int = 30
FIXED_DIFF_LAG: int = 5
WARMUP_ROWS_PER_RUN: int = DEFAULT_WINDOW - 1  # 29 rows


# Columns strictly forbidden from the feature matrix.
#
# Some of these are labels, some are identifiers/temporal metadata,
# and some are environmental variables intentionally excluded pending Step 1b.
FORBIDDEN_COLUMNS = frozenset(
    [
        # Labels / targets
        "fault_type_code",
        "fault_type",
        "fault_severity",

        # Run / temporal metadata
        "run_id",
        "time_s",

        # Environment columns excluded pending Step 1b
        "altitude_m",
        "ambient_temp_c",
        "air_pressure_pa",
        "air_density_kgm3",

        # Other target/metadata leakage candidates
        "fault_flag",
        "temp_offset_c",
        "source",
        "TargetSeverity",
    ]
)


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _validate_fixed_mission_vocabulary() -> None:
    """
    Protect the canonical mission-phase vocabulary from accidental mutation.
    """
    expected = [
        "endurance_low_throttle",
        "steady_cruise",
        "high_power_climb",
        "rapid_throttle_transitions",
    ]

    if MISSION_PHASES != expected:
        raise AssertionError(
            "CRITICAL: MISSION_PHASES has been modified. "
            f"Expected exactly: {expected}; got: {MISSION_PHASES}"
        )


def _validate_window(window: int) -> None:
    """Validate the rolling-window argument."""
    if not isinstance(window, int):
        raise TypeError(
            f"window must be an integer, got {type(window).__name__}"
        )

    if window <= 0:
        raise ValueError(
            f"window must be greater than 0, got {window}"
        )


def _validate_required_columns(
    df: pd.DataFrame,
    sensor_columns: List[str],
    context_columns: List[str],
) -> None:
    """
    Validate that all required input columns exist.

    fault_type_code is intentionally mandatory because every run must
    be validated against exactly one non-null fault label.
    """

    required_inputs = [
        "run_id",
        "time_s",
        "mission_phase",
        "fault_type_code",
        *sensor_columns,
        *context_columns,
    ]

    missing = [column for column in required_inputs if column not in df.columns]

    if missing:
        raise ValueError(
            "compute_features: input dataframe is missing required "
            f"columns: {missing}"
        )


def _validate_fault_type_code_per_run(df: pd.DataFrame) -> None:
    """
    Assert that every run_id has exactly one unique, non-null
    fault_type_code.
    """

    if df["fault_type_code"].isna().any():
        null_runs = (
            df.loc[df["fault_type_code"].isna(), "run_id"]
            .unique()
            .tolist()
        )

        raise AssertionError(
            "Validation failed: missing/null fault_type_code "
            f"in run_ids: {null_runs[:20]}"
        )

    codes_per_run = (
        df.groupby("run_id", sort=False)["fault_type_code"]
        .nunique(dropna=False)
    )

    invalid_runs = codes_per_run[codes_per_run != 1]

    if not invalid_runs.empty:
        raise AssertionError(
            "Validation failed: "
            f"{len(invalid_runs)} run_id(s) do not have exactly "
            "1 unique fault_type_code. "
            f"Examples: {invalid_runs.head().to_dict()}"
        )


def _validate_mission_phases(
    df: pd.DataFrame,
) -> None:
    """
    Validate mission_phase against the fixed canonical vocabulary.
    """

    _validate_fixed_mission_vocabulary()

    invalid_phases = (
        set(df["mission_phase"].dropna().unique())
        - set(MISSION_PHASES)
    )

    if invalid_phases:
        raise ValueError(
            "compute_features: input contains unrecognized "
            f"mission_phase values: {sorted(invalid_phases)}. "
            f"Allowed vocabulary: {MISSION_PHASES}"
        )

    if df["mission_phase"].isna().any():
        null_count = int(df["mission_phase"].isna().sum())

        raise ValueError(
            "compute_features: mission_phase contains "
            f"{null_count} null value(s)."
        )


def _validate_sampling_rate(sorted_df: pd.DataFrame) -> None:
    """
    Enforce the expected 1 Hz sampling rate.

    Every consecutive sample within every run must have:
        delta_t == 1 second

    A tolerance of 0.05 seconds is used for floating-point representation.
    """

    dt = (
        sorted_df
        .groupby("run_id", sort=False)["time_s"]
        .diff()
        .dropna()
    )

    if dt.empty:
        raise AssertionError(
            "Sampling-rate validation failed: no time differences found."
        )

    invalid_dt = ~np.isclose(
        dt.to_numpy(dtype=float),
        1.0,
        atol=0.05,
        rtol=0.0,
    )

    if invalid_dt.any():
        invalid_values = dt.iloc[np.flatnonzero(invalid_dt)[:10]].tolist()

        raise AssertionError(
            "Sampling-rate validation failed: expected 1 Hz "
            "(1.0 second between samples), but found "
            f"{int(invalid_dt.sum())} invalid time steps. "
            f"Examples: {invalid_values}"
        )


# ---------------------------------------------------------------------------
# Public API 1: Deterministic ordered list of feature columns
# ---------------------------------------------------------------------------

def feature_column_names() -> List[str]:
    """
    Return the deterministic ordered list of canonical feature columns.

    Total: 30 features

    24 sensor temporal features:
        8 sensors × 3 features

    2 context features:
        throttle_frac
        power_kw

    4 mission-phase one-hot features:
        fixed canonical vocabulary
    """

    _validate_fixed_mission_vocabulary()

    columns: List[str] = []

    # ---------------------------------------------------------------
    # 1. Sensor temporal features
    # ---------------------------------------------------------------

    for sensor in SENSOR_COLUMNS:
        columns.append(f"{sensor}_roll_mean")
        columns.append(f"{sensor}_roll_var")
        columns.append(f"{sensor}_diff_5")

    # ---------------------------------------------------------------
    # 2. Context features
    # ---------------------------------------------------------------

    columns.extend(CONTEXT_COLUMNS)

    # ---------------------------------------------------------------
    # 3. Fixed mission-phase one-hot features
    # ---------------------------------------------------------------

    for phase in MISSION_PHASES:
        columns.append(f"mission_phase_{phase}")

    # ---------------------------------------------------------------
    # Final integrity check
    # ---------------------------------------------------------------

    leaks = set(columns) & FORBIDDEN_COLUMNS

    assert not leaks, (
        "CRITICAL: Forbidden column(s) detected in feature list: "
        f"{sorted(leaks)}"
    )

    assert len(columns) == 30, (
        f"CRITICAL: Expected exactly 30 feature columns, "
        f"got {len(columns)}"
    )

    assert len(columns) == len(set(columns)), (
        "CRITICAL: Duplicate feature column names detected."
    )

    return columns


# ---------------------------------------------------------------------------
# Public API 2: Feature computation
# ---------------------------------------------------------------------------

def compute_features(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """
    Compute canonical temporal features independently within each run_id.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned telemetry dataframe containing:

        - run_id
        - time_s
        - mission_phase
        - fault_type_code
        - eight sensor columns
        - throttle_frac
        - power_kw

    window : int, default=30
        Rolling window size in samples.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe with the first 29 rows of
        every run removed.

        Labels and metadata remain available in this dataframe so
        the caller can separately construct y.

    Notes
    -----
    Temporal calculations are strictly isolated by run_id.

    The canonical difference is always:

        value(t) - value(t-5)

    The difference lag is intentionally fixed at 5 samples.
    """

    # ---------------------------------------------------------------
    # 1. Validate constants and arguments
    # ---------------------------------------------------------------

    _validate_fixed_mission_vocabulary()
    _validate_window(window)

    if window != DEFAULT_WINDOW:
        raise ValueError(
            f"Canonical feature pipeline requires window={DEFAULT_WINDOW}. "
            f"Received window={window}."
        )

    # ---------------------------------------------------------------
    # 2. Validate required input columns
    # ---------------------------------------------------------------

    _validate_required_columns(
        df,
        SENSOR_COLUMNS,
        CONTEXT_COLUMNS,
    )

    # ---------------------------------------------------------------
    # 3. Validate mission-phase vocabulary
    # ---------------------------------------------------------------

    _validate_mission_phases(df)

    # ---------------------------------------------------------------
    # 4. Validate fault label integrity
    # ---------------------------------------------------------------

    _validate_fault_type_code_per_run(df)

    # ---------------------------------------------------------------
    # 5. Sort strictly by run_id and time_s
    # ---------------------------------------------------------------

    sorted_df = (
        df
        .sort_values(
            ["run_id", "time_s"],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )

    # ---------------------------------------------------------------
    # 6. Enforce 1 Hz sampling
    # ---------------------------------------------------------------

    _validate_sampling_rate(sorted_df)

    # ---------------------------------------------------------------
    # 7. Group sensor data by run_id
    # ---------------------------------------------------------------

    grouped = sorted_df.groupby(
        "run_id",
        sort=False,
    )[SENSOR_COLUMNS]

    # ---------------------------------------------------------------
    # 8. Rolling mean
    # ---------------------------------------------------------------

    roll_mean = (
        grouped
        .rolling(
            window=DEFAULT_WINDOW,
            min_periods=DEFAULT_WINDOW,
        )
        .mean()
        .reset_index(level=0, drop=True)
    )

    roll_mean.columns = [
        f"{sensor}_roll_mean"
        for sensor in SENSOR_COLUMNS
    ]

    # ---------------------------------------------------------------
    # 9. Rolling variance
    # ---------------------------------------------------------------

    roll_var = (
        grouped
        .rolling(
            window=DEFAULT_WINDOW,
            min_periods=DEFAULT_WINDOW,
        )
        .var(ddof=0)
        .reset_index(level=0, drop=True)
    )

    roll_var.columns = [
        f"{sensor}_roll_var"
        for sensor in SENSOR_COLUMNS
    ]

    # ---------------------------------------------------------------
    # 10. Five-sample difference
    # ---------------------------------------------------------------

    diff_5 = grouped.diff(FIXED_DIFF_LAG)

    diff_5.columns = [
        f"{sensor}_diff_5"
        for sensor in SENSOR_COLUMNS
    ]

    # ---------------------------------------------------------------
    # 11. Combine temporal features
    # ---------------------------------------------------------------

    sorted_df = pd.concat(
        [
            sorted_df,
            roll_mean,
            roll_var,
            diff_5,
        ],
        axis=1,
    )

    # ---------------------------------------------------------------
    # 12. Fixed-vocabulary mission-phase one-hot encoding
    # ---------------------------------------------------------------

    phase_cat = pd.Categorical(
        sorted_df["mission_phase"],
        categories=MISSION_PHASES,
    )

    dummies = pd.get_dummies(
        phase_cat,
        prefix="mission_phase",
        dtype=int,
    )

    dummy_columns = [
        f"mission_phase_{phase}"
        for phase in MISSION_PHASES
    ]

    # Guarantee all four canonical columns exist.
    for column in dummy_columns:
        if column not in dummies.columns:
            dummies[column] = 0

    dummies = dummies[dummy_columns]

    dummies.index = sorted_df.index

    sorted_df = pd.concat(
        [
            sorted_df,
            dummies,
        ],
        axis=1,
    )

    # ---------------------------------------------------------------
    # 13. Drop first 29 rows of every run
    # ---------------------------------------------------------------

    sorted_df["_run_row_idx"] = (
        sorted_df
        .groupby(
            "run_id",
            sort=False,
        )
        .cumcount()
    )

    clean_df = (
        sorted_df[
            sorted_df["_run_row_idx"] >= WARMUP_ROWS_PER_RUN
        ]
        .drop(columns=["_run_row_idx"])
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # 14. Final integrity checks
    # ---------------------------------------------------------------

    expected_feature_columns = feature_column_names()

    missing_features = [
        column
        for column in expected_feature_columns
        if column not in clean_df.columns
    ]

    if missing_features:
        raise AssertionError(
            "Feature computation failed. Missing generated "
            f"feature columns: {missing_features}"
        )

    # Verify no NaNs in model features after warm-up.
    feature_nan_count = int(
        clean_df[expected_feature_columns]
        .isna()
        .sum()
        .sum()
    )

    if feature_nan_count != 0:
        raise AssertionError(
            "Feature computation produced "
            f"{feature_nan_count} NaN values after warm-up."
        )

    return clean_df


# ---------------------------------------------------------------------------
# Public API 3: Model-ready feature matrix
# ---------------------------------------------------------------------------

def to_feature_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract the canonical model-ready feature matrix X.

    The returned dataframe contains exactly 30 columns:

        24 sensor temporal features
        2 context features
        4 mission-phase features

    Labels, identifiers, time, and environmental columns are excluded.
    """

    feature_cols = feature_column_names()

    # ---------------------------------------------------------------
    # 1. Ensure all canonical features exist
    # ---------------------------------------------------------------

    missing = [
        column
        for column in feature_cols
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "to_feature_matrix: missing feature columns. "
            "Run compute_features() first. Missing: "
            f"{missing}"
        )

    # ---------------------------------------------------------------
    # 2. Select ONLY canonical feature columns
    # ---------------------------------------------------------------

    X = df[feature_cols].copy()

    # ---------------------------------------------------------------
    # 3. Global forbidden-column leak check
    # ---------------------------------------------------------------

    leak_check = set(X.columns) & FORBIDDEN_COLUMNS

    assert not leak_check, (
        "SECURITY ASSERTION FAILED: "
        "Label/forbidden columns in feature matrix: "
        f"{sorted(leak_check)}"
    )

    # ---------------------------------------------------------------
    # 4. Explicit target/metadata checks
    # ---------------------------------------------------------------

    explicitly_forbidden = [
        "fault_type_code",
        "fault_type",
        "fault_severity",
        "run_id",
        "time_s",
        "altitude_m",
        "ambient_temp_c",
        "air_pressure_pa",
        "air_density_kgm3",
    ]

    leaked = [
        column
        for column in explicitly_forbidden
        if column in X.columns
    ]

    assert not leaked, (
        "SECURITY ASSERTION FAILED: forbidden column(s) "
        f"leaked into feature matrix: {leaked}"
    )

    # ---------------------------------------------------------------
    # 5. NaN check
    # ---------------------------------------------------------------

    nan_count = int(
        X.isna().sum().sum()
    )

    assert nan_count == 0, (
        "Integrity check failed: "
        f"{nan_count} NaN values found in feature matrix."
    )

    # ---------------------------------------------------------------
    # 6. Final shape/order check
    # ---------------------------------------------------------------

    assert list(X.columns) == feature_cols, (
        "CRITICAL: Feature-column ordering does not match "
        "the canonical feature definition."
    )

    assert X.shape[1] == 30, (
        f"CRITICAL: Expected 30 feature columns, "
        f"got {X.shape[1]}"
    )

    return X
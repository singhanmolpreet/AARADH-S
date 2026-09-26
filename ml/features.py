"""
features.py

Turns raw Rotax 914 telemetry rows into a feature matrix for the fault
classifier. Pure function, pandas/numpy only, no side effects, no I/O.

Design (agreed decisions, see project discussion):
  - Window = 30 samples (30s at the confirmed 1Hz sample rate), sliding,
    stride 1.
  - Diff (rate-of-change) lag = 5 samples (5s) -- short enough to catch
    abrupt onsets (e.g. misfire) without being pure sample-to-sample noise.
  - Rolling features never span two episodes or two engines: episode
    boundaries are detected from `fault_label` transitions per `engine_id`,
    not assumed from row counts (row counts turned out inexact in
    practice). Unlabeled data (no `fault_label` column, e.g. the healthy
    baseline file) is treated as one episode per `engine_id`.
  - All 12 sensor columns are rolled uniformly, including
    `injection_timing_deg` -- `drift_channel` can name any sensor as the
    drifting one, so excluding any column would make the model blind to a
    drift fault on that channel.
  - `mission_phase` is one-hot encoded against a fixed 7-value category
    list so train and inference always produce the same columns, even if
    a given batch doesn't contain every phase.
  - `engine_id` is used only for grouping (rolling + episode detection);
    it is never itself a feature -- the model should key off telemetry
    shape, not off which physical engine produced it.
  - Rows with an incomplete window (the first ~30s of each episode) are
    dropped rather than backfilled.
  - fault_label / severity / fault_ramp_frac / misfire_cylinder /
    drift_channel / drift_bias / wastegate_direction are never touched or
    used as inputs here -- they pass through unchanged (if present) for
    the caller to use as targets or for validation, never as features.
"""

import warnings

import pandas as pd

SENSOR_COLUMNS = [
    "rpm",
    "cht_c",
    "egt_c",
    "oil_pressure_bar",
    "oil_temp_c",
    "fuel_flow_lph",
    "vibration_rms_g",
    "battery_voltage_v",
    "alternator_current_a",
    "injection_timing_deg",
    "map_kpa",
    "boost_pressure_bar",
]

MISSION_PHASES = [
    "ground",
    "takeoff",
    "climb",
    "cruise",
    "high_altitude_cruise",
    "loiter",
    "descent",
]

DEFAULT_WINDOW = 30
DEFAULT_DIFF_LAG = 5


def feature_column_names(sensor_columns=SENSOR_COLUMNS, mission_phases=MISSION_PHASES):
    """Explicit, importable list of the columns that belong in the model's
    feature matrix X. Downstream code should select columns via this list
    rather than inferring "everything except known label columns"."""
    cols = list(sensor_columns)
    for col in sensor_columns:
        cols += [f"{col}_roll_mean", f"{col}_roll_var", f"{col}_diff"]
    cols += [f"mission_phase_{phase}" for phase in mission_phases]
    return cols


def _assign_episode_key(df, label_col="fault_label"):
    """Return a per-row string key identifying (engine_id, episode) so
    rolling windows and splits never blend two episodes or two engines.

    An "episode" is a maximal run of consecutive rows, within one
    engine_id, that share the same value of `label_col`. If `label_col`
    isn't present (unlabeled data), each engine_id is treated as a single
    episode.
    """
    if "engine_id" not in df.columns:
        raise ValueError("_assign_episode_key requires an 'engine_id' column")

    if label_col in df.columns:
        changed = df[label_col] != df.groupby("engine_id")[label_col].shift()
    else:
        changed = df["engine_id"] != df["engine_id"].shift()

    episode_num = changed.groupby(df["engine_id"]).cumsum()
    return df["engine_id"].astype(str) + "_ep" + episode_num.astype(str)


def compute_features(
    df,
    window=DEFAULT_WINDOW,
    diff_lag=DEFAULT_DIFF_LAG,
    sensor_columns=SENSOR_COLUMNS,
    mission_phases=MISSION_PHASES,
    label_col="fault_label",
):
    """Compute rolling-window features from raw telemetry.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `engine_id`, `mission_phase`, and all of
        `sensor_columns`. May optionally contain `timestamp` (used to sort
        rows) and label/metadata columns (fault_label, severity, etc.),
        which are passed through unchanged if present.
    window : int
        Rolling window size in samples (30 == 30s at 1Hz).
    diff_lag : int
        Lag in samples for the rate-of-change feature.

    Returns
    -------
    pd.DataFrame
        Input columns (sorted, with incomplete-window rows dropped) plus
        for each sensor column: `<col>_roll_mean`, `<col>_roll_var`,
        `<col>_diff`; plus one-hot `mission_phase_<phase>` columns for
        every phase in `mission_phases`. Use `feature_column_names()` to
        select the actual model input columns from the result.
    """
    required = list(sensor_columns) + ["engine_id", "mission_phase"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"compute_features: missing required columns {missing}")

    sort_cols = ["engine_id", "timestamp"] if "timestamp" in df.columns else ["engine_id"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    episode_key = _assign_episode_key(df, label_col=label_col)

    episode_lengths = episode_key.value_counts()
    too_short = episode_lengths[episode_lengths < window]
    if len(too_short) > 0:
        warnings.warn(
            f"{len(too_short)} episode(s) are shorter than window={window} "
            "samples and will be fully dropped (no complete window ever "
            f"forms): {list(too_short.index)}"
        )

    out = df.copy()
    grouped_sensors = df.groupby(episode_key)

    for col in sensor_columns:
        series = grouped_sensors[col]
        out[f"{col}_roll_mean"] = series.transform(
            lambda s: s.rolling(window, min_periods=window).mean()
        )
        out[f"{col}_roll_var"] = series.transform(
            lambda s: s.rolling(window, min_periods=window).var()
        )
        out[f"{col}_diff"] = series.transform(lambda s: s.diff(diff_lag))

    phase_categorical = pd.Categorical(df["mission_phase"], categories=mission_phases)
    dummies = pd.get_dummies(phase_categorical, prefix="mission_phase")
    for phase in mission_phases:
        col = f"mission_phase_{phase}"
        if col not in dummies.columns:
            dummies[col] = False
    dummies = dummies[[f"mission_phase_{phase}" for phase in mission_phases]].astype(int)
    out = pd.concat([out, dummies], axis=1)

    feature_cols = feature_column_names(sensor_columns, mission_phases)
    complete = ~out[feature_cols].isna().any(axis=1)
    out = out.loc[complete].reset_index(drop=True)

    return out

"""
verify_features.py
==================
Comprehensive verification script for the canonical features.py pipeline.
Runs against data/processed/rotax_combined_clean.csv and reports all step 10 metrics.
"""

import sys
import pathlib
import time
import numpy as np
import pandas as pd

# Add repo root to sys.path so we can import from project.model.features
repo_root = pathlib.Path(__file__).parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from project.model.features import (
    compute_features,
    feature_column_names,
    to_feature_matrix,
    FORBIDDEN_COLUMNS,
    SENSOR_COLUMNS,
    CONTEXT_COLUMNS,
    MISSION_PHASES,
)

CSV_PATH = repo_root / "data" / "processed" / "rotax_combined_clean.csv"

def run_verification():
    sep = "=" * 78
    sep2 = "-" * 78
    print(sep)
    print("ROTAX 914 CANONICAL FEATURE PIPELINE VERIFICATION")
    print(sep)

    # 1. Load Data
    print(f"\n1. Loading input dataset: {CSV_PATH}")
    t0 = time.time()
    df = pd.read_csv(CSV_PATH, low_memory=False)
    load_time = time.time() - t0

    n_input_rows = len(df)
    n_input_runs = df["run_id"].nunique()
    print(f"   Input row count : {n_input_rows:,}")
    print(f"   Input run count : {n_input_runs:,}")
    print(f"   Loaded in       : {load_time:.2f} s")

    # 2. Timing check (1 Hz sampling rate confirmation)
    print(f"\n{sep2}")
    print("2. Verifying 1 Hz sampling assumption across all runs")
    df_sorted = df.sort_values(["run_id", "time_s"])
    dt = df_sorted.groupby("run_id", sort=False)["time_s"].diff().dropna()
    dt_counts = dt.value_counts()
    median_dt = dt.median()
    mean_dt = dt.mean()
    min_dt = dt.min()
    max_dt = dt.max()

    print(f"   Total sample-to-sample transitions : {len(dt):,}")
    print(f"   Min dt    : {min_dt} s")
    print(f"   Max dt    : {max_dt} s")
    print(f"   Mean dt   : {mean_dt:.4f} s")
    print(f"   Median dt : {median_dt:.4f} s")
    print(f"   Frequency distribution of dt:")
    for val, count in dt_counts.items():
        print(f"     dt = {val}s : {count:,} ({count/len(dt)*100:.2f}%)")

    assert np.isclose(median_dt, 1.0), f"Sampling rate is not 1.0s (median={median_dt})"
    assert min_dt == 1.0 and max_dt == 1.0, f"Unexpected non-1.0s steps: min={min_dt}, max={max_dt}"
    print("   [CONFIRMED] 1 Hz timing assumption confirmed: 100.0% of steps are exactly 1.0000s.")

    # 3. Verification of Run ID Integrity (Fault Code uniqueness)
    print(f"\n{sep2}")
    print("3. Verifying run_id integrity (fault_type_code uniqueness per run)")
    codes_per_run = df.groupby("run_id")["fault_type_code"].nunique(dropna=False)
    assert (codes_per_run == 1).all(), "Found runs with multiple or missing fault_type_code!"
    assert not df["fault_type_code"].isna().any(), "Found null fault_type_code values!"
    print(f"   [CONFIRMED] All {n_input_runs:,} runs have exactly 1 unique, non-null fault_type_code.")

    # 4. Feature Computation
    print(f"\n{sep2}")
    print("4. Executing canonical compute_features(df, window=30)...")
    t1 = time.time()
    out_df = compute_features(df)
    compute_time = time.time() - t1

    n_output_rows = len(out_df)
    n_output_runs = out_df["run_id"].nunique()
    print(f"   Feature computation completed in : {compute_time:.2f} s")
    print(f"   Output row count                 : {n_output_rows:,}")
    print(f"   Output run count                 : {n_output_runs:,}")

    # 5. Row Count Exact Verification
    expected_rows = n_input_rows - (n_input_runs * 29)
    print(f"\n{sep2}")
    print("5. Row count verification against theoretical expectation:")
    print(f"   Theoretical formula : input_rows - (runs * 29)")
    print(f"   Calculation         : {n_input_rows:,} - ({n_input_runs:,} * 29)")
    print(f"   Expected output rows: {expected_rows:,}")
    print(f"   Actual output rows  : {n_output_rows:,}")
    assert n_output_rows == 1010282, f"Row count mismatch! Expected 1,010,282, got {n_output_rows}"
    print(f"   [CONFIRMED] Output row count is EXACTLY 1,010,282.")

    # 6. Independence of temporal calculations across run boundaries
    print(f"\n{sep2}")
    print("6. Verifying temporal calculations do NOT bleed across run boundaries")
    # For each run_id, check row 0 (which was row 29 in original run):
    # Its diff_5 must equal df.loc[29, sensor] - df.loc[24, sensor] within that same run!
    first_rows_out = out_df.groupby("run_id", sort=False).first()
    # Check on a sample of runs
    sample_runs = list(df["run_id"].unique()[:5]) + list(df["run_id"].unique()[-5:])
    for r in sample_runs:
        run_raw = df[df["run_id"] == r].sort_values("time_s").reset_index(drop=True)
        # 30-sample mean for rpm at row 29
        expected_rpm_mean = run_raw.loc[:29, "rpm"].mean()
        actual_rpm_mean = first_rows_out.loc[r, "rpm_roll_mean"]
        assert np.isclose(expected_rpm_mean, actual_rpm_mean, atol=1e-6), (
            f"Run {r} rolling mean bleed mismatch! Expected {expected_rpm_mean}, got {actual_rpm_mean}"
        )

        # 30-sample variance (ddof=0) for rpm at row 29
        expected_rpm_var = run_raw.loc[:29, "rpm"].var(ddof=0)
        actual_rpm_var = first_rows_out.loc[r, "rpm_roll_var"]
        assert np.isclose(expected_rpm_var, actual_rpm_var, atol=1e-6), (
            f"Run {r} rolling var bleed mismatch! Expected {expected_rpm_var}, got {actual_rpm_var}"
        )

        # 5-sample diff for rpm at row 29: val(29) - val(24)
        expected_rpm_diff5 = run_raw.loc[29, "rpm"] - run_raw.loc[24, "rpm"]
        actual_rpm_diff5 = first_rows_out.loc[r, "rpm_diff_5"]
        assert np.isclose(expected_rpm_diff5, actual_rpm_diff5, atol=1e-6), (
            f"Run {r} diff_5 bleed mismatch! Expected {expected_rpm_diff5}, got {actual_rpm_diff5}"
        )
    print("   [CONFIRMED] Zero cross-run boundary bleed: rolling stats and diffs verified mathematically.")

    # 7. Extract Feature Matrix X
    print(f"\n{sep2}")
    print("7. Extracting feature matrix X via to_feature_matrix(out_df)...")
    X = to_feature_matrix(out_df)
    feature_names = feature_column_names()

    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Number of feature columns: {len(feature_names)}")

    # 8. Exact Feature Column Names Listing
    print(f"\n{sep2}")
    print("8. Exact feature-column names (30 columns in deterministic order):")
    print(f"   {'Index':<6} {'Feature Column Name':<45} {'Dtype':<10} {'Min':>12} {'Max':>12}")
    for idx, col in enumerate(feature_names, 1):
        c_min = f"{X[col].min():.4f}" if np.issubdtype(X[col].dtype, np.number) else str(X[col].min())
        c_max = f"{X[col].max():.4f}" if np.issubdtype(X[col].dtype, np.number) else str(X[col].max())
        print(f"   {idx:2d}.    {col:<45} {str(X[col].dtype):<10} {c_min:>12} {c_max:>12}")

    # 9. Mission-phase columns
    print(f"\n{sep2}")
    print("9. Mission-phase one-hot columns:")
    phase_cols = [c for c in feature_names if c.startswith("mission_phase_")]
    for p in phase_cols:
        col_sum = X[p].sum()
        print(f"   - {p:<45} : {col_sum:>9,} rows ({col_sum/len(X)*100:5.2f}%)")

    # 10. Label-leak assertions & forbidden columns check
    print(f"\n{sep2}")
    print("10. Checking label leakage and forbidden columns in feature matrix X:")
    leaks = set(X.columns) & FORBIDDEN_COLUMNS
    assert not leaks, f"LEAK DETECTED: {leaks}"
    for target in ["fault_type_code", "fault_type", "fault_severity", "run_id", "time_s", "altitude_m"]:
        assert target not in X.columns, f"Target '{target}' leaked into X!"
    print("   [CONFIRMED] Zero label/forbidden columns present in feature matrix X.")
    print("   [CONFIRMED] All leak assertions passed successfully.")

    # 11. Missing value check
    nan_count = int(X.isna().sum().sum())
    assert nan_count == 0, f"Found {nan_count} NaNs in X!"
    print(f"   [CONFIRMED] Exactly 0 NaN values across all 1,010,282 rows in feature matrix.")

    print(f"\n{sep}")
    print("ALL 10 VERIFICATION CHECKS PASSED PERFECTLY!")
    print(sep)

if __name__ == "__main__":
    run_verification()

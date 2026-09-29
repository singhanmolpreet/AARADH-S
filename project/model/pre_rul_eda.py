from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "rotax_combined_clean.csv"
OUTPUT_PATH = SCRIPT_DIR / "rul_run_level_eda.csv"


# ============================================================
# Configuration
# ============================================================

SEVERITY_FAILURE_THRESHOLD = 0.99
TREND_WINDOW = 60

FAULT_CODES = {
    1: "misfire",
    2: "lubrication_issue",
    3: "sensor_drift",
    4: "overheating",
    5: "wastegate_fault",
    6: "injector_fault",
    7: "alternator_fault",
    8: "air_filter_blockage",
}


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("RUL RUN-LEVEL EDA")
print("=" * 70)

print(f"\nLoading:\n{DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)

required_columns = [
    "run_id",
    "time_s",
    "fault_type_code",
    "fault_type",
    "fault_severity",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )

print(f"Rows: {len(df):,}")
print(f"Runs: {df['run_id'].nunique():,}")


# ============================================================
# Basic cleanup / type conversion
# ============================================================

df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
df["fault_type_code"] = pd.to_numeric(
    df["fault_type_code"], errors="coerce"
)
df["fault_severity"] = pd.to_numeric(
    df["fault_severity"], errors="coerce"
)

df = df.dropna(
    subset=[
        "run_id",
        "time_s",
        "fault_type_code",
        "fault_severity",
    ]
)

df = df.sort_values(["run_id", "time_s"]).reset_index(drop=True)


# ============================================================
# 1. Sampling / time-axis analysis
# ============================================================

print("\n" + "=" * 70)
print("1. TIME AXIS / SAMPLING")
print("=" * 70)

time_diffs = (
    df.groupby("run_id")["time_s"]
    .diff()
    .dropna()
)

print("\nGlobal time_s difference statistics:")
print(time_diffs.describe().to_string())

print("\nUnique time_s differences:")
print(
    time_diffs
    .value_counts()
    .sort_index()
    .head(20)
    .to_string()
)

duplicate_times = (
    df.groupby("run_id")["time_s"]
    .apply(lambda x: x.duplicated().sum())
)

print(
    f"\nRuns containing duplicate time_s values: "
    f"{(duplicate_times > 0).sum():,}"
)

gap_runs = (
    df.groupby("run_id")["time_s"]
    .apply(lambda x: (x.diff().dropna() > 1.0).any())
)

print(
    f"Runs containing gaps > 1 second: "
    f"{gap_runs.sum():,}"
)


# ============================================================
# 2. Run-level summary
# ============================================================

print("\n" + "=" * 70)
print("2. RUN-LEVEL SUMMARY")
print("=" * 70)

run_rows = []

for run_id, run in df.groupby("run_id", sort=True):

    run = run.sort_values("time_s")

    severity = run["fault_severity"].to_numpy(dtype=float)
    time = run["time_s"].to_numpy(dtype=float)

    fault_code = int(run["fault_type_code"].iloc[0])
    fault_name = str(run["fault_type"].iloc[0])

    max_idx = int(np.argmax(severity))
    max_severity = float(severity[max_idx])

    reaches_failure = max_severity >= SEVERITY_FAILURE_THRESHOLD

    failure_time_s = (
        float(time[max_idx])
        if reaches_failure
        else np.nan
    )

    # First point at or above the threshold.
    threshold_indices = np.where(
        severity >= SEVERITY_FAILURE_THRESHOLD
    )[0]

    if len(threshold_indices) > 0:
        first_failure_idx = int(threshold_indices[0])
        first_failure_time_s = float(time[first_failure_idx])
    else:
        first_failure_idx = np.nan
        first_failure_time_s = np.nan

    # Monotonicity statistics.
    severity_diff = np.diff(severity)

    increasing_steps = int(np.sum(severity_diff > 0))
    decreasing_steps = int(np.sum(severity_diff < 0))
    flat_steps = int(np.sum(severity_diff == 0))

    total_steps = len(severity_diff)

    increasing_fraction = (
        increasing_steps / total_steps
        if total_steps > 0
        else np.nan
    )

    decreasing_fraction = (
        decreasing_steps / total_steps
        if total_steps > 0
        else np.nan
    )

    # Number of usable samples for the intended trend window.
    enough_for_window = len(run) >= TREND_WINDOW

    run_rows.append(
        {
            "run_id": run_id,
            "fault_type_code": fault_code,
            "fault_type": fault_name,
            "n_rows": len(run),
            "start_time_s": float(time[0]),
            "end_time_s": float(time[-1]),
            "duration_s": float(time[-1] - time[0]),
            "duration_min": float((time[-1] - time[0]) / 60.0),
            "initial_severity": float(severity[0]),
            "final_severity": float(severity[-1]),
            "min_severity": float(np.min(severity)),
            "max_severity": max_severity,
            "reaches_failure": reaches_failure,
            "first_failure_time_s": first_failure_time_s,
            "first_failure_time_min": (
                first_failure_time_s / 60.0
                if not np.isnan(first_failure_time_s)
                else np.nan
            ),
            "increasing_fraction": increasing_fraction,
            "decreasing_fraction": decreasing_fraction,
            "increasing_steps": increasing_steps,
            "decreasing_steps": decreasing_steps,
            "flat_steps": flat_steps,
            "enough_for_60_point_window": enough_for_window,
        }
    )

run_summary = pd.DataFrame(run_rows)

print(f"\nRun-level rows: {len(run_summary):,}")

print("\nRun duration:")
print(
    run_summary[
        ["duration_s", "duration_min", "n_rows"]
    ].describe().to_string()
)


# ============================================================
# 3. Overall faulty-run failure/censoring analysis
# ============================================================

print("\n" + "=" * 70)
print("3. FAILURE VS RIGHT-CENSORED RUNS")
print("=" * 70)

faulty_runs = run_summary[
    run_summary["fault_type_code"] != 0
].copy()

failure_count = int(faulty_runs["reaches_failure"].sum())
censored_count = int(
    len(faulty_runs) - failure_count
)

print(f"\nFaulty runs: {len(faulty_runs):,}")
print(f"Runs reaching >= {SEVERITY_FAILURE_THRESHOLD}: {failure_count:,}")
print(f"Runs NOT reaching threshold: {censored_count:,}")

if len(faulty_runs) > 0:
    print(
        f"Failure observed fraction: "
        f"{failure_count / len(faulty_runs):.2%}"
    )

    print(
        f"Right-censored fraction: "
        f"{censored_count / len(faulty_runs):.2%}"
    )


# ============================================================
# 4. Per-fault-type analysis
# ============================================================

print("\n" + "=" * 70)
print("4. FAILURE / CENSORING BY FAULT TYPE")
print("=" * 70)

fault_report = []

for code, group in faulty_runs.groupby(
    "fault_type_code", sort=True
):

    fault_name = FAULT_CODES.get(
        int(code),
        str(group["fault_type"].iloc[0])
    )

    total = len(group)
    reached = int(group["reaches_failure"].sum())
    censored = total - reached

    fault_report.append(
        {
            "fault_type_code": int(code),
            "fault_type": fault_name,
            "runs": total,
            "reaches_0_99": reached,
            "censored": censored,
            "failure_fraction": reached / total,
            "censored_fraction": censored / total,
            "mean_max_severity": group["max_severity"].mean(),
            "median_max_severity": group["max_severity"].median(),
            "min_max_severity": group["max_severity"].min(),
            "max_max_severity": group["max_severity"].max(),
            "mean_duration_min": group["duration_min"].mean(),
            "median_duration_min": group["duration_min"].median(),
        }
    )

fault_report_df = pd.DataFrame(fault_report)

print(
    fault_report_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# 5. Severity progression / monotonicity
# ============================================================

print("\n" + "=" * 70)
print("5. SEVERITY PROGRESSION")
print("=" * 70)

print("\nFaulty runs — monotonicity statistics:")

progression = (
    faulty_runs
    .groupby(["fault_type_code", "fault_type"])
    .agg(
        runs=("run_id", "count"),
        mean_increasing_fraction=(
            "increasing_fraction",
            "mean",
        ),
        median_increasing_fraction=(
            "increasing_fraction",
            "median",
        ),
        mean_decreasing_fraction=(
            "decreasing_fraction",
            "mean",
        ),
        median_decreasing_fraction=(
            "decreasing_fraction",
            "median",
        ),
        mean_initial_severity=(
            "initial_severity",
            "mean",
        ),
        mean_final_severity=(
            "final_severity",
            "mean",
        ),
        mean_max_severity=(
            "max_severity",
            "mean",
        ),
    )
    .reset_index()
)

print(
    progression.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# 6. 60-point window suitability
# ============================================================

print("\n" + "=" * 70)
print("6. 60-POINT WINDOW CHECK")
print("=" * 70)

faulty_with_window = faulty_runs[
    faulty_runs["fault_type_code"] != 0
]

window_count = int(
    faulty_with_window[
        "enough_for_60_point_window"
    ].sum()
)

print(
    f"\nFaulty runs with >= {TREND_WINDOW} observations: "
    f"{window_count:,}/{len(faulty_with_window):,}"
)

if len(faulty_with_window) > 0:
    print(
        f"Fraction usable for a {TREND_WINDOW}-point window: "
        f"{window_count / len(faulty_with_window):.2%}"
    )


# ============================================================
# 7. Failure timing for runs that actually fail
# ============================================================

print("\n" + "=" * 70)
print("7. OBSERVED FAILURE TIMING")
print("=" * 70)

failed_runs = faulty_runs[
    faulty_runs["reaches_failure"]
].copy()

if len(failed_runs) == 0:

    print("\nNo faulty runs reached the failure threshold.")

else:

    print(
        "\nTime to first severity >= "
        f"{SEVERITY_FAILURE_THRESHOLD}:"
    )

    print(
        failed_runs[
            "first_failure_time_min"
        ].describe().to_string()
    )

    print("\nBy fault type:")

    failure_timing = (
        failed_runs
        .groupby(["fault_type_code", "fault_type"])
        ["first_failure_time_min"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
        .reset_index()
    )

    print(
        failure_timing.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )


# ============================================================
# 8. Censored-run final severity
# ============================================================

print("\n" + "=" * 70)
print("8. RIGHT-CENSORED RUNS")
print("=" * 70)

if len(faulty_runs) > 0:

    censored = faulty_runs[
        ~faulty_runs["reaches_failure"]
    ].copy()

    print(
        f"\nCensored faulty runs: {len(censored):,}"
    )

    print("\nFinal severity of censored runs:")

    print(
        censored["final_severity"]
        .describe()
        .to_string()
    )

    print("\nMaximum severity of censored runs:")

    print(
        censored["max_severity"]
        .describe()
        .to_string()
    )

    print("\nCensored runs by fault type:")

    censored_report = (
        censored
        .groupby(["fault_type_code", "fault_type"])
        .agg(
            runs=("run_id", "count"),
            mean_final_severity=(
                "final_severity",
                "mean",
            ),
            median_final_severity=(
                "final_severity",
                "median",
            ),
            mean_max_severity=(
                "max_severity",
                "mean",
            ),
            median_max_severity=(
                "max_severity",
                "median",
            ),
        )
        .reset_index()
    )

    print(
        censored_report.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )


# ============================================================
# 9. Check severity behavior near the end of runs
# ============================================================

print("\n" + "=" * 70)
print("9. END-OF-RUN SEVERITY TREND")
print("=" * 70)

end_trends = []

for run_id, run in df.groupby("run_id", sort=True):

    run = run.sort_values("time_s")

    if len(run) < TREND_WINDOW:
        continue

    recent = run.tail(TREND_WINDOW)

    x = recent["time_s"].to_numpy(dtype=float)
    y = recent["fault_severity"].to_numpy(dtype=float)

    if np.ptp(x) == 0:
        slope = np.nan
    else:
        slope = np.polyfit(x, y, 1)[0]

    end_trends.append(
        {
            "run_id": run_id,
            "fault_type_code": int(
                run["fault_type_code"].iloc[0]
            ),
            "fault_type": str(
                run["fault_type"].iloc[0]
            ),
            "final_severity": float(
                run["fault_severity"].iloc[-1]
            ),
            "max_severity": float(
                run["fault_severity"].max()
            ),
            "recent_60_slope_per_s": slope,
        }
    )

end_trends_df = pd.DataFrame(end_trends)

if len(end_trends_df) > 0:

    faulty_end_trends = end_trends_df[
        end_trends_df["fault_type_code"] != 0
    ]

    print(
        "\nFaulty runs with >= 60 observations:"
    )

    print(
        faulty_end_trends[
            "recent_60_slope_per_s"
        ].describe().to_string()
    )

    print(
        "\nFraction with positive slope:"
    )

    positive_fraction = (
        faulty_end_trends[
            "recent_60_slope_per_s"
        ] > 0
    ).mean()

    print(f"{positive_fraction:.2%}")

    print(
        "\nFraction with non-positive slope:"
    )

    non_positive_fraction = (
        faulty_end_trends[
            "recent_60_slope_per_s"
        ] <= 0
    ).mean()

    print(f"{non_positive_fraction:.2%}")


# ============================================================
# 10. Save run-level EDA table
# ============================================================

run_summary.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("10. OUTPUT")
print("=" * 70)

print(f"\nSaved run-level EDA to:")
print(OUTPUT_PATH)

print("\nEDA complete.")
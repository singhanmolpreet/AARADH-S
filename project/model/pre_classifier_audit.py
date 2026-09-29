import pandas as pd
from pathlib import Path

CSV_PATH = Path("../../data/processed/rotax_combined_clean.csv")

df = pd.read_csv(CSV_PATH, low_memory=False)

print("=" * 80)
print("DATASET TARGET AUDIT")
print("=" * 80)

# 1. Columns
print("\n[1] Required columns")
required = [
    "run_id",
    "fault_type_code",
    "fault_severity",
    "battery_voltage_v",
]
for col in required:
    print(f"  {col:25} {'YES' if col in df.columns else 'NO'}")

# 2. Shape
print("\n[2] Dataset")
print(f"  Rows: {len(df):,}")
print(f"  Runs: {df['run_id'].nunique():,}")

# 3. Fault type distribution
print("\n[3] fault_type_code — rows")
print(df["fault_type_code"].value_counts().sort_index())

print("\n[3] fault_type_code — runs")
print(
    df.groupby("fault_type_code")["run_id"]
      .nunique()
      .sort_index()
)

# 4. Severity
print("\n[4] fault_severity")
if "fault_severity" in df.columns:
    print(f"  Non-null: {df['fault_severity'].notna().sum():,}")
    print(f"  Null:     {df['fault_severity'].isna().sum():,}")
    print(f"  Unique:   {df['fault_severity'].nunique(dropna=True):,}")
    print(f"  Min:      {df['fault_severity'].min()}")
    print(f"  Max:      {df['fault_severity'].max()}")
    print("\n  By fault class:")
    print(
        df.groupby("fault_type_code")["fault_severity"]
          .agg(["count", "nunique", "min", "max", "mean"])
          .to_string()
    )
else:
    print("  ERROR: fault_severity column does NOT exist.")

# 5. Healthy severity
print("\n[5] Healthy severity")
healthy = df[df["fault_type_code"] == 0]

if "fault_severity" in df.columns:
    print(
        healthy["fault_severity"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

# 6. Alternator voltage
print("\n[6] Alternator voltage check")

voltage_stats = (
    df.groupby("fault_type_code")["battery_voltage_v"]
      .agg(["count", "mean", "std", "min", "max", "median"])
)

print(voltage_stats.to_string())

# 7. Run-level class consistency
print("\n[7] Run-level class consistency")

codes_per_run = df.groupby("run_id")["fault_type_code"].nunique(dropna=False)
bad_runs = codes_per_run[codes_per_run != 1]

print(f"  Runs with exactly one class: {(codes_per_run == 1).sum():,}")
print(f"  Bad runs: {len(bad_runs):,}")

# 8. XGBoost availability
print("\n[8] XGBoost")

try:
    import xgboost
    print(f"  Installed version: {xgboost.__version__}")
except ImportError:
    print("  ERROR: xgboost is NOT installed.")

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)
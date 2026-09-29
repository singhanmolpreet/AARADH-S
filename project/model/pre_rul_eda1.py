import pandas as pd

df = pd.read_csv(
    "../../data/processed/rotax_combined_clean.csv",
    low_memory=False
)

# Number of distinct severity values within each run
severity_variation = (
    df.groupby("run_id")["fault_severity"]
    .agg(
        unique_values="nunique",
        first="first",
        last="last",
        minimum="min",
        maximum="max",
    )
)

print("=== Severity variation within runs ===")
print(severity_variation["unique_values"].describe())

print("\nRuns with changing severity:")
print(
    (severity_variation["unique_values"] > 1).sum()
)

print("\nRuns with constant severity:")
print(
    (severity_variation["unique_values"] == 1).sum()
)

print("\n=== Example runs ===")

for run_id in severity_variation.head(10).index:
    run = df[df["run_id"] == run_id]

    print(
        f"\nRun {run_id}: "
        f"fault={run['fault_type'].iloc[0]}, "
        f"severity values="
        f"{run['fault_severity'].unique()[:10]}"
    )
    
print("\n=== Severity by run and fault ===")

run_level = (
    df.groupby(
        ["run_id", "fault_type_code", "fault_type"]
    )
    .agg(
        severity=("fault_severity", "first"),
        n_unique_severity=("fault_severity", "nunique"),
        n_rows=("fault_severity", "size"),
        duration_s=("time_s", lambda x: x.max() - x.min()),
    )
    .reset_index()
)

print(
    run_level[
        run_level["fault_type_code"] != 0
    ].head(30).to_string(index=False)
)
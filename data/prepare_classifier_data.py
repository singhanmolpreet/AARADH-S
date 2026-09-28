import pandas as pd

# ============================================================
# CONFIG
# ============================================================

INPUT = "processed/rotax_combined_clean.csv"
OUTPUT = "processed/rotax_classifier_train.csv"

HEALTHY = 0

ENV_COLS = [
    "altitude_m",
    "ambient_temp_c",
    "air_density_kgm3",
]

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

print("=" * 70)
print("PREPARING CLASSIFIER DATASET")
print("=" * 70)

print(f"Input rows: {len(df):,}")
print(f"Input runs: {df.run_id.nunique():,}")


# ============================================================
# FIND ENVIRONMENTAL ENVELOPE OF FAULTY DATA
# ============================================================

faulty = df[df["fault_type_code"] != HEALTHY].copy()
healthy = df[df["fault_type_code"] == HEALTHY].copy()

print("\nFaulty environmental envelope:")
print("-" * 70)

limits = {}

for col in ENV_COLS:

    minimum = faulty[col].min()
    maximum = faulty[col].max()

    limits[col] = {
        "min": minimum,
        "max": maximum,
    }

    print(
        f"{col:25s}: "
        f"{minimum:.4f} → {maximum:.4f}"
    )


# ============================================================
# RESTRICT HEALTHY DATA
# ============================================================

healthy_mask = pd.Series(
    True,
    index=healthy.index
)

for col in ENV_COLS:

    minimum = limits[col]["min"]
    maximum = limits[col]["max"]

    healthy_mask &= (
        healthy[col] >= minimum
    ) & (
        healthy[col] <= maximum
    )

healthy_filtered = healthy[healthy_mask].copy()


# ============================================================
# COMBINE
# ============================================================

classifier_df = pd.concat(
    [
        healthy_filtered,
        faulty,
    ],
    ignore_index=True
)

# Preserve deterministic ordering
classifier_df = classifier_df.sort_values(
    ["run_id", "fault_type_code"]
).reset_index(drop=True)


# ============================================================
# REPORT
# ============================================================

print("\nHealthy rows:")
print(f"Before filtering : {len(healthy):,}")
print(f"After filtering  : {len(healthy_filtered):,}")
print(
    f"Removed          : "
    f"{len(healthy) - len(healthy_filtered):,}"
)

print("\nFaulty rows:")
print(f"Kept             : {len(faulty):,}")

print("\nFinal dataset:")
print(f"Rows             : {len(classifier_df):,}")
print(f"Unique run_ids   : {classifier_df.run_id.nunique():,}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")
print(
    classifier_df["fault_type_code"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# VERIFY ENVIRONMENTAL RANGES
# ============================================================

print("\nFinal environmental ranges:")
print("-" * 70)

for col in ENV_COLS:

    print(
        f"\n{col}"
    )

    print(
        classifier_df
        .groupby("fault_type_code")[col]
        .agg(["min", "max"])
        .to_string()
    )


# ============================================================
# VERIFY NO HEALTHY ROWS OUTSIDE FAULTY ENVELOPE
# ============================================================

print("\nVerification:")
print("-" * 70)

for col in ENV_COLS:

    minimum = limits[col]["min"]
    maximum = limits[col]["max"]

    bad = classifier_df[
        (classifier_df["fault_type_code"] == HEALTHY)
        & (
            (classifier_df[col] < minimum)
            | (classifier_df[col] > maximum)
        )
    ]

    print(
        f"{col:25s}: "
        f"{len(bad):,} healthy rows outside faulty range"
    )


# ============================================================
# SAVE NEW DATASET
# ============================================================

classifier_df.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"Saved: {OUTPUT}")
print("Original dataset was NOT modified.")
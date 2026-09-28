import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

PATH = "data/processed/rotax_combined_clean.csv"

FAULT_CLASS = 7
HEALTHY_CLASS = 0
N_BINS = 5

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(PATH)

print("=" * 80)
print("ROTAX DATA AUDIT")
print("=" * 80)

print(f"\nFile: {PATH}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Unique run_ids: {df['run_id'].nunique():,}")

print("\nFault type counts:")
print(df["fault_type_code"].value_counts().sort_index())


# ============================================================
# BASIC COLUMN CHECK
# ============================================================

required = [
    "fault_type_code",
    "battery_voltage_v",
    "rpm",
    "fault_severity",
    "altitude_m",
    "ambient_temp_c",
    "air_density_kgm3",
    "mission_phase",
    "run_id",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        "Missing required columns:\n" + "\n".join(missing)
    )


# ============================================================
# HELPERS
# ============================================================

def summary_stats(series):
    """Return min / median / p95 / max."""
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        return {
            "min": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    return {
        "min": s.min(),
        "median": s.median(),
        "p95": s.quantile(0.95),
        "max": s.max(),
    }


def print_stats(title, data):
    print(f"\n{title}")
    print("-" * len(title))

    stats = summary_stats(data)

    print(f"min    : {stats['min']:.4f}")
    print(f"median : {stats['median']:.4f}")
    print(f"p95    : {stats['p95']:.4f}")
    print(f"max    : {stats['max']:.4f}")


def make_quantile_bins(series, n=5):
    """
    Create approximately equal-frequency bins.
    Duplicated edges are dropped so the function
    remains safe when many values are identical.
    """
    try:
        return pd.qcut(
            series,
            q=n,
            duplicates="drop"
        )
    except ValueError:
        return pd.Series(
            ["unavailable"] * len(series),
            index=series.index
        )


# ============================================================
# SECTION 1
# BATTERY VOLTAGE — CLASS 7 VS CLASS 0
# ============================================================

print("\n\n")
print("=" * 80)
print("SECTION 1 — BATTERY VOLTAGE: CLASS 7 VS CLASS 0")
print("=" * 80)

vdf = df[
    df["fault_type_code"].isin([HEALTHY_CLASS, FAULT_CLASS])
].copy()

for cls in [HEALTHY_CLASS, FAULT_CLASS]:

    subset = vdf[vdf["fault_type_code"] == cls]

    label = "HEALTHY (0)" if cls == 0 else "FAULT CLASS 7"

    print_stats(
        f"{label} — battery_voltage_v",
        subset["battery_voltage_v"]
    )


# ------------------------------------------------------------
# Overall comparison
# ------------------------------------------------------------

print("\nOverall voltage comparison:")
print("-" * 40)

overall_voltage = (
    vdf.groupby("fault_type_code")["battery_voltage_v"]
    .agg(
        count="count",
        min="min",
        median="median",
        p95=lambda x: x.quantile(0.95),
        max="max",
    )
)

print(overall_voltage)


# ============================================================
# VOLTAGE BY RPM — 5 BINS
# ============================================================

print("\n\nBattery voltage by RPM — 5 quantile bins")
print("-" * 80)

vdf["rpm_bin"] = make_quantile_bins(vdf["rpm"], N_BINS)

rpm_voltage = (
    vdf.groupby(
        ["rpm_bin", "fault_type_code"],
        observed=True
    )["battery_voltage_v"]
    .agg(
        count="count",
        min="min",
        median="median",
        p95=lambda x: x.quantile(0.95),
        max="max",
    )
)

print(rpm_voltage.to_string())


# ============================================================
# VOLTAGE BY FAULT SEVERITY — 5 BINS
# ============================================================

print("\n\nBattery voltage by fault_severity — 5 quantile bins")
print("-" * 80)

# Severity may be NaN for healthy rows.
# For the requested comparison, bin only rows where severity exists.
severity_mask = vdf["fault_severity"].notna()

vdf.loc[severity_mask, "severity_bin"] = make_quantile_bins(
    vdf.loc[severity_mask, "fault_severity"],
    N_BINS
)

severity_voltage = (
    vdf[severity_mask]
    .groupby(
        ["severity_bin", "fault_type_code"],
        observed=True
    )["battery_voltage_v"]
    .agg(
        count="count",
        min="min",
        median="median",
        p95=lambda x: x.quantile(0.95),
        max="max",
    )
)

print(severity_voltage.to_string())


# ============================================================
# >20V ANALYSIS
# ============================================================

print("\n\nRows with battery_voltage_v > 20V")
print("-" * 80)

high_v = vdf[vdf["battery_voltage_v"] > 20].copy()

print(f"Total rows >20V: {len(high_v):,}")

print("\nBy fault_type_code:")
print(
    high_v["fault_type_code"]
    .value_counts()
    .sort_index()
)

print("\nBy fault_severity:")
if high_v["fault_severity"].notna().any():
    print(
        high_v["fault_severity"]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.95])
        .to_string()
    )
else:
    print("No non-null fault_severity values.")


print("\nBy mission_phase:")
print(
    high_v["mission_phase"]
    .value_counts(dropna=False)
    .to_string()
)


print("\nBy run_id — top 20:")
run_high = (
    high_v["run_id"]
    .value_counts()
    .head(20)
)

print(run_high.to_string())


# ------------------------------------------------------------
# >20V by severity
# ------------------------------------------------------------

print("\n>20V rows by fault severity:")
print(
    high_v.groupby("fault_severity", dropna=False)
    .size()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# Correlations
# ------------------------------------------------------------

print("\nCorrelation of battery_voltage_v >20V with severity/rpm")
print("-" * 80)

# Binary indicator: 1 if voltage >20V, otherwise 0
vdf["voltage_above_20"] = (
    vdf["battery_voltage_v"] > 20
).astype(int)

corr_cols = [
    "voltage_above_20",
    "fault_severity",
    "rpm",
]

print(
    vdf[corr_cols]
    .corr(numeric_only=True)["voltage_above_20"]
    .sort_values(ascending=False)
    .to_string()
)


# ============================================================
# CLASS 7 BELOW HEALTHY VOLTAGE RANGE AT SAME RPM
# ============================================================

print("\n\nClass-7 rows below healthy voltage range at the same RPM")
print("-" * 80)

healthy = vdf[
    vdf["fault_type_code"] == HEALTHY_CLASS
].copy()

fault7 = vdf[
    vdf["fault_type_code"] == FAULT_CLASS
].copy()

# Create RPM bins based on combined class 0 + class 7 data.
combined_rpm_bins = make_quantile_bins(
    vdf["rpm"],
    N_BINS
)

vdf["comparison_rpm_bin"] = combined_rpm_bins

healthy_rpm = (
    vdf[vdf["fault_type_code"] == HEALTHY_CLASS]
    .groupby("comparison_rpm_bin", observed=True)
    ["battery_voltage_v"]
    .agg(
        healthy_min="min",
        healthy_median="median",
        healthy_p95=lambda x: x.quantile(0.95),
        healthy_max="max",
    )
)

fault7_rpm = (
    vdf[vdf["fault_type_code"] == FAULT_CLASS]
    .copy()
)

fault7_rpm = fault7_rpm.merge(
    healthy_rpm,
    left_on="comparison_rpm_bin",
    right_index=True,
    how="left"
)

fault7_rpm["below_healthy_min"] = (
    fault7_rpm["battery_voltage_v"]
    < fault7_rpm["healthy_min"]
)

below = fault7_rpm[
    fault7_rpm["below_healthy_min"]
].copy()

print(
    f"Class-7 rows below healthy minimum in same RPM bin: "
    f"{len(below):,}"
)

if len(below) > 0:

    print("\nBreakdown by RPM bin:")
    print(
        below.groupby("comparison_rpm_bin", observed=True)
        .size()
        .to_string()
    )

    print("\nExample rows:")
    cols = [
        "run_id",
        "rpm",
        "battery_voltage_v",
        "fault_severity",
        "mission_phase",
        "altitude_m",
    ]

    print(
        below[cols]
        .head(20)
        .to_string(index=False)
    )

else:
    print(
        "WARNING: No class-7 rows fall below the healthy "
        "minimum within their RPM bin."
    )


# ============================================================
# SECTION 2
# ENVIRONMENTAL RANGE AUDIT
# ============================================================

print("\n\n")
print("=" * 80)
print("SECTION 2 — ALTITUDE / TEMPERATURE / AIR DENSITY")
print("=" * 80)

environment_cols = [
    "altitude_m",
    "ambient_temp_c",
    "air_density_kgm3",
]

for col in environment_cols:

    print(f"\n{col}")
    print("-" * 60)

    stats = (
        df.groupby("fault_type_code")[col]
        .agg(
            count="count",
            min="min",
            median="median",
            max="max",
        )
    )

    print(stats.to_string())


# ============================================================
# RANGE GAPS: HEALTHY EXISTS WHERE FAULTY DOES NOT
# ============================================================

print("\n\nEnvironment range gaps")
print("-" * 80)

for col in environment_cols:

    healthy_values = df.loc[
        df["fault_type_code"] == HEALTHY_CLASS,
        col
    ].dropna()

    faulty_values = df.loc[
        df["fault_type_code"] != HEALTHY_CLASS,
        col
    ].dropna()

    if len(healthy_values) == 0 or len(faulty_values) == 0:
        print(f"\n{col}: insufficient data")
        continue

    healthy_min = healthy_values.min()
    healthy_max = healthy_values.max()

    faulty_min = faulty_values.min()
    faulty_max = faulty_values.max()

    print(f"\n{col}")
    print(f"Healthy range : {healthy_min:.4f} → {healthy_max:.4f}")
    print(f"Faulty range  : {faulty_min:.4f} → {faulty_max:.4f}")

    # Healthy values below faulty minimum
    healthy_below_fault = healthy_values[
        healthy_values < faulty_min
    ]

    # Healthy values above faulty maximum
    healthy_above_fault = healthy_values[
        healthy_values > faulty_max
    ]

    if len(healthy_below_fault) > 0:

        print(
            f"FLAG: healthy rows exist BELOW faulty minimum: "
            f"{len(healthy_below_fault):,}"
        )

        print(
            f"      healthy-only lower gap: "
            f"{healthy_below_fault.min():.4f} → "
            f"{healthy_below_fault.max():.4f}"
        )

    if len(healthy_above_fault) > 0:

        print(
            f"FLAG: healthy rows exist ABOVE faulty maximum: "
            f"{len(healthy_above_fault):,}"
        )

        print(
            f"      healthy-only upper gap: "
            f"{healthy_above_fault.min():.4f} → "
            f"{healthy_above_fault.max():.4f}"
        )

    if (
        len(healthy_below_fault) == 0
        and len(healthy_above_fault) == 0
    ):
        print("No healthy-only range outside faulty range.")


# ============================================================
# ALTITUDE-SPECIFIC GAP
# ============================================================

print("\n\nAltitude gap details")
print("-" * 80)

healthy_alt = df.loc[
    df["fault_type_code"] == HEALTHY_CLASS,
    "altitude_m"
].dropna()

faulty_alt = df.loc[
    df["fault_type_code"] != HEALTHY_CLASS,
    "altitude_m"
].dropna()

healthy_above_faulty_ceiling = healthy_alt[
    healthy_alt > faulty_alt.max()
]

print(f"Healthy maximum altitude: {healthy_alt.max():.2f} m")
print(f"Faulty maximum altitude : {faulty_alt.max():.2f} m")
print(
    f"Healthy rows above faulty ceiling: "
    f"{len(healthy_above_faulty_ceiling):,}"
)

if len(healthy_above_faulty_ceiling) > 0:
    print(
        f"Healthy-only altitude range: "
        f"{healthy_above_faulty_ceiling.min():.2f} → "
        f"{healthy_above_faulty_ceiling.max():.2f} m"
    )


# ============================================================
# SECTION 3
# MISSION PHASE MIX
# ============================================================

print("\n\n")
print("=" * 80)
print("SECTION 3 — MISSION PHASE MIX + RUN LENGTH")
print("=" * 80)

phase_counts = pd.crosstab(
    df["fault_type_code"],
    df["mission_phase"]
)

print("\nRaw phase counts:")
print(phase_counts.to_string())


# Row-normalised crosstab
phase_mix = pd.crosstab(
    df["fault_type_code"],
    df["mission_phase"],
    normalize="index"
)

print("\n\nRow-normalised fault_type_code × mission_phase")
print("-" * 80)

print(
    phase_mix
    .round(4)
    .to_string()
)


# Percentage version
phase_percent = phase_mix * 100

print("\n\nPercentage version:")
print(
    phase_percent
    .round(2)
    .to_string()
)


# ============================================================
# COMPARE EVERY CLASS AGAINST HEALTHY
# ============================================================

print("\n\nPhase mix difference from healthy")
print("-" * 80)

if HEALTHY_CLASS not in phase_mix.index:

    print("Healthy class 0 not found.")

else:

    healthy_phase = phase_mix.loc[HEALTHY_CLASS]

    phase_flags = []

    for cls in phase_mix.index:

        if cls == HEALTHY_CLASS:
            continue

        class_phase = phase_mix.loc[cls]

        # Align columns in case a class lacks a phase
        class_phase, healthy_aligned = class_phase.align(
            healthy_phase,
            fill_value=0
        )

        delta = (class_phase - healthy_aligned) * 100

        for phase, diff in delta.items():

            if abs(diff) > 10:

                phase_flags.append({
                    "fault_type_code": cls,
                    "mission_phase": phase,
                    "class_percent": class_phase[phase] * 100,
                    "healthy_percent": healthy_aligned[phase] * 100,
                    "difference_pp": diff,
                })

    if phase_flags:

        phase_flags_df = pd.DataFrame(phase_flags)

        print(
            phase_flags_df
            .sort_values(
                ["fault_type_code", "mission_phase"]
            )
            .round(2)
            .to_string(index=False)
        )

    else:

        print(
            "No fault class differs from healthy by "
            "more than 10 percentage points in any phase."
        )


# ============================================================
# RUN LENGTH
# ============================================================

print("\n\nMean run length per fault_type_code")
print("-" * 80)

run_lengths = (
    df.groupby(
        ["run_id", "fault_type_code"]
    )
    .size()
    .reset_index(name="run_length")
)

mean_run_length = (
    run_lengths
    .groupby("fault_type_code")["run_length"]
    .agg(
        number_of_runs="count",
        mean="mean",
        median="median",
        min="min",
        max="max",
    )
)

print(
    mean_run_length
    .round(2)
    .to_string()
)


# ============================================================
# SECTION 4
# AUTOMATED RECOMMENDATION SIGNALS
# ============================================================

print("\n\n")
print("=" * 80)
print("SECTION 4 — RECOMMENDATION SIGNALS")
print("=" * 80)

# ------------------------------------------------------------
# Signal 1: altitude gap
# ------------------------------------------------------------

altitude_gap = (
    healthy_alt.max() > faulty_alt.max()
)

print("\nAltitude gap:")
if altitude_gap:
    print(
        "YES — healthy data extends beyond the faulty altitude ceiling."
    )
else:
    print(
        "NO — healthy data does not extend beyond faulty altitude ceiling."
    )


# ------------------------------------------------------------
# Signal 2: class 7 collapse
# ------------------------------------------------------------

class7_below_healthy = len(below) > 0

print("\nClass-7 voltage collapse below healthy RPM-matched minimum:")

if class7_below_healthy:
    print(
        "YES — class 7 contains rows below the healthy voltage "
        "minimum at the same RPM."
    )
else:
    print(
        "NO — no class-7 rows fall below the healthy minimum "
        "within the RPM bins."
    )


# ------------------------------------------------------------
# Signal 3: phase mismatch
# ------------------------------------------------------------

phase_mismatch = bool(phase_flags)

print("\nPhase-mix mismatch >10 percentage points:")

if phase_mismatch:
    print(
        "YES — at least one fault class differs from healthy "
        "by >10 percentage points in a mission phase."
    )
else:
    print(
        "NO — no class differs from healthy by >10 percentage points."
    )


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

print("\n\n")
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)

print("""
Options:

(a) Keep as is
(b) Restrict healthy rows to the faulty altitude/temperature range
    for classifier training
(c) Drop environment columns from classifier features
(d) Regenerate class 7 at source in Simulink

Decision signals:
""")

if altitude_gap:
    print(
        "→ (b) is strongly indicated because healthy data covers "
        "environmental conditions not represented by faulty data."
    )
else:
    print(
        "→ (b) is not required based on altitude range alone."
    )


if class7_below_healthy:
    print(
        "→ Class 7 shows the expected downward/collapse behaviour "
        "relative to healthy voltage within RPM bins."
    )
else:
    print(
        "→ (d) should be investigated: class 7 does NOT show "
        "voltage below the healthy floor within RPM bins."
    )


if phase_mismatch:
    print(
        "→ Phase imbalance should also be addressed before "
        "using phase/environment variables as classifier signals."
    )
else:
    print(
        "→ No >10 percentage-point phase imbalance detected."
    )


print("\nSuggested interpretation:")
print(
    "Do not automatically drop environment columns just because "
    "the healthy set has a wider range. First remove the "
    "healthy-only environmental region from classifier training "
    "if the classifier is intended to distinguish faults under "
    "the same operating envelope."
)

print(
    "If class 7 lacks the documented voltage-collapse behaviour, "
    "fix the source Simulink fault model rather than trying to "
    "repair the behaviour statistically in Python."
)

print("\nAudit complete. No data was modified.")
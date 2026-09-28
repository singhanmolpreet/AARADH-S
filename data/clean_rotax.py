"""
clean_rotax.py
==============
Cleaning pipeline for rotax_dataset_healthy.csv and rotax_dataset_faulty.csv.

Execution order
---------------
1.  Load both raw CSVs.
2.  Drop TargetSeverity from the faulty file (exact duplicate of FaultSeverity).
3.  Drop the first row (min Time_s) of every RunID — simulation-init artifact.
4.  Clip MeasuredFuelFlow and MeasuredOilPressure to floor 0 (physically non-negative).
    -> Report rows affected per file.
5.  Audit MeasuredOilTemp negatives AFTER step 3 — deliberately NOT clipped.
    -> Report count + AmbientTemp values.
6.  Audit MeasuredVoltage > 20 V — plot histogram, report row count.
    -> Script STOPS here and waits for your decision.
7.  Concatenate both files, rename columns to snake_case spec, add 'source' tag.
8.  Save to data/processed/rotax_combined_clean.csv.
9.  Print per-fault_type_code and per-mission_phase row counts.

What this script does NOT handle
---------------------------------
- MeasuredVoltage outliers  -> shown in histogram, user decides threshold.
- MeasuredCHT / EGT limits  -> not requested; left raw.
- Missing-value imputation  -> no NaN filling performed.
- Train/val/test splits      -> not done here.
- Feature engineering        -> not done here.
"""

import sys
import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — saves PNG, no GUI needed
import matplotlib.pyplot as plt

# -- Paths ----------------------------------------------------------------------
BASE      = pathlib.Path(__file__).parent
RAW       = BASE / "raw"
PROCESSED = BASE / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

HEALTHY_CSV = RAW / "rotax_dataset_healthy.csv"
FAULTY_CSV  = RAW / "rotax_dataset_faulty.csv"
OUT_CSV     = PROCESSED / "rotax_combined_clean.csv"
VOLT_HIST   = PROCESSED / "voltage_above_20v_distribution.png"

# -- Column rename map (original -> final snake_case) ---------------------------
COL_RENAME = {
    "Time_s":                 "time_s",
    "MeasuredRPM":            "rpm",
    "MeasuredCHT":            "cht_c",
    "MeasuredEGT":            "egt_c",
    "MeasuredOilTemp":        "oil_temp_c",
    "MeasuredOilPressure":    "oil_pressure_bar",
    "MeasuredVoltage":        "battery_voltage_v",
    "MeasuredVibration":      "vibration_rms_g",
    "MeasuredFuelFlow":       "fuel_flow_lph",
    "FaultFlag":              "fault_flag",          # kept for reference, not in final spec
    "FaultType":              "fault_type_code",
    "FaultSeverity":          "fault_severity",
    "InjectionTiming":        "injection_timing_deg",
    "Throttle":               "throttle_frac",
    "Power":                  "power_kw",
    "AmbientTemp":            "ambient_temp_c",
    "AirPressure":            "air_pressure_pa",
    "AirDensity":             "air_density_kgm3",
    "RunID":                  "run_id",
    "MissionProfile":         "mission_phase",
    "Altitude_m":             "altitude_m",
    "TempOffset_C":           "temp_offset_c",       # kept as supplementary
    "Label":                  "fault_type",
}

# Final columns to keep (in order) — FaultFlag / TempOffset_C retained as extras
FINAL_COLS = [
    "rpm", "cht_c", "egt_c", "oil_temp_c", "oil_pressure_bar",
    "battery_voltage_v", "vibration_rms_g", "fuel_flow_lph",
    "injection_timing_deg", "throttle_frac", "power_kw",
    "ambient_temp_c", "air_pressure_pa", "air_density_kgm3",
    "altitude_m", "mission_phase",
    "fault_type_code", "fault_type", "fault_severity",
    "run_id", "time_s",
    "fault_flag", "temp_offset_c",   # supplementary — easy to drop later
    "source",                         # 'healthy' | 'faulty'
]

SEP = "=" * 70


# ==============================================================================
# 1 · Load raw CSVs
# ==============================================================================
print(SEP)
print("STEP 1 · Loading raw CSVs")
print(SEP)

healthy = pd.read_csv(HEALTHY_CSV, low_memory=False)
faulty  = pd.read_csv(FAULTY_CSV,  low_memory=False)

print(f"  healthy : {len(healthy):>9,} rows × {healthy.shape[1]} cols")
print(f"  faulty  : {len(faulty):>9,} rows × {faulty.shape[1]} cols")


# ==============================================================================
# 2 · Drop TargetSeverity (exact duplicate of FaultSeverity, faulty only)
# ==============================================================================
print(f"\n{SEP}")
print("STEP 2 · Dropping TargetSeverity (faulty only)")
print(SEP)

if "TargetSeverity" in faulty.columns:
    match = (faulty["TargetSeverity"] == faulty["FaultSeverity"]).all()
    print(f"  TargetSeverity == FaultSeverity in every row? {match}")
    faulty = faulty.drop(columns=["TargetSeverity"])
    print("  Dropped TargetSeverity.")
else:
    print("  TargetSeverity not found — nothing to drop.")


# ==============================================================================
# 3 · Drop first row of each RunID (simulation-init artifact)
# ==============================================================================
print(f"\n{SEP}")
print("STEP 3 · Dropping min-Time_s row per RunID (init artifact)")
print(SEP)

def drop_init_rows(df: pd.DataFrame, label: str) -> pd.DataFrame:
    init_idx = df.groupby("RunID")["Time_s"].idxmin()
    n_init   = len(init_idx)
    df_clean = df.drop(index=init_idx)
    n_runs   = df["RunID"].nunique()
    print(f"  [{label}] runs={n_runs:,}  init rows dropped={n_init:,}  "
          f"remaining={len(df_clean):,}")
    return df_clean.reset_index(drop=True)

healthy = drop_init_rows(healthy, "healthy")
faulty  = drop_init_rows(faulty,  "faulty")


# ==============================================================================
# 4 · Clip MeasuredFuelFlow and MeasuredOilPressure to floor 0
# ==============================================================================
print(f"\n{SEP}")
print("STEP 4 · Clipping FuelFlow & OilPressure to >= 0")
print(SEP)

def clip_floor(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    neg_mask  = df[col] < 0
    n_neg     = neg_mask.sum()
    if n_neg:
        df[col] = df[col].clip(lower=0)
    print(f"  [{label}] {col}: {n_neg:,} rows clipped to 0  "
          f"(was < 0; min_before clip shown in audit above)")
    return df

for col in ["MeasuredFuelFlow", "MeasuredOilPressure"]:
    neg_healthy = (healthy[col] < 0).sum()
    neg_faulty  = (faulty[col]  < 0).sum()
    print(f"\n  {col}")
    print(f"    healthy: {neg_healthy:,} rows < 0  ->  all clipped to 0")
    print(f"    faulty : {neg_faulty:,}  rows < 0  ->  all clipped to 0")
    healthy[col] = healthy[col].clip(lower=0)
    faulty[col]  = faulty[col].clip(lower=0)


# ==============================================================================
# 5 · Audit MeasuredOilTemp negatives (deliberately NOT clipped)
# ==============================================================================
print(f"\n{SEP}")
print("STEP 5 · Negative MeasuredOilTemp audit (NOT clipped — cold-start real)")
print(SEP)

for df, label in [(healthy, "healthy"), (faulty, "faulty")]:
    neg_mask = df["MeasuredOilTemp"] < 0
    n_neg    = neg_mask.sum()
    print(f"\n  [{label}] rows with OilTemp < 0 : {n_neg:,}")
    if n_neg:
        neg_sub = df.loc[neg_mask, ["MeasuredOilTemp", "AmbientTemp"]]
        print(f"    OilTemp range : [{neg_sub['MeasuredOilTemp'].min():.2f}, "
              f"{neg_sub['MeasuredOilTemp'].max():.2f}]°C")
        print(f"    AmbientTemp range : [{neg_sub['AmbientTemp'].min():.2f}, "
              f"{neg_sub['AmbientTemp'].max():.2f}]°C")
        # Bin AmbientTemp to show distribution of cold-start rows
        bins  = [-50, -30, -20, -10, -5, 0, 5, 15, 50]
        labels_b = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins)-1)]
        ambient_bin = pd.cut(neg_sub["AmbientTemp"], bins=bins, labels=labels_b,
                             right=False, include_lowest=True)
        print(f"    AmbientTemp histogram for negative-OilTemp rows:")
        for b, cnt in ambient_bin.value_counts().sort_index().items():
            print(f"      {b:>12s} : {cnt:,}")
    else:
        print("    -> No negative OilTemp rows.")


# ==============================================================================
# 6 · Voltage audit — histogram above 20 V, then STOP
# ==============================================================================
print(f"\n{SEP}")
print("STEP 6 · MeasuredVoltage > 20 V audit & histogram")
print(SEP)

# Combine raw (pre-rename) just for the voltage audit
volt_all = pd.concat([
    healthy[["MeasuredVoltage"]].assign(source="healthy"),
    faulty[["MeasuredVoltage"]].assign(source="faulty"),
], ignore_index=True)

above20 = volt_all[volt_all["MeasuredVoltage"] > 20]

print(f"\n  Total rows > 20 V : {len(above20):,}")
print(f"    healthy : {(above20['source']=='healthy').sum():,}")
print(f"    faulty  : {(above20['source']=='faulty').sum():,}")
print(f"\n  Voltage > 20 V statistics:")
print(above20["MeasuredVoltage"].describe().to_string())

# Percentile table
pcts = [50, 75, 90, 95, 99, 99.5, 99.9, 100]
print(f"\n  Percentiles of rows > 20 V:")
for p in pcts:
    v = np.percentile(above20["MeasuredVoltage"], p)
    print(f"    p{p:5.1f} : {v:.4f} V")

# -- Histogram -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MeasuredVoltage Distribution (rows > 20 V)", fontsize=14, fontweight="bold")

for ax, src, color in zip(axes, ["healthy", "faulty"], ["steelblue", "tomato"]):
    subset = above20[above20["source"] == src]["MeasuredVoltage"]
    ax.hist(subset, bins=80, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_title(f"{src.capitalize()}  (n={len(subset):,})", fontsize=12)
    ax.set_xlabel("MeasuredVoltage (V)", fontsize=10)
    ax.set_ylabel("Row count", fontsize=10)
    ax.axvline(subset.median(), color="black", linestyle="--", linewidth=1.2,
               label=f"median={subset.median():.2f} V")
    ax.axvline(subset.quantile(0.99), color="red", linestyle=":", linewidth=1.2,
               label=f"p99={subset.quantile(0.99):.2f} V")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(VOLT_HIST, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Histogram saved -> {VOLT_HIST}")

print(f"""
{'-'*70}
[!]  STOPPING HERE as requested.

The voltage distribution is saved to:
  {VOLT_HIST}

Once you decide on a voltage threshold / treatment (e.g. clip to X V, flag
as outlier, or leave as-is), re-run this script with the flag:

    python data/clean_rotax.py --continue-after-voltage --volt-cap <VALUE>

or edit the VOLT_CAP variable at the top of the script.
{'-'*70}
""")

# ==============================================================================
# Only proceed if --continue-after-voltage flag is present
# ==============================================================================
import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--continue-after-voltage", action="store_true")
parser.add_argument("--volt-cap", type=float, default=None,
                    help="Hard cap for MeasuredVoltage (e.g. 30.0)")
args, _ = parser.parse_known_args()

if not args.continue_after_voltage:
    sys.exit(0)

# ==============================================================================
# 7 · (Optional) Apply voltage cap if provided
# ==============================================================================
if args.volt_cap is not None:
    print(f"\n{SEP}")
    print(f"STEP 7 · Capping MeasuredVoltage at {args.volt_cap} V")
    print(SEP)
    for df, label in [(healthy, "healthy"), (faulty, "faulty")]:
        over = (df["MeasuredVoltage"] > args.volt_cap).sum()
        df["MeasuredVoltage"] = df["MeasuredVoltage"].clip(upper=args.volt_cap)
        print(f"  [{label}] {over:,} rows capped to {args.volt_cap} V")
else:
    print("\nSTEP 7 · No --volt-cap supplied; MeasuredVoltage left unchanged.")


# ==============================================================================
# 8 · Concatenate, rename columns, add source tag
# ==============================================================================
print(f"\n{SEP}")
print("STEP 8 · Concatenating & renaming columns")
print(SEP)

healthy["source"] = "healthy"
faulty["source"]  = "faulty"

combined = pd.concat([healthy, faulty], ignore_index=True)
print(f"  Combined rows (pre-rename): {len(combined):,}")

# Rename
combined = combined.rename(columns=COL_RENAME)

# Select / reorder final columns (silently skip any that aren't present)
available = [c for c in FINAL_COLS if c in combined.columns]
missing   = [c for c in FINAL_COLS if c not in combined.columns]
if missing:
    print(f"  NOTE: columns not found and skipped: {missing}")
combined = combined[available]

print(f"  Final column set ({len(available)} cols): {available}")


# ==============================================================================
# 9 · Save & report
# ==============================================================================
print(f"\n{SEP}")
print("STEP 9 · Saving & row-count report")
print(SEP)

combined.to_csv(OUT_CSV, index=False)
print(f"\n  Saved -> {OUT_CSV}  ({len(combined):,} rows)")

print(f"\n  Rows per fault_type_code:")
ft_counts = combined["fault_type_code"].value_counts().sort_index()
for code, cnt in ft_counts.items():
    ft_label = combined.loc[combined["fault_type_code"] == code, "fault_type"].iloc[0]
    print(f"    {code:>2} ({ft_label:<20s}) : {cnt:>9,}")

print(f"\n  Rows per mission_phase:")
mp_counts = combined["mission_phase"].value_counts().sort_index()
for phase, cnt in mp_counts.items():
    print(f"    {phase:<25s} : {cnt:>9,}")

print(f"\n{SEP}")
print("Done.")
print(SEP)

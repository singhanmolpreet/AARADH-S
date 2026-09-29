"""
fault_classifier.py
===================

XGBoost fault classification + fault-severity regression for the
canonical Rotax 914 feature pipeline.

Models
------
1. Multi-class XGBoost classifier
   - Features: canonical to_feature_matrix() output
   - Target: fault_type_code
   - Fixed class map from docs/schemas.md
   - No LabelEncoder
   - Run-level 80/20 stratified split
   - Healthy class weight = 1.3
   - Fault class weights = 1.0
   - Evaluation ONLY on held-out runs

2. XGBoost regressor
   - Features: same canonical feature matrix
   - Target: fault_severity
   - Healthy severity = 0
   - health_score = 100 * (1 - severity_hat)
   - Evaluation ONLY on held-out runs

Artifacts
---------
project/model/models/
    fault_classifier.joblib
    fault_severity_regressor.joblib
    metadata.json
    holdout_runs.json

The script deliberately uses the canonical feature pipeline and does
not independently construct features.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor


# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project.model.features import (  # noqa: E402
    compute_features,
    feature_column_names,
    to_feature_matrix,
)


DATA_PATH = _REPO_ROOT / "data" / "processed" / "rotax_combined_clean.csv"
MODELS_DIR = _HERE / "models"

CLASSIFIER_PATH = MODELS_DIR / "fault_classifier.joblib"
REGRESSOR_PATH = MODELS_DIR / "fault_severity_regressor.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
HOLDOUT_RUNS_PATH = MODELS_DIR / "holdout_runs.json"


# ---------------------------------------------------------------------------
# Fixed schema
# ---------------------------------------------------------------------------

CLASS_MAP: Dict[int, str] = {
    0: "healthy",
    1: "misfire",
    2: "lubrication_issue",
    3: "sensor_drift",
    4: "overheating",
    5: "wastegate_fault",
    6: "injector_fault",
    7: "alternator_fault",
    8: "air_filter_blockage",
}

CLASS_CODES = list(CLASS_MAP.keys())

HEALTHY_CODE = 0
ALTERNATOR_CODE = 7

# Explicit interpretation of the requested 1.3x healthy weighting.
HEALTHY_CLASS_WEIGHT = 1.3
FAULT_CLASS_WEIGHT = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_required_columns(df: pd.DataFrame) -> None:
    required = {
        "run_id",
        "fault_type_code",
        "fault_severity",
        "battery_voltage_v",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )


def _validate_fault_schema(df: pd.DataFrame) -> None:
    """Validate the fixed fault-type and severity contract."""

    _assert_required_columns(df)

    # fault_type_code must be integer-like and within the fixed map.
    codes = set(df["fault_type_code"].dropna().astype(int).unique())

    unexpected = codes - set(CLASS_MAP)

    if unexpected:
        raise ValueError(
            f"Unexpected fault_type_code values: {sorted(unexpected)}. "
            f"Expected only {CLASS_CODES}."
        )

    if df["fault_type_code"].isna().any():
        raise ValueError("fault_type_code contains null values.")

    # Every run must contain exactly one fault class.
    codes_per_run = (
        df.groupby("run_id")["fault_type_code"]
        .nunique(dropna=False)
    )

    bad_runs = codes_per_run[codes_per_run != 1]

    if not bad_runs.empty:
        raise ValueError(
            f"{len(bad_runs)} run(s) contain multiple fault_type_code values."
        )

    # Severity must exist and be within [0, 1].
    if df["fault_severity"].isna().any():
        raise ValueError("fault_severity contains null values.")

    severity_min = float(df["fault_severity"].min())
    severity_max = float(df["fault_severity"].max())

    if severity_min < 0.0 or severity_max > 1.0:
        raise ValueError(
            "fault_severity must be within [0, 1]. "
            f"Observed range: [{severity_min}, {severity_max}]"
        )

    # Healthy must have severity exactly zero.
    healthy_severity = df.loc[
        df["fault_type_code"] == HEALTHY_CODE,
        "fault_severity",
    ]

    if not np.allclose(healthy_severity.to_numpy(), 0.0):
        raise ValueError(
            "Healthy rows must have fault_severity == 0."
        )


def _split_run_ids(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split at run level.

    Stratification happens on one fault_type_code label per run.
    No run can appear in both train and holdout.
    """

    run_table = (
        df.groupby("run_id", as_index=False)["fault_type_code"]
        .first()
    )

    train_runs, holdout_runs = train_test_split(
        run_table["run_id"].to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=run_table["fault_type_code"],
    )

    train_runs = np.sort(train_runs)
    holdout_runs = np.sort(holdout_runs)

    overlap = set(train_runs) & set(holdout_runs)

    if overlap:
        raise AssertionError(
            f"Run leakage detected: {len(overlap)} overlapping run IDs."
        )

    if len(train_runs) + len(holdout_runs) != len(run_table):
        raise AssertionError(
            "Run split does not cover all dataset runs."
        )

    return train_runs, holdout_runs


def _class_weights(y: pd.Series) -> np.ndarray:
    """
    Explicit requested weighting:

        healthy = 1.3
        every fault class = 1.0
    """

    return np.where(
        y.to_numpy() == HEALTHY_CODE,
        HEALTHY_CLASS_WEIGHT,
        FAULT_CLASS_WEIGHT,
    ).astype(np.float32)


def _json_safe(value):
    """Convert NumPy values to JSON-serializable Python values."""

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def _class_distribution(
    feats: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    """Return row and run counts per fault class."""

    result = {}

    for code, name in CLASS_MAP.items():
        subset = feats[feats["fault_type_code"] == code]

        result[str(code)] = {
            "name": name,
            "rows": int(len(subset)),
            "runs": int(subset["run_id"].nunique()),
        }

    return result


def _print_classifier_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """Print and return held-out classification metrics."""

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_CODES,
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASS_CODES,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_CODES,
    )

    print("\n" + "=" * 80)
    print("HELD-OUT FAULT CLASSIFICATION")
    print("=" * 80)

    print("\nPer-class metrics:")
    print(
        f"{'Code':>4}  "
        f"{'Class':<25} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'Support':>10}"
    )

    per_class = {}

    for i, code in enumerate(CLASS_CODES):
        name = CLASS_MAP[code]

        print(
            f"{code:>4}  "
            f"{name:<25} "
            f"{precision[i]:>10.4f} "
            f"{recall[i]:>10.4f} "
            f"{f1[i]:>10.4f} "
            f"{support[i]:>10,}"
        )

        per_class[str(code)] = {
            "name": name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    print(f"\nMacro-F1: {macro_f1:.6f}")

    print("\nConfusion matrix")
    print("Rows = true class, columns = predicted class")
    print()

    header = "      " + "".join(f"{code:>8}" for code in CLASS_CODES)
    print(header)

    for i, code in enumerate(CLASS_CODES):
        values = "".join(f"{v:>8,}" for v in cm[i])
        print(f"{code:>4}  {values}")

    # ---------------------------------------------------------------
    # Class 7 dedicated report
    # ---------------------------------------------------------------

    class7 = per_class[str(ALTERNATOR_CODE)]

    print("\n" + "-" * 80)
    print("CLASS 7 — ALTERNATOR_FAULT")
    print("-" * 80)

    print(f"Precision : {class7['precision']:.6f}")
    print(f"Recall    : {class7['recall']:.6f}")
    print(f"F1        : {class7['f1']:.6f}")
    print(f"Support   : {class7['support']:,}")

    # Class 7 row of the confusion matrix.
    class7_row = cm[ALTERNATOR_CODE]

    print("\nClass-7 confusion row:")
    for predicted_code, count in zip(CLASS_CODES, class7_row):
        if count:
            print(
                f"  predicted {predicted_code} "
                f"({CLASS_MAP[predicted_code]}): {count:,}"
            )

    return {
        "per_class": per_class,
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "labels": CLASS_CODES,
    }


def _analyze_class7_voltage(
    holdout_feats: pd.DataFrame,
) -> Dict:
    """
    Check whether class 7 is trivially separable using raw battery voltage.

    We deliberately do NOT claim trivial separability merely because
    class 7 has a lower mean voltage.

    The strict test here is whether the held-out class-7 voltage range
    is completely separated from the held-out non-class-7 voltage range.
    """

    class7_voltage = (
        holdout_feats.loc[
            holdout_feats["fault_type_code"] == ALTERNATOR_CODE,
            "battery_voltage_v",
        ]
        .dropna()
    )

    non7_voltage = (
        holdout_feats.loc[
            holdout_feats["fault_type_code"] != ALTERNATOR_CODE,
            "battery_voltage_v",
        ]
        .dropna()
    )

    if class7_voltage.empty or non7_voltage.empty:
        return {
            "available": False,
            "trivially_separable": False,
            "reason": "Insufficient held-out voltage data.",
        }

    class7_min = float(class7_voltage.min())
    class7_max = float(class7_voltage.max())

    non7_min = float(non7_voltage.min())
    non7_max = float(non7_voltage.max())

    # Complete interval separation.
    separated = (
        class7_max < non7_min
        or non7_max < class7_min
    )

    # Count the class-7 rows that fall below the minimum non-7 voltage.
    below_non7_min = int(
        (class7_voltage < non7_min).sum()
    )

    result = {
        "available": True,
        "trivially_separable": bool(separated),
        "class_7_voltage_min": class7_min,
        "class_7_voltage_max": class7_max,
        "non_class_7_voltage_min": non7_min,
        "non_class_7_voltage_max": non7_max,
        "class_7_rows_below_non7_min": below_non7_min,
        "class_7_rows": int(len(class7_voltage)),
    }

    print("\n" + "-" * 80)
    print("CLASS 7 VOLTAGE SEPARABILITY")
    print("-" * 80)

    print(
        f"Class 7 voltage range : "
        f"{class7_min:.6f} – {class7_max:.6f} V"
    )

    print(
        f"Non-class-7 range     : "
        f"{non7_min:.6f} – {non7_max:.6f} V"
    )

    if separated:
        print(
            "\nRESULT: Class 7 is trivially separable by the raw "
            "battery-voltage range in the held-out data."
        )
    else:
        print(
            "\nRESULT: Class 7 is NOT trivially range-separable by "
            "battery voltage alone; the voltage ranges overlap."
        )

    print(
        f"Class-7 rows below non-class-7 minimum: "
        f"{below_non7_min:,} / {len(class7_voltage):,}"
    )

    return result


def _print_regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    holdout_feats: pd.DataFrame,
) -> Dict:
    """Print and return held-out severity-regression metrics."""

    y_pred = np.clip(y_pred, 0.0, 1.0)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    health_score = 100.0 * (1.0 - y_pred)

    print("\n" + "=" * 80)
    print("HELD-OUT FAULT-SEVERITY REGRESSION")
    print("=" * 80)

    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.6f}")

    print("\nHealth-score definition:")
    print("  health_score = 100 * (1 - severity_hat)")

    print(
        f"\nPredicted severity range: "
        f"{y_pred.min():.6f} – {y_pred.max():.6f}"
    )

    print(
        f"Predicted health-score range: "
        f"{health_score.min():.2f} – {health_score.max():.2f}"
    )

    # ---------------------------------------------------------------
    # Class 7 regression metrics separately
    # ---------------------------------------------------------------

    class7_mask = (
        holdout_feats["fault_type_code"].to_numpy()
        == ALTERNATOR_CODE
    )

    class7_true = y_true[class7_mask]
    class7_pred = y_pred[class7_mask]

    if len(class7_true):
        class7_mae = mean_absolute_error(
            class7_true,
            class7_pred,
        )

        class7_rmse = np.sqrt(
            mean_squared_error(
                class7_true,
                class7_pred,
            )
        )

        print("\n" + "-" * 80)
        print("CLASS 7 — ALTERNATOR_FAULT SEVERITY")
        print("-" * 80)
        print(f"Rows : {len(class7_true):,}")
        print(f"MAE  : {class7_mae:.6f}")
        print(f"RMSE : {class7_rmse:.6f}")

        class7_metrics = {
            "rows": int(len(class7_true)),
            "mae": float(class7_mae),
            "rmse": float(class7_rmse),
        }
    else:
        class7_metrics = {
            "rows": 0,
            "mae": None,
            "rmse": None,
        }

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "predicted_severity_min": float(y_pred.min()),
        "predicted_severity_max": float(y_pred.max()),
        "predicted_health_score_min": float(health_score.min()),
        "predicted_health_score_max": float(health_score.max()),
        "class_7": class7_metrics,
    }


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_and_evaluate(
    data_path: Path = DATA_PATH,
    models_dir: Path = MODELS_DIR,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict:
    """
    Train both XGBoost models and evaluate only on held-out runs.
    """

    models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------

    print("=" * 80)
    print("FAULT CLASSIFIER + SEVERITY REGRESSOR")
    print("=" * 80)

    print(f"\n[1/7] Loading dataset:")
    print(f"  {data_path}")

    t0 = time.perf_counter()

    df = pd.read_csv(
        data_path,
        low_memory=False,
    )

    print(
        f"  Rows: {len(df):,}"
        f"  Runs: {df['run_id'].nunique():,}"
        f"  Load time: {time.perf_counter() - t0:.1f}s"
    )

    _validate_fault_schema(df)

    # ------------------------------------------------------------------
    # 2. Split by run_id BEFORE feature generation
    # ------------------------------------------------------------------

    print("\n[2/7] Creating run-level 80/20 stratified split …")

    train_run_ids, holdout_run_ids = _split_run_ids(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    train_raw = df[
        df["run_id"].isin(train_run_ids)
    ].copy()

    holdout_raw = df[
        df["run_id"].isin(holdout_run_ids)
    ].copy()

    # Explicit leakage guard.
    if set(train_run_ids) & set(holdout_run_ids):
        raise AssertionError(
            "Training and holdout run IDs overlap."
        )

    print(f"  Training runs : {len(train_run_ids):,}")
    print(f"  Holdout runs  : {len(holdout_run_ids):,}")

    print("\n  Training runs by class:")
    print(
        train_raw.groupby("fault_type_code")["run_id"]
        .nunique()
        .sort_index()
        .to_string()
    )

    print("\n  Holdout runs by class:")
    print(
        holdout_raw.groupby("fault_type_code")["run_id"]
        .nunique()
        .sort_index()
        .to_string()
    )

    # ------------------------------------------------------------------
    # 3. Compute canonical features
    # ------------------------------------------------------------------

    print("\n[3/7] Computing canonical features …")

    t0 = time.perf_counter()

    train_feats = compute_features(train_raw)
    holdout_feats = compute_features(holdout_raw)

    print(
        f"  Training feature rows : {len(train_feats):,}"
    )
    print(
        f"  Holdout feature rows  : {len(holdout_feats):,}"
    )
    print(
        f"  Feature generation time: "
        f"{time.perf_counter() - t0:.1f}s"
    )

    X_train = to_feature_matrix(train_feats)
    X_holdout = to_feature_matrix(holdout_feats)

    feature_cols = list(X_train.columns)

    # Canonical feature-order assertion.
    expected_features = feature_column_names()

    if feature_cols != expected_features:
        raise AssertionError(
            "Feature order does not match canonical feature_column_names()."
        )

    if list(X_holdout.columns) != feature_cols:
        raise AssertionError(
            "Training and holdout feature column order differs."
        )

    print(f"  Feature count: {len(feature_cols)}")

    # ------------------------------------------------------------------
    # 4. Prepare targets and weights
    # ------------------------------------------------------------------

    print("\n[4/7] Preparing classifier/regressor targets …")

    y_train_cls = (
        train_feats["fault_type_code"]
        .astype(int)
        .to_numpy()
    )

    y_holdout_cls = (
        holdout_feats["fault_type_code"]
        .astype(int)
        .to_numpy()
    )

    y_train_reg = (
        train_feats["fault_severity"]
        .astype(float)
        .to_numpy()
    )

    y_holdout_reg = (
        holdout_feats["fault_severity"]
        .astype(float)
        .to_numpy()
    )

    sample_weights = _class_weights(
        train_feats["fault_type_code"]
    )

    print(
        f"  Healthy classifier weight: "
        f"{HEALTHY_CLASS_WEIGHT}"
    )
    print(
        f"  Fault classifier weight: "
        f"{FAULT_CLASS_WEIGHT}"
    )

    # ------------------------------------------------------------------
    # 5. Train classifier
    # ------------------------------------------------------------------

    print("\n[5/7] Training XGBoost multi-class classifier …")

    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_CODES),
        n_estimators=300,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )

    classifier.fit(
        X_train,
        y_train_cls,
        sample_weight=sample_weights,
        verbose=False,
    )

    print("  Classifier training complete.")

    # ------------------------------------------------------------------
    # 6. Train severity regressor
    # ------------------------------------------------------------------

    print("\n[6/7] Training XGBoost severity regressor …")

    regressor = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.0,
        eval_metric="rmse",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )

    regressor.fit(
        X_train,
        y_train_reg,
        verbose=False,
    )

    print("  Regressor training complete.")

    # ------------------------------------------------------------------
    # 7. Held-out evaluation
    # ------------------------------------------------------------------

    print("\n[7/7] Evaluating ONLY on held-out runs …")

    # ---------------------------------------------------------------
    # Classification
    # ---------------------------------------------------------------

    y_pred_cls = classifier.predict(X_holdout)

    classifier_metrics = _print_classifier_report(
        y_holdout_cls,
        y_pred_cls,
    )

    # ---------------------------------------------------------------
    # Severity regression
    # ---------------------------------------------------------------

    y_pred_severity = regressor.predict(X_holdout)

    regression_metrics = _print_regression_report(
        y_holdout_reg,
        y_pred_severity,
        holdout_feats,
    )

    # ---------------------------------------------------------------
    # Class 7 voltage analysis
    # ---------------------------------------------------------------

    class7_voltage = _analyze_class7_voltage(
        holdout_feats
    )

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SAVING ARTIFACTS")
    print("=" * 80)

    classifier_payload = {
        "model": classifier,
        "feature_cols": feature_cols,
        "class_map": CLASS_MAP,
        "healthy_class_weight": HEALTHY_CLASS_WEIGHT,
        "fault_class_weight": FAULT_CLASS_WEIGHT,
        "random_state": random_state,
    }

    joblib.dump(
        classifier_payload,
        CLASSIFIER_PATH,
    )

    regressor_payload = {
        "model": regressor,
        "feature_cols": feature_cols,
        "target": "fault_severity",
        "health_score_formula": "100 * (1 - severity_hat)",
        "random_state": random_state,
    }

    joblib.dump(
        regressor_payload,
        REGRESSOR_PATH,
    )

    # ------------------------------------------------------------------
    # Holdout run manifest
    # ------------------------------------------------------------------

    holdout_run_table = (
        holdout_raw.groupby("run_id")["fault_type_code"]
        .first()
    )

    holdout_runs_by_class = {}

    for code, name in CLASS_MAP.items():
        runs = (
            holdout_run_table[
                holdout_run_table == code
            ]
            .index
            .tolist()
        )

        holdout_runs_by_class[str(code)] = {
            "name": name,
            "run_ids": [
                int(run_id)
                for run_id in sorted(runs)
            ],
        }

    holdout_manifest = {
        "split": {
            "strategy": "run_id",
            "test_size": test_size,
            "random_state": random_state,
            "stratified_by": "fault_type_code",
        },
        "train_run_count": int(len(train_run_ids)),
        "holdout_run_count": int(len(holdout_run_ids)),
        "holdout_runs_by_fault_type_code": holdout_runs_by_class,
    }

    with open(
        HOLDOUT_RUNS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            holdout_manifest,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    metadata = {
        "project": "AI-Enabled Digital Twin for MALE UAV Aero Piston Engines",
        "model_type": {
            "classifier": "XGBoost multi-class",
            "regressor": "XGBoost regressor",
        },
        "data": {
            "path": str(
                data_path.relative_to(_REPO_ROOT)
            ),
            "raw_rows": int(len(df)),
            "raw_runs": int(df["run_id"].nunique()),
            "train_raw_rows": int(len(train_raw)),
            "holdout_raw_rows": int(len(holdout_raw)),
            "train_feature_rows": int(len(train_feats)),
            "holdout_feature_rows": int(len(holdout_feats)),
            "train_runs": int(len(train_run_ids)),
            "holdout_runs": int(len(holdout_run_ids)),
        },
        "features": {
            "source": "project/model/features.py",
            "feature_count": len(feature_cols),
            "feature_list": feature_cols,
        },
        "class_map": {
            str(code): name
            for code, name in CLASS_MAP.items()
        },
        "classifier": {
            "target": "fault_type_code",
            "objective": "multi:softprob",
            "num_class": len(CLASS_CODES),
            "healthy_class_weight": HEALTHY_CLASS_WEIGHT,
            "fault_class_weight": FAULT_CLASS_WEIGHT,
            "class_weight_interpretation": (
                "Healthy rows receive weight 1.3; all fault rows "
                "receive weight 1.0."
            ),
            "metrics": classifier_metrics,
            "class_7": classifier_metrics["per_class"][
                str(ALTERNATOR_CODE)
            ],
        },
        "regressor": {
            "target": "fault_severity",
            "target_range": [0.0, 1.0],
            "healthy_target": 0.0,
            "health_score_formula": (
                "100 * (1 - severity_hat)"
            ),
            "metrics": regression_metrics,
        },
        "class_7_voltage_analysis": class7_voltage,
        "split": {
            "method": "run-level train_test_split",
            "test_size": test_size,
            "stratified_by": "fault_type_code",
            "random_state": random_state,
            "run_leakage_checked": True,
        },
        "artifacts": {
            "classifier": str(
                CLASSIFIER_PATH.relative_to(_REPO_ROOT)
            ),
            "regressor": str(
                REGRESSOR_PATH.relative_to(_REPO_ROOT)
            ),
            "metadata": str(
                METADATA_PATH.relative_to(_REPO_ROOT)
            ),
            "holdout_runs": str(
                HOLDOUT_RUNS_PATH.relative_to(_REPO_ROOT)
            ),
        },
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            default=_json_safe,
        )

    print(f"\nSaved:")
    print(f"  {CLASSIFIER_PATH}")
    print(f"  {REGRESSOR_PATH}")
    print(f"  {METADATA_PATH}")
    print(f"  {HOLDOUT_RUNS_PATH}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_and_evaluate()
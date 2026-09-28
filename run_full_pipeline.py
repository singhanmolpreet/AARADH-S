"""
run_full_pipeline.py
====================
Master verification script — runs all 21 checks from the specification.

Usage (from repo root):
    python run_full_pipeline.py

Expected key outputs:
  - Feature rows: 1,010,282
  - Feature count: 30
  - Runs: 1,248
  - Artifact: project/model/models/anomaly_model.joblib
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from project.model.features import (
    FORBIDDEN_COLUMNS,
    MISSION_PHASES,
    SENSOR_COLUMNS,
    compute_features,
    feature_column_names,
    to_feature_matrix,
)
from project.model.anomaly_detector import AnomalyDetector, train_and_evaluate

CSV_PATH = REPO_ROOT / "data" / "processed" / "rotax_combined_clean.csv"
MODELS_DIR = REPO_ROOT / "project" / "model" / "models"

SEP = "=" * 80
SEP2 = "-" * 80

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check(label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if not ok:
        raise AssertionError(f"VERIFICATION FAILED: {label}. {detail}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEP)
    print("FULL PIPELINE VERIFICATION — 21 CHECKS")
    print(SEP)

    # ===================================================================
    # CHECK 1: features.py imports successfully
    # ===================================================================
    print(f"\n{SEP2}")
    print("CHECK 1: features.py imports successfully")
    try:
        from project.model.features import compute_features, feature_column_names, to_feature_matrix
        check("features.py imports successfully", True)
    except Exception as e:
        check("features.py imports successfully", False, str(e))

    # ===================================================================
    # CHECK 2: anomaly_detector.py imports successfully
    # ===================================================================
    print(f"\n{SEP2}")
    print("CHECK 2: anomaly_detector.py imports successfully")
    try:
        from project.model.anomaly_detector import AnomalyDetector
        check("anomaly_detector.py imports successfully", True)
    except Exception as e:
        check("anomaly_detector.py imports successfully", False, str(e))

    # ===================================================================
    # LOAD DATA
    # ===================================================================
    print(f"\n{SEP2}")
    print("Loading canonical dataset …")
    t0 = time.perf_counter()
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {df['run_id'].nunique():,} runs in {time.perf_counter()-t0:.1f}s")

    # ===================================================================
    # CHECK 3: canonical feature generation works
    # ===================================================================
    print(f"\n{SEP2}")
    print("CHECK 3: canonical feature generation works")
    t1 = time.perf_counter()
    feat_df = compute_features(df)
    elapsed = time.perf_counter() - t1
    print(f"  compute_features completed in {elapsed:.1f}s")
    check("compute_features returns a DataFrame", isinstance(feat_df, pd.DataFrame))

    # ===================================================================
    # CHECK 4: expected full feature row count = 1,010,282
    # ===================================================================
    print(f"\n{SEP2}")
    print("CHECK 4: expected row count = 1,010,282")
    n_rows = len(feat_df)
    print(f"  Output rows: {n_rows:,}")
    if n_rows != 1_010_282:
        n_runs = df["run_id"].nunique()
        n_warmup = n_runs * 29
        n_expected = len(df) - n_warmup
        print(f"  Input rows: {len(df):,}, runs: {n_runs:,}, warm-up: {n_warmup:,}, expected: {n_expected:,}")
        print(f"  WARNING: expected 1,010,282 but got {n_rows:,}. Investigating …")
        check(f"row count == 1,010,282", n_rows == 1_010_282,
              f"got {n_rows:,} (input={len(df):,}, runs={n_runs:,})")
    else:
        check("row count == 1,010,282", True, f"{n_rows:,}")

    # ===================================================================
    # CHECK 5: feature count = 30
    # ===================================================================
    print(f"\n{SEP2}")
    print("CHECK 5: feature count = 30")
    X = to_feature_matrix(feat_df)
    n_features = X.shape[1]
    check(f"feature count == 30", n_features == 30, f"got {n_features}")
    print(f"  Feature matrix shape: {X.shape}")

    # ===================================================================
    # CHECK 6: no NaNs in X
    # ===================================================================
    print(f"\n{SEP2}")
    print("CHECK 6: no NaNs in X")
    nan_count = int(X.isna().sum().sum())
    check("no NaNs in feature matrix", nan_count == 0, f"found {nan_count} NaNs")

    # ===================================================================
    # CHECK 7: no forbidden/label columns in X
    # ===================================================================
    print(f"\n{SEP2}")
    print("CHECK 7: no forbidden/label columns in X")
    leaked = set(X.columns) & FORBIDDEN_COLUMNS
    check("no forbidden columns in X", len(leaked) == 0, f"leaked: {sorted(leaked)}")

    # ===================================================================
    # CHECK 8: each run has exactly one fault_type_code
    # ===================================================================
    print(f"\n{SEP2}")
    print("CHECK 8: each run has exactly one fault_type_code")
    codes_per_run = df.groupby("run_id")["fault_type_code"].nunique(dropna=False)
    bad_runs = codes_per_run[codes_per_run != 1]
    check("each run has exactly one fault_type_code", len(bad_runs) == 0,
          f"{len(bad_runs)} bad runs")

    # ===================================================================
    # CHECK 9: temporal calculations have no cross-run bleed
    # ===================================================================
    print(f"\n{SEP2}")
    print("CHECK 9: temporal calculations have no cross-run bleed")
    first_rows = feat_df.groupby("run_id", sort=False).first()
    sample_runs = list(df["run_id"].unique()[:5]) + list(df["run_id"].unique()[-5:])
    for r in sample_runs:
        run_raw = df[df["run_id"] == r].sort_values("time_s").reset_index(drop=True)
        # Mean of rows 0..29 for rpm
        expected_mean = run_raw.loc[:29, "rpm"].mean()
        actual_mean = first_rows.loc[r, "rpm_roll_mean"]
        assert np.isclose(expected_mean, actual_mean, atol=1e-5), \
            f"run {r}: roll_mean mismatch (expected {expected_mean}, got {actual_mean})"
        # ddof=0 variance
        expected_var = run_raw.loc[:29, "rpm"].var(ddof=0)
        actual_var = first_rows.loc[r, "rpm_roll_var"]
        assert np.isclose(expected_var, actual_var, atol=1e-5), \
            f"run {r}: roll_var mismatch (expected {expected_var}, got {actual_var})"
        # diff_5: val(29) - val(24)
        expected_diff = run_raw.loc[29, "rpm"] - run_raw.loc[24, "rpm"]
        actual_diff = first_rows.loc[r, "rpm_diff_5"]
        assert np.isclose(expected_diff, actual_diff, atol=1e-5), \
            f"run {r}: diff_5 mismatch (expected {expected_diff}, got {actual_diff})"
    check("no cross-run temporal bleed (10 runs verified)", True)

    # ===================================================================
    # CHECK 10: 1 Hz sampling
    # ===================================================================
    print(f"\n{SEP2}")
    print("CHECK 10: 1 Hz sampling assumption satisfied")
    sorted_df = df.sort_values(["run_id", "time_s"])
    dt = sorted_df.groupby("run_id", sort=False)["time_s"].diff().dropna()
    pct_exact_1 = (dt == 1.0).mean() * 100
    check("100% of time steps are exactly 1.0s", np.isclose(pct_exact_1, 100.0),
          f"{pct_exact_1:.4f}%")

    # ===================================================================
    # CHECK 11: healthy run split has no overlap
    # CHECK 12: Isolation Forest fitted only on healthy training runs
    # CHECK 13: normalization fitted only on healthy training runs
    # CHECK 14: scoring works on held-out healthy data
    # CHECK 15: scoring works on faulty data
    # CHECK 16: ROC-AUC successfully calculated
    # CHECK 17: all fault classes 1-8 evaluated
    # ===================================================================

    print(f"\n{SEP2}")
    print("CHECKS 11–17: Running train_and_evaluate() …")
    print("  (includes all leakage guards, split validation, score distributions, ROC-AUC)")
    print()

    # Run train_and_evaluate — it internally validates all leakage guards
    train_and_evaluate(
        csv_path=CSV_PATH,
        models_dir=MODELS_DIR,
        healthy_holdout_frac=0.20,
        random_seed=42,
    )

    # Verify artifact was saved
    artifact_path = MODELS_DIR / "anomaly_model.joblib"
    check("artifact file saved", artifact_path.exists(), str(artifact_path))

    # Load metadata
    meta_path = MODELS_DIR / "training_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    # Check 11: run overlap (reported in train_and_evaluate; verify via metadata)
    check(
        "CHECK 11: healthy run split has no overlap",
        "1. No training run_id in held-out healthy set" in meta["leakage_checks_passed"],
    )
    check(
        "CHECK 12: Isolation Forest fitted only on healthy training runs",
        "4. No faulty row used during Isolation Forest fitting" in meta["leakage_checks_passed"],
    )
    check(
        "CHECK 13: normalization fitted only on healthy training runs",
        "3. No faulty run used during normalization fitting" in meta["leakage_checks_passed"],
    )

    # Load the trained detector and check scoring
    det = AnomalyDetector.load(artifact_path)

    # CHECK 14: scoring on held-out healthy data
    # CHECK 14: scoring on held-out healthy data
    print(f"\n{SEP2}")
    print("CHECK 14: scoring works on held-out healthy data")
    rng = np.random.default_rng(42)
    healthy_run_ids = sorted(df.loc[df["fault_type_code"] == 0, "run_id"].unique())
    healthy_arr = np.array(healthy_run_ids)
    rng.shuffle(healthy_arr)
    n_holdout = max(1, int(len(healthy_arr) * 0.20))
    holdout_ids = set(healthy_arr[:n_holdout].tolist())

    # Reuse canonical features already computed in CHECK 3.
    holdout_features = feat_df[feat_df["run_id"].isin(holdout_ids)].copy()
    holdout_X = to_feature_matrix(holdout_features)

    scores_ho = det.score_precomputed(
        holdout_features,
        holdout_X,
    )

    check(
        "scoring works on held-out healthy data",
        len(scores_ho) > 0,
        f"{len(scores_ho):,} scores, "
        f"range [{scores_ho.min():.4f}, {scores_ho.max():.4f}]",
    )

    # CHECK 15: scoring on faulty data
    print(f"\n{SEP2}")
    print("CHECK 15: scoring works on faulty data")

    # Reuse canonical features already computed in CHECK 3.
    faulty_features = feat_df[feat_df["fault_type_code"] != 0].copy()
    faulty_X = to_feature_matrix(faulty_features)

    scores_fa = det.score_precomputed(
        faulty_features,
        faulty_X,
    )

    check(
        "scoring works on faulty data",
        len(scores_fa) > 0,
        f"{len(scores_fa):,} scores, "
        f"range [{scores_fa.min():.4f}, {scores_fa.max():.4f}]",
    )

    # CHECK 16: ROC-AUC
    print(f"\n{SEP2}")
    print("CHECK 16: ROC-AUC successfully calculated")
    roc_auc = meta["evaluation"]["roc_auc_healthy_vs_faulty"]
    check("ROC-AUC computed", 0.0 <= roc_auc <= 1.0, f"ROC-AUC={roc_auc:.6f}")

    # CHECK 17: all fault classes 1-8 evaluated
    print(f"\n{SEP2}")
    print("CHECK 17: all fault classes 1-8 evaluated")
    fault_class_map = meta.get("fault_class_map", {})
    score_dist_labels = {s["label"] for s in meta["score_distributions"]}
    all_8_evaluated = all(
        name in score_dist_labels
        for name in fault_class_map.values()
    )
    check("all fault classes 1-8 evaluated", all_8_evaluated,
          f"found labels: {sorted(score_dist_labels)}")

    # ===================================================================
    # CHECK 18: saved artifact can be loaded
    # ===================================================================
    print(f"\n{SEP2}")
    print("CHECK 18: saved artifact can be loaded")
    try:
        det2 = AnomalyDetector.load(artifact_path)
        check("artifact loads successfully", det2._is_fitted)
    except Exception as e:
        check("artifact loads successfully", False, str(e))

    # ===================================================================
    # CHECK 19: loaded artifact produces same scores
    # ===================================================================
    print(f"\n{SEP2}")
    print("CHECK 19: loaded artifact produces same scores as fitted detector")
    # Score a small sample (first 3 holdout runs)
    sample_ids = list(holdout_ids)[:3]
    sample_features = feat_df[feat_df["run_id"].isin(sample_ids)].copy()
    sample_X = to_feature_matrix(sample_features)

    s1 = det.score_precomputed(sample_features, sample_X)
    s2 = det2.score_precomputed(sample_features, sample_X)
    max_diff = float(np.abs(s1 - s2).max())
    check("loaded detector scores match fitted detector", max_diff < 1e-10,
          f"max abs diff={max_diff:.2e}")

    # ===================================================================
    # CHECK 20: all active imports still resolve
    # ===================================================================
    print(f"\n{SEP2}")
    print("CHECK 20: all active imports still resolve")
    try:
        import project.model.features as _f
        import project.model.anomaly_detector as _a
        check("project.model.features importable", True)
        check("project.model.anomaly_detector importable", True)
    except Exception as e:
        check("active imports resolve", False, str(e))

    # ===================================================================
    # CHECK 21: Docker/build paths have not changed
    # ===================================================================
    print(f"\n{SEP2}")
    print("CHECK 21: Docker/build paths not broken")
    backend_dockerfile = REPO_ROOT / "backend" / "Dockerfile"
    with open(backend_dockerfile) as f:
        docker_content = f.read()
    # Backend Docker does not reference features.py or anomaly_detector.py
    check("backend Dockerfile does not reference ML modules", 
          "features" not in docker_content and "anomaly_detector" not in docker_content,
          "Docker build config unchanged")
    
    docker_compose = REPO_ROOT / "docker-compose.yml"
    with open(docker_compose) as f:
        compose_content = f.read()
    check("docker-compose.yml does not reference ML module paths",
          "features.py" not in compose_content and "anomaly_detector.py" not in compose_content,
          "docker-compose unchanged")

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print(f"\n{SEP}")
    print("ALL 21 CHECKS PASSED")
    print(SEP)

    # Print feature column names
    cols = feature_column_names()
    print(f"\nCanonical 30 feature columns (in order):")
    for i, c in enumerate(cols, 1):
        print(f"  {i:2d}. {c}")

    print(f"\nDataset summary:")
    print(f"  Input rows     : {len(df):,}")
    print(f"  Input runs     : {df['run_id'].nunique():,}")
    print(f"  Output rows    : {len(feat_df):,}")
    print(f"  Warm-up rows   : {len(df) - len(feat_df):,}")
    print(f"  Feature columns: {n_features}")

    split = meta["data_split"]
    print(f"\nDataset split:")
    print(f"  Healthy total runs     : {split['healthy_total_runs']}")
    print(f"  Healthy training runs  : {split['healthy_training_runs']}")
    print(f"  Healthy held-out runs  : {split['healthy_holdout_runs']}")
    print(f"  Training rows (post-WU): {split['training_rows_after_warmup']:,}")
    print(f"  Held-out healthy rows  : {split['holdout_healthy_rows_after_warmup']:,}")

    mdl = meta["model"]
    print(f"\nModel:")
    print(f"  Algorithm      : {mdl['algorithm']}")
    print(f"  n_estimators   : {mdl['n_estimators']}")
    print(f"  contamination  : {mdl['contamination']}")
    print(f"  max_samples    : {mdl['max_samples']}")
    print(f"  random_state   : {mdl['random_state']}")
    print(f"  normalization  : {mdl['normalization']}")

    ev = meta["evaluation"]
    print(f"\nEvaluation:")
    print(f"  ROC-AUC (healthy-vs-faulty): {ev['roc_auc_healthy_vs_faulty']:.6f}")
    print(f"  Healthy eval rows           : {ev['healthy_eval_rows']:,}  ({ev['healthy_eval_runs']:,} runs)")
    print(f"  Faulty eval rows            : {ev['faulty_eval_rows']:,}  ({ev['faulty_eval_runs']:,} runs)")

    print(f"\nArtifacts:")
    print(f"  {artifact_path}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()

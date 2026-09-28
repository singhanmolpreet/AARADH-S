"""
project/model/anomaly_detector.py
==================================

Isolation-Forest anomaly detector for the Rotax 914 telemetry pipeline.

Architecture
------------
- Trains ONLY on healthy (fault_type_code == 0) telemetry.
- Healthy runs are split by run_id (NOT by individual rows):
      ~80 % training healthy run_ids
      ~20 % held-out healthy run_ids
- Normalization (per-mission-phase z-score) is fitted ONLY on the
  healthy training runs.
- Isolation Forest is fitted ONLY on the healthy training rows.
- Score calibration ([0, 1]) is derived ONLY from the healthy training
  score distribution.
- Held-out healthy data and faulty data are used ONLY for evaluation.

Score convention
----------------
  score ≈ 0  →  looks like healthy training data for that mission phase
  score ≈ 1  →  as anomalous as (or more anomalous than) the most
               extreme ~0.1 % of the healthy training distribution

Per-phase normalization
-----------------------
Every feature is z-scored relative to the statistics of the SAME mission
phase in the healthy training set.  This prevents legitimate phase-level
differences (e.g. RPM 1 400 at idle vs 5 800 at climb) from appearing as
anomalies: the forest only sees "is this row unusual for its own phase?"

Unknown phases at inference raise a hard error; the vocabulary is fixed
and must always match the four canonical phases.

Leakage guards (fail-loud)
--------------------------
Eight explicit assertions are checked before any fitting step:
  1. No healthy training run_id in held-out healthy run_ids.
  2. No held-out healthy run_id used during normalization fitting.
  3. No faulty run used during normalization fitting.
  4. No faulty row used during Isolation Forest fitting.
  5. Feature engineering does not bleed across run_id boundaries
     (enforced by compute_features inside features.py).
  6. Feature matrix contains no label columns.
  7. Evaluation does not influence score calibration.
  8. Feature column order is identical during fit and score.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Import canonical feature pipeline
# ---------------------------------------------------------------------------

# Support running from the repo root (python project/model/anomaly_detector.py)
# as well as via `python -m project.model.anomaly_detector`.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project.model.features import (  # noqa: E402
    MISSION_PHASES,
    compute_features,
    feature_column_names,
    to_feature_matrix,
    FORBIDDEN_COLUMNS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-8          # guards divide-by-zero when a phase feature is constant
_FAULT_CLASSES = {
    1: "misfire",
    2: "lubrication_issue",
    3: "sensor_drift",
    4: "overheating",
    5: "wastegate_fault",
    6: "injector_fault",
    7: "alternator_fault",
    8: "air_filter_blockage",
}
_HEALTHY_CODE = 0
_DEFAULT_MODELS_DIR = _HERE / "models"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assert_no_overlap(set_a: set, set_b: set, label_a: str, label_b: str) -> None:
    """Fail loudly if two run_id sets overlap."""
    overlap = set_a & set_b
    if overlap:
        raise AssertionError(
            f"LEAKAGE GUARD FAILED: {len(overlap)} run_id(s) appear in BOTH "
            f"{label_a} and {label_b}: {sorted(overlap)[:10]}"
        )


def _assert_no_forbidden_columns(X: pd.DataFrame, context: str) -> None:
    """Fail loudly if any forbidden/label column is present in X."""
    leaked = set(X.columns) & FORBIDDEN_COLUMNS
    if leaked:
        raise AssertionError(
            f"LEAKAGE GUARD FAILED ({context}): forbidden columns in feature "
            f"matrix: {sorted(leaked)}"
        )


def _assert_feature_order(X: pd.DataFrame, expected_cols: List[str], context: str) -> None:
    """Fail loudly if column order diverges from canonical order."""
    if list(X.columns) != expected_cols:
        raise AssertionError(
            f"LEAKAGE GUARD FAILED ({context}): feature column order does not "
            "match canonical order.\n"
            f"  Expected: {expected_cols}\n"
            f"  Got:      {list(X.columns)}"
        )


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Isolation-Forest anomaly detector over Rotax 914 telemetry, with
    per-mission-phase normalization so phase transitions are not themselves
    treated as anomalies.

    Usage (training)
    ----------------
        det = AnomalyDetector()
        det.fit(train_healthy_df)

    Usage (scoring)
    ---------------
        scores = det.score(new_df)   # -> np.ndarray in [0, 1]

    Usage (persistence)
    -------------------
        det.save("project/model/models/anomaly_model.joblib")
        det2 = AnomalyDetector.load("project/model/models/anomaly_model.joblib")
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.01,
        max_samples: Union[str, int] = "auto",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state

        self._feature_cols: List[str] = feature_column_names()
        self._mission_phases: List[str] = list(MISSION_PHASES)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )

        # Per-phase normalization stats (fitted from training healthy data only)
        self._phase_stats: Dict[str, Dict[str, pd.Series]] = {}
        # Score calibration (from training healthy data only)
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Private: normalization
    # ------------------------------------------------------------------

    def _fit_normalization(self, feats: pd.DataFrame, X: pd.DataFrame) -> None:
        """
        Compute per-phase mean and std over the 30 feature columns.
        Fitted ONLY from healthy training rows.
        """
        self._phase_stats = {}
        for phase in self._mission_phases:
            mask = feats["mission_phase"] == phase
            if mask.sum() == 0:
                raise ValueError(
                    "AnomalyDetector._fit_normalization: canonical mission phase "
                    f"'{phase}' has zero healthy training rows. "
                    f"Expected all four phases: {self._mission_phases}. "
                    "The fixed phase vocabulary requires every phase to be "
                    "represented in the healthy training set."
                )

            grp = X.loc[mask]
            std = grp.std()
            # Constant-variance guard: clip near-zero std.
            std = std.clip(lower=_EPS)
            self._phase_stats[phase] = {
                "mean": grp.mean(),
                "std": std,
            }

    def _normalize(self, feats: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        """
        Z-score normalize each row using the statistics of its own mission phase.
        Unknown phase → hard error (vocabulary is fixed; this must never happen).
        Rows are returned in the same order as feats.index.
        """
        out = np.empty(X.shape, dtype=float)

        for phase in feats["mission_phase"].unique():
            if phase not in self._phase_stats:
                raise ValueError(
                    f"AnomalyDetector._normalize: unknown mission phase '{phase}'. "
                    f"Known phases: {self._mission_phases}. "
                    "The vocabulary is fixed; do not invent new phase labels."
                )
            mask = feats["mission_phase"] == phase
            idx_positions = np.where(mask.values)[0]
            stats = self._phase_stats[phase]
            block = X.iloc[idx_positions].to_numpy(dtype=float)
            z = (block - stats["mean"].to_numpy(dtype=float)) / stats["std"].to_numpy(dtype=float)
            out[idx_positions] = z

        return out

    # ------------------------------------------------------------------
    # Public API: fit
    # ------------------------------------------------------------------

    def fit(self, train_healthy_df: pd.DataFrame) -> "AnomalyDetector":
        """
        Train on healthy training telemetry ONLY.

        This is the raw-data convenience API. It computes the canonical
        features once, then delegates to fit_precomputed().

        Parameters
        ----------
        train_healthy_df : pd.DataFrame
            Raw 1 Hz telemetry for HEALTHY TRAINING runs only.
            Must NOT include held-out healthy or faulty runs.
        """
        print("[AnomalyDetector.fit] Running compute_features on training data …")
        t0 = time.perf_counter()
        feats = compute_features(train_healthy_df)
        elapsed = time.perf_counter() - t0
        X = to_feature_matrix(feats)
        print(
            f"  compute_features: {len(feats):,} rows, "
            f"{X.shape[1]} features in {elapsed:.1f}s"
        )
        return self.fit_precomputed(feats, X)

    def fit_precomputed(
        self,
        feats: pd.DataFrame,
        X: pd.DataFrame,
    ) -> "AnomalyDetector":
        """
        Fit using already-computed canonical features.

        This is the preferred API when the caller already has the output of
        compute_features(), because it avoids recomputing the expensive
        rolling/difference feature pipeline.

        Parameters
        ----------
        feats : pd.DataFrame
            Output of compute_features(). Must include run_id,
            mission_phase, and fault_type_code.
        X : pd.DataFrame
            Output of to_feature_matrix(feats), exactly 30 canonical columns.
        """
        if "mission_phase" not in feats.columns:
            raise ValueError(
                "AnomalyDetector.fit_precomputed: feats must contain 'mission_phase'."
            )

        if "run_id" not in feats.columns:
            raise ValueError(
                "AnomalyDetector.fit_precomputed: feats must contain 'run_id'."
            )

        # Leakage guard 6: no forbidden columns in X
        _assert_no_forbidden_columns(X, context="fit_precomputed")

        # Leakage guard 8: feature column order
        _assert_feature_order(X, self._feature_cols, context="fit_precomputed")

        if len(feats) != len(X):
            raise ValueError(
                "AnomalyDetector.fit_precomputed: feats and X must have the "
                f"same number of rows (got {len(feats):,} and {len(X):,})."
            )

        # Leakage guard 4: confirm all rows are healthy.
        if "fault_type_code" not in feats.columns:
            raise ValueError(
                "AnomalyDetector.fit_precomputed: feats must contain "
                "'fault_type_code' so healthy-only training can be verified."
            )

        faulty_mask = feats["fault_type_code"] != _HEALTHY_CODE
        if faulty_mask.any():
            raise AssertionError(
                "LEAKAGE GUARD FAILED: faulty rows (fault_type_code != 0) "
                f"passed to AnomalyDetector.fit_precomputed — "
                f"{int(faulty_mask.sum()):,} row(s)."
            )

        n_runs = feats["run_id"].nunique()
        print(
            f"[AnomalyDetector.fit_precomputed] Training on "
            f"{len(feats):,} healthy feature rows from {n_runs:,} runs "
            f"with {X.shape[1]} features."
        )

        # Fit per-phase normalization (healthy training rows only).
        self._fit_normalization(feats, X)

        # Normalize.
        X_norm = self._normalize(feats, X)

        # Fit Isolation Forest.
        print(
            f"[AnomalyDetector.fit_precomputed] Fitting IsolationForest "
            f"on {X_norm.shape[0]:,} rows …"
        )
        t1 = time.perf_counter()
        self.model.fit(X_norm)
        print(
            f"  IsolationForest fit in "
            f"{time.perf_counter() - t1:.1f}s"
        )

        # Calibrate score range from training data ONLY.
        # sklearn score_samples: higher = more normal → negate for
        # "higher = more anomalous".
        train_raw = -self.model.score_samples(X_norm)
        self._score_min = float(np.percentile(train_raw, 1))
        self._score_max = float(np.percentile(train_raw, 99.9))
        if self._score_max <= self._score_min:
            self._score_max = self._score_min + _EPS

        self._is_fitted = True
        print(
            f"[AnomalyDetector.fit_precomputed] Done. "
            f"score_min(1%)={self._score_min:.6f}  "
            f"score_max(99.9%)={self._score_max:.6f}"
        )
        return self

    # ------------------------------------------------------------------
    # Public API: score
    # ------------------------------------------------------------------

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return an anomaly score in [0, 1] per input row (same order as df).

        score ≈ 0  →  looks like healthy training data for that mission phase
        score ≈ 1  →  at least as anomalous as the top 0.1% of training data

        Parameters
        ----------
        df : pd.DataFrame
            Raw 1 Hz telemetry.  Must contain all required columns.
        """
        if not self._is_fitted:
            raise RuntimeError("AnomalyDetector.score: call .fit() first")

        feats = compute_features(df)
        X = to_feature_matrix(feats)

        # Leakage guard 6: no forbidden columns in X
        _assert_no_forbidden_columns(X, context="score")

        # Leakage guard 8: feature column order must match fit-time order
        _assert_feature_order(X, self._feature_cols, context="score")

        X_norm = self._normalize(feats, X)

        # Leakage guard 7: score calibration is never recomputed here
        raw = -self.model.score_samples(X_norm)
        scaled = (raw - self._score_min) / (self._score_max - self._score_min)
        return np.clip(scaled, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Public API: score_precomputed
    # ------------------------------------------------------------------

    def score_precomputed(
        self,
        feats: pd.DataFrame,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Score using already-computed feature DataFrame and matrix.
        Avoids redundant compute_features() call when feats/X are already known.

        Parameters
        ----------
        feats : pd.DataFrame
            Output of compute_features() — must include 'mission_phase'.
        X : pd.DataFrame
            Output of to_feature_matrix(feats) — exactly 30 columns.
        """
        if not self._is_fitted:
            raise RuntimeError("AnomalyDetector.score_precomputed: call .fit() first")

        _assert_no_forbidden_columns(X, context="score_precomputed")
        _assert_feature_order(X, self._feature_cols, context="score_precomputed")

        X_norm = self._normalize(feats, X)
        raw = -self.model.score_samples(X_norm)
        scaled = (raw - self._score_min) / (self._score_max - self._score_min)
        return np.clip(scaled, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted model and all parameters to a single joblib file."""
        if not self._is_fitted:
            raise RuntimeError("AnomalyDetector.save: nothing fitted yet")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            # Model
            "model": self.model,
            # Normalization
            "phase_stats": self._phase_stats,
            # Score calibration
            "score_min": self._score_min,
            "score_max": self._score_max,
            # Feature spec (for reproducibility checks on load)
            "feature_cols": self._feature_cols,
            "mission_phases": self._mission_phases,
            # Hyper-parameters
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
        }
        joblib.dump(payload, path)
        print(f"[AnomalyDetector.save] Saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AnomalyDetector":
        """Load a detector previously saved with .save()."""
        state = joblib.load(path)
        det = cls(
            n_estimators=state["n_estimators"],
            contamination=state["contamination"],
            max_samples=state["max_samples"],
            random_state=state["random_state"],
        )
        det.model = state["model"]
        det._phase_stats = state["phase_stats"]
        det._score_min = state["score_min"]
        det._score_max = state["score_max"]
        det._feature_cols = state["feature_cols"]
        det._mission_phases = state["mission_phases"]
        det._is_fitted = True

        # Leakage guard 8: verify saved feature order matches current canonical order
        current_cols = feature_column_names()
        if det._feature_cols != current_cols:
            raise AssertionError(
                "LEAKAGE GUARD FAILED (load): saved feature column order does not "
                "match the current canonical feature_column_names().\n"
                f"  Saved:   {det._feature_cols}\n"
                f"  Current: {current_cols}"
            )

        print(f"[AnomalyDetector.load] Loaded from {path}")
        return det


# ---------------------------------------------------------------------------
# Training / evaluation entry point
# ---------------------------------------------------------------------------

def _score_distribution_summary(scores: np.ndarray, label: str) -> dict:
    """Return a summary dict for an array of anomaly scores."""
    return {
        "label": label,
        "count": len(scores),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "p5": float(np.percentile(scores, 5)),
        "p25": float(np.percentile(scores, 25)),
        "p75": float(np.percentile(scores, 75)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }


def _print_summary(s: dict) -> None:
    print(
        f"  {s['label']:<30s}  n={s['count']:>8,}  "
        f"mean={s['mean']:.4f}  median={s['median']:.4f}  "
        f"std={s['std']:.4f}  "
        f"min={s['min']:.4f}  max={s['max']:.4f}  "
        f"p90={s['p90']:.4f}  p99={s['p99']:.4f}"
    )


def train_and_evaluate(
    csv_path: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    healthy_holdout_frac: float = 0.20,
    random_seed: int = 42,
    n_estimators: int = 200,
    contamination: float = 0.01,
    max_samples: Union[str, int] = "auto",
) -> None:
    """
    Full training and evaluation pipeline.

    1. Load canonical CSV.
    2. Split healthy run_ids (80/20) — leakage guards enforced.
    3. Compute features for each split.
    4. Fit AnomalyDetector on healthy training runs.
    5. Evaluate on held-out healthy runs and all faulty runs.
    6. Report score distributions and ROC-AUC.
    7. Save artifacts to models_dir.
    """
    sep = "=" * 80
    sep2 = "-" * 80

    if csv_path is None:
        csv_path = _REPO_ROOT / "data" / "processed" / "rotax_combined_clean.csv"
    if models_dir is None:
        models_dir = _DEFAULT_MODELS_DIR

    csv_path = Path(csv_path)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print(sep)
    print("ROTAX 914 ANOMALY DETECTOR — TRAINING & EVALUATION")
    print(sep)
    print(f"\n[1/8] Loading {csv_path} …")
    t0 = time.perf_counter()
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {df['run_id'].nunique():,} runs in {time.perf_counter()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # 2. Validate pre-split
    # -----------------------------------------------------------------------
    print(f"\n[2/8] Validating dataset …")
    assert not df["fault_type_code"].isna().any(), "Null fault_type_code found"
    codes_per_run = df.groupby("run_id")["fault_type_code"].nunique(dropna=False)
    assert (codes_per_run == 1).all(), "Some run_ids have multiple fault_type_code values"
    print(f"  All {df['run_id'].nunique():,} runs have exactly 1 fault_type_code. OK.")

    # -----------------------------------------------------------------------
    # 3. Split healthy run_ids
    # -----------------------------------------------------------------------
    print(f"\n[3/8] Splitting healthy run_ids (hold-out fraction={healthy_holdout_frac}) …")
    healthy_run_ids = sorted(df.loc[df["fault_type_code"] == _HEALTHY_CODE, "run_id"].unique())
    faulty_run_ids = sorted(df.loc[df["fault_type_code"] != _HEALTHY_CODE, "run_id"].unique())

    rng = np.random.default_rng(random_seed)
    healthy_arr = np.array(healthy_run_ids)
    rng.shuffle(healthy_arr)

    n_holdout = max(1, int(len(healthy_arr) * healthy_holdout_frac))
    holdout_healthy_ids = set(healthy_arr[:n_holdout].tolist())
    train_healthy_ids = set(healthy_arr[n_holdout:].tolist())

    # Leakage guard 1: no overlap between training and held-out healthy run_ids
    _assert_no_overlap(train_healthy_ids, holdout_healthy_ids,
                       "train_healthy_ids", "holdout_healthy_ids")

    # Leakage guard 3: no faulty run_id in training set
    _assert_no_overlap(train_healthy_ids, set(faulty_run_ids),
                       "train_healthy_ids", "faulty_run_ids")

    print(f"  Total healthy runs     : {len(healthy_run_ids):,}")
    print(f"  Training healthy runs  : {len(train_healthy_ids):,}")
    print(f"  Held-out healthy runs  : {len(holdout_healthy_ids):,}")
    print(f"  Faulty runs            : {len(faulty_run_ids):,}")

    train_healthy_raw = df[df["run_id"].isin(train_healthy_ids)].copy()
    holdout_healthy_raw = df[df["run_id"].isin(holdout_healthy_ids)].copy()
    faulty_raw = df[df["fault_type_code"] != _HEALTHY_CODE].copy()

    print(f"  Training healthy rows  : {len(train_healthy_raw):,}")
    print(f"  Held-out healthy rows  : {len(holdout_healthy_raw):,}")
    print(f"  Faulty rows            : {len(faulty_raw):,}")

    # -----------------------------------------------------------------------
    # 4. Compute features for each split
    # -----------------------------------------------------------------------
    print(f"\n[4/8] Computing features for each split …")

    t1 = time.perf_counter()
    print("  compute_features(train_healthy) …")
    feats_train = compute_features(train_healthy_raw)
    X_train = to_feature_matrix(feats_train)
    print(f"    → {len(feats_train):,} rows, {X_train.shape[1]} features in {time.perf_counter()-t1:.1f}s")

    t2 = time.perf_counter()
    print("  compute_features(holdout_healthy) …")
    feats_holdout = compute_features(holdout_healthy_raw)
    X_holdout = to_feature_matrix(feats_holdout)
    print(f"    → {len(feats_holdout):,} rows in {time.perf_counter()-t2:.1f}s")

    t3 = time.perf_counter()
    print("  compute_features(faulty) …")
    feats_faulty = compute_features(faulty_raw)
    X_faulty = to_feature_matrix(feats_faulty)
    print(f"    → {len(feats_faulty):,} rows in {time.perf_counter()-t3:.1f}s")

    # Leakage guard 2: no held-out run_id in training feature frame
    train_run_ids_in_feats = set(feats_train["run_id"].unique())
    _assert_no_overlap(train_run_ids_in_feats, holdout_healthy_ids,
                       "training feature frame run_ids", "holdout_healthy_ids")

    # Leakage guard 3 (again, on feature frames): no faulty run in training frame
    faulty_run_ids_in_feats = set(feats_faulty["run_id"].unique())
    _assert_no_overlap(train_run_ids_in_feats, faulty_run_ids_in_feats,
                       "training feature frame run_ids", "faulty feature frame run_ids")

    # Leakage guard 6: no forbidden columns in any feature matrix
    _assert_no_forbidden_columns(X_train, "training X")
    _assert_no_forbidden_columns(X_holdout, "holdout X")
    _assert_no_forbidden_columns(X_faulty, "faulty X")

    # -----------------------------------------------------------------------
    # 5. Fit AnomalyDetector
    # -----------------------------------------------------------------------
    print(f"\n[5/8] Fitting AnomalyDetector …")
    det = AnomalyDetector(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_seed,
    )
    # Features were already computed above. Reuse them directly so the
    # expensive rolling/difference feature pipeline is not run twice.
    det.fit_precomputed(feats_train, X_train)

    # -----------------------------------------------------------------------
    # 6. Score evaluation sets
    # -----------------------------------------------------------------------
    print(f"\n[6/8] Scoring evaluation sets …")

    # Leakage guard 7: score calibration is NOT recomputed here; only det._score_min/max are used.
    scores_train = det.score_precomputed(feats_train, X_train)
    scores_holdout = det.score_precomputed(feats_holdout, X_holdout)
    scores_faulty = det.score_precomputed(feats_faulty, X_faulty)

    # Leakage guard 4: confirm no faulty fault_type_code in training feature frame
    faulty_in_train = feats_train[feats_train["fault_type_code"] != _HEALTHY_CODE]
    if len(faulty_in_train) > 0:
        raise AssertionError(
            f"LEAKAGE GUARD FAILED: {len(faulty_in_train)} faulty rows found in "
            "training feature frame."
        )

    # -----------------------------------------------------------------------
    # 7. Report score distributions
    # -----------------------------------------------------------------------
    print(f"\n[7/8] Score distributions …")
    print(sep2)
    summaries = []

    # Held-out healthy
    s = _score_distribution_summary(scores_holdout, "held-out healthy")
    _print_summary(s)
    summaries.append(s)

    # Per-fault-class
    for code, name in sorted(_FAULT_CLASSES.items()):
        mask = feats_faulty["fault_type_code"] == code
        if mask.sum() == 0:
            print(f"  {name:<30s}  NOT FOUND IN DATASET")
            continue
        s = _score_distribution_summary(scores_faulty[mask.values], name)
        _print_summary(s)
        summaries.append(s)

    # -----------------------------------------------------------------------
    # 7b. ROC-AUC
    # -----------------------------------------------------------------------
    print(sep2)
    # y=0 for healthy, y=1 for faulty
    y_true = np.concatenate([
        np.zeros(len(scores_holdout)),
        np.ones(len(scores_faulty)),
    ])
    y_score = np.concatenate([scores_holdout, scores_faulty])
    roc_auc = roc_auc_score(y_true, y_score)

    n_healthy_eval_rows = len(scores_holdout)
    n_faulty_eval_rows = len(scores_faulty)
    n_healthy_eval_runs = len(holdout_healthy_ids)
    n_faulty_eval_runs = feats_faulty["run_id"].nunique()

    print(f"\n  ROC-AUC (healthy-vs-faulty, all classes): {roc_auc:.6f}")
    print(f"  Healthy eval rows : {n_healthy_eval_rows:,}  (runs: {n_healthy_eval_runs:,})")
    print(f"  Faulty eval rows  : {n_faulty_eval_rows:,}  (runs: {n_faulty_eval_runs:,})")

    # -----------------------------------------------------------------------
    # 8. Save artifacts
    # -----------------------------------------------------------------------
    print(f"\n[8/8] Saving artifacts to {models_dir} …")

    artifact_path = models_dir / "anomaly_model.joblib"
    det.save(artifact_path)

    # Save metadata
    import json
    metadata = {
        "model": {
            "algorithm": "IsolationForest",
            "n_estimators": det.n_estimators,
            "contamination": det.contamination,
            "max_samples": str(det.max_samples),
            "random_state": det.random_state,
            "normalization": "per-mission-phase z-score (mean/std from healthy training)",
        },
        "data_split": {
            "healthy_total_runs": len(healthy_run_ids),
            "healthy_training_runs": len(train_healthy_ids),
            "healthy_holdout_runs": len(holdout_healthy_ids),
            "faulty_total_runs": len(faulty_run_ids),
            "training_rows_after_warmup": int(len(feats_train)),
            "holdout_healthy_rows_after_warmup": int(len(feats_holdout)),
            "faulty_rows_after_warmup": int(len(feats_faulty)),
            "holdout_fraction": healthy_holdout_frac,
            "random_seed": random_seed,
        },
        "evaluation": {
            "roc_auc_healthy_vs_faulty": float(roc_auc),
            "healthy_eval_rows": n_healthy_eval_rows,
            "faulty_eval_rows": n_faulty_eval_rows,
            "healthy_eval_runs": n_healthy_eval_runs,
            "faulty_eval_runs": n_faulty_eval_runs,
            "note": "Detector trained ONLY on healthy training runs. Held-out healthy and faulty used only for evaluation.",
        },
        "score_calibration": {
            "score_min_p1": det._score_min,
            "score_max_p999": det._score_max,
            "source": "healthy training data ONLY",
        },
        "feature_order": det._feature_cols,
        "mission_phases": det._mission_phases,
        "score_distributions": summaries,
        "fault_class_map": {str(k): v for k, v in _FAULT_CLASSES.items()},
        "leakage_checks_passed": [
            "1. No training run_id in held-out healthy set",
            "2. No held-out run_id used during normalization fitting",
            "3. No faulty run used during normalization fitting",
            "4. No faulty row used during Isolation Forest fitting",
            "5. Feature engineering isolated per run_id (enforced by compute_features)",
            "6. Feature matrix contains no label/forbidden columns",
            "7. Score calibration derived from training data only",
            "8. Feature column order verified identical at fit and score",
        ],
    }
    meta_path = models_dir / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {meta_path}")

    print(sep)
    print("TRAINING & EVALUATION COMPLETE")
    print(sep)
    print(f"\nArtifacts:")
    print(f"  {artifact_path}")
    print(f"  {meta_path}")
    print(f"\nROC-AUC: {roc_auc:.6f}")
    print(f"\nNote: Detector was trained ONLY on healthy training runs.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_and_evaluate()
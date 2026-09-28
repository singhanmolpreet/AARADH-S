"""
anomaly_detector.py

Isolation-Forest anomaly detector for the P3 Rotax 914 telemetry pipeline.

Trains ONLY on healthy telemetry. Uses features.compute_features() for
rolling-window features, then normalizes every feature *within its own
mission_phase* before handing anything to the Isolation Forest. That
normalization step is what stops legitimate phase differences (ground
idle at 1,400 rpm vs. takeoff at 5,800 rpm) from being flagged as
anomalies: each phase's own typical value is mapped close to 0, so the
forest is judging "is this row weird for ITS phase", not "is this row
weird compared to the whole flight".
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from features import compute_features, ROLLING_COLUMNS

# Raw telemetry channels used directly as features, alongside their
# rolling-window derivatives from compute_features().
RAW_FEATURE_COLUMNS = [
    "rpm",
    "cht_c",
    "egt_c",
    "oil_pressure_bar",
    "oil_temp_c",
    "fuel_flow_lph",
    "vibration_rms_g",
    "battery_voltage_v",
    "alternator_current_a",
    "injection_timing_deg",
    "map_kpa",
    "boost_pressure_bar",
]

# {col}_roll_mean / {col}_roll_var / {col}_roll_slope for each ROLLING_COLUMNS entry.
ENGINEERED_FEATURE_COLUMNS = [
    f"{col}_{stat}"
    for col in ROLLING_COLUMNS
    for stat in ("roll_mean", "roll_var", "roll_slope")
]

FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS

_EPS = 1e-6  # guards divide-by-zero for a phase where some channel is ~constant


def _run_compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    compute_features() sorts its output by [engine_id, timestamp] and drops
    the original row order/index. Since score() promises to return results
    in the same order as the input, we tag rows before calling it and
    restore order afterward.
    """
    tmp = df.copy()
    tmp["_orig_pos"] = np.arange(len(tmp))
    feats = compute_features(tmp)
    feats = feats.sort_values("_orig_pos").drop(columns="_orig_pos").reset_index(drop=True)
    return feats


class AnomalyDetector:
    """
    Isolation-Forest anomaly detector over Rotax 914 telemetry, with
    per-mission-phase normalization so phase transitions aren't themselves
    treated as anomalies.

    Usage
    -----
        det = AnomalyDetector()
        det.fit(healthy_df)          # healthy_df: raw 1 Hz telemetry, healthy rows only
        scores = det.score(new_df)   # -> np.ndarray of anomaly scores in [0, 1]
        det.save("anomaly_model.joblib")
        det2 = AnomalyDetector.load("anomaly_model.joblib")
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

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self._phase_stats: Dict[str, Dict[str, pd.Series]] = {}
        self._global_stats: Dict[str, pd.Series] = {}
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _compute_stats(self, feats: pd.DataFrame) -> None:
        """Store per-phase and global median/mean/std over FEATURE_COLUMNS."""
        self._global_stats = {
            "median": feats[FEATURE_COLUMNS].median(),
            "mean": feats[FEATURE_COLUMNS].mean(),
            "std": feats[FEATURE_COLUMNS].std().clip(lower=_EPS),
        }
        self._phase_stats = {}
        for phase, group in feats.groupby("mission_phase"):
            self._phase_stats[phase] = {
                "median": group[FEATURE_COLUMNS].median(),
                "mean": group[FEATURE_COLUMNS].mean(),
                "std": group[FEATURE_COLUMNS].std().clip(lower=_EPS),
            }

    def _stats_for_phase(self, phase: str) -> Dict[str, pd.Series]:
        # Falls back to global stats for a phase never seen during fit
        # (e.g. a new phase name appears at inference time).
        return self._phase_stats.get(phase, self._global_stats)

    def _prepare_matrix(self, feats: pd.DataFrame) -> np.ndarray:
        """
        Impute NaNs (per-phase median, falling back to global median, then 0)
        and z-score every feature within its own mission_phase. This is the
        step that neutralizes "which phase is this" as a source of anomaly
        signal, leaving only "is this row unusual for its phase".
        """
        blocks = []
        for phase, group in feats.groupby("mission_phase", sort=False):
            stats = self._stats_for_phase(phase)
            block = group[FEATURE_COLUMNS].fillna(stats["median"])
            block = block.fillna(self._global_stats["median"]).fillna(0.0)
            z = (block - stats["mean"]) / stats["std"]
            blocks.append(z)
        z_all = pd.concat(blocks).loc[feats.index]  # restore original row order
        return z_all.to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, healthy_df: pd.DataFrame) -> "AnomalyDetector":
        """
        Train on healthy telemetry only. Runs compute_features(), learns
        per-mission-phase normalization stats, fits the Isolation Forest
        on the normalized features, and calibrates the 0-1 output scale
        against the training data's own score distribution.
        """
        if "mission_phase" not in healthy_df.columns:
            raise ValueError("AnomalyDetector.fit: healthy_df must have a mission_phase column")

        feats = _run_compute_features(healthy_df)
        self._compute_stats(feats)
        X = self._prepare_matrix(feats)

        self.model.fit(X)

        # sklearn's score_samples is higher = more normal; flip the sign so
        # "higher = more anomalous" matches the score() contract below.
        train_raw = -self.model.score_samples(X)
        self._score_min = float(np.percentile(train_raw, 1))
        self._score_max = float(np.percentile(train_raw, 99.9))
        if self._score_max <= self._score_min:
            self._score_max = self._score_min + _EPS

        self._is_fitted = True
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return an anomaly score in [0, 1] per input row, in the same order
        as df. ~0 = looks like the healthy training distribution for that
        row's mission phase; 1 = as anomalous as, or more anomalous than,
        the most extreme ~0.1% of the healthy training data.
        """
        if not self._is_fitted:
            raise RuntimeError("AnomalyDetector.score: call .fit() first")
        if "mission_phase" not in df.columns:
            raise ValueError("AnomalyDetector.score: df must have a mission_phase column")

        feats = _run_compute_features(df)
        X = self._prepare_matrix(feats)

        raw = -self.model.score_samples(X)
        scaled = (raw - self._score_min) / (self._score_max - self._score_min)
        return np.clip(scaled, 0.0, 1.0)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted model and normalization stats to a single file."""
        if not self._is_fitted:
            raise RuntimeError("AnomalyDetector.save: nothing fitted yet")
        joblib.dump(
            {
                "model": self.model,
                "phase_stats": self._phase_stats,
                "global_stats": self._global_stats,
                "score_min": self._score_min,
                "score_max": self._score_max,
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "max_samples": self.max_samples,
                "random_state": self.random_state,
            },
            path,
        )

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
        det._global_stats = state["global_stats"]
        det._score_min = state["score_min"]
        det._score_max = state["score_max"]
        det._is_fitted = True
        return det

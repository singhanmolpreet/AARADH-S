# Model Baseline Evaluation & Technical Debt Summary

**Module:** `project/model/fault_classifier.py`  
**Status:** Functional Prototype Delivered  
**Evaluation Scope:** Held-out run-level 80/20 split (250 runs, 205,006 feature rows)

---

## 1. Core Model Performance

* **Classification Macro-F1:** `0.928376`
* **Severity Regression:** MAE = `0.054569`, RMSE = `0.095899`, R² = `0.935594`
* **Health Score Range:** `0.00` to `100.00` via $100 \times (1 - \hat{\text{severity}})$
* **Artifacts Persisted:**
  * `project/model/models/fault_classifier.joblib`
  * `project/model/models/fault_severity_regressor.joblib`
  * `project/model/models/metadata.json`
  * `project/model/models/holdout_runs.json`

---

## 2. Key Findings & Class Special Notes

### Class 7 (`alternator_fault`) Separability
* **Model Accuracy:** Precision = `1.0000`, Recall = `1.0000`, F1 = `1.0000`
* **Voltage Analysis Verdict:** **Not trivially separable by voltage range alone.**
  * Class 7 voltage interval: `8.23 V – 12.46 V`
  * Non-Class 7 voltage interval: `11.67 V – 31.66 V`
  * **Overlap:** An interval overlap exists between `11.67 V` and `12.46 V`. Although 94.4% (`13,275 / 14,055`) of alternator fault rows fall strictly below the minimum non-fault voltage threshold, tree-based multivariate splits (e.g., combining voltage with engine speed/electrical load) are required for 100% boundary separation.

---

## 3. Downsides & Issues (For Next Optimization Phase)

### A. Airflow / Pressure Confusion Mode
* **Issue:** Severe class confusion exists between Class 5 (`wastegate_fault`) and Class 8 (`air_filter_blockage`).
  * 5,159 `wastegate_fault` rows classified as `air_filter_blockage`.
  * 4,106 `air_filter_blockage` rows classified as `wastegate_fault`.
  * Class 5 F1: `0.6695` | Class 8 F1: `0.7152`
* **Root Cause:** Both faults degrade manifold air pressure (MAP) and intake airflow dynamics, making static time-domain summary statistics insufficient to separate turbo-side control dynamics from intake-side restriction.
* **Future Work:** Engineer differential pressure features, MAP error relative to expected boost curves, or transient derivative features ($\Delta \text{MAP} / \Delta \text{RPM}$).

### B. Minor False Positive Leakage to Healthy (Class 0)
* **Issue:** 789 total fault rows were falsely predicted as `healthy` (Class 0):
  * 429 rows from `injector_fault` (Class 6)
  * 322 rows from `overheating` (Class 4)
  * 38 rows from `air_filter_blockage` (Class 8)
* **Impact:** Low-severity onset stages of thermal and fuel-injection faults mimic nominal baseline telemetry.
* **Future Work:** Evaluate lowering the decision threshold for non-zero fault detection or introducing dynamic sequence tracking.

### C. Sample Weight Implementation Scope
* **Issue:** The `1.3x` healthy class weight was applied strictly to the multi-class classifier. The severity regressor currently trains with uniform sample weighting.
* **Future Work:** Evaluate error-weighted loss functions for severity estimation on low-severity fault initiation points.
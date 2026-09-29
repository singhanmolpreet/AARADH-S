"""
Phase-1 RUL proxy — see project notes for why: fault_severity is a per-run
CONFIGURED target, constant across every row (confirmed: mean fitted slope
~= -7.9e-20 across 1,248 runs), so a trend-fitted time-to-failure RUL is not
estimable from this dataset as generated. This maps the LIVE severity
regressor's prediction to a bounded number so mission_advisory.py has
something real to react to today. NOT a validated time-to-failure estimate
— say so if asked, don't present it as measured.
"""

MAX_RUL_MINUTES = 180  # ASSUMPTION, arbitrary ceiling — tune or replace later

def estimate_rul(severity_hat: float, max_rul_minutes: float = MAX_RUL_MINUTES) -> float:
    severity_hat = max(0.0, min(1.0, severity_hat))
    return round(max_rul_minutes * (1.0 - severity_hat), 1)
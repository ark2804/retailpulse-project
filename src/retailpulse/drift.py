"""
Drift Detection using Population Stability Index (PSI).
Optionally wraps Evidently AI when installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_pcts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pcts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_pcts = np.where(expected_pcts == 0, 1e-4, expected_pcts)
    actual_pcts = np.where(actual_pcts == 0, 1e-4, actual_pcts)

    psi = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
    return float(psi)


def interpret_psi(psi: float) -> str:
    if psi < 0.1:
        return "Stable – no significant drift"
    elif psi < 0.2:
        return "Moderate drift – monitor closely"
    else:
        return "Significant drift – consider retraining"


def run_drift_detection(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    output_dir: str | Path = "reports",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if numeric_cols is None:
        common = list(set(reference_df.columns) & set(current_df.columns))
        numeric_cols = [c for c in common
                        if reference_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]][:10]

    report = {"columns": {}, "overall_status": "Stable"}
    drift_count = 0

    for col in numeric_cols:
        ref_vals = reference_df[col].dropna().values
        cur_vals = current_df[col].dropna().values
        psi = compute_psi(ref_vals, cur_vals)
        status = interpret_psi(psi)
        report["columns"][col] = {"psi": round(psi, 4), "status": status}
        if psi >= 0.2:
            drift_count += 1

    if drift_count > len(numeric_cols) * 0.3:
        report["overall_status"] = "Drift Detected"
    elif drift_count > 0:
        report["overall_status"] = "Moderate Drift"

    # Try Evidently AI
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        evidently_report = Report(metrics=[DataDriftPreset()])
        evidently_report.run(reference_data=reference_df[numeric_cols],
                             current_data=current_df[numeric_cols])
        evidently_report.save_html(str(output_dir / "evidently_drift_report.html"))
        report["evidently_report"] = "evidently_drift_report.html"
        print("   Evidently AI report generated")
    except Exception:
        pass

    out_path = output_dir / "drift_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Drift report → {out_path}")
    return report

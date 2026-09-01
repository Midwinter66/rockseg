"""Evaluation metrics for volume estimation.

Reports the metrics specified in E5 (``02_experiment_ledger.md``):

* MAE  — Mean Absolute Error
* RMSE — Root Mean Squared Error
* MAPE — Mean Absolute Percentage Error
* SMAPE — Symmetric Mean Absolute Percentage Error
* Median relative error
* R²   — Coefficient of determination

All metrics are computed on the **true scale** (mm³) unless the caller
explicitly rescales the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class MetricSet:
    """Container for a complete set of regression metrics."""

    mae: float
    rmse: float
    mape: float       # %
    smape: float      # %
    median_rel_error: float  # %
    r2: float
    mean_rel_error: float    # %
    n_samples: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_relative_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Per-sample relative errors E_i = |V_pred − V_true| / V_true.

    Returns
    -------
    (n,) array of relative errors (fraction, not %).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> MetricSet:
    """Compute all E5 metrics for a single method.

    Parameters
    ----------
    y_true : (n,) ground-truth volumes.
    y_pred : (n,) predicted volumes.

    Returns
    -------
    MetricSet
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)

    residuals = y_pred - y_true
    abs_residuals = np.abs(residuals)
    abs_true = np.abs(y_true)

    mae = float(np.mean(abs_residuals))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    rel_errors = abs_residuals / (abs_true + 1e-12)
    mean_rel = float(np.mean(rel_errors) * 100)
    med_rel = float(np.median(rel_errors) * 100)
    mape = mean_rel  # MAPE = mean relative error × 100

    smape_vals = abs_residuals / (
        (np.abs(y_pred) + np.abs(y_true)) / 2.0 + 1e-12
    )
    smape = float(np.mean(smape_vals) * 100)

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return MetricSet(
        mae=mae,
        rmse=rmse,
        mape=mape,
        smape=smape,
        median_rel_error=med_rel,
        r2=r2,
        mean_rel_error=mean_rel,
        n_samples=n,
    )


def format_metrics_table(
    method_names: list[str],
    metric_sets: list[MetricSet],
) -> str:
    """Pretty-print an ablation comparison table.

    Returns
    -------
    A formatted string table.
    """
    header = (
        f"{'Method':<20} {'MAE':>12} {'RMSE':>12} "
        f"{'MAPE%':>8} {'SMAPE%':>8} {'MedRE%':>8} "
        f"{'MeanRE%':>8} {'R²':>8} {'N':>5}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for name, ms in zip(method_names, metric_sets):
        lines.append(
            f"{name:<20} {ms.mae:>12.2f} {ms.rmse:>12.2f} "
            f"{ms.mape:>8.2f} {ms.smape:>8.2f} {ms.median_rel_error:>8.2f} "
            f"{ms.mean_rel_error:>8.2f} {ms.r2:>8.4f} {ms.n_samples:>5}"
        )
    return "\n".join(lines)

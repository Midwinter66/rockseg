"""Visualization helpers for volume validation results.

All figures are saved as PNG (300 dpi) to the configured output directory
and can also be returned as matplotlib Figure objects for inline use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import MetricSet


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method_name: str,
    color: str = "steelblue",
    save_path: Path | None = None,
) -> plt.Figure:
    """Scatter plot of predicted vs true volume with 1:1 line.

    Parameters
    ----------
    y_true, y_pred : volume arrays (same units).
    method_name : title label.
    color : scatter point colour.
    save_path : if given, save PNG to this path.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, s=20, alpha=0.5, color=color, edgecolors="none")

    # 1:1 reference line
    lim_min = min(float(y_true.min()), float(y_pred.min()))
    lim_max = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            "k--", lw=1, alpha=0.5, label="y = x")

    # Linear fit
    if len(y_true) > 2:
        z = np.polyfit(y_true, y_pred, 1)
        x_fit = np.linspace(lim_min, lim_max, 100)
        ax.plot(x_fit, np.polyval(z, x_fit), color="red", lw=1.5, alpha=0.7,
                label=f"fit: slope={z[0]:.3f}")

    ax.set_xlabel("True Volume (mm³)")
    ax.set_ylabel("Predicted Volume (mm³)")
    ax.set_title(f"{method_name}: Predicted vs True")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_error_boxplot(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    save_path: Path | None = None,
) -> plt.Figure:
    """Boxplot of per-sample relative errors for each method.

    Parameters
    ----------
    y_true : (n,) ground-truth volumes.
    predictions : {method_name: (n,) predicted volumes}.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    rel_errors = {}
    for name, y_pred in predictions.items():
        re = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12) * 100
        rel_errors[name] = re

    bp = ax.boxplot(
        list(rel_errors.values()),
        labels=list(rel_errors.keys()),
        patch_artist=True,
        showfliers=True,
        widths=0.5,
    )

    colors = ["#ff9999", "#99ccff", "#99ff99", "#ffcc99"]
    for patch, c in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Volume Estimation: Per-Sample Relative Error")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=20, color="red", linestyle="--", alpha=0.5, label="20% threshold")
    ax.legend(fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_ablation_bars(
    method_names: list[str],
    metric_sets: list[MetricSet],
    save_path: Path | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing MAPE and R² across methods.

    Parameters
    ----------
    method_names : e.g. ["Box", "Ellipsoid", "2.5D", "Shape-aware"].
    metric_sets : corresponding MetricSet objects.
    """
    n_methods = len(method_names)
    x = np.arange(n_methods)
    width = 0.18

    fig, ax1 = plt.subplots(figsize=(10, 5))

    mapes = [ms.mape for ms in metric_sets]
    smapes = [ms.smape for ms in metric_sets]
    med_res = [ms.median_rel_error for ms in metric_sets]

    bars1 = ax1.bar(x - width, mapes, width, label="MAPE (%)", color="#ff6b6b", alpha=0.8)
    bars2 = ax1.bar(x, smapes, width, label="SMAPE (%)", color="#4ecdc4", alpha=0.8)
    bars3 = ax1.bar(x + width, med_res, width, label="Median RE (%)", color="#45b7d1", alpha=0.8)

    ax1.set_xlabel("Method")
    ax1.set_ylabel("Error (%)")
    ax1.set_title("Volume Ablation: Error Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names, rotation=15)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    # Add R² as text annotation
    ax2 = ax1.twinx()
    r2s = [ms.r2 for ms in metric_sets]
    ax2.plot(x, r2s, "ko-", markersize=8, linewidth=2, label="R²", zorder=5)
    ax2.set_ylabel("R²", fontsize=10)
    ax2.set_ylim(min(0, min(r2s) - 0.1), 1.05)
    ax2.legend(loc="upper right", fontsize=8)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_per_group_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    method_name: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of mean relative error per group.

    Parameters
    ----------
    y_true, y_pred : (n,) volume arrays.
    groups : (n,) group labels.
    method_name : title label.
    """
    unique_groups = np.unique(groups)
    mean_errors = []
    for g in unique_groups:
        mask = groups == g
        re = np.abs(y_pred[mask] - y_true[mask]) / (np.abs(y_true[mask]) + 1e-12) * 100
        mean_errors.append(float(np.mean(re)))

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    bars = ax.bar(unique_groups, mean_errors, color=colors[:len(unique_groups)], alpha=0.8)

    for bar, val in zip(bars, mean_errors):
        ax.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Group")
    ax.set_ylabel("Mean Relative Error (%)")
    ax.set_title(f"{method_name}: Error by Source Group")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_feature_importance(
    importance: dict[str, float],
    save_path: Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of LightGBM feature importance.

    Parameters
    ----------
    importance : {feature_name: importance_value}.
    """
    if not importance:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No importance data", ha="center", va="center")
        return fig

    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Split Importance")
    ax.set_title("Shape-Aware Model: Feature Importance")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

"""
Stone statistics report and chart generation

Outputs:
  - Diameter distribution histogram
  - Volume distribution histogram
  - Volume vs diameter scatter plot
  - Cumulative volume curve
  - Stone size box plot
  - JSON summary report
"""

from __future__ import annotations
import json, math
from pathlib import Path
from typing import Any
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed, skipping chart generation")


def generate_all_charts(stones: list[dict], output_dir: str | Path,
                         config: dict | None = None) -> dict[str, Path]:
    """Generate all statistical charts

    Args:
        stones: list of stone dicts with volume_m3, equivalent_diameter_m, etc.
        output_dir: output directory for chart images
        config: optional config dict for bin settings

    Returns:
        {chart_name: file_path}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    charts = {}

    if not HAS_MPL:
        print("  [SKIP] matplotlib not installed")
        return charts

    # Extract data
    diams = np.array([s.get("equivalent_diameter_m", 0) for s in stones])
    vols = np.array([s.get("volume_m3", 0) for s in stones])
    areas = np.array([s.get("area_m2", 0) for s in stones])
    scores = np.array([s.get("score", s.get("score_avg", 1)) for s in stones])

    valid = (diams > 0) & (vols >= 0)
    diams, vols, areas, scores = diams[valid], vols[valid], areas[valid], scores[valid]

    if len(diams) == 0:
        print("  [WARN] No valid stone data for charts")
        return charts

    # ── 1. Diameter distribution ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = config.get("stats", {}).get("diameter_bins", 10) if config else 10
    ax.hist(diams, bins=bins, color="#3498db", edgecolor="white", alpha=0.85)
    ax.axvline(diams.mean(), color="#e74c3c", ls="--", lw=1.5,
               label=f"Mean: {diams.mean():.2f}m")
    ax.set_xlabel("Equivalent Diameter (m)")
    ax.set_ylabel("Stone Count")
    ax.set_title(f"Diameter Distribution (n={len(diams)})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    p = out / "diameter_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    charts["diameter_distribution"] = p

    # ── 2. Volume distribution ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    bins_v = config.get("stats", {}).get("volume_bins", 10) if config else 10
    ax.hist(vols, bins=bins_v, color="#2ecc71", edgecolor="white", alpha=0.85)
    ax.axvline(vols.mean(), color="#e74c3c", ls="--", lw=1.5,
               label=f"Mean: {vols.mean():.4f}m")
    ax.set_xlabel("Volume (m3)")
    ax.set_ylabel("Stone Count")
    ax.set_title(f"Volume Distribution (n={len(vols)})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    p = out / "volume_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    charts["volume_distribution"] = p

    # ── 3. Volume vs diameter scatter ────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(diams, vols, c=scores, cmap="viridis",
                         s=30, alpha=0.7, edgecolors="none")
    cbar = plt.colorbar(scatter, ax=ax, label="Confidence")
    if len(diams) > 3:
        coeffs = np.polyfit(diams, vols, 3)
        x_fit = np.linspace(diams.min(), diams.max(), 100)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, "r--", lw=1.5, alpha=0.7, label="Cubic fit")
    ax.set_xlabel("Equivalent Diameter (m)")
    ax.set_ylabel("Volume (m3)")
    ax.set_title("Volume vs Diameter")
    ax.legend()
    ax.grid(alpha=0.3)
    p = out / "volume_vs_diameter.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    charts["volume_vs_diameter"] = p

    # ── 4. Cumulative volume curve ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_idx = np.argsort(diams)[::-1]
    cum_vol = np.cumsum(vols[sorted_idx])
    cum_pct = cum_vol / cum_vol[-1] * 100
    ax.plot(np.arange(1, len(cum_pct)+1), cum_pct, color="#9b59b6", lw=2)
    ax.axhline(80, color="#e74c3c", ls="--", alpha=0.5, label="80% cumulative")
    ax.axhline(50, color="#f39c12", ls="--", alpha=0.5, label="50% cumulative")
    ax.set_xlabel("Stones (sorted by diameter descending)")
    ax.set_ylabel("Cumulative Volume (%)")
    ax.set_title(f"Cumulative Volume — Total: {cum_vol[-1]:.2f}m")
    ax.legend()
    ax.grid(alpha=0.3)
    p = out / "cumulative_volume.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    charts["cumulative_volume"] = p

    # ── 5. Box plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bp1 = axes[0].boxplot(diams, vert=True, patch_artist=True,
                          boxprops=dict(facecolor="#3498db", alpha=0.7),
                          medianprops=dict(color="red", lw=2))
    axes[0].set_ylabel("Equivalent Diameter (m)")
    axes[0].set_title(f"Diameter (n={len(diams)})")
    axes[0].grid(axis="y", alpha=0.3)

    bp2 = axes[1].boxplot(vols, vert=True, patch_artist=True,
                          boxprops=dict(facecolor="#2ecc71", alpha=0.7),
                          medianprops=dict(color="red", lw=2))
    axes[1].set_ylabel("Volume (m3)")
    axes[1].set_title(f"Volume (n={len(vols)})")
    axes[1].grid(axis="y", alpha=0.3)
    p = out / "boxplot.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    charts["boxplot"] = p

    # ── 6. Summary JSON ──────────────────────────────────────────
    stats_summary = compute_statistics(stones, diams, vols, areas)
    report_path = out / "statistics_summary.json"
    report_path.write_text(
        json.dumps(stats_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    charts["statistics_summary"] = report_path

    print(f"  Charts saved: {out}")
    for name, p in charts.items():
        print(f"    {name}: {p.name}")
    return charts


def compute_statistics(stones: list[dict], diams: np.ndarray | None = None,
                       vols: np.ndarray | None = None,
                       areas: np.ndarray | None = None) -> dict:
    """Compute comprehensive statistics summary"""
    if diams is None:
        diams = np.array([s.get("equivalent_diameter_m", 0) for s in stones])
    if vols is None:
        vols = np.array([s.get("volume_m3", 0) for s in stones])
    if areas is None:
        areas = np.array([s.get("area_m2", 0) for s in stones])

    valid_d = diams[diams > 0]
    valid_v = vols[vols >= 0]
    valid_a = areas[areas > 0]

    summary = {
        "total_stones": len(stones),
        "valid_stones": int(len(valid_d)),
        "diameter": {
            "min_m": round(float(valid_d.min()), 3) if len(valid_d) > 0 else 0,
            "max_m": round(float(valid_d.max()), 3) if len(valid_d) > 0 else 0,
            "mean_m": round(float(valid_d.mean()), 3) if len(valid_d) > 0 else 0,
            "median_m": round(float(np.median(valid_d)), 3) if len(valid_d) > 0 else 0,
            "std_m": round(float(valid_d.std()), 3) if len(valid_d) > 0 else 0,
        },
        "volume": {
            "min_m3": round(float(valid_v.min()), 4) if len(valid_v) > 0 else 0,
            "max_m3": round(float(valid_v.max()), 4) if len(valid_v) > 0 else 0,
            "mean_m3": round(float(valid_v.mean()), 4) if len(valid_v) > 0 else 0,
            "median_m3": round(float(np.median(valid_v)), 4) if len(valid_v) > 0 else 0,
            "total_m3": round(float(valid_v.sum()), 4) if len(valid_v) > 0 else 0,
            "std_m3": round(float(valid_v.std()), 4) if len(valid_v) > 0 else 0,
        },
        "area": {
            "min_m2": round(float(valid_a.min()), 3) if len(valid_a) > 0 else 0,
            "max_m2": round(float(valid_a.max()), 3) if len(valid_a) > 0 else 0,
            "mean_m2": round(float(valid_a.mean()), 3) if len(valid_a) > 0 else 0,
            "total_m2": round(float(valid_a.sum()), 3) if len(valid_a) > 0 else 0,
        },
        "size_buckets": {},
    }

    buckets = [
        (1.2, 1.5, "small (1.2-1.5m)"),
        (1.5, 2.0, "medium (1.5-2.0m)"),
        (2.0, 3.0, "large (2.0-3.0m)"),
        (3.0, 99, "xlarge (>3.0m)"),
    ]
    for lo, hi, label in buckets:
        cnt = int(np.sum((valid_d >= lo) & (valid_d < hi)))
        summary["size_buckets"][label] = {
            "count": cnt,
            "pct": round(cnt / max(len(valid_d), 1) * 100, 1),
        }

    return summary

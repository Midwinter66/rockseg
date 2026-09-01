"""Apply external-validation correction factor to real-scene volume estimates.

This is the bridge between E5 (external validation) and the actual scene:
    1. External validation proved: V_2.5D overestimates by ~37% (α=0.731)
    2. This script applies: V_corrected = α × V_2.5D to every stone
    3. Reports confidence intervals based on external validation error bounds

Usage::
    python -m research_v2.volume_validation.apply_correction_to_scene
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    _project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_project_root))
    from research_v2.volume_validation.config import Config
else:
    from .config import Config

logger = logging.getLogger("apply_correction")


# From external validation (E5 on 79 T01 rocks)
ALPHA = 0.731          # median correction ratio
ALPHA_STD = 0.053      # std of correction ratio (train set)
E_LINEAR_PCT = 8.05    # linear correction mean relative error (%)
E_SHAPE_PCT = 7.89     # shape-aware mean relative error (%)
MEDIAN_RE_PCT = 5.94   # median relative error (%)


def load_scene_volumes(stone_volumes_path: Path) -> list[dict]:
    """Load stone_volumes.json from the volume pipeline output."""
    with open(stone_volumes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["stones"]


def extract_volumes(stones: list[dict]) -> pd.DataFrame:
    """Extract volume and shape info from stone records."""
    rows = []
    for s in stones:
        m = s.get("methods", {})
        d25 = m.get("2d5", {})
        if not isinstance(d25, dict) or d25.get("status") != "ok":
            continue

        v_raw = float(d25.get("volume_m3", 0))
        if v_raw <= 0:
            continue

        pc = s.get("point_cloud", {})
        geo = s.get("geometry", {})
        proj = s.get("projected_shape", {})
        height_stats = d25.get("height_stats", {})

        rows.append({
            "stone_id": s.get("stone_id", ""),
            "V_2_5d_raw": v_raw,
            "V_corrected": v_raw * ALPHA,
            "V_corrected_lower": v_raw * (ALPHA - 2 * ALPHA_STD),
            "V_corrected_upper": v_raw * (ALPHA + 2 * ALPHA_STD),
            "point_count": pc.get("point_count", 0),
            "z_range_m": pc.get("z_range_m", 0),
            "equivalent_diameter_m": geo.get("equivalent_diameter_m", 0),
            "area_m2": geo.get("area_m2", proj.get("area_m2", 0)),
            "occupied_area_m2": d25.get("occupied_area_m2", 0),
            "grid_nx": d25.get("grid_nx", 0),
            "grid_ny": d25.get("grid_ny", 0),
            "occupied_cells": d25.get("occupied_cells", 0),
            "H_mean_m": height_stats.get("mean_m", 0),
            "H_max_m": height_stats.get("max_m", 0),
            "H_p90_m": height_stats.get("p90_m", 0),
            "H_std_m": height_stats.get("std_m", 0),
        })

    return pd.DataFrame(rows)


def compute_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute shape features needed for per-stone correction (transferable)."""
    df = df.copy()

    # H_mean / H_max (dimensionless, how flat)
    df["H_mean_norm"] = df["H_mean_m"] / (df["H_max_m"] + 1e-9)

    # H_std / H_max (dimensionless, how uneven)
    df["H_std_norm"] = df["H_std_m"] / (df["H_max_m"] + 1e-9)

    # Approximate circularity from area and perimeter
    # P ≈ 2×sqrt(π×A) for circle; C = 4πA/P²
    # We don't have P directly, but we can approximate from grid
    if "occupied_cells" in df.columns and "occupied_area_m2" in df.columns:
        # Approximate perimeter from grid boundary
        # Each boundary cell contributes ~grid_resolution of perimeter
        # Rough estimate: P ≈ 4 * sqrt(occupied_cells) * grid_res
        # This is an approximation
        df["C_approx"] = np.minimum(
            4.0 * np.pi * df["occupied_area_m2"] /
            (4.0 * np.sqrt(df["occupied_cells"] + 1) * np.sqrt(df["occupied_area_m2"] / (df["occupied_cells"] + 1)) + 1e-9) ** 2,
            1.0
        )
    else:
        df["C_approx"] = 0.5

    # Per-stone correction ratio prediction (simplified model)
    # Based on external validation feature importance: H_std_norm and AR matter most
    # Since we only have H_std_norm reliably, use a simple heuristic:
    # - Low H_std_norm (flat top) → more overestimation → lower r
    # - High H_std_norm (uneven top) → less overestimation → higher r
    r_base = ALPHA
    r_adjustment = (df["H_std_norm"] - 0.15) * 0.3  # centered on typical value
    df["r_predicted"] = np.clip(r_base + r_adjustment, 0.5, 1.0)
    df["V_shape_corrected"] = df["V_2_5d_raw"] * df["r_predicted"]

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Apply external-validation correction to scene volume estimates."
    )
    parser.add_argument(
        "--input", type=str,
        default="experiments/volume/outputs/quadtree_dom/correlation_clustering/stone_volumes.json",
        help="Path to stone_volumes.json",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="research_v2/volume_validation/output",
        help="Output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Applying External-Validation Correction to Scene Volumes")
    logger.info("=" * 70)
    logger.info("Input:  %s", input_path)
    logger.info("Alpha:  %.4f (from E5 external validation)", ALPHA)
    logger.info("Error:  %.1f%% (linear correction on external test set)", E_LINEAR_PCT)

    # Load
    stones = load_scene_volumes(input_path)
    logger.info("Loaded %d stones", len(stones))

    # Extract
    df = extract_volumes(stones)
    logger.info("Valid stones with 2.5D volume: %d", len(df))

    # Compute shape features and per-stone correction
    df = compute_shape_features(df)

    # ── Summary ────────────────────────────────────────────────────
    total_raw = df["V_2_5d_raw"].sum()
    total_linear = df["V_corrected"].sum()
    total_shape = df["V_shape_corrected"].sum()
    total_lower = df["V_corrected_lower"].sum()
    total_upper = df["V_corrected_upper"].sum()

    print("\n" + "=" * 70)
    print("Scene Volume Summary (6932 stones)")
    print("=" * 70)
    print(f"\n  2.5D raw (no correction):       {total_raw:>12.2f} m³")
    print(f"  Linear corrected (α=0.731):     {total_linear:>12.2f} m³")
    print(f"  Shape-aware corrected:          {total_shape:>12.2f} m³")
    print(f"\n  95% CI (linear):                [{total_lower:.2f}, {total_upper:.2f}] m³")
    print(f"\n  Correction reduced total by:    {(1 - total_linear/total_raw)*100:.1f}%")
    print(f"  Expected error per stone:       ±{E_LINEAR_PCT:.1f}% (from E5)")
    print(f"  Expected error (median):        ±{MEDIAN_RE_PCT:.1f}% (from E5)")

    # ── Size distribution ──────────────────────────────────────────
    print(f"\n{'Size Distribution':}")
    eq_d = df["equivalent_diameter_m"]
    for label, lo, hi in [
        ("fine  (<0.5m)", 0, 0.5),
        ("small (0.5-1m)", 0.5, 1.0),
        ("medium(1-2m)", 1.0, 2.0),
        ("large (2-5m)", 2.0, 5.0),
        ("boulder(>5m)", 5.0, 999),
    ]:
        mask = (eq_d >= lo) & (eq_d < hi)
        n = mask.sum()
        v = df.loc[mask, "V_corrected"].sum()
        if n > 0:
            print(f"    {label}: {n:>5} stones, {v:>10.2f} m³ ({v/total_linear*100:.1f}%)")

    # ── Save corrected CSV ─────────────────────────────────────────
    csv_path = output_dir / "scene_volumes_corrected.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved corrected volumes to %s", csv_path)

    # ── Save summary JSON ──────────────────────────────────────────
    summary = {
        "n_stones": len(df),
        "total_raw_m3": float(total_raw),
        "total_linear_corrected_m3": float(total_linear),
        "total_shape_corrected_m3": float(total_shape),
        "ci_95_lower_m3": float(total_lower),
        "ci_95_upper_m3": float(total_upper),
        "correction_alpha": ALPHA,
        "correction_alpha_std": ALPHA_STD,
        "expected_error_pct": E_LINEAR_PCT,
        "expected_median_error_pct": MEDIAN_RE_PCT,
        "source": "E5 external validation on 79 T01 rocks (Čapek et al. 2025)",
        "per_stone": {
            "mean_V_raw_m3": float(df["V_2_5d_raw"].mean()),
            "mean_V_corrected_m3": float(df["V_corrected"].mean()),
            "median_V_corrected_m3": float(df["V_corrected"].median()),
            "max_V_corrected_m3": float(df["V_corrected"].max()),
        },
    }
    json_path = output_dir / "scene_volume_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", json_path)

    print(f"\nResults saved to:")
    print(f"  {csv_path}")
    print(f"  {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

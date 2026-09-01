"""Validate V2 stones with the same 2.5D volume method used for V1.

Takes a subset of V2 rock_instances, converts pixel bbox → world coords,
crops point cloud, runs estimate_2d5_with_ground, applies α=0.731 correction,
and checks if fill ratio / height ratio match V1 and external validation.

Usage::
    python -m research_v2.volume_validation.validate_v2_subset --n-samples 200
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from experiments.common.scene_reference import CURRENT_SCENE
from experiments.common.stone_region import crop_stone_point_cloud, PointCloudXYGridIndex
from experiments.volume.ground_estimator import GroundDEM
from experiments.volume.estimators import estimate_2d5_with_ground

logger = logging.getLogger("validate_v2")

ALPHA = 0.731
ALPHA_STD = 0.053


def pixel_bbox_to_world(bbox_px, gt):
    """Convert pixel bbox [x1,y1,x2,y2] to world [x0,y0,x1,y1]."""
    x1_px, y1_px, x2_px, y2_px = bbox_px
    # Four corners
    wx1 = gt[0] + x1_px * gt[1] + y1_px * gt[2]
    wy1 = gt[3] + x1_px * gt[4] + y1_px * gt[5]
    wx2 = gt[0] + x2_px * gt[1] + y2_px * gt[2]
    wy2 = gt[3] + x2_px * gt[4] + y2_px * gt[5]
    return [min(wx1, wx2), min(wy1, wy2), max(wx1, wx2), max(wy1, wy2)]


def _mask_to_world_polygon(mask, bbox_px, gt, xy_transform):
    """Convert a local bbox-sized binary mask to a world-coordinate polygon.

    Uses the same logic as experiments.common.stone_region.mask_to_laz_polygon:
    extract boundary → convex hull → pixel_to_world → point_xy.
    """
    from scipy import ndimage
    from experiments.common.stone_region import pixel_to_world

    # Find boundary pixels
    if mask.sum() < 3:
        return None
    # Use convex hull of True pixels directly (faster than boundary tracing)
    coords = np.argwhere(mask)  # (row, col) = (y, x)
    if len(coords) < 3:
        return None

    # Convert to (x, y) for hull
    pts_xy = coords[:, ::-1].astype(np.float64)  # (col, row) = (x, y)

    # Convex hull
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts_xy)
        hull_pts = pts_xy[hull.vertices]
    except Exception:
        hull_pts = pts_xy

    if len(hull_pts) < 3:
        return None

    # Add bbox origin offset and convert to world coords
    ox, oy = bbox_px[0], bbox_px[1]  # pixel origin
    polygon = []
    for px, py in hull_pts:
        wx, wy = pixel_to_world(gt, px + ox, py + oy)
        px_pc, py_pc = xy_transform.world_to_point_xy(wx, wy)
        polygon.append([px_pc, py_pc])

    if len(polygon) < 3:
        return None
    return np.asarray([polygon], dtype=np.float32)


def load_point_cloud():
    """Load point cloud from LAZ files."""
    import laspy
    all_pts = []
    for path in CURRENT_SCENE.pointcloud_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        logger.info("Loading %s ...", path.name)
        las = laspy.read(str(path))
        pts = np.column_stack([las.x, las.y, las.z]).astype(np.float64, copy=False)
        all_pts.append(pts)
        logger.info("  %d points", len(pts))
    pc = np.vstack(all_pts)
    logger.info("Total points: %d", len(pc))
    return pc


def main():
    parser = argparse.ArgumentParser(description="Validate V2 stones with 2.5D volume")
    parser.add_argument("--n-samples", type=int, default=200, help="Number of V2 stones to sample")
    parser.add_argument("--min-diameter-m", type=float, default=0.3, help="Min equivalent diameter")
    parser.add_argument("--grid-resolution", type=float, default=0.05, help="Volume grid resolution (m)")
    parser.add_argument("--output-dir", type=str, default="research_v2/volume_validation/output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load V2 rock instances ──────────────────────────────────────
    v2_path = PROJECT_ROOT / "output" / "dom2_full" / "rock_instances.json"
    with open(v2_path, "r") as f:
        v2_rocks = json.load(f)
    logger.info("Loaded %d V2 rock instances", len(v2_rocks))

    GSD = 0.01
    for r in v2_rocks:
        r["area_m2"] = r["area"] * GSD**2
        r["eq_d_m"] = 2 * np.sqrt(r["area_m2"] / np.pi)

    # Filter by diameter and sample
    eligible = [r for r in v2_rocks if r["eq_d_m"] >= args.min_diameter_m]
    logger.info("Eligible (>= %.1fm): %d stones", args.min_diameter_m, len(eligible))

    # Sort by area descending (prefer larger stones) and take top N
    eligible.sort(key=lambda r: r["area_m2"], reverse=True)
    n_take = min(args.n_samples, len(eligible))
    # Sample across size range: take evenly spaced
    if len(eligible) > n_take:
        indices = np.linspace(0, len(eligible) - 1, n_take, dtype=int)
        sample = [eligible[i] for i in indices]
    else:
        sample = eligible[:n_take]
    logger.info("Sampled %d stones for volume estimation", len(sample))

    # ── Load point cloud and build GroundDEM ───────────────────────
    gt = CURRENT_SCENE.load_gt()
    logger.info("GT: origin=(%.1f, %.1f)  pixel=%.4fm", gt[0], gt[3], gt[1])

    t0 = time.time()
    pc = load_point_cloud()
    logger.info("Point cloud loaded in %.1fs", time.time() - t0)

    # Build spatial index
    t0 = time.time()
    pc_index = PointCloudXYGridIndex.build(pc, cell_size=1.0)
    logger.info("Spatial index built in %.1fs", time.time() - t0)

    # Build GroundDEM
    t0 = time.time()
    ground_dem = GroundDEM(
        pc,
        resolution=0.5,
        percentile=5,
        subsample_step=100,
        min_points_per_cell=3,
    )
    logger.info("GroundDEM built in %.1fs", time.time() - t0)

    # ── Load V2 masks ───────────────────────────────────────────────
    masks_path = PROJECT_ROOT / "output" / "dom2_full" / "rock_masks.npz"
    masks_data = np.load(masks_path, allow_pickle=True) if masks_path.exists() else None
    if masks_data is not None:
        logger.info("Loaded V2 masks: %d masks", len(masks_data.keys()))
    else:
        logger.warning("No V2 masks found, will use bbox only")

    # ── Estimate volume for each stone ────────────────────────────
    results = []
    xy_transform = CURRENT_SCENE.xy_transform

    for i, rock in enumerate(sample):
        bbox_px = rock["bbox"]
        bbox_world = pixel_bbox_to_world(bbox_px, gt)

        # Get bbox candidates from spatial index
        x0, y0, x1, y1 = xy_transform.world_bbox_to_point_bbox(bbox_world, pad_m=0.5)
        if pc_index is not None:
            candidate_indices = pc_index.query_bbox_indices(x0, y0, x1, y1)
            candidates = pc[candidate_indices].copy() if len(candidate_indices) > 0 else np.empty((0, 3))
        else:
            mask_xy = ((pc[:, 0] >= x0) & (pc[:, 0] <= x1) & (pc[:, 1] >= y0) & (pc[:, 1] <= y1))
            candidates = pc[mask_xy].copy()

        # If we have a mask, use it for precise polygon filtering
        if masks_data is not None:
            mask_key = f"{rock['instance_id']}_mask"
            if mask_key in masks_data:
                stone_mask = masks_data[mask_key]
                # Convert mask to world polygon
                polygon = _mask_to_world_polygon(stone_mask, bbox_px, gt, xy_transform)
                if polygon is not None and len(candidates) > 0:
                    from experiments.common.stone_region import _points_in_polygon
                    inside = _points_in_polygon(candidates[:, :2], polygon[0])
                    stone_pts = candidates[inside].copy()
                else:
                    stone_pts = candidates
            else:
                stone_pts = candidates
        else:
            stone_pts = candidates

        pt_count = len(stone_pts)
        if pt_count < 10:
            results.append({
                "stone_id": rock["instance_id"],
                "scale_level": rock["scale_level"],
                "eq_d_m": rock["eq_d_m"],
                "area_m2": rock["area_m2"],
                "point_count": pt_count,
                "status": "too_few_points",
                "V_2_5d": 0,
            })
            continue

        # Estimate 2.5D volume
        vol_result = estimate_2d5_with_ground(
            stone_pts, ground_dem, grid_resolution=args.grid_resolution
        )

        hs = vol_result.get("height_stats", {})
        results.append({
            "stone_id": rock["instance_id"],
            "scale_level": rock["scale_level"],
            "eq_d_m": rock["eq_d_m"],
            "area_m2": rock["area_m2"],
            "point_count": pt_count,
            "status": vol_result.get("status", "unknown"),
            "V_2_5d": vol_result.get("volume_m3", 0),
            "occupied_area_m2": vol_result.get("occupied_area_m2", 0),
            "h_mean_m": hs.get("mean_m", 0),
            "h_max_m": hs.get("max_m", 0),
            "h_std_m": hs.get("std_m", 0),
            "z_range_m": float(np.ptp(stone_pts[:, 2])),
        })

        if (i + 1) % 50 == 0:
            logger.info("Processed %d/%d stones", i + 1, len(sample))

    # ── Analyze results ─────────────────────────────────────────────
    df = pd.DataFrame(results)
    valid = df[df["status"] == "ok"].copy()
    valid["V_corrected"] = valid["V_2_5d"] * ALPHA
    valid["fill_ratio"] = valid["V_2_5d"] / (valid["occupied_area_m2"] * valid["h_max_m"] + 1e-9)
    valid["h_mean_norm"] = valid["h_mean_m"] / (valid["h_max_m"] + 1e-9)

    print("\n" + "=" * 70)
    print(f"V2 Volume Validation ({len(sample)} stones sampled, {len(valid)} valid)")
    print("=" * 70)

    print(f"\nStatus distribution:")
    for st, cnt in df["status"].value_counts().items():
        print(f"  {st}: {cnt} ({cnt/len(df)*100:.1f}%)")

    if len(valid) == 0:
        print("\nNo valid volume estimates. Check point cloud coverage.")
        return 1

    print(f"\n--- V2 Volume Results (n={len(valid)}) ---")
    print(f"  V_2.5D raw:   sum={valid['V_2_5d'].sum():.2f} m3  mean={valid['V_2_5d'].mean():.4f}  median={valid['V_2_5d'].median():.4f}")
    print(f"  V_corrected:  sum={valid['V_corrected'].sum():.2f} m3  mean={valid['V_corrected'].mean():.4f}  median={valid['V_corrected'].median():.4f}")

    print(f"\n--- Shape Consistency (V2 vs V1 vs External) ---")
    print(f"  {'Metric':<25} {'V2 (this)':<15} {'V1 (6929)':<15} {'External (79)':<15}")
    print(f"  {'-'*70}")
    print(f"  {'Fill ratio mean':<25} {valid['fill_ratio'].mean():<15.3f} {'0.489':<15} {'0.49':<15}")
    print(f"  {'Fill ratio median':<25} {valid['fill_ratio'].median():<15.3f} {'0.500':<15} {'—':<15}")
    print(f"  {'Fill ratio std':<25} {valid['fill_ratio'].std():<15.3f} {'0.156':<15} {'—':<15}")
    print(f"  {'H_mean/H_max mean':<25} {valid['h_mean_norm'].mean():<15.3f} {'0.532':<15} {'0.53':<15}")
    print(f"  {'H_mean/H_max std':<25} {valid['h_mean_norm'].std():<15.3f} {'0.043':<15} {'—':<15}")

    print(f"\n--- Point Cloud Quality ---")
    print(f"  Point count: mean={valid['point_count'].mean():.0f}  median={valid['point_count'].median():.0f}")
    print(f"  Z range:    mean={valid['z_range_m'].mean():.3f}m  median={valid['z_range_m'].median():.3f}m")

    print(f"\n--- Scale Level Distribution ---")
    for sl in ["coarse", "medium", "fine"]:
        sub = valid[valid["scale_level"] == sl]
        if len(sub) > 0:
            print(f"  {sl}: n={len(sub)}  V_mean={sub['V_2_5d'].mean():.4f}  fill={sub['fill_ratio'].mean():.3f}")

    # Outlier detection
    normal = (valid["fill_ratio"] > 0.15) & (valid["fill_ratio"] < 0.90) & \
             (valid["h_mean_norm"] > 0.15) & (valid["h_mean_norm"] < 0.95)
    print(f"\n--- Quality ---")
    print(f"  Normal: {normal.sum()} ({normal.sum()/len(valid)*100:.1f}%)")
    print(f"  Outliers: {(~normal).sum()} ({(~normal).sum()/len(valid)*100:.1f}%)")

    clean = valid[normal]
    if len(clean) > 0:
        print(f"\n--- Corrected Volume (clean, n={len(clean)}) ---")
        print(f"  Raw 2.5D:  {clean['V_2_5d'].sum():.2f} m3")
        print(f"  Corrected: {clean['V_corrected'].sum():.2f} m3 (α={ALPHA})")
        print(f"  Error:     ~8% (from E5 external validation)")

    print(f"\n--- Conclusion ---")
    v2_fill = valid["fill_ratio"].mean()
    v2_h = valid["h_mean_norm"].mean()
    match_fill = abs(v2_fill - 0.49) < 0.1
    match_h = abs(v2_h - 0.53) < 0.1
    print(f"  Fill ratio match: {'YES' if match_fill else 'NO'} (V2={v2_fill:.3f} vs expected=0.49)")
    print(f"  Height ratio match: {'YES' if match_h else 'NO'} (V2={v2_h:.3f} vs expected=0.53)")
    if match_fill and match_h:
        print(f"  => α={ALPHA} correction is valid for V2 data")
    else:
        print(f"  => Shape distribution differs, α may need recalibration")

    # Save
    csv_path = output_dir / "v2_subset_volumes.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved to %s", csv_path)

    summary = {
        "n_sampled": len(sample),
        "n_valid": len(valid),
        "n_clean": int(normal.sum()),
        "total_raw_m3": float(valid["V_2_5d"].sum()),
        "total_corrected_m3": float(valid["V_corrected"].sum()),
        "fill_ratio_mean": float(v2_fill),
        "fill_ratio_median": float(valid["fill_ratio"].median()),
        "h_mean_norm_mean": float(v2_h),
        "v1_fill_ratio_mean": 0.489,
        "external_fill_ratio_mean": 0.49,
        "alpha": ALPHA,
        "match_v1": bool(match_fill and match_h),
    }
    json_path = output_dir / "v2_subset_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", json_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

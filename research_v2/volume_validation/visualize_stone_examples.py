"""Visualize individual stone volume estimation examples.

Picks 6 representative stones (small/medium/large) from V2,
shows: mask, point cloud, 2.5D height grid, volume breakdown.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from experiments.common.scene_reference import CURRENT_SCENE
from experiments.common.stone_region import PointCloudXYGridIndex, _points_in_polygon
from experiments.common.stone_region import pixel_to_world
from experiments.volume.ground_estimator import GroundDEM
from experiments.volume.estimators import estimate_2d5_with_ground

ALPHA = 0.731


def pixel_bbox_to_world(bbox_px, gt):
    x1_px, y1_px, x2_px, y2_px = bbox_px
    wx1 = gt[0] + x1_px * gt[1] + y1_px * gt[2]
    wy1 = gt[3] + x1_px * gt[4] + y1_px * gt[5]
    wx2 = gt[0] + x2_px * gt[1] + y2_px * gt[2]
    wy2 = gt[3] + x2_px * gt[4] + y2_px * gt[5]
    return [min(wx1, wx2), min(wy1, wy2), max(wx1, wx2), max(wy1, wy2)]


def mask_to_polygon(mask, bbox_px, gt, xy_transform):
    if mask.sum() < 3:
        return None
    coords = np.argwhere(mask)
    pts_xy = coords[:, ::-1].astype(np.float64)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts_xy)
        hull_pts = pts_xy[hull.vertices]
    except Exception:
        hull_pts = pts_xy
    if len(hull_pts) < 3:
        return None
    ox, oy = bbox_px[0], bbox_px[1]
    polygon = []
    for px, py in hull_pts:
        wx, wy = pixel_to_world(gt, px + ox, py + oy)
        px_pc, py_pc = xy_transform.world_to_point_xy(wx, wy)
        polygon.append([px_pc, py_pc])
    return np.asarray(polygon, dtype=np.float32) if len(polygon) >= 3 else None


def load_point_cloud():
    import laspy
    all_pts = []
    for path in CURRENT_SCENE.pointcloud_paths:
        las = laspy.read(str(path))
        pts = np.column_stack([las.x, las.y, las.z]).astype(np.float64, copy=False)
        all_pts.append(pts)
    return np.vstack(all_pts)


def crop_stone(pc, pc_index, ground_dem, mask, bbox_px, gt, xy_transform, grid_res=0.05):
    """Full pipeline for one stone: crop → ground → 2.5D volume."""
    bbox_world = pixel_bbox_to_world(bbox_px, gt)
    x0, y0, x1, y1 = xy_transform.world_bbox_to_point_bbox(bbox_world, pad_m=0.5)

    candidate_indices = pc_index.query_bbox_indices(x0, y0, x1, y1)
    candidates = pc[candidate_indices].copy() if len(candidate_indices) > 0 else np.empty((0, 3))

    polygon = mask_to_polygon(mask, bbox_px, gt, xy_transform)
    if polygon is not None and len(candidates) > 0:
        inside = _points_in_polygon(candidates[:, :2], polygon)
        stone_pts = candidates[inside].copy()
    else:
        stone_pts = candidates

    if len(stone_pts) < 10:
        return None

    vol_result = estimate_2d5_with_ground(stone_pts, ground_dem, grid_resolution=grid_res)

    # Also compute the 2.5D height grid manually for visualization
    xy = stone_pts[:, :2]
    z = stone_pts[:, 2]
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    nx = max(1, int(np.ceil((xmax - xmin) / grid_res)))
    ny = max(1, int(np.ceil((ymax - ymin) / grid_res)))

    xi = np.floor((xy[:, 0] - xmin) / grid_res).astype(np.int32)
    yi = np.floor((xy[:, 1] - ymin) / grid_res).astype(np.int32)
    xi = np.clip(xi, 0, nx - 1)
    yi = np.clip(yi, 0, ny - 1)

    cell_z_max = np.full((ny, nx), np.nan)
    np.maximum.at(cell_z_max, (yi, xi), z)

    # Ground heights
    cell_rows, cell_cols = np.nonzero(~np.isnan(cell_z_max))
    cx = xmin + (cell_cols + 0.5) * grid_res
    cy = ymin + (cell_rows + 0.5) * grid_res
    ground_z = np.asarray(ground_dem.get_ground_z(cx, cy), dtype=np.float64)

    height_grid = np.full((ny, nx), np.nan)
    valid = np.isfinite(ground_z) & np.isfinite(cell_z_max[cell_rows, cell_cols])
    heights = cell_z_max[cell_rows, cell_cols] - ground_z
    valid_pos = valid & (heights > 0)
    height_grid[cell_rows[valid_pos], cell_cols[valid_pos]] = heights[valid_pos]

    return {
        "stone_pts": stone_pts,
        "vol_result": vol_result,
        "height_grid": height_grid,
        "grid_res": grid_res,
        "xmin": xmin, "ymin": ymin, "nx": nx, "ny": ny,
        "polygon": polygon,
    }


def plot_stone_example(ax_mask, ax_pc, ax_grid, ax_info, rock, mask, result, stone_idx):
    """Plot one stone in 4 subplots."""
    # 1. Mask
    ax_mask.imshow(mask, cmap="gray_r")
    ax_mask.set_title(f"{rock['instance_id']} ({rock['scale_level']})", fontsize=9)
    ax_mask.axis("off")

    # 2. Point cloud (top-down view, colored by height)
    pts = result["stone_pts"]
    sc = ax_pc.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.3, cmap="terrain", alpha=0.6)
    ax_pc.set_aspect("equal")
    ax_pc.set_title("Point Cloud (top view)", fontsize=9)
    ax_pc.set_xlabel("X (m)", fontsize=7)
    ax_pc.set_ylabel("Y (m)", fontsize=7)
    ax_pc.tick_params(labelsize=6)
    plt.colorbar(sc, ax=ax_pc, label="Z (m)", shrink=0.8)

    # 3. Height grid
    hg = result["height_grid"]
    im = ax_grid.imshow(hg, origin="lower", cmap="YlOrRd", interpolation="nearest")
    ax_grid.set_title("2.5D Height Grid", fontsize=9)
    ax_grid.set_xlabel(f"X cell ({result['grid_res']}m)", fontsize=7)
    ax_grid.set_ylabel(f"Y cell ({result['grid_res']}m)", fontsize=7)
    ax_grid.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax_grid, label="Height (m)", shrink=0.8)

    # 4. Info text
    vr = result["vol_result"]
    hs = vr.get("height_stats", {})
    v_raw = vr.get("volume_m3", 0)
    v_corr = v_raw * ALPHA
    area = vr.get("occupied_area_m2", 0)
    h_mean = hs.get("mean_m", 0)
    h_max = hs.get("max_m", 0)
    fill = v_raw / (area * h_max + 1e-9) if area > 0 and h_max > 0 else 0
    pt_count = len(pts)

    info_text = (
        f"Stone: {rock['instance_id']}\n"
        f"Scale: {rock['scale_level']}\n"
        f"Eq diameter: {rock['eq_d_m']:.3f} m\n"
        f"Mask area: {rock['area_m2']:.3f} m2\n"
        f"\n--- Point Cloud ---\n"
        f"Points: {pt_count}\n"
        f"Z range: {pts[:,2].max()-pts[:,2].min():.3f} m\n"
        f"\n--- 2.5D Volume ---\n"
        f"Occupied area: {area:.3f} m2\n"
        f"H_mean: {h_mean:.3f} m\n"
        f"H_max:  {h_max:.3f} m\n"
        f"Fill ratio: {fill:.3f}\n"
        f"\n--- Volume ---\n"
        f"V_2.5D raw:   {v_raw:.4f} m3\n"
        f"V corrected:  {v_corr:.4f} m3\n"
        f"  (alpha={ALPHA})\n"
        f"Error: ~8%"
    )
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=8, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax_info.axis("off")
    ax_info.set_title("Volume Breakdown", fontsize=9)


def main():
    print("=== Stone Volume Visualization ===")

    # Load V2 data
    v2_path = PROJECT_ROOT / "output" / "dom2_full" / "rock_instances.json"
    with open(v2_path) as f:
        v2_rocks = json.load(f)
    GSD = 0.01
    for r in v2_rocks:
        r["area_m2"] = r["area"] * GSD**2
        r["eq_d_m"] = 2 * np.sqrt(r["area_m2"] / np.pi)

    masks_data = np.load(PROJECT_ROOT / "output" / "dom2_full" / "rock_masks.npz", allow_pickle=True)

    # Load point cloud
    gt = CURRENT_SCENE.load_gt()
    print("Loading point cloud...")
    pc = load_point_cloud()
    print(f"  {len(pc):,} points")

    print("Building spatial index...")
    pc_index = PointCloudXYGridIndex.build(pc, cell_size=1.0)

    print("Building GroundDEM...")
    ground_dem = GroundDEM(pc, resolution=0.5, percentile=5, subsample_step=100, min_points_per_cell=3)

    # Select 6 stones: 2 small, 2 medium, 2 large
    eligible = [r for r in v2_rocks if r["eq_d_m"] >= 0.3]
    eligible.sort(key=lambda r: r["area_m2"], reverse=True)

    # Pick from different size ranges
    large = [r for r in eligible if r["eq_d_m"] >= 1.0][:2]
    medium = [r for r in eligible if 0.5 <= r["eq_d_m"] < 1.0][:2]
    small = [r for r in eligible if 0.3 <= r["eq_d_m"] < 0.5][:2]
    selected = large + medium + small
    print(f"Selected {len(selected)} stones: {len(large)} large, {len(medium)} medium, {len(small)} small")

    xy_transform = CURRENT_SCENE.xy_transform
    output_dir = PROJECT_ROOT / "research_v2" / "volume_validation" / "output" / "stone_examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, rock in enumerate(selected):
        mask_key = f"{rock['instance_id']}_mask"
        if mask_key not in masks_data:
            print(f"  Skip {rock['instance_id']}: no mask")
            continue

        mask = masks_data[mask_key]
        print(f"  [{idx+1}/{len(selected)}] {rock['instance_id']} eq_d={rock['eq_d_m']:.2f}m ...", end=" ")

        result = crop_stone(pc, pc_index, ground_dem, mask, rock["bbox"], gt, xy_transform)
        if result is None or result["vol_result"].get("status") != "ok":
            print("FAILED")
            continue

        # Create figure with 4 subplots
        fig = plt.figure(figsize=(16, 4))
        gs = GridSpec(1, 4, width_ratios=[1, 1.2, 1.2, 1.5], figure=fig)
        ax_mask = fig.add_subplot(gs[0])
        ax_pc = fig.add_subplot(gs[1])
        ax_grid = fig.add_subplot(gs[2])
        ax_info = fig.add_subplot(gs[3])

        plot_stone_example(ax_mask, ax_pc, ax_grid, ax_info, rock, mask, result, idx)

        vr = result["vol_result"]
        v_raw = vr.get("volume_m3", 0)
        v_corr = v_raw * ALPHA
        fig.suptitle(
            f"{rock['instance_id']}  |  eq_d={rock['eq_d_m']:.2f}m  |  "
            f"V_2.5D={v_raw:.3f} m3  |  V_corrected={v_corr:.3f} m3  |  "
            f"pts={len(result['stone_pts'])}",
            fontsize=10, fontweight="bold"
        )
        plt.tight_layout()

        out_path = output_dir / f"stone_{idx+1}_{rock['instance_id']}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved → {out_path.name}")

    # Summary figure: all 6 stones side by side (mask + volume bar)
    print("\nCreating summary figure...")
    fig, axes = plt.subplots(2, len(selected), figsize=(4*len(selected), 8))
    if len(selected) == 1:
        axes = axes.reshape(2, 1)

    for idx, rock in enumerate(selected):
        mask_key = f"{rock['instance_id']}_mask"
        if mask_key not in masks_data:
            continue
        mask = masks_data[mask_key]
        result = crop_stone(pc, pc_index, ground_dem, mask, rock["bbox"], gt, xy_transform)
        if result is None or result["vol_result"].get("status") != "ok":
            continue

        # Top row: mask
        axes[0, idx].imshow(mask, cmap="gray_r")
        axes[0, idx].set_title(f"{rock['instance_id']}\n({rock['scale_level']}, d={rock['eq_d_m']:.2f}m)", fontsize=8)
        axes[0, idx].axis("off")

        # Bottom row: height grid
        hg = result["height_grid"]
        im = axes[1, idx].imshow(hg, origin="lower", cmap="YlOrRd", interpolation="nearest")
        vr = result["vol_result"]
        v_raw = vr.get("volume_m3", 0)
        v_corr = v_raw * ALPHA
        axes[1, idx].set_title(f"V_raw={v_raw:.3f} m3\nV_corr={v_corr:.3f} m3", fontsize=8)
        axes[1, idx].axis("off")
        plt.colorbar(im, ax=axes[1, idx], shrink=0.6, label="H (m)")

    fig.suptitle("Stone Volume Examples: Mask (top) → 2.5D Height Grid (bottom)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    summary_path = output_dir / "summary_all_stones.png"
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary saved → {summary_path.name}")

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

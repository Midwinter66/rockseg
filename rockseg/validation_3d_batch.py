"""3D point cloud screening — optimized batch version.

Processes all rock instances in batches for much better performance.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Validation3DConfig:
    """Configuration for 3D point cloud validation."""
    dem_resolution_m: float = 0.5
    dem_percentile: int = 5
    dem_subsample_step: int = 100
    dem_min_points_per_cell: int = 3
    bbox_pad_m: float = 0.5
    index_cell_size_m: float = 1.0
    min_points: int = 60
    min_z_range_m: float = 0.18
    elevated_height_m: float = 0.08
    min_p90_height_m: float = 0.12
    min_elevated_ratio: float = 0.2


def load_point_cloud(laz_paths: Sequence[str | Path]) -> np.ndarray:
    """Load LAZ files into Nx3 float64 array."""
    import laspy
    all_pts = []
    for p in laz_paths:
        p = Path(p)
        if not p.exists():
            logger.warning("Point cloud file not found: %s", p)
            continue
        logger.info("Loading point cloud: %s", p.name)
        las = laspy.read(str(p))
        pts = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
        all_pts.append(pts)
        logger.info("  %d points", len(pts))
    if not all_pts:
        raise RuntimeError("No point cloud files loaded")
    result = np.vstack(all_pts)
    logger.info("Total point cloud: %d points", len(result))
    return result


class GroundDEM:
    """Ground elevation model from point cloud."""

    def __init__(self, pc, resolution=0.5, percentile=5, subsample_step=100, min_points_per_cell=3):
        pts = np.asarray(pc, dtype=np.float64)
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        sub = pts[::subsample_step]
        self.resolution = float(resolution)
        self.percentile = int(percentile)
        xmin, ymin = float(sub[:, 0].min()), float(sub[:, 1].min())
        xmax, ymax = float(sub[:, 0].max()), float(sub[:, 1].max())
        nx = max(2, int(np.ceil((xmax - xmin) / self.resolution)))
        ny = max(2, int(np.ceil((ymax - ymin) / self.resolution)))
        self.xmin, self.ymin = xmin, ymin
        self.xmax, self.ymax = xmin + nx * self.resolution, ymin + ny * self.resolution
        self.nx, self.ny = nx, ny

        xi = np.clip(np.floor((sub[:, 0] - xmin) / self.resolution).astype(np.int32), 0, nx - 1)
        yi = np.clip(np.floor((sub[:, 1] - ymin) / self.resolution).astype(np.int32), 0, ny - 1)
        flat_idx = yi * nx + xi
        order = np.argsort(flat_idx)
        sorted_idx = flat_idx[order]
        sorted_z = sub[order, 2]
        unique_idx, starts, counts = np.unique(sorted_idx, return_index=True, return_counts=True)

        self.grid = np.full((ny, nx), np.nan, dtype=np.float64)
        for uid, start, cnt in zip(unique_idx, starts, counts):
            if cnt < min_points_per_cell:
                continue
            self.grid.flat[int(uid)] = float(np.percentile(sorted_z[start:start + cnt], self.percentile))
        self._fill_holes()
        logger.info("GroundDEM: %d x %d = %d cells, coverage %.1f%%",
                     nx, ny, nx * ny, np.sum(~np.isnan(self.grid)) / (nx * ny) * 100)

    def _fill_holes(self):
        valid = ~np.isnan(self.grid)
        if valid.all():
            return
        grid = self.grid.copy()
        ny, nx = grid.shape
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for _ in range(max(nx, ny)):
            mask = np.isnan(grid)
            if not mask.any():
                break
            updated = False
            for dy, dx in offsets:
                src = np.roll(grid, shift=(-dy, -dx), axis=(0, 1))
                fill = mask & ~np.isnan(src)
                if dy > 0: fill[-dy:, :] = False
                elif dy < 0: fill[:-dy, :] = False
                if dx > 0: fill[:, -dx:] = False
                elif dx < 0: fill[:, :-dx] = False
                if fill.any():
                    grid[fill] = src[fill]
                    updated = True
            if not updated:
                break
        self.grid = grid

    def get_ground_z(self, x, y):
        xa = np.atleast_1d(np.asarray(x, dtype=np.float64))
        ya = np.atleast_1d(np.asarray(y, dtype=np.float64))
        fx = (xa - self.xmin) / self.resolution
        fy = (ya - self.ymin) / self.resolution
        x0 = np.clip(np.floor(fx).astype(np.int32), 0, self.nx - 1)
        y0 = np.clip(np.floor(fy).astype(np.int32), 0, self.ny - 1)
        x1 = np.clip(x0 + 1, 0, self.nx - 1)
        y1 = np.clip(y0 + 1, 0, self.ny - 1)
        wx = fx - x0
        wy = fy - y0
        z = (self.grid[y0, x0] * (1 - wx) * (1 - wy)
             + self.grid[y0, x1] * wx * (1 - wy)
             + self.grid[y1, x0] * (1 - wx) * wy
             + self.grid[y1, x1] * wx * wy)
        outside = (fx < 0) | (fy < 0) | (fx > self.nx - 1) | (fy > self.ny - 1)
        z[outside] = np.nan
        return z


class PointCloudGridIndex:
    """Fast XY grid index for point cloud bbox queries."""

    def __init__(self, points, cell_size=1.0):
        pts = np.asarray(points, dtype=np.float64)
        self.points = pts
        self.cell_size = float(cell_size)
        self.xmin, self.ymin = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
        self.xmax, self.ymax = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
        nx = max(1, int(np.floor((self.xmax - self.xmin) / cell_size)) + 1)
        ny = max(1, int(np.floor((self.ymax - self.ymin) / cell_size)) + 1)
        self.nx, self.ny = nx, ny
        xi = np.clip(np.floor((pts[:, 0] - self.xmin) / cell_size).astype(np.int32), 0, nx - 1)
        yi = np.clip(np.floor((pts[:, 1] - self.ymin) / cell_size).astype(np.int32), 0, ny - 1)
        flat = yi.astype(np.int64) * np.int64(nx) + xi.astype(np.int64)
        order = np.argsort(flat, kind="mergesort")
        flat_sorted = flat[order]
        unique_ids, starts, counts = np.unique(flat_sorted, return_index=True, return_counts=True)
        self._order = order.astype(np.int32)
        self._unique_ids = unique_ids.astype(np.int64)
        self._starts = starts.astype(np.int64)
        self._counts = counts.astype(np.int64)

    def query_bbox_indices(self, x0, y0, x1, y1):
        gx0 = max(0, min(self.nx - 1, int(np.floor((x0 - self.xmin) / self.cell_size))))
        gy0 = max(0, min(self.ny - 1, int(np.floor((y0 - self.ymin) / self.cell_size))))
        gx1 = max(0, min(self.nx - 1, int(np.floor((x1 - self.xmin) / self.cell_size))))
        gy1 = max(0, min(self.ny - 1, int(np.floor((y1 - self.ymin) / self.cell_size))))
        if gx1 < gx0 or gy1 < gy0:
            return np.empty(0, dtype=np.int32)
        xs = np.arange(gx0, gx1 + 1, dtype=np.int64)
        ys = np.arange(gy0, gy1 + 1, dtype=np.int64)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        cell_ids = (gy.reshape(-1) * np.int64(self.nx) + gx.reshape(-1)).astype(np.int64)
        pos = np.searchsorted(self._unique_ids, cell_ids)
        valid = ((pos >= 0) & (pos < len(self._unique_ids))
                 & (self._unique_ids[np.clip(pos, 0, len(self._unique_ids) - 1)] == cell_ids))
        if not np.any(valid):
            return np.empty(0, dtype=np.int32)
        parts = [self._order[int(self._starts[idx]):int(self._starts[idx]) + int(self._counts[idx])]
                 for idx in pos[valid]]
        if not parts:
            return np.empty(0, dtype=np.int32)
        candidates = np.concatenate(parts)
        pts = self.points[candidates]
        mask = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1)
                & (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
        return candidates[mask]


# ---------------------------------------------------------------------------
# Batch 3D validation (much faster than per-instance)
# ---------------------------------------------------------------------------

def run_3d_validation(
    instances: list[dict],
    masks: list[np.ndarray],
    laz_paths: Sequence[str | Path],
    transform,  # rasterio Affine
    config: Validation3DConfig | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Batch 3D validation using spatial grouping for speed.

    Strategy: group instances by point-cloud grid cell, process all instances
    in a cell together (single point cloud query per cell).
    """
    if config is None:
        config = Validation3DConfig()

    logger.info("=== 3D Point Cloud Validation (batch) ===")

    pc = load_point_cloud(laz_paths)
    ground_dem = GroundDEM(pc, resolution=config.dem_resolution_m,
                           percentile=config.dem_percentile,
                           subsample_step=config.dem_subsample_step,
                           min_points_per_cell=config.dem_min_points_per_cell)
    pc_index = PointCloudGridIndex(pc, cell_size=config.index_cell_size_m)
    inv_transform = ~transform

    n = len(instances)
    results_passed = np.ones(n, dtype=bool)
    results_reasons: list[list[str]] = [[] for _ in range(n)]
    results_point_count = np.zeros(n, dtype=np.int32)
    results_z_range = np.zeros(n, dtype=np.float32)
    results_p90 = np.zeros(n, dtype=np.float32)
    results_elev_ratio = np.zeros(n, dtype=np.float32)

    # Group instances by point cloud index cell for batch processing
    # Use larger cells (5m) to batch more instances per query
    batch_cell_m = 5.0
    cell_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, inst in enumerate(instances):
        x1, y1, x2, y2 = inst["bbox"]
        # Center of bbox in world coords
        cx_px = (x1 + x2) / 2
        cy_px = (y1 + y2) / 2
        cx_w, cy_w = transform * (cx_px, cy_px)
        gx = int(np.floor((cx_w - pc_index.xmin) / batch_cell_m))
        gy = int(np.floor((cy_w - pc_index.ymin) / batch_cell_m))
        cell_groups[(gx, gy)].append(i)

    logger.info("Grouped %d instances into %d cells (%.1fm grid)",
                 n, len(cell_groups), batch_cell_m)

    processed = 0
    for (gx, gy), inst_indices in cell_groups.items():
        # Get bbox covering all instances in this cell
        xmins, ymins, xmaxs, ymaxs = [], [], [], []
        for i in inst_indices:
            x1, y1, x2, y2 = instances[i]["bbox"]
            wx0, wy0 = transform * (x1 - config.bbox_pad_m / transform[0],
                                     y1 - config.bbox_pad_m / abs(transform[4]))
            wx1, wy1 = transform * (x2 + config.bbox_pad_m / transform[0],
                                     y2 + config.bbox_pad_m / abs(transform[4]))
            if wx0 > wx1: wx0, wx1 = wx1, wx0
            if wy0 > wy1: wy0, wy1 = wy1, wy0
            xmins.append(wx0); ymins.append(wy0); xmaxs.append(wx1); ymaxs.append(wy1)

        # Query all points in the combined bbox
        qx0 = min(xmins)
        qy0 = min(ymins)
        qx1 = max(xmaxs)
        qy1 = max(ymaxs)

        point_idx = pc_index.query_bbox_indices(qx0, qy0, qx1, qy1)
        if len(point_idx) == 0:
            for i in inst_indices:
                results_passed[i] = False
                results_reasons[i] = ["no_points_in_bbox"]
            processed += len(inst_indices)
            continue

        pts = pc_index.points[point_idx]  # (M, 3)

        # Ground elevation for all points
        ground_z = ground_dem.get_ground_z(pts[:, 0], pts[:, 1])
        valid_g = ~np.isnan(ground_z)

        # For each instance in this cell
        for idx_i, i in enumerate(inst_indices):
            inst = instances[i]
            mask = masks[i]
            x1, y1, x2, y2 = inst["bbox"]

            # Get instance bbox in world coords
            wx0 = xmins[idx_i] + config.bbox_pad_m / transform[0]  # approximate
            wy0 = ymins[idx_i] + config.bbox_pad_m / abs(transform[4])
            # Better: compute from mask bbox
            wx0a, wy0a = transform * (x1, y1)
            wx1a, wy1a = transform * (x2, y2)
            if wx0a > wx1a: wx0a, wx1a = wx1a, wx0a
            if wy0a > wy1a: wy0a, wy1a = wy1a, wy0a

            # Filter points to instance bbox
            in_bbox = ((pts[:, 0] >= wx0a - config.bbox_pad_m) &
                       (pts[:, 0] <= wx1a + config.bbox_pad_m) &
                       (pts[:, 1] >= wy0a - config.bbox_pad_m) &
                       (pts[:, 1] <= wy1a + config.bbox_pad_m))

            if in_bbox.sum() == 0:
                results_passed[i] = False
                results_reasons[i] = ["no_points_in_bbox"]
                continue

            # Convert to pixel coords and test mask
            pts_local = pts[in_bbox]
            px_coords = np.zeros((len(pts_local), 2), dtype=np.float64)
            for p_i in range(len(pts_local)):
                px, py = inv_transform * (pts_local[p_i, 0], pts_local[p_i, 1])
                px_coords[p_i, 0] = px
                px_coords[p_i, 1] = py

            h, w = mask.shape
            px = np.clip(np.floor(px_coords[:, 0] - x1).astype(np.int32), 0, w - 1)
            py = np.clip(np.floor(px_coords[:, 1] - y1).astype(np.int32), 0, h - 1)
            inside = mask[py, px]
            stone_pts = pts_local[inside]

            if len(stone_pts) < config.min_points:
                results_passed[i] = False
                results_reasons[i].append("too_few_points")
                results_point_count[i] = len(stone_pts)
                continue

            results_point_count[i] = len(stone_pts)

            # Compute heights relative to ground
            stone_gz = ground_z[np.where(in_bbox)[0][inside.nonzero()[0]]] if valid_g.sum() > 0 else np.zeros(len(stone_pts))
            # Simpler: recompute ground for stone points
            stone_gz = ground_dem.get_ground_z(stone_pts[:, 0], stone_pts[:, 1])
            sg_valid = ~np.isnan(stone_gz)

            if sg_valid.sum() < max(3, len(stone_pts) // 10):
                results_passed[i] = False
                results_reasons[i].append("no_ground_elevation_data")
                continue

            rel_h = np.maximum(stone_pts[sg_valid, 2] - stone_gz[sg_valid], 0.0)
            z_range = float(np.max(stone_pts[sg_valid, 2]) - np.min(stone_pts[sg_valid, 2]))
            p90 = float(np.percentile(rel_h, 90))
            elev_ratio = float(np.mean(rel_h >= config.elevated_height_m))

            results_z_range[i] = z_range
            results_p90[i] = p90
            results_elev_ratio[i] = elev_ratio

            reasons = []
            if z_range < config.min_z_range_m:
                reasons.append("insufficient_z_range")
            if p90 < config.min_p90_height_m:
                reasons.append("insufficient_p90_height")
            if elev_ratio < config.min_elevated_ratio:
                reasons.append("insufficient_elevated_ratio")

            if reasons:
                results_passed[i] = False
                results_reasons[i].extend(reasons)

        processed += len(inst_indices)
        if processed % 5000 < len(inst_indices) or processed == n:
            logger.info("  Validated %d/%d (%.0f%%)", processed, n, processed / n * 100)

    # Build output
    accepted = []
    rejected = []
    reason_counts: dict[str, int] = {}

    for i, inst in enumerate(instances):
        inst_out = dict(inst)
        inst_out["validation_3d"] = {
            "passed": bool(results_passed[i]),
            "reasons": results_reasons[i],
            "point_count": int(results_point_count[i]),
            "z_range_m": round(float(results_z_range[i]), 4),
            "p90_height_m": round(float(results_p90[i]), 4),
            "elevated_ratio": round(float(results_elev_ratio[i]), 4),
        }
        if results_passed[i]:
            accepted.append(inst_out)
        else:
            rejected.append(inst_out)
            for r in results_reasons[i]:
                reason_counts[r] = reason_counts.get(r, 0) + 1

    summary = {
        "total_instances": n,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / n if n > 0 else 0.0,
        "rejection_reasons": reason_counts,
        "config": {
            "min_points": config.min_points,
            "min_z_range_m": config.min_z_range_m,
            "elevated_height_m": config.elevated_height_m,
            "min_p90_height_m": config.min_p90_height_m,
            "min_elevated_ratio": config.min_elevated_ratio,
        },
    }

    logger.info("3D validation done: %d accepted / %d rejected (%.1f%%)",
                len(accepted), len(rejected),
                len(accepted) / n * 100 if n > 0 else 0)
    logger.info("Rejection reasons: %s", reason_counts)

    return accepted, rejected, summary

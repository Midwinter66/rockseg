"""3D point cloud screening for rock instances.

Validates 2D rock detections against 3D point cloud elevation data.
Rocks that don't have sufficient 3D elevation signature are rejected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Validation3DConfig:
    """Configuration for 3D point cloud validation."""

    # Ground DEM
    dem_resolution_m: float = 0.5
    dem_percentile: int = 5
    dem_subsample_step: int = 100
    dem_min_points_per_cell: int = 3

    # Point cloud cropping
    bbox_pad_m: float = 0.5
    index_cell_size_m: float = 1.0

    # Filter thresholds (all must pass)
    min_points: int = 60
    min_z_range_m: float = 0.18
    elevated_height_m: float = 0.08
    min_p90_height_m: float = 0.12
    min_elevated_ratio: float = 0.2


@dataclass
class Validation3DResult:
    """Result of 3D validation for a single rock instance."""

    passed: bool
    reasons: list[str]
    point_count: int
    z_range_m: float
    p50_height_m: float
    p90_height_m: float
    max_height_m: float
    elevated_ratio: float


# ---------------------------------------------------------------------------
# Ground DEM
# ---------------------------------------------------------------------------

class GroundDEM:
    """Scene-level ground digital elevation model from point cloud."""

    def __init__(
        self,
        pc: np.ndarray,
        resolution: float = 0.5,
        percentile: int = 5,
        subsample_step: int = 100,
        min_points_per_cell: int = 3,
    ) -> None:
        pts = np.asarray(pc, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("pc must be Nx3 array")

        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            raise RuntimeError("No valid points for GroundDEM")

        self.resolution = float(resolution)
        self.percentile = int(percentile)

        sub = pts[::subsample_step]
        logger.info("GroundDEM: %d points -> %d sampled", len(pts), len(sub))

        xmin, ymin = float(sub[:, 0].min()), float(sub[:, 1].min())
        xmax, ymax = float(sub[:, 0].max()), float(sub[:, 1].max())
        nx = max(2, int(np.ceil((xmax - xmin) / self.resolution)))
        ny = max(2, int(np.ceil((ymax - ymin) / self.resolution)))

        self.xmin, self.ymin = xmin, ymin
        self.xmax = xmin + nx * self.resolution
        self.ymax = ymin + ny * self.resolution
        self.nx, self.ny = nx, ny

        logger.info("GroundDEM grid: %d x %d = %d cells @ %.1fm",
                     nx, ny, nx * ny, self.resolution)

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
        valid = np.sum(~np.isnan(self.grid))
        logger.info("GroundDEM coverage: %.1f%%", valid / (nx * ny) * 100)

    def _fill_holes(self) -> None:
        valid = ~np.isnan(self.grid)
        if valid.all():
            return
        grid = self.grid.copy()
        ny, nx = grid.shape
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),            (0, 1),
                   (1, -1),  (1, 0),   (1, 1)]
        for _ in range(max(nx, ny)):
            mask = np.isnan(grid)
            if not mask.any():
                break
            updated = False
            for dy, dx in offsets:
                src = np.roll(grid, shift=(-dy, -dx), axis=(0, 1))
                fill = mask & ~np.isnan(src)
                if dy > 0:
                    fill[-dy:, :] = False
                elif dy < 0:
                    fill[:-dy, :] = False
                if dx > 0:
                    fill[:, -dx:] = False
                elif dx < 0:
                    fill[:, :-dx] = False
                if fill.any():
                    grid[fill] = src[fill]
                    updated = True
            if not updated:
                break
        self.grid = grid

    def get_ground_z(self, x, y) -> np.ndarray:
        """Bilinear interpolation of ground elevation."""
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


# ---------------------------------------------------------------------------
# Point cloud spatial index
# ---------------------------------------------------------------------------

class PointCloudGridIndex:
    """XY grid index for fast point cloud bbox queries."""

    def __init__(self, points: np.ndarray, cell_size: float = 1.0):
        pts = np.asarray(points, dtype=np.float64)
        self.points = pts
        self.cell_size = float(cell_size)
        self.xmin = float(np.min(pts[:, 0]))
        self.ymin = float(np.min(pts[:, 1]))
        self.xmax = float(np.max(pts[:, 0]))
        self.ymax = float(np.max(pts[:, 1]))
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

    def query_bbox(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        """Return point indices within the XY bounding box."""
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

        parts = []
        for idx in pos[valid]:
            s, c = int(self._starts[idx]), int(self._counts[idx])
            parts.append(self._order[s:s + c])
        if not parts:
            return np.empty(0, dtype=np.int32)

        candidates = np.concatenate(parts)
        pts = self.points[candidates]
        mask = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1)
                & (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
        return candidates[mask]


# ---------------------------------------------------------------------------
# Point cloud loading
# ---------------------------------------------------------------------------

def load_point_cloud(laz_paths: Sequence[str | Path]) -> np.ndarray:
    """Load one or more LAZ files and return Nx3 point cloud array."""
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


# ---------------------------------------------------------------------------
# Per-instance point cloud cropping & validation
# ---------------------------------------------------------------------------

def _points_in_mask(
    pts_px: np.ndarray,
    mask: np.ndarray,
    mask_x0: int,
    mask_y0: int,
) -> np.ndarray:
    """Check which pixel-coordinate points fall inside the binary mask.

    Parameters
    ----------
    pts_px : (N, 2) array of pixel coordinates (float)
    mask : (H, W) bool array
    mask_x0, mask_y0 : pixel offset of the mask's top-left corner

    Returns
    -------
    bool array of length N
    """
    h, w = mask.shape
    px = np.floor(pts_px[:, 0] - mask_x0).astype(np.int32)
    py = np.floor(pts_px[:, 1] - mask_y0).astype(np.int32)
    valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    inside = np.zeros(len(pts_px), dtype=bool)
    if valid.any():
        inside[valid] = mask[py[valid], px[valid]]
    return inside


def validate_instance_3d(
    instance: dict,
    mask: np.ndarray,
    pc_index: PointCloudGridIndex,
    ground_dem: GroundDEM,
    transform,  # rasterio Affine transform
    config: Validation3DConfig,
) -> Validation3DResult:
    """Validate a single rock instance against 3D point cloud.

    Parameters
    ----------
    instance : rock instance dict with bbox, centroid, etc.
    mask : 2D bool array (tight bbox crop)
    pc_index : point cloud spatial index
    ground_dem : ground DEM
    transform : rasterio Affine (pixel -> world)
    config : validation parameters
    """
    x1, y1, x2, y2 = instance["bbox"]
    mask_h, mask_w = mask.shape

    # Convert bbox to world coordinates (with padding)
    wx0, wy0 = transform * (x1 - config.bbox_pad_m / transform[0],
                             y1 - config.bbox_pad_m / abs(transform[4]))
    wx1, wy1 = transform * (x2 + config.bbox_pad_m / transform[0],
                             y2 + config.bbox_pad_m / abs(transform[4]))
    # Ensure correct ordering
    if wx0 > wx1:
        wx0, wx1 = wx1, wx0
    if wy0 > wy1:
        wy0, wy1 = wy1, wy0

    # Query point cloud bbox
    idx = pc_index.query_bbox(wx0, wy0, wx1, wy1)
    if len(idx) == 0:
        return Validation3DResult(
            passed=False, reasons=["no_points_in_bbox"],
            point_count=0, z_range_m=0.0,
            p50_height_m=0.0, p90_height_m=0.0, max_height_m=0.0,
            elevated_ratio=0.0,
        )

    pts = pc_index.points[idx]

    # Convert world coords to pixel coords for mask test
    # inverse transform: world -> pixel
    inv = ~transform
    px_coords = np.zeros((len(pts), 2), dtype=np.float64)
    for i in range(len(pts)):
        px, py = inv * (pts[i, 0], pts[i, 1])
        px_coords[i, 0] = px
        px_coords[i, 1] = py

    inside = _points_in_mask(px_coords, mask, x1, y1)
    stone_pts = pts[inside]

    if len(stone_pts) == 0:
        return Validation3DResult(
            passed=False, reasons=["no_points_inside_mask"],
            point_count=0, z_range_m=0.0,
            p50_height_m=0.0, p90_height_m=0.0, max_height_m=0.0,
            elevated_ratio=0.0,
        )

    point_count = len(stone_pts)
    z = stone_pts[:, 2]
    z_range = float(np.max(z) - np.min(z))

    # Ground elevation at each point
    ground_z = ground_dem.get_ground_z(stone_pts[:, 0], stone_pts[:, 1])
    valid_ground = ~np.isnan(ground_z)
    if valid_ground.sum() < max(3, point_count // 10):
        return Validation3DResult(
            passed=False, reasons=["no_ground_elevation_data"],
            point_count=point_count, z_range_m=z_range,
            p50_height_m=0.0, p90_height_m=0.0, max_height_m=0.0,
            elevated_ratio=0.0,
        )

    rel_height = z[valid_ground] - ground_z[valid_ground]
    rel_height = np.maximum(rel_height, 0.0)  # clamp to non-negative

    p50 = float(np.percentile(rel_height, 50))
    p90 = float(np.percentile(rel_height, 90))
    max_h = float(np.max(rel_height))
    elevated_ratio = float(np.mean(rel_height >= config.elevated_height_m))

    reasons = []
    if point_count < config.min_points:
        reasons.append("too_few_points")
    if z_range < config.min_z_range_m:
        reasons.append("insufficient_z_range")
    if p90 < config.min_p90_height_m:
        reasons.append("insufficient_p90_height")
    if elevated_ratio < config.min_elevated_ratio:
        reasons.append("insufficient_elevated_ratio")

    return Validation3DResult(
        passed=len(reasons) == 0,
        reasons=reasons,
        point_count=point_count,
        z_range_m=z_range,
        p50_height_m=p50,
        p90_height_m=p90,
        max_height_m=max_h,
        elevated_ratio=elevated_ratio,
    )


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

def run_3d_validation(
    instances: list[dict],
    masks: np.ndarray,
    laz_paths: Sequence[str | Path],
    transform,  # rasterio Affine
    config: Validation3DConfig | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Run 3D point cloud validation on all instances.

    Returns
    -------
    (accepted_instances, rejected_instances, summary)
    """
    if config is None:
        config = Validation3DConfig()

    logger.info("=== 3D Point Cloud Validation ===")

    # Load point cloud
    pc = load_point_cloud(laz_paths)

    # Build ground DEM
    ground_dem = GroundDEM(
        pc,
        resolution=config.dem_resolution_m,
        percentile=config.dem_percentile,
        subsample_step=config.dem_subsample_step,
        min_points_per_cell=config.dem_min_points_per_cell,
    )

    # Build spatial index
    pc_index = PointCloudGridIndex(pc, cell_size=config.index_cell_size_m)
    logger.info("Point cloud index: %d cells", len(pc_index._unique_ids))

    # Validate each instance
    accepted = []
    rejected = []
    reason_counts: dict[str, int] = {}

    total = len(instances)
    for i, inst in enumerate(instances):
        if (i + 1) % 5000 == 0 or i == total - 1:
            logger.info("  Validated %d/%d (%.0f%%)", i + 1, total, (i + 1) / total * 100)

        mask = masks[i]
        result = validate_instance_3d(inst, mask, pc_index, ground_dem, transform, config)

        inst_out = dict(inst)
        inst_out["validation_3d"] = {
            "passed": result.passed,
            "reasons": result.reasons,
            "point_count": result.point_count,
            "z_range_m": round(result.z_range_m, 4),
            "p50_height_m": round(result.p50_height_m, 4),
            "p90_height_m": round(result.p90_height_m, 4),
            "max_height_m": round(result.max_height_m, 4),
            "elevated_ratio": round(result.elevated_ratio, 4),
        }

        if result.passed:
            accepted.append(inst_out)
        else:
            rejected.append(inst_out)
            for r in result.reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1

    summary = {
        "total_instances": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / total if total > 0 else 0.0,
        "rejection_reasons": reason_counts,
        "config": {
            "min_points": config.min_points,
            "min_z_range_m": config.min_z_range_m,
            "elevated_height_m": config.elevated_height_m,
            "min_p90_height_m": config.min_p90_height_m,
            "min_elevated_ratio": config.min_elevated_ratio,
        },
    }

    logger.info("3D validation: %d accepted / %d rejected (%.1f%% acceptance)",
                len(accepted), len(rejected),
                len(accepted) / total * 100 if total > 0 else 0)
    logger.info("Rejection reasons: %s", reason_counts)

    return accepted, rejected, summary

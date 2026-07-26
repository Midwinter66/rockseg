from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import open3d as o3d


o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def _as_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    mask = np.isfinite(arr).all(axis=1)
    return arr[mask]


def _to_array2d(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("2D points must be an Nx2 array")
    return arr


def _numeric_summary(values: Sequence[float]) -> dict:
    vals = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "sum": 0.0,
        }

    vals.sort()
    return {
        "count": int(vals.size),
        "min": round(float(vals[0]), 4),
        "max": round(float(vals[-1]), 4),
        "mean": round(float(vals.mean()), 4),
        "median": round(float(np.median(vals)), 4),
        "std": round(float(vals.std(ddof=0)), 4),
        "p25": round(float(np.quantile(vals, 0.25)), 4),
        "p75": round(float(np.quantile(vals, 0.75)), 4),
        "sum": round(float(vals.sum()), 4),
    }


def _xy_convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)

    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        return np.unique(hull, axis=0)
    return hull


def _polygon_area_xy(points: np.ndarray) -> float:
    pts = _to_array2d(points)
    if len(pts) < 3:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return area


def _prepare_height_stats(heights: np.ndarray) -> dict:
    vals = np.asarray(heights, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "status": "invalid",
            "count": 0,
            "min_m": 0.0,
            "max_m": 0.0,
            "mean_m": 0.0,
            "median_m": 0.0,
            "std_m": 0.0,
            "p25_m": 0.0,
            "p75_m": 0.0,
        }

    return {
        "status": "ok",
        "count": int(vals.size),
        "min_m": round(float(np.min(vals)), 4),
        "max_m": round(float(np.max(vals)), 4),
        "mean_m": round(float(np.mean(vals)), 4),
        "median_m": round(float(np.median(vals)), 4),
        "std_m": round(float(np.std(vals, ddof=0)), 4),
        "p25_m": round(float(np.quantile(vals, 0.25)), 4),
        "p75_m": round(float(np.quantile(vals, 0.75)), 4),
    }


def _safe_mesh_volume(mesh: o3d.geometry.TriangleMesh) -> float:
    if mesh is None or len(mesh.vertices) < 4 or len(mesh.triangles) < 4:
        return 0.0
    if not mesh.is_watertight():
        return 0.0
    try:
        volume = float(mesh.get_volume())
    except Exception:
        return 0.0
    return volume if volume > 0 else 0.0


def estimate_projected_footprint(points: np.ndarray) -> dict:
    pts = _as_points(points)
    if len(pts) == 0:
        return {
            "status": "invalid",
            "point_count": 0,
            "span_x_m": 0.0,
            "span_y_m": 0.0,
            "span_z_m": 0.0,
            "bbox_area_m2": 0.0,
            "convex_hull_area_m2": 0.0,
            "equivalent_diameter_m": 0.0,
        }

    span = np.ptp(pts, axis=0)
    bbox_area = float(span[0] * span[1])
    xy_hull = _xy_convex_hull(pts[:, :2])
    hull_area = _polygon_area_xy(xy_hull) if len(xy_hull) >= 3 else 0.0
    equivalent_diameter = math.sqrt((4.0 * hull_area) / math.pi) if hull_area > 0 else 0.0

    return {
        "status": "ok",
        "point_count": int(len(pts)),
        "span_x_m": round(float(span[0]), 4),
        "span_y_m": round(float(span[1]), 4),
        "span_z_m": round(float(span[2]), 4),
        "bbox_area_m2": round(bbox_area, 4),
        "convex_hull_area_m2": round(float(hull_area), 4),
        "equivalent_diameter_m": round(float(equivalent_diameter), 4),
    }


def estimate_convex_hull(points: np.ndarray) -> dict:
    """Estimate convex-hull volume and record failure reasons explicitly."""

    pts = _as_points(points)
    footprint = estimate_projected_footprint(pts)
    result = {
        "method": "convex_hull",
        "status": "invalid",
        "reason": "unknown",
        "volume_m3": 0.0,
        "point_count": int(len(pts)),
        "mesh_vertex_count": 0,
        "mesh_triangle_count": 0,
        "planarity_ratio": 0.0,
        "footprint": footprint,
    }
    if len(pts) < 4:
        result["reason"] = "point_count_too_small"
        return result

    span = np.ptp(pts, axis=0)
    span_x = float(span[0])
    span_y = float(span[1])
    span_z = float(span[2])
    max_xy = max(span_x, span_y)
    result["planarity_ratio"] = round(float(span_z / max_xy), 4) if max_xy > 0 else 0.0

    if max_xy <= 0 or span_z <= 0:
        result["reason"] = "degenerate_span"
        return result
    if max_xy > 0 and span_z / max_xy < 0.02:
        result["reason"] = "nearly_planar"
        return result

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        hull, _ = pcd.compute_convex_hull()
    except Exception:
        result["reason"] = "convex_hull_failed"
        return result

    if hull is None:
        result["reason"] = "empty_hull"
        return result

    result["mesh_vertex_count"] = int(len(hull.vertices))
    result["mesh_triangle_count"] = int(len(hull.triangles))
    if len(hull.vertices) < 4 or len(hull.triangles) < 4:
        result["reason"] = "degenerate_hull_mesh"
        return result
    if not hull.is_watertight():
        result["reason"] = "non_watertight_hull"
        return result

    volume = _safe_mesh_volume(hull)
    if volume <= 0:
        result["reason"] = "non_positive_volume"
        return result

    result["status"] = "ok"
    result["reason"] = "ok"
    result["volume_m3"] = round(float(volume), 4)
    return result


def estimate_2d_proxy_from_diameter(
    equivalent_diameter_m: float,
    area_m2: float = 0.0,
) -> dict:
    """Simple 2D baseline: equivalent sphere volume from projected diameter."""

    diameter = float(equivalent_diameter_m)
    area = float(area_m2)
    result = {
        "method": "equivalent_sphere_from_detection_diameter",
        "status": "invalid",
        "reason": "invalid_diameter",
        "equivalent_diameter_m": round(diameter, 4),
        "projected_area_m2": round(area, 4),
        "volume_m3": 0.0,
        "formula": "pi/6 * d^3",
        "note": "2D proxy baseline derived from fused equivalent diameter.",
    }
    if not np.isfinite(diameter) or diameter <= 0:
        return result

    volume = float((math.pi / 6.0) * diameter**3)
    if not np.isfinite(volume) or volume <= 0:
        result["reason"] = "non_positive_volume"
        return result

    result["status"] = "ok"
    result["reason"] = "ok"
    result["volume_m3"] = round(volume, 4)
    return result


def estimate_2d5_with_ground(
    points: np.ndarray,
    ground_dem,
    grid_resolution: float = 0.05,
    return_debug: bool = False,
) -> dict:
    """2.5D grid integration using the shared ground DEM."""

    pts = _as_points(points)
    footprint = estimate_projected_footprint(pts)
    result = {
        "method": "2d5_with_ground",
        "status": "invalid",
        "reason": "unknown",
        "volume_m3": 0.0,
        "surface_area_m2": 0.0,
        "occupied_area_m2": 0.0,
        "occupancy_ratio": 0.0,
        "point_count": int(len(pts)),
        "grid_resolution_m": float(grid_resolution),
        "grid_nx": 0,
        "grid_ny": 0,
        "occupied_cells": 0,
        "ground_supported_cells": 0,
        "valid_cells": 0,
        "height_stats": _prepare_height_stats(np.empty(0, dtype=np.float64)),
        "footprint": footprint,
        "note": "2.5D grid integration using the shared ground DEM",
    }

    if len(pts) < 4:
        result["reason"] = "point_count_too_small"
        return result
    if not np.isfinite(grid_resolution) or grid_resolution <= 0:
        result["reason"] = "invalid_grid_resolution"
        return result

    xy = pts[:, :2]
    z = pts[:, 2]

    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    nx = max(1, int(np.ceil((xmax - xmin) / grid_resolution)))
    ny = max(1, int(np.ceil((ymax - ymin) / grid_resolution)))
    result["grid_nx"] = int(nx)
    result["grid_ny"] = int(ny)

    xi = np.floor((xy[:, 0] - xmin) / grid_resolution).astype(np.int32)
    yi = np.floor((xy[:, 1] - ymin) / grid_resolution).astype(np.int32)
    xi = np.clip(xi, 0, nx - 1)
    yi = np.clip(yi, 0, ny - 1)

    cell_z_max = np.full((ny, nx), -np.inf, dtype=np.float64)
    cell_count = np.zeros((ny, nx), dtype=np.int32)
    np.maximum.at(cell_z_max, (yi, xi), z)
    np.add.at(cell_count, (yi, xi), 1)

    occupied_mask = cell_count > 0
    occupied_cells = int(np.count_nonzero(occupied_mask))
    result["occupied_cells"] = occupied_cells
    if occupied_cells == 0:
        result["reason"] = "no_occupied_cells"
        return result

    cell_area = float(grid_resolution**2)
    result["occupied_area_m2"] = round(float(occupied_cells * cell_area), 4)

    cell_rows, cell_cols = np.nonzero(occupied_mask)
    centers_x = xmin + (cell_cols.astype(np.float64) + 0.5) * grid_resolution
    centers_y = ymin + (cell_rows.astype(np.float64) + 0.5) * grid_resolution
    rock_top = cell_z_max[cell_rows, cell_cols]
    ground_z = np.asarray(ground_dem.get_ground_z(centers_x, centers_y), dtype=np.float64)

    finite_ground = np.isfinite(ground_z)
    result["ground_supported_cells"] = int(np.count_nonzero(finite_ground))
    if not np.any(finite_ground):
        result["reason"] = "ground_lookup_failed"
        return result

    heights = rock_top[finite_ground] - ground_z[finite_ground]
    positive_mask = np.isfinite(heights) & (heights > 0)
    positive_heights = heights[positive_mask]
    positive_cells = int(np.count_nonzero(positive_mask))
    result["valid_cells"] = positive_cells
    result["occupancy_ratio"] = round(float(positive_cells / occupied_cells), 4) if occupied_cells > 0 else 0.0
    result["surface_area_m2"] = round(float(positive_cells * cell_area), 4)
    result["height_stats"] = _prepare_height_stats(positive_heights)

    if positive_cells == 0:
        result["reason"] = "no_positive_heights"
        return result

    volume = float(np.sum(positive_heights) * cell_area)
    if not np.isfinite(volume) or volume <= 0:
        result["reason"] = "non_positive_volume"
        return result

    result["status"] = "ok"
    result["reason"] = "ok"
    result["volume_m3"] = round(volume, 4)
    if return_debug:
        finite_centers = np.column_stack((centers_x[finite_ground], centers_y[finite_ground]))
        result["debug"] = {
            "grid_origin_xy": np.asarray([xmin, ymin], dtype=np.float64),
            "grid_resolution_m": float(grid_resolution),
            "occupied_centers_xy": np.column_stack((centers_x, centers_y)),
            "occupied_ground_z": ground_z,
            "occupied_rock_top_z": rock_top,
            "finite_ground_centers_xy": finite_centers,
            "finite_ground_z": ground_z[finite_ground],
            "finite_rock_top_z": rock_top[finite_ground],
            "positive_centers_xy": finite_centers[positive_mask],
            "positive_ground_z": ground_z[finite_ground][positive_mask],
            "positive_rock_top_z": rock_top[finite_ground][positive_mask],
            "positive_heights_m": positive_heights,
        }
    return result

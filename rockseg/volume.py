"""Volume estimation for rock instances from 3D point cloud data.

Uses 2.5D integration + shape-aware LightGBM correction model.
The shape-aware model is trained on the Čapek et al. (2025) OBJ dataset
and predicts a correction ratio r = V_true / V_2_5D from dimensionless
shape features (circularity, aspect ratio, height distribution, etc.).

Final volume: V = r_predicted × V_2_5D
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .validation_3d import GroundDEM, PointCloudGridIndex
from research_v2.volume_validation.shape_features_v2 import (
    FEATURE_NAMES,
    FeatureSchemaError,
    compute_surface_descriptors,
    extract_features as extract_v2_features,
    validate_model_feature_names,
)

logger = logging.getLogger(__name__)

_EPS = 1e-9


@dataclass
class VolumeResult:
    """Volume estimation result for a single rock."""

    volume_2_5d_m3: float
    volume_shape_aware_m3: float
    volume_linear_m3: float
    v_2_5d: float
    correction_ratio: float
    linear_alpha: float
    height_max_m: float
    height_mean_m: float
    footprint_area_m2: float
    circularity: float
    aspect_ratio: float
    n_valid_cells: int


# ---------------------------------------------------------------------------
# Height map extraction from point cloud
# ---------------------------------------------------------------------------

def extract_height_map(
    mask: np.ndarray,
    mask_x0: int,
    mask_y0: int,
    pc_index: PointCloudGridIndex,
    ground_dem: GroundDEM,
    transform,  # rasterio Affine
    grid_resolution_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Extract 2.5D height map for a rock from point cloud.

    Returns
    -------
    (height_map_m, footprint_mask, grid_x_min_m, grid_y_min_m)
    height_map_m: 2D array of heights above ground (meters), NaN where no data
    footprint_mask: bool array of the rock's footprint
    """
    h, w = mask.shape

    # Build grid in world coordinates (meters)
    # Grid covers the mask bbox
    x_world_0, y_world_0 = transform * (mask_x0, mask_y0)
    x_world_1, y_world_1 = transform * (mask_x0 + w, mask_y0 + h)
    # Ensure correct ordering
    if x_world_0 > x_world_1:
        x_world_0, x_world_1 = x_world_1, x_world_0
    if y_world_0 > y_world_1:
        y_world_0, y_world_1 = y_world_1, y_world_0

    nx = max(2, int(np.ceil((x_world_1 - x_world_0) / grid_resolution_m)))
    ny = max(2, int(np.ceil((y_world_1 - y_world_0) / grid_resolution_m)))

    # Query point cloud
    pad = grid_resolution_m * 2
    idx = pc_index.query_bbox(x_world_0 - pad, y_world_0 - pad,
                              x_world_1 + pad, y_world_1 + pad)
    if len(idx) == 0:
        return (np.full((ny, nx), np.nan), mask.astype(bool),
                x_world_0, y_world_0)

    pts = pc_index.points[idx]

    # Assign points to grid cells (max height per cell = top surface)
    gx = np.clip(np.floor((pts[:, 0] - x_world_0) / grid_resolution_m).astype(np.int32), 0, nx - 1)
    gy = np.clip(np.floor((pts[:, 1] - y_world_0) / grid_resolution_m).astype(np.int32), 0, ny - 1)

    height_map = np.full((ny, nx), np.nan, dtype=np.float64)

    # Compute ground elevation per point
    ground_z = ground_dem.get_ground_z(pts[:, 0], pts[:, 1])
    valid_g = ~np.isnan(ground_z)

    if valid_g.sum() > 0:
        rel_height = np.maximum(pts[valid_g, 2] - ground_z[valid_g], 0.0)
        gxv = gx[valid_g]
        gyv = gy[valid_g]

        flat = gyv * nx + gxv
        order = np.argsort(flat)
        flat_sorted = flat[order]
        h_sorted = rel_height[order]
        unique_flat, indices = np.unique(flat_sorted, return_index=True)
        if len(unique_flat) > 0:
            max_per_group = np.maximum.reduceat(h_sorted, indices)
            height_map.flat[unique_flat.astype(np.intp)] = max_per_group

    # Footprint mask: resize the binary mask to grid resolution
    from scipy import ndimage
    footprint = ndimage.zoom(mask.astype(np.float32),
                              (ny / h, nx / w),
                              order=0) > 0.5

    return height_map, footprint, x_world_0, y_world_0


# ---------------------------------------------------------------------------
# Shape descriptor computation
# ---------------------------------------------------------------------------

def compute_shape_descriptors(
    height_map: np.ndarray,
    footprint: np.ndarray,
    grid_resolution_m: float,
) -> dict:
    """Adapt a ground-referenced production surface to the canonical V2 schema."""
    descriptors = compute_surface_descriptors(
        height_map,
        footprint,
        grid_resolution_m,
        ground_z=0.0,
    )
    if descriptors is not None:
        return descriptors
    return {
        "L": 0.0, "W": 0.0, "H": 0.0, "A": 0.0, "P": 0.0,
        "A_convex": 0.0, "C": 0.0, "AR": 0.0, "solidity": 0.0,
        "compactness": 0.0, "eq_diam_ratio": 0.0, "H_mean": 0.0,
        "H_max": 0.0, "H_std": 0.0, "H_p25": 0.0, "H_p75": 0.0,
        "H_skew": 0.0, "fill_ratio": 0.0, "ellipsoid_ratio": 0.0,
        "V_box": 0.0, "V_ellipsoid": 0.0, "V_2_5d": 0.0,
        "n_valid_cells": 0,
    }


# ---------------------------------------------------------------------------
# Shape-aware volume model
# ---------------------------------------------------------------------------

class SimpleTreeModel:
    """Pure-numpy single-tree predictor for LightGBM text models.

    Parses the ``Tree=0`` block from a LightGBM model file and implements
    a ``predict`` method with the same interface as ``lightgbm.Booster``.
    Used as fallback when the lightgbm package is not installed.
    """

    def __init__(self, model_text: str):
        trees = {}
        current_tree = None
        fields: dict[str, str] = {}
        for line in model_text.splitlines():
            line = line.strip()
            if line.startswith("Tree="):
                if current_tree is not None and fields:
                    trees[current_tree] = fields
                current_tree = int(line.split("=")[1])
                fields = {}
            elif "=" in line and current_tree is not None:
                key, _, val = line.partition("=")
                fields[key] = val
        if current_tree is not None and fields:
            trees[current_tree] = fields

        self.trees = []
        for tid in sorted(trees):
            f = trees[tid]
            n_leaves = int(f["num_leaves"])
            split_feature = [int(x) for x in f["split_feature"].split()]
            threshold = [float(x) for x in f["threshold"].split()]
            left_child = [int(x) for x in f["left_child"].split()]
            right_child = [int(x) for x in f["right_child"].split()]
            leaf_value = [float(x) for x in f["leaf_value"].split()]
            self.trees.append({
                "split_feature": split_feature,
                "threshold": threshold,
                "left_child": left_child,
                "right_child": right_child,
                "leaf_value": leaf_value,
            })

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        out = np.zeros(n)
        for t in self.trees:
            sf = t["split_feature"]
            th = t["threshold"]
            lc = t["left_child"]
            rc = t["right_child"]
            lv = t["leaf_value"]
            for i in range(n):
                node = 0
                while node >= 0:
                    feat = sf[node]
                    if X[i, feat] <= th[node]:
                        node = lc[node]
                    else:
                        node = rc[node]
                out[i] += lv[-node - 1]
        return out


def _read_model_feature_names(model_path: Path) -> list[str]:
    for line in model_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("feature_names="):
            return line.partition("=")[2].split()
    raise FeatureSchemaError(f"Model does not declare feature_names: {model_path}")


def load_shape_aware_model(model_path: str | Path):
    """Load a pre-trained LightGBM shape-aware model.

    The model predicts correction ratio r = V_true / V_2_5D.
    Only models with the canonical 12-feature Shape-Aware V2 schema are accepted.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    validate_model_feature_names(_read_model_feature_names(model_path))

    try:
        import lightgbm as lgb
        return lgb.Booster(model_file=str(model_path))
    except ImportError:
        logger.info("lightgbm not installed, using SimpleTreeModel fallback")
        text = model_path.read_text()
        return SimpleTreeModel(text)


def predict_shape_aware(
    model,
    descriptors: list[dict],
) -> np.ndarray:
    """Predict correction ratios and volumes for multiple rocks.

    Returns array of predicted volumes in m³.
    """
    X = np.asarray([extract_v2_features(descriptor) for descriptor in descriptors], dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
        raise FeatureSchemaError("Production feature adapter did not produce 12 Shape-Aware V2 features")
    ratios = model.predict(X)
    ratios = np.clip(ratios, 0.01, 10.0)

    v_2_5d = np.array([d["V_2_5d"] for d in descriptors])
    return ratios * v_2_5d


# ---------------------------------------------------------------------------
# Batch volume estimation
# ---------------------------------------------------------------------------

def estimate_volumes(
    instances: list[dict],
    masks: list[np.ndarray],
    laz_paths: Sequence[str | Path],
    transform,
    model_path: str | Path,
    linear_alpha: float = 0.731,
    grid_resolution_m: float = 0.05,
) -> tuple[list[dict], dict]:
    """Estimate volume for all rock instances.

    Parameters
    ----------
    instances : list of rock instance dicts
    masks : list of mask arrays (tight bbox cropped)
    laz_paths : point cloud LAZ file paths
    transform : rasterio Affine
    model_path : path to trained LightGBM model
    linear_alpha : global linear correction factor (default 0.731 from T01 training)
    grid_resolution_m : height map grid resolution in meters

    Returns
    -------
    (instances_with_volume, summary)
    """
    from .validation_3d import load_point_cloud

    logger.info("=== Volume Estimation ===")

    # Load point cloud and build index + DEM
    pc = load_point_cloud(laz_paths)
    ground_dem = GroundDEM(pc)
    pc_index = PointCloudGridIndex(pc, cell_size=1.0)

    # Load shape-aware model
    model = load_shape_aware_model(model_path)
    logger.info("Shape-aware model loaded from %s", model_path)

    # Process each instance
    total = len(instances)
    descriptors_list = []
    height_info_list = []

    for i in range(total):
        if (i + 1) % 5000 == 0 or i == total - 1:
            logger.info("  Height maps: %d/%d (%.0f%%)", i + 1, total, (i + 1) / total * 100)

        inst = instances[i]
        mask = masks[i]
        x1, y1, x2, y2 = inst["bbox"]

        try:
            height_map, footprint, _, _ = extract_height_map(
                mask, x1, y1, pc_index, ground_dem, transform, grid_resolution_m
            )
            desc = compute_shape_descriptors(height_map, footprint, grid_resolution_m)
        except Exception:
            desc = {
                "L": 0, "W": 0, "H": 0, "A": 0, "P": 0,
                "C": 0.5, "AR": 1.0, "H_mean": 0, "H_max": 0, "H_std": 0,
                "V_box": 0, "V_ellipsoid": 0, "V_2_5d": 0,
                "n_valid_cells": 0,
            }

        descriptors_list.append(desc)
        height_info_list.append({
            "H_max": desc["H"],
            "H_mean": desc["H_mean"],
            "footprint_area": desc["A"],
            "circularity": desc["C"],
            "aspect_ratio": desc["AR"],
            "n_valid_cells": desc["n_valid_cells"],
        })

    # Predict volumes
    v_2_5d_arr = np.array([d["V_2_5d"] for d in descriptors_list])
    v_linear_arr = linear_alpha * v_2_5d_arr
    v_shape_arr = predict_shape_aware(model, descriptors_list)

    # Add to instances
    out_instances = []
    for i, inst in enumerate(instances):
        inst_out = dict(inst)
        inst_out["volume"] = {
            "v_2_5d_m3": round(float(v_2_5d_arr[i]), 6),
            "v_linear_m3": round(float(v_linear_arr[i]), 6),
            "v_shape_aware_m3": round(float(v_shape_arr[i]), 6),
            "correction_ratio_shape_aware": round(float(v_shape_arr[i] / (v_2_5d_arr[i] + _EPS)), 4),
            "height_max_m": round(height_info_list[i]["H_max"], 4),
            "height_mean_m": round(height_info_list[i]["H_mean"], 4),
            "footprint_area_m2": round(height_info_list[i]["footprint_area"], 4),
            "circularity": round(height_info_list[i]["circularity"], 4),
            "aspect_ratio": round(height_info_list[i]["aspect_ratio"], 4),
            "n_height_cells": height_info_list[i]["n_valid_cells"],
        }
        out_instances.append(inst_out)

    # Summary
    valid = v_2_5d_arr > 0
    total_volume_shape = float(np.sum(v_shape_arr[valid]))
    total_volume_2_5d = float(np.sum(v_2_5d_arr[valid]))
    total_volume_linear = float(np.sum(v_linear_arr[valid]))

    summary = {
        "total_instances": total,
        "instances_with_volume": int(valid.sum()),
        "total_volume_shape_aware_m3": round(total_volume_shape, 3),
        "total_volume_2_5d_m3": round(total_volume_2_5d, 3),
        "total_volume_linear_m3": round(total_volume_linear, 3),
        "mean_correction_ratio": round(float(np.mean(v_shape_arr[valid] / v_2_5d_arr[valid])), 4),
        "linear_alpha": linear_alpha,
        "grid_resolution_m": grid_resolution_m,
        "model_path": str(model_path),
    }

    logger.info("Volume estimation complete:")
    logger.info("  Instances with volume: %d/%d", valid.sum(), total)
    logger.info("  Total volume (shape-aware): %.2f m³", total_volume_shape)
    logger.info("  Total volume (2.5D):      %.2f m³", total_volume_2_5d)
    logger.info("  Mean correction ratio: %.4f", summary["mean_correction_ratio"])

    return out_instances, summary

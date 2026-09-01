"""Canonical Shape-Aware V2 descriptor and feature schema.

The Dataset B training implementation is the reference for all definitions in
this module. Length, area, and volume inputs may use millimetres or metres,
provided a single unit system is used within one surface.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


FEATURE_NAMES = [
    "C",
    "AR",
    "solidity",
    "compactness",
    "eq_diam_ratio",
    "H_mean_norm",
    "H_std_norm",
    "H_p25_norm",
    "H_p75_norm",
    "H_skew_norm",
    "fill_ratio",
    "ellipsoid_ratio",
]

FEATURE_FORMULAS = {
    "C": "min(4*pi*A/P^2, 1)",
    "AR": "L/W",
    "solidity": "min(A/A_convex, 1)",
    "compactness": "P/sqrt(A)",
    "eq_diam_ratio": "sqrt(4*A/pi)/L",
    "H_mean_norm": "H_mean/H",
    "H_std_norm": "H_std/H",
    "H_p25_norm": "H_p25/H",
    "H_p75_norm": "H_p75/H",
    "H_skew_norm": "H_skew",
    "fill_ratio": "V_2_5D/V_box",
    "ellipsoid_ratio": "V_2_5D/V_ellipsoid",
}

REQUIRED_DESCRIPTOR_FIELDS = (
    "C", "AR", "solidity", "compactness", "eq_diam_ratio", "H_mean",
    "H", "H_std", "H_p25", "H_p75", "H_skew", "fill_ratio",
    "ellipsoid_ratio",
)
_EPS = 1e-9


class FeatureSchemaError(ValueError):
    """Raised when a feature vector cannot satisfy the V2 model schema."""


def feature_schema() -> dict:
    return {
        "feature_names": list(FEATURE_NAMES),
        "feature_formulas": dict(FEATURE_FORMULAS),
        "unit_rule": (
            "All geometry quantities for one surface use one length unit; "
            "the 12 exported features are dimensionless."
        ),
        "height_skew_rule": "H_skew_norm is the unnormalised H_skew value.",
    }


def height_skewness(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return 0.0
    mean = np.mean(values)
    std = np.std(values)
    if std < 1e-12:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 3))


def _binary_erosion_cross(mask: np.ndarray) -> np.ndarray:
    """Equivalent to SciPy's default 4-neighbour binary erosion in Dataset B."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    height, width = mask.shape
    return (
        padded[1 : 1 + height, 1 : 1 + width]
        & padded[0:height, 1 : 1 + width]
        & padded[2 : 2 + height, 1 : 1 + width]
        & padded[1 : 1 + height, 0:width]
        & padded[1 : 1 + height, 2 : 2 + width]
    )


def _convex_hull_area(mask: np.ndarray, cell_size: float) -> float:
    """Area of the convex hull of occupied cell indices, as in Dataset B."""
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return float(len(xs) * cell_size * cell_size)

    points = sorted(set(zip(xs.tolist(), ys.tolist())))

    def cross(origin, first, second) -> float:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return float(len(xs) * cell_size * cell_size)

    area2 = 0.0
    for index, point in enumerate(hull):
        next_point = hull[(index + 1) % len(hull)]
        area2 += point[0] * next_point[1] - point[1] * next_point[0]
    return float(abs(area2) * 0.5 * cell_size * cell_size)


def compute_surface_descriptors(
    height_map: np.ndarray,
    footprint_mask: np.ndarray,
    cell_size: float,
    ground_z: float = 0.0,
) -> dict | None:
    """Compute Dataset B descriptors from one ground-referenced 2.5D surface."""
    height_map = np.asarray(height_map, dtype=np.float64)
    mask = np.asarray(footprint_mask, dtype=bool)
    if height_map.shape != mask.shape or cell_size <= 0:
        raise FeatureSchemaError("height_map, footprint_mask, and cell_size are incompatible")

    valid = mask & np.isfinite(height_map)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return None

    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    y_min = int(rows.argmax())
    y_max = int(len(rows) - 1 - rows[::-1].argmax())
    x_min = int(cols.argmax())
    x_max = int(len(cols) - 1 - cols[::-1].argmax())
    extent_x = (x_max - x_min + 1) * float(cell_size)
    extent_y = (y_max - y_min + 1) * float(cell_size)
    length = max(extent_x, extent_y)
    width = min(extent_x, extent_y)

    heights = height_map[valid] - float(ground_z)
    positive = np.maximum(heights, 0.0)
    height_max = float(np.max(positive))
    height_mean = float(np.mean(positive))
    height_std = float(np.std(positive))
    height_p25 = float(np.percentile(positive, 25))
    height_p75 = float(np.percentile(positive, 75))
    height_skew = height_skewness(positive)

    area = float(n_valid * cell_size * cell_size)
    perimeter = float(np.count_nonzero(valid & ~_binary_erosion_cross(valid)) * cell_size)
    convex_area = _convex_hull_area(valid, float(cell_size))
    circularity = float(min(4.0 * math.pi * area / (perimeter * perimeter), 1.0)) if perimeter > 0 else 0.0
    aspect_ratio = float(length / width) if width > 0 else 0.0
    solidity = float(min(area / convex_area, 1.0)) if convex_area > 0 else 0.0
    compactness = float(perimeter / math.sqrt(area)) if area > 0 else 0.0
    eq_diam = math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0
    eq_diam_ratio = float(eq_diam / length) if length > 0 else 0.0
    v_box = float(length * width * height_max)
    v_ellipsoid = float(math.pi / 6.0 * length * width * height_max)
    v_2_5d = float(np.sum(positive) * cell_size * cell_size)
    fill_ratio = float(v_2_5d / v_box) if v_box > 0 else 0.0
    ellipsoid_ratio = float(v_2_5d / v_ellipsoid) if v_ellipsoid > 0 else 0.0

    return {
        "L": length, "W": width, "H": height_max,
        "A": area, "P": perimeter, "A_convex": convex_area,
        "C": circularity, "AR": aspect_ratio, "solidity": solidity,
        "compactness": compactness, "eq_diam_ratio": eq_diam_ratio,
        "H_mean": height_mean, "H_max": height_max, "H_std": height_std,
        "H_p25": height_p25, "H_p75": height_p75, "H_skew": height_skew,
        "fill_ratio": fill_ratio, "ellipsoid_ratio": ellipsoid_ratio,
        "V_box": v_box, "V_ellipsoid": v_ellipsoid, "V_2_5d": v_2_5d,
        "n_valid_cells": n_valid,
    }


def extract_features(descriptors: Mapping[str, float]) -> np.ndarray:
    missing = [name for name in REQUIRED_DESCRIPTOR_FIELDS if name not in descriptors]
    if missing:
        raise FeatureSchemaError(f"Missing V2 descriptor fields: {missing}")
    height = float(descriptors["H"])
    height = height if height > _EPS else _EPS
    features = np.asarray([
        descriptors["C"], descriptors["AR"], descriptors["solidity"],
        descriptors["compactness"], descriptors["eq_diam_ratio"],
        descriptors["H_mean"] / height, descriptors["H_std"] / height,
        descriptors["H_p25"] / height, descriptors["H_p75"] / height,
        descriptors["H_skew"], descriptors["fill_ratio"],
        descriptors["ellipsoid_ratio"],
    ], dtype=np.float64)
    return validate_feature_vector(features)


def validate_feature_vector(features: Sequence[float]) -> np.ndarray:
    vector = np.asarray(features, dtype=np.float64)
    if vector.shape != (len(FEATURE_NAMES),):
        raise FeatureSchemaError(
            f"Expected {len(FEATURE_NAMES)} Shape-Aware V2 features, got shape {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise FeatureSchemaError("Shape-Aware V2 feature vector contains NaN or Inf")
    return vector


def validate_model_feature_names(feature_names: Sequence[str]) -> None:
    names = list(feature_names)
    if names != FEATURE_NAMES:
        raise FeatureSchemaError(
            "Model feature order does not match Shape-Aware V2 schema: "
            f"expected {FEATURE_NAMES}, got {names}"
        )

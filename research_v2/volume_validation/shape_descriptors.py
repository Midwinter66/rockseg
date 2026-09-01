"""Shape descriptor extraction from a 2.5-D surface.

These features mirror the ``Shape descriptors`` table in the V2 master plan
(``00_master_plan.md`` §3.C3):

| Category       | Features                         |
|----------------|----------------------------------|
| Dimensions     | L, W, H                          |
| Area/perimeter | A, P                             |
| Circularly     | C = 4π A / P²                    |
| Aspect ratio   | AR = L / W                       |
| Height stats   | H_mean, H_max, H_std             |

All values are in mm / mm². They are **scale-invariant in normalised form**
(e.g. circularity, aspect ratio), which is what allows transfer from mm-scale
validation rocks to cm/m-scale blast rocks.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage

from .simulate_2_5d import Surface2_5D


@dataclass
class ShapeDescriptors:
    """All shape descriptors extracted from a 2.5-D surface."""

    # ── Dimensions [mm] ────────────────────────────────────────────
    L: float  # footprint length (longest bbox side)
    W: float  # footprint width  (shorter bbox side)
    H: float  # maximum height above ground

    # ── Footprint geometry ────────────────────────────────────────
    A: float  # footprint area [mm²]
    P: float  # footprint perimeter [mm]
    A_convex: float  # convex hull area [mm²]

    # ── Shape ratios (dimensionless) ──────────────────────────────
    C: float  # circularity = 4πA / P²   (1 = circle)
    AR: float  # aspect ratio = L / W      (1 = square footprint)
    solidity: float  # A / A_convex (1 = perfectly convex)
    compactness: float  # P / sqrt(A) — boundary complexity
    eq_diam_ratio: float  # sqrt(4A/π) / L — how close to circle

    # ── Height statistics [mm] ─────────────────────────────────────
    H_mean: float  # mean height over valid cells
    H_max: float   # max height (same as H)
    H_std: float   # std of height over valid cells
    H_p25: float   # 25th percentile height
    H_p75: float   # 75th percentile height
    H_skew: float  # skewness of height distribution

    # ── Volume fill ratios (dimensionless) ────────────────────────
    fill_ratio: float       # V_2.5d / V_box — how "full" the bbox
    ellipsoid_ratio: float  # V_2.5d / V_ellipsoid — how close to ellipsoid

    # ── Derived volume proxies [mm³] ──────────────────────────────
    V_box: float        # L × W × H
    V_ellipsoid: float  # (π/6) × L × W × H
    V_2_5d: float       # deterministic 2.5-D integration

    # ── Metadata ──────────────────────────────────────────────────
    n_valid_cells: int

    # Feature names in canonical order (for ML models)
    FEATURE_NAMES = [
        "L", "W", "H",
        "A", "P", "A_convex",
        "C", "AR", "solidity", "compactness", "eq_diam_ratio",
        "H_mean", "H_max", "H_std", "H_p25", "H_p75", "H_skew",
        "fill_ratio", "ellipsoid_ratio",
        "V_box", "V_ellipsoid", "V_2_5d",
    ]

    def to_vector(self) -> np.ndarray:
        """Return features as a 1-D array in canonical order."""
        return np.array([getattr(self, name) for name in self.FEATURE_NAMES])

    def to_dict(self) -> dict:
        return asdict(self)


def _compute_perimeter(footprint_mask: np.ndarray, cell_size: float) -> float:
    """Approximate the footprint perimeter from the binary mask.

    Uses binary erosion to find boundary cells, then converts to physical
    length. For a grid-aligned mask this overestimates slightly; the bias
    is consistent across all samples so comparisons remain fair.
    """
    if footprint_mask.sum() == 0:
        return 0.0
    eroded = ndimage.binary_erosion(footprint_mask)
    boundary = footprint_mask & ~eroded
    n_boundary = int(boundary.sum())
    # Each boundary cell contributes ~cell_size of perimeter
    # (diagonal neighbours slightly more, but this is a fair approximation)
    return n_boundary * cell_size


def _compute_convex_area(mask: np.ndarray, cell_size: float) -> float:
    """Compute convex hull area of the footprint from binary mask.

    Uses scipy ConvexHull on the coordinates of valid cells.
    """
    from scipy.spatial import ConvexHull
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return float(len(xs) * cell_size * cell_size)
    points = np.column_stack([xs, ys]).astype(np.float64)
    try:
        hull = ConvexHull(points)
        return float(hull.volume * cell_size * cell_size)  # 2D hull: volume = area
    except Exception:
        return float(len(xs) * cell_size * cell_size)


def _compute_skewness(values: np.ndarray) -> float:
    """Compute skewness of a distribution (Fisher-Pearson standardized)."""
    if len(values) < 3:
        return 0.0
    mean = np.mean(values)
    std = np.std(values)
    if std < 1e-12:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 3))


def extract_descriptors(surface: Surface2_5D) -> ShapeDescriptors:
    """Extract all shape descriptors from a 2.5-D surface.

    Parameters
    ----------
    surface : the simulated 2.5-D observation.

    Returns
    -------
    ShapeDescriptors
    """
    hm = surface.height_map
    mask = surface.footprint_mask
    cs = surface.cell_size
    ground = surface.ground_z

    valid = mask
    n_valid = int(valid.sum())

    if n_valid == 0:
        return ShapeDescriptors(
            L=0, W=0, H=0, A=0, P=0, A_convex=0,
            C=0, AR=0, solidity=0, compactness=0, eq_diam_ratio=0,
            H_mean=0, H_max=0, H_std=0, H_p25=0, H_p75=0, H_skew=0,
            fill_ratio=0, ellipsoid_ratio=0,
            V_box=0, V_ellipsoid=0, V_2_5d=0,
            n_valid_cells=0,
        )

    # ── Footprint bounding box ────────────────────────────────────
    # Find rows and columns with at least one valid cell
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    x_min = surface.grid_x[rows.argmax()]
    x_max = surface.grid_x[len(rows) - 1 - rows[::-1].argmax()]
    y_min = surface.grid_y[cols.argmax()]
    y_max = surface.grid_y[len(cols) - 1 - cols[::-1].argmax()]

    extent_x = x_max - x_min + cs  # add one cell width
    extent_y = y_max - y_min + cs
    L = max(extent_x, extent_y)
    W = min(extent_x, extent_y)

    # ── Height statistics ──────────────────────────────────────────
    heights_above_ground = hm[valid] - ground
    positive_heights = np.maximum(heights_above_ground, 0.0)
    H_mean = float(np.mean(positive_heights))
    H_max = float(np.max(positive_heights))
    H_std = float(np.std(positive_heights))
    H_p25 = float(np.percentile(positive_heights, 25))
    H_p75 = float(np.percentile(positive_heights, 75))
    H_skew = _compute_skewness(positive_heights)
    H = H_max  # H is defined as max height

    # ── Footprint area, perimeter, convex area ─────────────────────
    A = float(n_valid * cs * cs)
    P = _compute_perimeter(mask, cs)
    A_convex = _compute_convex_area(mask, cs)

    # ── Shape ratios (dimensionless) ──────────────────────────────
    C = float(4.0 * np.pi * A / (P * P)) if P > 0 else 0.0
    C = min(C, 1.0)  # clamp to [0, 1]
    AR = float(L / W) if W > 0 else 0.0
    solidity = float(A / A_convex) if A_convex > 0 else 0.0
    solidity = min(solidity, 1.0)
    compactness = float(P / np.sqrt(A)) if A > 0 else 0.0
    eq_diam = np.sqrt(4.0 * A / np.pi) if A > 0 else 0.0
    eq_diam_ratio = float(eq_diam / L) if L > 0 else 0.0

    # ── Volume proxies ────────────────────────────────────────────
    V_box = float(L * W * H)
    V_ellipsoid = float(np.pi / 6.0 * L * W * H)

    # Deterministic 2.5-D integration: Σ max(z_top - z_ground, 0) × Δ²
    V_2_5d = float(np.sum(positive_heights) * cs * cs)

    # ── Volume fill ratios (dimensionless) ────────────────────────
    fill_ratio = float(V_2_5d / V_box) if V_box > 0 else 0.0
    ellipsoid_ratio = float(V_2_5d / V_ellipsoid) if V_ellipsoid > 0 else 0.0

    return ShapeDescriptors(
        L=L, W=W, H=H,
        A=A, P=P, A_convex=A_convex,
        C=C, AR=AR, solidity=solidity, compactness=compactness,
        eq_diam_ratio=eq_diam_ratio,
        H_mean=H_mean, H_max=H_max, H_std=H_std,
        H_p25=H_p25, H_p75=H_p75, H_skew=H_skew,
        fill_ratio=fill_ratio, ellipsoid_ratio=ellipsoid_ratio,
        V_box=V_box, V_ellipsoid=V_ellipsoid, V_2_5d=V_2_5d,
        n_valid_cells=n_valid,
    )

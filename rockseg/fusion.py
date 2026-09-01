"""Multi-feature fusion scoring and instance merging.

Two-stage fusion:

1.  **Within-scale** (boundary-aware) — merges duplicates caused by
    overlapping tiles within the same scale.
2.  **Cross-scale** — merges duplicates across different scales that
    detected the same physical rock.

Fusion score for a pair of instances:

.. math::
    S = w_1 \\cdot \\text{IoU}
      + w_2 \\cdot e^{-d_c^2 / (2\\sigma^2)}
      + w_3 \\cdot r_A
      + w_4 \\cdot B
      + w_5 \\cdot C

where ``IoU`` is mask intersection-over-union, ``d_c`` is centroid
distance, ``r_A`` is area ratio, ``B`` is boundary completeness, and
``C`` is average confidence.
"""

from __future__ import annotations

import numpy as np

from .config import PipelineConfig
from .models import RockInstance


# ---------------------------------------------------------------------------
# Individual score components
# ---------------------------------------------------------------------------

def compute_mask_iou(inst1: RockInstance, inst2: RockInstance) -> float:
    """Mask IoU between two instances.

    Works with local masks and global bounding boxes — extracts the
    overlapping region in both local masks and computes IoU.
    """
    x_min = max(inst1.bbox[0], inst2.bbox[0])
    y_min = max(inst1.bbox[1], inst2.bbox[1])
    x_max = min(inst1.bbox[2], inst2.bbox[2])
    y_max = min(inst1.bbox[3], inst2.bbox[3])

    if x_min >= x_max or y_min >= y_max:
        return 0.0

    region1 = inst1.mask_local[
        y_min - inst1.bbox[1] : y_max - inst1.bbox[1],
        x_min - inst1.bbox[0] : x_max - inst1.bbox[0],
    ]
    region2 = inst2.mask_local[
        y_min - inst2.bbox[1] : y_max - inst2.bbox[1],
        x_min - inst2.bbox[0] : x_max - inst2.bbox[0],
    ]

    intersection = np.logical_and(region1, region2).sum()
    union = np.logical_or(region1, region2).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_bbox_iou(inst1: RockInstance, inst2: RockInstance) -> float:
    """Bounding-box IoU — used as a fast pre-filter."""
    x_min = max(inst1.bbox[0], inst2.bbox[0])
    y_min = max(inst1.bbox[1], inst2.bbox[1])
    x_max = min(inst1.bbox[2], inst2.bbox[2])
    y_max = min(inst1.bbox[3], inst2.bbox[3])

    if x_min >= x_max or y_min >= y_max:
        return 0.0

    inter = (x_max - x_min) * (y_max - y_min)
    area1 = (inst1.bbox[2] - inst1.bbox[0]) * (inst1.bbox[3] - inst1.bbox[1])
    area2 = (inst2.bbox[2] - inst2.bbox[0]) * (inst2.bbox[3] - inst2.bbox[1])
    union = area1 + area2 - inter

    return float(inter / union) if union > 0 else 0.0


def compute_centroid_score(
    inst1: RockInstance,
    inst2: RockInstance,
    sigma: float,
) -> float:
    """Gaussian centroid proximity.  ``1.0`` for identical centroids."""
    c1 = inst1.centroid
    c2 = inst2.centroid
    dist_sq = (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2
    return float(np.exp(-dist_sq / (2.0 * sigma ** 2)))


def compute_area_ratio(inst1: RockInstance, inst2: RockInstance) -> float:
    """Area similarity ratio.  ``1.0`` for equal areas."""
    a1, a2 = inst1.area, inst2.area
    if a1 == 0 or a2 == 0:
        return 0.0
    return float(min(a1, a2) / max(a1, a2))


def compute_boundary_score(inst1: RockInstance, inst2: RockInstance) -> float:
    """Average boundary completeness of the two instances."""
    return (inst1.boundary_completeness + inst2.boundary_completeness) / 2.0


def compute_confidence_score(inst1: RockInstance, inst2: RockInstance) -> float:
    """Average confidence."""
    return (inst1.confidence + inst2.confidence) / 2.0


# ---------------------------------------------------------------------------
# Combined fusion score
# ---------------------------------------------------------------------------

def compute_fusion_score(
    inst1: RockInstance,
    inst2: RockInstance,
    config: PipelineConfig,
) -> float:
    """Multi-feature fusion score for a pair of instances."""
    w = config.fusion_weights

    return float(
        w.iou * compute_mask_iou(inst1, inst2)
        + w.centroid * compute_centroid_score(inst1, inst2, config.centroid_sigma_px)
        + w.area_ratio * compute_area_ratio(inst1, inst2)
        + w.boundary * compute_boundary_score(inst1, inst2)
        + w.confidence * compute_confidence_score(inst1, inst2)
    )


# ---------------------------------------------------------------------------
# Union-Find for efficient grouping
# ---------------------------------------------------------------------------

class _UnionFind:
    """Disjoint-set data structure with path compression and union by rank."""

    def __init__(self, n: int):
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression
        while self._parent[x] != root:
            next_x = self._parent[x]
            self._parent[x] = root
            x = next_x
        return root

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self._rank[px] < self._rank[py]:
            px, py = py, px
        self._parent[py] = px
        if self._rank[px] == self._rank[py]:
            self._rank[px] += 1
        return True


# ---------------------------------------------------------------------------
# Canonical mask selection
# ---------------------------------------------------------------------------

def select_canonical(instances: list[RockInstance]) -> RockInstance:
    """Select the best instance from a merged group.

    Strategy ``"best_mask"`` (confirmed by user): pick the instance with
    the highest ``confidence * boundary_completeness`` score.
    """
    return max(instances, key=lambda inst: inst.quality_score())


# ---------------------------------------------------------------------------
# Core fusion engine
# ---------------------------------------------------------------------------

def _build_spatial_grid(
    instances: list[RockInstance],
    cell_size: int,
) -> dict[tuple[int, int], list[int]]:
    """Build a grid-based spatial index for O(1) neighbour lookup.

    Each instance is registered in every grid cell its bbox overlaps.
    """
    grid: dict[tuple[int, int], list[int]] = {}
    for i, inst in enumerate(instances):
        x1, y1, x2, y2 = inst.bbox
        cx1 = int(x1) // cell_size
        cy1 = int(y1) // cell_size
        cx2 = int(x2) // cell_size
        cy2 = int(y2) // cell_size
        for cx in range(cx1, cx2 + 1):
            for cy in range(cy1, cy2 + 1):
                grid.setdefault((cx, cy), []).append(i)
    return grid


def fuse_instances(
    instances: list[RockInstance],
    config: PipelineConfig,
    threshold: float,
    cross_scale_only: bool = False,
) -> list[RockInstance]:
    """Fuse duplicate instances using the multi-feature score.

    Uses :class:`_UnionFind` for O(n * α(n)) grouping.  A spatial
    grid index avoids the O(n²) all-pairs scan — only instances
    sharing a grid cell are compared, then a bbox-IoU pre-filter
    rejects non-overlapping pairs before the expensive mask-IoU.

    Parameters
    ----------
    instances
        Raw or partially-fused instances.
    config
        Pipeline configuration (for weights, sigma, etc.).
    threshold
        Minimum fusion score to trigger a merge.
    cross_scale_only
        If ``True``, only consider pairs from **different** scale levels
        (used by :func:`cross_scale_fusion`).
    """
    n = len(instances)
    if n <= 1:
        return list(instances)

    uf = _UnionFind(n)

    # Grid cell size: large enough to capture overlap-zone duplicates.
    # The maximum tile is 1024 px with 20 % overlap ≈ 205 px margin,
    # so a 512 px cell guarantees co-locating candidates.
    cell_size = 512
    grid = _build_spatial_grid(instances, cell_size)

    for i in range(n):
        inst_i = instances[i]
        x1, y1, x2, y2 = inst_i.bbox
        cx1 = int(x1) // cell_size
        cy1 = int(y1) // cell_size
        cx2 = int(x2) // cell_size
        cy2 = int(y2) // cell_size

        # Collect candidate indices from all overlapping cells
        candidates: set[int] = set()
        for cx in range(cx1, cx2 + 1):
            for cy in range(cy1, cy2 + 1):
                bucket = grid.get((cx, cy))
                if bucket is None:
                    continue
                for j in bucket:
                    if j > i:
                        candidates.add(j)

        for j in candidates:
            inst_j = instances[j]

            if cross_scale_only and inst_i.scale_level == inst_j.scale_level:
                continue

            if compute_bbox_iou(inst_i, inst_j) < 0.05:
                continue

            mask_iou = compute_mask_iou(inst_i, inst_j)
            if mask_iou == 0.0:
                continue

            score = compute_fusion_score(inst_i, inst_j, config)
            if score >= threshold:
                uf.union(i, j)

    # Group by root
    groups: dict[int, list[RockInstance]] = {}
    for i, inst in enumerate(instances):
        root = uf.find(i)
        groups.setdefault(root, []).append(inst)

    return [select_canonical(group) for group in groups.values()]


def within_scale_fusion(
    instances: list[RockInstance],
    config: PipelineConfig,
) -> list[RockInstance]:
    """Fuse duplicate instances within the same scale.

    Resolves tile-boundary truncation from overlapping tiles.
    """
    return fuse_instances(instances, config, config.within_scale_threshold)


def cross_scale_fusion(
    instances: list[RockInstance],
    config: PipelineConfig,
) -> list[RockInstance]:
    """Fuse duplicate instances across different scales.

    Only considers pairs from **different** scale levels.
    """
    return fuse_instances(
        instances,
        config,
        config.cross_scale_threshold,
        cross_scale_only=True,
    )


# ---------------------------------------------------------------------------
# Cascade deduplication (alternative to cross-scale fusion)
# ---------------------------------------------------------------------------

_COARSE_MEDIUM_BOUND_M = 0.50
_MEDIUM_FINE_BOUND_M = 0.30


def _primary_scale_for_diameter(diameter_m: float) -> str:
    """Return the scale level best suited for a rock of this diameter."""
    if diameter_m >= _COARSE_MEDIUM_BOUND_M:
        return "coarse"
    if diameter_m >= _MEDIUM_FINE_BOUND_M:
        return "medium"
    return "fine"


def cascade_deduplication(
    instances: list[RockInstance],
    config: PipelineConfig,
) -> list[RockInstance]:
    """Size-aware cross-scale deduplication without mask fusion.

    Instead of merging masks across scales (which can incorrectly
    merge distinct nearby rocks), this function:

    1.  Groups spatially-overlapping cross-scale detections.
    2.  For each group, determines the *primary scale* from the
        largest detection's equivalent diameter.
    3.  Keeps the detection from the primary scale; discards others.

    Boundaries are data-driven (crossover analysis of V2-v3):
    - ≥0.50 m → coarse is primary
    - 0.30–0.50 m → medium is primary
    - <0.30 m → fine is primary
    """
    n = len(instances)
    if n <= 1:
        return list(instances)

    gsd = config.gsd
    uf = _UnionFind(n)
    cell_size = 512
    grid = _build_spatial_grid(instances, cell_size)

    for i in range(n):
        inst_i = instances[i]
        x1, y1, x2, y2 = inst_i.bbox
        cx1, cy1 = int(x1) // cell_size, int(y1) // cell_size
        cx2, cy2 = int(x2) // cell_size, int(y2) // cell_size

        candidates: set[int] = set()
        for cx in range(cx1, cx2 + 1):
            for cy in range(cy1, cy2 + 1):
                bucket = grid.get((cx, cy))
                if bucket is None:
                    continue
                for j in bucket:
                    if j > i:
                        candidates.add(j)

        for j in candidates:
            inst_j = instances[j]
            if inst_i.scale_level == inst_j.scale_level:
                continue
            if compute_bbox_iou(inst_i, inst_j) < 0.05:
                continue
            if compute_mask_iou(inst_i, inst_j) == 0.0:
                continue

            # Size compatibility: only group if diameters are similar
            d_i = 2 * (inst_i.area * gsd * gsd / np.pi) ** 0.5
            d_j = 2 * (inst_j.area * gsd * gsd / np.pi) ** 0.5
            if d_i > 0 and d_j > 0:
                ratio = min(d_i, d_j) / max(d_i, d_j)
                if ratio < 0.3:
                    continue

            # Centroid proximity: centroids should be within the larger rock's radius
            ci = inst_i.centroid
            cj = inst_j.centroid
            dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
            max_radius_px = max(d_i, d_j) / gsd / 2
            if dist > max_radius_px:
                continue

            uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    result: list[RockInstance] = []
    for member_indices in groups.values():
        if len(member_indices) == 1:
            result.append(instances[member_indices[0]])
            continue

        members = [instances[i] for i in member_indices]
        diameters = [2 * (m.area * gsd * gsd / np.pi) ** 0.5 for m in members]
        max_diam = max(diameters)
        primary = _primary_scale_for_diameter(max_diam)

        primary_members = [
            (m, d) for m, d in zip(members, diameters)
            if m.scale_level == primary
        ]

        if primary_members:
            best = max(primary_members, key=lambda md: md[0].quality_score())
            result.append(best[0])
        else:
            best = max(members, key=lambda m: m.quality_score())
            result.append(best)

    return result

"""
Fusion stage experiment.

This script reads detection outputs, merges repeated detections into stone
candidates, optionally validates them in 3D, and writes both:
1. a fusion result file compatible with volume estimation
2. an analysis-friendly summary for paper/report use

Examples:
  python experiments/fusion/run_fusion_experiment.py --source quadtree_dom --method correlation_clustering
  python experiments/fusion/run_fusion_experiment.py --source sahi --method heuristic
  python experiments/fusion/run_fusion_experiment.py --source all --method all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.common.pointcloud_index import PointCloudXYGridIndex
from experiments.common.scene_reference import CURRENT_SCENE
from experiments.common.stone_region import crop_stone_point_cloud
from experiments.volume.ground_estimator import GroundDEM

SELF_DIR = Path(__file__).resolve().parent
DETECTION_OUTPUTS = PROJECT_ROOT / "experiments" / "detection" / "outputs"
SOURCES = ["sahi", "quadtree_dom"]
FUSION_METHODS = ["heuristic", "correlation_clustering"]

_CACHED_PC: np.ndarray | None = None
_CACHED_PC_INDEX: PointCloudXYGridIndex | None = None


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_fusion_config(method: str) -> dict:
    path = PROJECT_ROOT / "experiments" / "configs" / "fusion" / f"{method}.json"
    if not path.exists():
        raise FileNotFoundError(f"Fusion config not found: {path}")
    return _load_json(path)


def _load_detections(source: str) -> list[dict]:
    path = DETECTION_OUTPUTS / source / "detections.json"
    if not path.exists():
        raise FileNotFoundError(f"No detections for {source}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_output_dir(source: str, method: str) -> Path:
    path = SELF_DIR / "outputs" / source / method
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "stats": out_dir / "fusion_stats.json",
        "summary": out_dir / "fusion_summary.json",
        "candidate_stones": out_dir / "candidate_stones.json",
        "accepted_stones": out_dir / "accepted_stones.json",
        "rejected_stones": out_dir / "rejected_stones.json",
    }


def _summarize_numeric(values: list[float], digits: int = 4) -> dict | None:
    vals = [float(value) for value in values if np.isfinite(value)]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": round(float(np.min(arr)), digits),
        "max": round(float(np.max(arr)), digits),
        "mean": round(float(np.mean(arr)), digits),
        "median": round(float(np.median(arr)), digits),
        "p25": round(float(np.percentile(arr, 25)), digits),
        "p75": round(float(np.percentile(arr, 75)), digits),
    }


def _safe_ratio(numerator: float, denominator: float, digits: int = 4) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator / denominator), digits)


def _load_point_cloud() -> np.ndarray:
    global _CACHED_PC
    if _CACHED_PC is not None:
        return _CACHED_PC

    import laspy

    parts: list[np.ndarray] = []
    for path in CURRENT_SCENE.pointcloud_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing point cloud file: {path}")
        las = laspy.read(str(path))
        points = np.column_stack([las.x, las.y, las.z]).astype(np.float32)
        parts.append(points)
        print(f"    loaded {path.name}: {len(points):,} points")

    if not parts:
        raise RuntimeError("No point cloud files were loaded")

    _CACHED_PC = np.vstack(parts)
    print(f"    full scene points: {len(_CACHED_PC):,}")
    return _CACHED_PC


def _load_point_cloud_index(cell_size: float) -> PointCloudXYGridIndex:
    global _CACHED_PC_INDEX
    if _CACHED_PC_INDEX is not None and abs(_CACHED_PC_INDEX.cell_size - float(cell_size)) < 1e-9:
        return _CACHED_PC_INDEX

    pc = _load_point_cloud()
    print(f"    building XY grid index @ {cell_size:.2f} m ...")
    _CACHED_PC_INDEX = PointCloudXYGridIndex.build(pc, cell_size=cell_size)
    meta = _CACHED_PC_INDEX.to_dict()
    print(f"    index ready: {meta['indexed_cell_count']} cells for {meta['point_count']:,} points")
    return _CACHED_PC_INDEX


def _bbox_intersects(a: list[float], b: list[float], pad: float = 0.0) -> bool:
    return not (
        a[2] + pad < b[0] - pad
        or b[2] + pad < a[0] - pad
        or a[3] + pad < b[1] - pad
        or b[3] + pad < a[1] - pad
    )


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def groups(self) -> list[list[int]]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(self.parent)):
            grouped[self.find(idx)].append(idx)
        return list(grouped.values())


def _heuristic_fuse(detections: list[dict], config: dict) -> tuple[list[list[int]], dict]:
    association_cfg = config["association"]
    distance_threshold = float(association_cfg["cross_tile_distance_m"])
    iou_threshold = float(association_cfg["cross_tile_iou_threshold"])

    detection_count = len(detections)
    if detection_count == 0:
        return [], {
            "candidate_pairs": 0,
            "cross_tile_candidate_pairs": 0,
            "positive_edges": 0,
            "distance_threshold_m": distance_threshold,
            "iou_threshold": iou_threshold,
        }

    uf = UnionFind(detection_count)
    candidate_pairs = 0
    cross_tile_pairs = 0
    positive_edges = 0

    for i in range(detection_count):
        det_i = detections[i]
        for j in range(i + 1, detection_count):
            candidate_pairs += 1
            det_j = detections[j]
            if det_i.get("source_patch_id") == det_j.get("source_patch_id"):
                continue
            cross_tile_pairs += 1

            center_i = det_i.get("centroid_world", [0.0, 0.0])
            center_j = det_j.get("centroid_world", [0.0, 0.0])
            distance = math.sqrt((center_i[0] - center_j[0]) ** 2 + (center_i[1] - center_j[1]) ** 2)
            if distance > distance_threshold:
                continue

            bbox_i = det_i.get("bbox_world", [0, 0, 0, 0])
            bbox_j = det_j.get("bbox_world", [0, 0, 0, 0])
            if not _bbox_intersects(bbox_i, bbox_j):
                continue
            if iou_threshold > 0 and _bbox_iou(bbox_i, bbox_j) < iou_threshold:
                continue

            if uf.union(i, j):
                positive_edges += 1

    diagnostics = {
        "candidate_pairs": candidate_pairs,
        "cross_tile_candidate_pairs": cross_tile_pairs,
        "positive_edges": positive_edges,
        "distance_threshold_m": distance_threshold,
        "iou_threshold": iou_threshold,
    }
    return uf.groups(), diagnostics


def _correlation_clustering_with_diagnostics(
    detections: list[dict],
    config: dict,
) -> tuple[list[list[int]], dict]:
    correlation_cfg = config["correlation"]
    sigma = float(correlation_cfg["distance_sigma"])
    positive_threshold = float(correlation_cfg["positive_weight_threshold"])
    iou_weight = float(correlation_cfg.get("iou_weight", 0.3))
    use_iou = bool(correlation_cfg.get("use_iou", True))
    max_distance = float(correlation_cfg.get("max_distance_m", 5.0))
    min_pair_iou = float(correlation_cfg.get("min_pair_iou", 0.0))
    require_bbox_intersect = bool(correlation_cfg.get("require_bbox_intersect", False))
    enforce_one_detection_per_tile = bool(correlation_cfg.get("enforce_one_detection_per_tile", True))
    cell_size = float(correlation_cfg.get("spatial_index_cell_m", max_distance))

    if sigma <= 0 or positive_threshold < 0 or max_distance <= 0 or cell_size <= 0:
        raise ValueError("Invalid correlation-clustering config values")

    detection_count = len(detections)
    if detection_count <= 1:
        groups = [list(range(detection_count))] if detection_count == 1 else []
        return groups, {
            "candidate_pairs": 0,
            "cross_tile_candidate_pairs": 0,
            "positive_edges": 0,
            "spatial_index_cells": detection_count,
            "spatial_index_cell_m": cell_size,
            "neighbor_cell_radius": 0,
            "min_pair_iou": min_pair_iou,
            "enforce_one_detection_per_tile": enforce_one_detection_per_tile,
            "require_bbox_intersect": require_bbox_intersect,
        }

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(detection_count)]
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    neighbor_radius = max(1, int(math.ceil(max_distance / cell_size)))
    max_distance_sq = max_distance * max_distance
    candidate_pairs = 0
    cross_tile_pairs = 0
    positive_edges = 0

    for i, det_i in enumerate(detections):
        center_i = det_i.get("centroid_world", [0.0, 0.0])
        bbox_i = det_i.get("bbox_world", [0, 0, 0, 0])
        cell = (
            math.floor(float(center_i[0]) / cell_size),
            math.floor(float(center_i[1]) / cell_size),
        )

        for gx in range(cell[0] - neighbor_radius, cell[0] + neighbor_radius + 1):
            for gy in range(cell[1] - neighbor_radius, cell[1] + neighbor_radius + 1):
                for j in grid.get((gx, gy), []):
                    candidate_pairs += 1
                    det_j = detections[j]
                    if det_i.get("source_patch_id") == det_j.get("source_patch_id"):
                        continue
                    cross_tile_pairs += 1

                    center_j = det_j.get("centroid_world", [0.0, 0.0])
                    dx = float(center_i[0]) - float(center_j[0])
                    dy = float(center_i[1]) - float(center_j[1])
                    distance_sq = dx * dx + dy * dy
                    if distance_sq > max_distance_sq:
                        continue

                    bbox_j = det_j.get("bbox_world", [0, 0, 0, 0])
                    intersects = _bbox_intersects(bbox_i, bbox_j)
                    if require_bbox_intersect and not intersects:
                        continue

                    iou = _bbox_iou(bbox_i, bbox_j) if use_iou and intersects else 0.0
                    if iou < min_pair_iou:
                        continue

                    weight = math.exp(-distance_sq / (2.0 * sigma**2)) * (1.0 + iou_weight * iou)
                    if weight < positive_threshold:
                        continue

                    adjacency[i].append((j, weight))
                    adjacency[j].append((i, weight))
                    positive_edges += 1

        grid[cell].append(i)

    active = [True] * detection_count
    groups: list[list[int]] = []

    for pivot in range(detection_count):
        if not active[pivot]:
            continue

        group = [pivot]
        active[pivot] = False

        if enforce_one_detection_per_tile:
            best_by_tile: dict[str, tuple[int, float, float]] = {}
            for neighbor, weight in adjacency[pivot]:
                if not active[neighbor]:
                    continue
                tile_id = detections[neighbor].get("source_patch_id") or f"__missing_{neighbor}"
                confidence = float(detections[neighbor].get("score", 0.0))
                current = best_by_tile.get(tile_id)
                if current is None or (weight, confidence, -neighbor) > (current[1], current[2], -current[0]):
                    best_by_tile[tile_id] = (neighbor, weight, confidence)
            selected = sorted(entry[0] for entry in best_by_tile.values())
        else:
            selected = sorted(neighbor for neighbor, _ in adjacency[pivot] if active[neighbor])

        for neighbor in selected:
            if active[neighbor]:
                active[neighbor] = False
                group.append(neighbor)

        groups.append(group)

    diagnostics = {
        "candidate_pairs": candidate_pairs,
        "cross_tile_candidate_pairs": cross_tile_pairs,
        "positive_edges": positive_edges,
        "spatial_index_cells": len(grid),
        "spatial_index_cell_m": cell_size,
        "neighbor_cell_radius": neighbor_radius,
        "distance_sigma": sigma,
        "positive_weight_threshold": positive_threshold,
        "min_pair_iou": min_pair_iou,
        "use_iou": use_iou,
        "iou_weight": iou_weight,
        "require_bbox_intersect": require_bbox_intersect,
        "enforce_one_detection_per_tile": enforce_one_detection_per_tile,
        "max_distance_m": max_distance,
    }
    return groups, diagnostics


def _build_candidate_stones(groups: list[list[int]], detections: list[dict], method: str) -> list[dict]:
    stones: list[dict] = []
    for group_index, indices in enumerate(groups):
        members = [detections[idx] for idx in indices]
        source_patch_ids = sorted({member.get("source_patch_id", "") for member in members})
        bboxes = [member.get("bbox_world", [0, 0, 0, 0]) for member in members]
        scores = [float(member.get("score", 0.0)) for member in members]
        diameters = [float(member.get("equivalent_diameter_m", 0.0)) for member in members]
        areas = [float(member.get("area_m2", 0.0)) for member in members]
        representative_local_idx = int(np.argmax(scores)) if scores else 0
        merged_bbox = [
            round(min(bbox[0] for bbox in bboxes), 4),
            round(min(bbox[1] for bbox in bboxes), 4),
            round(max(bbox[2] for bbox in bboxes), 4),
            round(max(bbox[3] for bbox in bboxes), 4),
        ]
        center_x = round((merged_bbox[0] + merged_bbox[2]) / 2.0, 4)
        center_y = round((merged_bbox[1] + merged_bbox[3]) / 2.0, 4)
        stones.append(
            {
                "stone_id": f"stone_{group_index:06d}",
                "merge_method": method,
                "source_detection_count": len(members),
                "source_patches_span": len(source_patch_ids),
                "source_patch_ids": source_patch_ids,
                "score_mean": round(float(np.mean(scores)) if scores else 0.0, 4),
                "score_max": round(float(max(scores)) if scores else 0.0, 4),
                "equivalent_diameter_m": round(float(np.median(diameters)) if diameters else 0.0, 4),
                "diameter_min_m": round(float(min(diameters)) if diameters else 0.0, 4),
                "diameter_max_m": round(float(max(diameters)) if diameters else 0.0, 4),
                "area_m2": round(float(np.median(areas)) if areas else 0.0, 4),
                "area_min_m2": round(float(min(areas)) if areas else 0.0, 4),
                "area_max_m2": round(float(max(areas)) if areas else 0.0, 4),
                "bbox_world": merged_bbox,
                "bbox_area_m2": round(max(0.0, merged_bbox[2] - merged_bbox[0]) * max(0.0, merged_bbox[3] - merged_bbox[1]), 4),
                "centroid_world": [center_x, center_y],
                "representative_detection_index": int(indices[representative_local_idx]),
                "detection_indices": indices,
            }
        )
    return stones


def _validate_stones_3d(
    stones: list[dict],
    detections: list[dict],
    config: dict,
) -> tuple[list[dict], list[dict], dict, list[dict]]:
    validation_cfg = config.get("validation_3d", {})
    enabled = bool(validation_cfg.get("enabled", False))
    if not enabled or not stones:
        summary = {
            "enabled": enabled,
            "candidate_stones": len(stones),
            "accepted_stones": len(stones),
            "rejected_stones": 0,
        }
        tile_reports = [
            {
                "stone_id": stone["stone_id"],
                "status": "skipped" if not enabled else "accepted",
                "reasons": [],
                "point_count": None,
            }
            for stone in stones
        ]
        return stones, [], summary, tile_reports

    gt = CURRENT_SCENE.load_gt()
    pc = _load_point_cloud()

    dem_cfg = validation_cfg.get("ground_dem", {})
    ground_dem = GroundDEM(
        pc,
        resolution=float(dem_cfg.get("resolution_m", 0.5)),
        percentile=int(dem_cfg.get("percentile", 5)),
        subsample_step=int(dem_cfg.get("subsample_step", 100)),
        min_points_per_cell=int(dem_cfg.get("min_points_per_cell", 3)),
    )

    crop_cfg = validation_cfg.get("crop", {})
    filter_cfg = validation_cfg.get("filter", {})
    bbox_pad_m = float(crop_cfg.get("bbox_pad_m", 0.5))
    min_points = int(filter_cfg.get("min_points", 30))
    min_z_range = float(filter_cfg.get("min_z_range_m", 0.1))
    elevated_height_m = float(filter_cfg.get("elevated_height_m", 0.05))
    min_p90_height = float(filter_cfg.get("min_p90_height_m", 0.08))
    min_elevated_ratio = float(filter_cfg.get("min_elevated_ratio", 0.1))
    index_cell_size = float(crop_cfg.get("index_cell_size_m", max(1.0, bbox_pad_m * 2.0)))
    pc_index = _load_point_cloud_index(index_cell_size)

    accepted: list[dict] = []
    rejected: list[dict] = []
    validation_reports: list[dict] = []
    reason_counter: Counter[str] = Counter()
    point_counts: list[float] = []
    z_ranges: list[float] = []
    p90_heights: list[float] = []
    elevated_ratios: list[float] = []

    for stone in stones:
        stone_points, crop_info = crop_stone_point_cloud(
            pc,
            stone,
            detections,
            gt,
            CURRENT_SCENE.xy_transform,
            bbox_pad_m=bbox_pad_m,
            pc_index=pc_index,
        )
        point_count = int(len(stone_points))
        z_range = float(np.ptp(stone_points[:, 2])) if point_count > 0 else 0.0

        if point_count > 0:
            ground_z = ground_dem.get_ground_z(stone_points[:, 0], stone_points[:, 1])
            rel_height = stone_points[:, 2].astype(np.float64) - ground_z
            rel_height = rel_height[np.isfinite(rel_height)]
        else:
            rel_height = np.asarray([], dtype=np.float64)

        if rel_height.size > 0:
            p50_height = float(np.median(rel_height))
            p90_height = float(np.quantile(rel_height, 0.9))
            max_height = float(np.max(rel_height))
            elevated_ratio = float(np.mean(rel_height >= elevated_height_m))
        else:
            p50_height = 0.0
            p90_height = 0.0
            max_height = 0.0
            elevated_ratio = 0.0

        reasons: list[str] = []
        if point_count < min_points:
            reasons.append("too_few_points")
        if z_range < min_z_range:
            reasons.append("insufficient_z_range")
        if p90_height < min_p90_height:
            reasons.append("insufficient_p90_height")
        if elevated_ratio < min_elevated_ratio:
            reasons.append("insufficient_elevated_ratio")

        stone_out = dict(stone)
        stone_out["validation_3d"] = {
            "passed": len(reasons) == 0,
            "reasons": reasons,
            "point_count": point_count,
            "z_range_m": round(z_range, 4),
            "height_above_ground": {
                "p50_m": round(p50_height, 4),
                "p90_m": round(p90_height, 4),
                "max_m": round(max_height, 4),
                "elevated_ratio": round(elevated_ratio, 4),
                "elevated_height_threshold_m": elevated_height_m,
            },
            "crop": crop_info,
        }

        validation_reports.append(
            {
                "stone_id": stone["stone_id"],
                "status": "accepted" if not reasons else "rejected",
                "reasons": reasons,
                "point_count": point_count,
                "z_range_m": round(z_range, 4),
                "p90_height_m": round(p90_height, 4),
                "elevated_ratio": round(elevated_ratio, 4),
                "source_detection_count": stone.get("source_detection_count", 0),
                "source_patches_span": stone.get("source_patches_span", 0),
            }
        )

        point_counts.append(float(point_count))
        z_ranges.append(z_range)
        p90_heights.append(p90_height)
        elevated_ratios.append(elevated_ratio)

        if reasons:
            for reason in reasons:
                reason_counter[reason] += 1
            rejected.append(stone_out)
        else:
            accepted.append(stone_out)

    summary = {
        "enabled": True,
        "candidate_stones": len(stones),
        "accepted_stones": len(accepted),
        "rejected_stones": len(rejected),
        "acceptance_ratio": _safe_ratio(len(accepted), len(stones)),
        "rejection_ratio": _safe_ratio(len(rejected), len(stones)),
        "rejection_reasons": dict(reason_counter),
        "metrics": {
            "point_count": _summarize_numeric(point_counts, digits=2),
            "z_range_m": _summarize_numeric(z_ranges, digits=4),
            "p90_height_m": _summarize_numeric(p90_heights, digits=4),
            "elevated_ratio": _summarize_numeric(elevated_ratios, digits=4),
        },
        "ground_dem": ground_dem.to_dict(),
        "filter": {
            "min_points": min_points,
            "min_z_range_m": min_z_range,
            "elevated_height_m": elevated_height_m,
            "min_p90_height_m": min_p90_height,
            "min_elevated_ratio": min_elevated_ratio,
            "bbox_pad_m": bbox_pad_m,
            "index_cell_size_m": index_cell_size,
        },
    }
    return accepted, rejected, summary, validation_reports


def _summarize_stone_set(stones: list[dict]) -> dict:
    detection_counts = [float(stone.get("source_detection_count", 0)) for stone in stones]
    tile_spans = [float(stone.get("source_patches_span", 0)) for stone in stones]
    diameters = [float(stone.get("equivalent_diameter_m", 0.0)) for stone in stones]
    areas = [float(stone.get("area_m2", 0.0)) for stone in stones]
    bbox_areas = [float(stone.get("bbox_area_m2", 0.0)) for stone in stones]
    score_means = [float(stone.get("score_mean", 0.0)) for stone in stones]
    return {
        "count": len(stones),
        "source_detection_count": _summarize_numeric(detection_counts, digits=4),
        "source_patches_span": _summarize_numeric(tile_spans, digits=4),
        "equivalent_diameter_m": _summarize_numeric(diameters, digits=4),
        "area_m2": _summarize_numeric(areas, digits=4),
        "bbox_area_m2": _summarize_numeric(bbox_areas, digits=4),
        "score_mean": _summarize_numeric(score_means, digits=4),
    }


def _build_summary(
    source: str,
    method: str,
    config: dict,
    detections: list[dict],
    candidate_stones: list[dict],
    accepted_stones: list[dict],
    rejected_stones: list[dict],
    fusion_diagnostics: dict,
    validation_summary: dict,
    elapsed_seconds: float,
    out_dir: Path,
) -> dict:
    detection_count = len(detections)
    candidate_count = len(candidate_stones)
    accepted_count = len(accepted_stones)
    rejected_count = len(rejected_stones)

    summary = {
        "analysis_version": "fusion_v2",
        "source": source,
        "method": method,
        "config": config,
        "input_detections": detection_count,
        "candidate_stones": candidate_count,
        "output_stones": accepted_count,
        "rejected_stones": rejected_count,
        "candidate_per_detection_ratio": _safe_ratio(candidate_count, detection_count),
        "accepted_per_detection_ratio": _safe_ratio(accepted_count, detection_count),
        "candidate_acceptance_ratio": _safe_ratio(accepted_count, candidate_count),
        "merge_ratio": round(1.0 - accepted_count / max(detection_count, 1), 4),
        "fusion_only_merge_ratio": round(1.0 - candidate_count / max(detection_count, 1), 4),
        "elapsed_seconds": round(elapsed_seconds, 4),
        "fusion_diagnostics": fusion_diagnostics,
        "validation_3d": validation_summary,
        "candidates": _summarize_stone_set(candidate_stones),
        "accepted": _summarize_stone_set(accepted_stones),
        "rejected": _summarize_stone_set(rejected_stones),
        "outputs": {
            "fusion_stats_json": str(out_dir / "fusion_stats.json"),
            "fusion_summary_json": str(out_dir / "fusion_summary.json"),
            "candidate_stones_json": str(out_dir / "candidate_stones.json"),
            "accepted_stones_json": str(out_dir / "accepted_stones.json"),
            "rejected_stones_json": str(out_dir / "rejected_stones.json"),
        },
    }
    return summary


def _write_outputs(
    out_dir: Path,
    summary: dict,
    candidate_stones: list[dict],
    accepted_stones: list[dict],
    rejected_stones: list[dict],
    validation_reports: list[dict],
) -> dict[str, Path]:
    paths = _artifact_paths(out_dir)
    full_stats = dict(summary)
    full_stats["stones"] = accepted_stones
    full_stats["rejected_stones_detail"] = rejected_stones
    full_stats["candidate_stones_detail"] = candidate_stones
    full_stats["validation_reports"] = validation_reports

    paths["stats"].write_text(json.dumps(full_stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    paths["candidate_stones"].write_text(json.dumps(candidate_stones, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["accepted_stones"].write_text(json.dumps(accepted_stones, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["rejected_stones"].write_text(json.dumps(rejected_stones, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def _run_fusion(source: str, method: str) -> dict:
    detections = _load_detections(source)
    config = _load_fusion_config(method)
    out_dir = _resolve_output_dir(source, method)

    start = time.perf_counter()
    if method == "heuristic":
        groups, fusion_diagnostics = _heuristic_fuse(detections, config)
    elif method == "correlation_clustering":
        groups, fusion_diagnostics = _correlation_clustering_with_diagnostics(detections, config)
    else:
        raise ValueError(f"Unsupported fusion method: {method}")
    elapsed_seconds = time.perf_counter() - start

    candidate_stones = _build_candidate_stones(groups, detections, method)
    accepted_stones, rejected_stones, validation_summary, validation_reports = _validate_stones_3d(
        candidate_stones,
        detections,
        config,
    )

    summary = _build_summary(
        source=source,
        method=method,
        config=config,
        detections=detections,
        candidate_stones=candidate_stones,
        accepted_stones=accepted_stones,
        rejected_stones=rejected_stones,
        fusion_diagnostics=fusion_diagnostics,
        validation_summary=validation_summary,
        elapsed_seconds=elapsed_seconds,
        out_dir=out_dir,
    )
    _write_outputs(
        out_dir=out_dir,
        summary=summary,
        candidate_stones=candidate_stones,
        accepted_stones=accepted_stones,
        rejected_stones=rejected_stones,
        validation_reports=validation_reports,
    )

    group_sizes = [len(group) for group in groups]
    group_mean = (sum(group_sizes) / len(group_sizes)) if group_sizes else 0.0
    print(
        f"  [{source}/{method}] {len(detections)} dets -> "
        f"{len(candidate_stones)} candidates -> {len(accepted_stones)} stones "
        f"(merge={summary['merge_ratio']:.1%}) in {elapsed_seconds:.2f}s"
    )
    print(
        f"    group size: min={min(group_sizes) if group_sizes else 0}  "
        f"max={max(group_sizes) if group_sizes else 0}  "
        f"mean={group_mean:.2f}"
    )
    if validation_summary.get("enabled"):
        print(
            f"    3D validation: kept={validation_summary['accepted_stones']}  "
            f"rejected={validation_summary['rejected_stones']}  "
            f"acceptance={validation_summary['acceptance_ratio']:.1%}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fusion experiment")
    parser.add_argument("--source", choices=["all"] + SOURCES, default="all", help="Slicing source")
    parser.add_argument("--method", choices=["all"] + FUSION_METHODS, default="all", help="Fusion method")
    args = parser.parse_args()

    methods = FUSION_METHODS if args.method == "all" else [args.method]
    sources = SOURCES if args.source == "all" else [args.source]

    print(f"\n{'=' * 64}")
    print("  Fusion Experiment")
    print(f"  Sources: {sources}")
    print(f"  Methods: {methods}")
    print(f"{'=' * 64}\n")

    all_results: dict[str, dict[str, dict]] = defaultdict(dict)
    for source in sources:
        for method in methods:
            try:
                stats = _run_fusion(source, method)
                all_results[source][method] = stats
            except FileNotFoundError as exc:
                print(f"  SKIP: {exc}")
            except Exception as exc:
                print(f"  FAILED: {exc}")
                import traceback

                traceback.print_exc()

    manifest_path = SELF_DIR / "outputs" / "fusion_manifest.json"
    manifest_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nFusion manifest: {manifest_path}")


if __name__ == "__main__":
    main()

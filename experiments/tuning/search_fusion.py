"""
Fusion parameter search.

This script runs a small grid search over fusion parameters and scores each
configuration with a lightweight 3D consistency check on the point cloud.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.pointcloud_index import PointCloudXYGridIndex
from experiments.common.scene_reference import CURRENT_SCENE
from experiments.common.stone_region import crop_stone_point_cloud
from experiments.fusion.run_fusion_experiment import (
    _build_candidate_stones,
    _correlation_clustering_with_diagnostics,
    _heuristic_fuse,
    _load_detections,
    _load_fusion_config,
)


SELF_DIR = Path(__file__).resolve().parent
FUSION_METHODS = ["heuristic", "correlation_clustering"]
_CACHED_PC: np.ndarray | None = None
_CACHED_PC_INDEX: dict[float, PointCloudXYGridIndex] = {}


def _grid_values(param: dict[str, Any]) -> list[float]:
    mn = float(param["min"])
    mx = float(param["max"])
    steps = int(param["steps"])
    if steps <= 1:
        return [round((mn + mx) / 2.0, 4)]
    return [round(v, 4) for v in np.linspace(mn, mx, steps)]


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
        parts.append(np.column_stack([las.x, las.y, las.z]).astype(np.float32))

    if not parts:
        raise RuntimeError("No point cloud files were loaded")

    _CACHED_PC = np.vstack(parts)
    print(f"  loaded point cloud: {len(_CACHED_PC):,} points")
    return _CACHED_PC


def _load_point_cloud_index(cell_size: float) -> PointCloudXYGridIndex:
    key = float(cell_size)
    cached = _CACHED_PC_INDEX.get(key)
    if cached is not None:
        return cached

    pc = _load_point_cloud()
    print(f"  building XY grid index @ {cell_size:.2f} m")
    index = PointCloudXYGridIndex.build(pc, cell_size=cell_size)
    _CACHED_PC_INDEX[key] = index
    return index


def _auto_precision(
    stones: list[dict],
    detections: list[dict],
    gt: tuple[float, float, float, float, float, float],
    z_thresh: float,
    min_points: int,
    bbox_pad_m: float,
    index_cell_size: float,
) -> tuple[float, int, int, int]:
    pc = _load_point_cloud()
    pc_index = _load_point_cloud_index(index_cell_size)

    tp = 0
    fp = 0
    skipped = 0
    for stone in stones:
        if int(stone.get("source_detection_count", 0)) <= 0:
            skipped += 1
            continue

        matched, crop_info = crop_stone_point_cloud(
            pc,
            stone,
            detections,
            gt,
            CURRENT_SCENE.xy_transform,
            bbox_pad_m=bbox_pad_m,
            pc_index=pc_index,
        )

        if int(crop_info.get("polygon_count", 0)) <= 0:
            skipped += 1
            continue
        if int(crop_info.get("bbox_candidate_count", 0)) < min_points:
            skipped += 1
            continue
        if len(matched) < min_points:
            skipped += 1
            continue

        z_range = float(np.ptp(matched[:, 2]))
        if z_range >= z_thresh:
            tp += 1
        else:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precision, tp, fp, skipped


def _build_runtime_config(method: str, params_dict: dict[str, float]) -> dict:
    config = copy.deepcopy(_load_fusion_config(method))
    section_name = "correlation" if method == "correlation_clustering" else "association"
    section = config.setdefault(section_name, {})
    for key, value in params_dict.items():
        section[key] = value
    return config


def _fuse_groups(method: str, detections: list[dict], config: dict) -> list[list[int]]:
    if method == "correlation_clustering":
        groups, _ = _correlation_clustering_with_diagnostics(detections, config)
        return groups
    return _heuristic_fuse(detections, config)


def _normalized_param_distance(a: dict, b: dict, param_names: list[str], grid_lookup: dict[str, list[float]]) -> int:
    distance = 0
    for name in param_names:
        values = grid_lookup[name]
        ia = values.index(round(float(a[name]), 4))
        ib = values.index(round(float(b[name]), 4))
        distance += abs(ia - ib)
    return distance


def _build_neighborhood_comparison(
    results: list[dict],
    param_names: list[str],
    grid_lookup: dict[str, list[float]],
    top_k: int = 5,
    max_distance: int = 2,
) -> list[dict]:
    comparisons: list[dict] = []
    for result in results[:top_k]:
        neighbors = [
            other for other in results
            if other is not result
            and _normalized_param_distance(result, other, param_names, grid_lookup) <= max_distance
        ]
        if neighbors:
            score_mean = float(np.mean([n["score"] for n in neighbors]))
            precision_mean = float(np.mean([n["precision"] for n in neighbors]))
            stability_mean = float(np.mean([n["stability"] for n in neighbors]))
        else:
            score_mean = 0.0
            precision_mean = 0.0
            stability_mean = 0.0

        comparisons.append(
            {
                "params": {name: result[name] for name in param_names},
                "score": result["score"],
                "precision": result["precision"],
                "stability": result["stability"],
                "neighbor_count": len(neighbors),
                "neighbor_score_mean": round(score_mean, 4),
                "neighbor_precision_mean": round(precision_mean, 4),
                "neighbor_stability_mean": round(stability_mean, 4),
                "score_advantage": round(float(result["score"] - score_mean), 4),
            }
        )
    return comparisons


def _search_fusion(
    method: str,
    detections: list[dict],
    gt: tuple[float, float, float, float, float, float],
    config: dict,
    fast: bool = False,
) -> tuple[list[dict], dict[str, list[float]]]:
    param_config = config.get("fusion", {}).get(method, {})
    param_names = [name for name in param_config.keys() if not name.startswith("_")]
    grid_lookup = {name: _grid_values(param_config[name]) for name in param_names}

    if fast:
        reduced: dict[str, list[float]] = {}
        for name, values in grid_lookup.items():
            if len(values) <= 3:
                reduced[name] = values
            else:
                reduced[name] = [values[0], values[len(values) // 2], values[-1]]
        grid_lookup = reduced

    combinations = list(itertools.product(*(grid_lookup[name] for name in param_names)))
    print(f"\n  {method} search: {len(combinations)} combinations")

    z_cfg = config.get("z_range", {})
    z_thresh = float(z_cfg.get("threshold_m", 0.3))
    min_points = int(z_cfg.get("min_points_per_stone", 10))
    bbox_pad_m = float(z_cfg.get("bbox_pad_m", 0.5))
    index_cell_size = float(z_cfg.get("index_cell_size_m", max(1.0, bbox_pad_m * 2.0)))

    results: list[dict] = []
    for rank, combo in enumerate(combinations, start=1):
        params_dict = dict(zip(param_names, combo))
        runtime_config = _build_runtime_config(method, params_dict)

        t0 = time.perf_counter()
        try:
            groups = _fuse_groups(method, detections, runtime_config)
            stones = _build_candidate_stones(groups, detections, method)
            precision, tp, fp, skipped = _auto_precision(
                stones,
                detections,
                gt,
                z_thresh=z_thresh,
                min_points=min_points,
                bbox_pad_m=bbox_pad_m,
                index_cell_size=index_cell_size,
            )
        except Exception as exc:
            param_str = "  ".join(f"{k}={v}" for k, v in params_dict.items())
            print(f"  [{rank:3d}/{len(combinations)}] {param_str}  FAILED: {exc}")
            continue

        elapsed_s = time.perf_counter() - t0
        merge_ratio = 1.0 - len(stones) / max(len(detections), 1)
        stability = max(0.0, 1.0 - abs(merge_ratio - 0.45))
        score = 0.7 * precision + 0.3 * stability

        result = {
            **params_dict,
            "precision": round(float(precision), 4),
            "tp": int(tp),
            "fp": int(fp),
            "skipped": int(skipped),
            "evaluated": int(tp + fp),
            "stones": int(len(stones)),
            "merge_ratio": round(float(merge_ratio), 4),
            "stability": round(float(stability), 4),
            "score": round(float(score), 4),
            "elapsed_s": round(float(elapsed_s), 2),
        }
        results.append(result)

        param_str = "  ".join(f"{k}={v}" for k, v in params_dict.items())
        print(
            f"  [{rank:3d}/{len(combinations)}] {param_str}  "
            f"score={result['score']:.3f}  P={result['precision']:.3f}  "
            f"stones={result['stones']}  skipped={result['skipped']}"
        )

    results.sort(key=lambda row: (row["score"], row["precision"], row["stability"]), reverse=True)
    return results, grid_lookup


def _summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(arr.std(ddof=0)), 4),
    }


def _report(results: list[dict], method: str, grid_lookup: dict[str, list[float]]) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {method} best parameters (Top 5)")
    print(f"{'=' * 64}")

    param_names = [name for name in grid_lookup.keys()]
    for rank, row in enumerate(results[:5], start=1):
        param_str = "  ".join(f"{name}={row[name]}" for name in param_names)
        print(
            f"  {rank:2d}  score={row['score']:.3f}  P={row['precision']:.3f}  "
            f"stab={row['stability']:.3f}  {param_str}"
        )
        print(
            f"      stones={row['stones']}  merge={row['merge_ratio']:.1%}  "
            f"tp={row['tp']}  fp={row['fp']}  skipped={row['skipped']}"
        )

    summary = {
        "score": _summarize_numeric([row["score"] for row in results]),
        "precision": _summarize_numeric([row["precision"] for row in results]),
        "stability": _summarize_numeric([row["stability"] for row in results]),
        "stones": _summarize_numeric([row["stones"] for row in results]),
    }
    neighborhood = _build_neighborhood_comparison(results, param_names, grid_lookup, top_k=5, max_distance=2)

    out = {
        "task": "fusion",
        "method": method,
        "scene": CURRENT_SCENE.to_dict(),
        "summary": summary,
        "top5": results[:5],
        "neighborhood_comparison": neighborhood,
        "all": results,
    }
    out_path = SELF_DIR / "outputs" / f"fusion_search_{method}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  output: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion parameter grid search")
    parser.add_argument("--method", choices=FUSION_METHODS + ["both"], default="both")
    parser.add_argument("--source", choices=["sahi", "quadtree_dom"], default="sahi")
    parser.add_argument("--fast", action="store_true", help="Reduce the search grid")
    args = parser.parse_args()

    config = json.loads((SELF_DIR / "config.json").read_text(encoding="utf-8"))
    gt = CURRENT_SCENE.load_gt()
    detections = _load_detections(args.source)
    print(f"\n  detections: {len(detections)}")

    methods = FUSION_METHODS if args.method == "both" else [args.method]
    for method in methods:
        results, grid_lookup = _search_fusion(method, detections, gt, config, args.fast)
        _report(results, method, grid_lookup)


if __name__ == "__main__":
    main()

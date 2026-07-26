"""
Volume experiment entry point.

Main pipeline:
  1. Load fused stone groups and detection masks.
  2. Reconstruct a scene-level ground DEM from the OSGB-derived point cloud.
  3. Crop each fused stone point cloud from the full scene.
  4. Estimate per-stone volume with:
       - 2.5D ground-referenced grid integration (main reported method)
       - 2D proxy from fused equivalent diameter (comparison baseline)
  5. Apply paper-oriented QC and export JSON summaries for manuscript tables.

Examples:
  python experiments/volume/run_volume.py --source quadtree_dom --method correlation_clustering
  python experiments/volume/run_volume.py --source quadtree_dom --method correlation_clustering --max-stones 10 --progress-every 5
  python experiments/volume/run_volume.py --source quadtree_dom --method correlation_clustering --debug-dem
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.pointcloud_index import PointCloudXYGridIndex
from experiments.common.scene_reference import CURRENT_SCENE
from experiments.common.stone_region import crop_stone_point_cloud
from experiments.volume.estimators import (
    estimate_2d5_with_ground,
    estimate_2d_proxy_from_diameter,
    estimate_projected_footprint,
)
from experiments.volume.ground_estimator import GroundDEM

SELF_DIR = Path(__file__).resolve().parent
FUSION_ROOT = PROJECT_ROOT / "experiments" / "fusion" / "outputs"
DETECTION_ROOT = PROJECT_ROOT / "experiments" / "detection" / "outputs"

SOURCES = ["sahi", "quadtree_dom"]
METHODS = ["heuristic", "correlation_clustering"]
ALL_TOKEN = "all"

_CACHED_PC_INDEX: PointCloudXYGridIndex | None = None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _load_point_cloud() -> np.ndarray:
    import laspy

    all_pts: list[np.ndarray] = []
    for path in CURRENT_SCENE.pointcloud_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing point cloud file: {path}")
        las = laspy.read(str(path))
        pts = np.column_stack([las.x, las.y, las.z]).astype(np.float64, copy=False)
        all_pts.append(pts)
        print(f"  loaded {path.name}: {len(pts):,} points")

    if not all_pts:
        raise RuntimeError("No point cloud data loaded")

    pc = np.vstack(all_pts)
    print(f"  full scene points: {len(pc):,}")
    return pc


def _load_point_cloud_index(pc: np.ndarray, cell_size: float) -> PointCloudXYGridIndex:
    global _CACHED_PC_INDEX
    if _CACHED_PC_INDEX is not None and abs(_CACHED_PC_INDEX.cell_size - float(cell_size)) < 1e-9:
        return _CACHED_PC_INDEX

    print(f"Building XY grid index @ {cell_size:.2f} m ...")
    _CACHED_PC_INDEX = PointCloudXYGridIndex.build(pc, cell_size=cell_size)
    meta = _CACHED_PC_INDEX.to_dict()
    print(f"  index ready: {meta['indexed_cell_count']} cells for {meta['point_count']:,} points")
    return _CACHED_PC_INDEX


def _load_fusion(source: str, method: str) -> tuple[dict, list[dict], list[dict]]:
    fusion_path = FUSION_ROOT / source / method / "fusion_stats.json"
    detections_path = DETECTION_ROOT / source / "detections.json"
    if not fusion_path.exists():
        raise FileNotFoundError(f"Missing fusion file: {fusion_path}")
    if not detections_path.exists():
        raise FileNotFoundError(f"Missing detections file: {detections_path}")

    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    detections = json.loads(detections_path.read_text(encoding="utf-8"))
    stones = fusion.get("stones") or fusion.get("validated_stones") or []
    return fusion, stones, detections


def _stats(values: list[float]) -> dict:
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


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0 or not np.isfinite(numerator) or not np.isfinite(denominator):
        return 0.0
    return float(numerator / denominator)


def _pearson_r(values_x: list[float], values_y: list[float]) -> float:
    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or y.size < 2:
        return 0.0
    x_std = float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=0))
    if x_std <= 0 or y_std <= 0:
        return 0.0
    return round(float(np.corrcoef(x, y)[0, 1]), 4)


def _count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _profile_specs(config: dict) -> tuple[str, dict[str, dict]]:
    profiles = config.get("qc_profiles") or {}
    if not profiles:
        profiles = {
            "default": {
                "description": "Legacy QC profile derived from filter settings.",
                "rules": dict(config.get("filter", {})),
            }
        }
    selected = str(config.get("selected_qc_profile", "paper"))
    if selected not in profiles:
        selected = next(iter(profiles))
    normalized: dict[str, dict] = {}
    for name, payload in profiles.items():
        normalized[name] = {
            "description": str(payload.get("description", "")),
            "rules": dict(payload.get("rules", {})),
        }
    return selected, normalized


def _evaluate_qc_profile(
    *,
    profile_name: str,
    profile_spec: dict,
    point_count: int,
    z_range_m: float,
    volume_2d5: dict,
) -> dict:
    rules = profile_spec.get("rules", {})
    flags: list[str] = []
    if point_count < int(rules.get("min_points", 0)):
        flags.append("too_few_points")
    if z_range_m < float(rules.get("min_z_range_m", 0.0)):
        flags.append("insufficient_z_range")
    if volume_2d5.get("status") != "ok" or float(volume_2d5.get("volume_m3", 0.0)) <= 0:
        flags.append("invalid_2d5_volume")

    min_surface_area = float(rules.get("min_surface_area_m2", 0.0))
    if min_surface_area > 0 and float(volume_2d5.get("surface_area_m2", 0.0)) < min_surface_area:
        flags.append("insufficient_surface_area")

    return {
        "name": profile_name,
        "description": str(profile_spec.get("description", "")),
        "passed": len(flags) == 0,
        "flags": flags,
        "rules": {
            "min_points": int(rules.get("min_points", 0)),
            "min_z_range_m": float(rules.get("min_z_range_m", 0.0)),
            "min_surface_area_m2": float(rules.get("min_surface_area_m2", 0.0)),
        },
    }


def _diameter_bin_rows(records: list[dict], profile_name: str, bins: list[float]) -> list[dict]:
    passed = [r for r in records if r["qc_profiles"][profile_name]["passed"]]
    total_n = len(passed)
    total_v_2d5 = sum(float(r["methods"]["2d5"]["volume_m3"]) for r in passed)
    total_v_2d = sum(float(r["methods"]["2d_proxy"]["volume_m3"]) for r in passed)

    rows: list[dict] = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        subset = [
            r for r in passed
            if float(r["fusion_prior"]["equivalent_diameter_m"]) >= lo
            and float(r["fusion_prior"]["equivalent_diameter_m"]) < hi
        ]
        vol_2d5_sum = sum(float(r["methods"]["2d5"]["volume_m3"]) for r in subset)
        vol_2d_sum = sum(float(r["methods"]["2d_proxy"]["volume_m3"]) for r in subset)
        count = len(subset)
        rows.append(
            {
                "bin_label": f"{lo:.2f}-{hi:.2f}",
                "diameter_lo_m": round(float(lo), 4),
                "diameter_hi_m": round(float(hi), 4),
                "stone_count": int(count),
                "count_ratio": round(_safe_ratio(count, total_n), 4),
                "volume_2d5_sum_m3": round(float(vol_2d5_sum), 4),
                "volume_2d5_ratio": round(_safe_ratio(vol_2d5_sum, total_v_2d5), 4),
                "volume_2d5_mean_m3": round(float(vol_2d5_sum / count), 4) if count > 0 else 0.0,
                "volume_2d_proxy_sum_m3": round(float(vol_2d_sum), 4),
                "volume_2d_proxy_ratio": round(_safe_ratio(vol_2d_sum, total_v_2d), 4),
                "volume_2d_proxy_mean_m3": round(float(vol_2d_sum / count), 4) if count > 0 else 0.0,
                "proxy_to_2d5_mean_ratio": round(_safe_ratio(vol_2d_sum, vol_2d5_sum), 4) if vol_2d5_sum > 0 else 0.0,
            }
        )
    return rows


def _comparison_summary(records: list[dict]) -> dict:
    comparable = [
        r for r in records
        if r["methods"]["2d5"]["status"] == "ok" and r["methods"]["2d_proxy"]["status"] == "ok"
    ]
    ratios = [r["ratios"]["proxy_2d_to_2d5"] for r in comparable]
    diffs = [r["differences"]["proxy_minus_2d5_m3"] for r in comparable]
    return {
        "comparable_stones": int(len(comparable)),
        "proxy_2d_to_2d5": _stats(ratios),
        "proxy_minus_2d5_m3": _stats(diffs),
        "pearson_r": _pearson_r(
            [r["methods"]["2d5"]["volume_m3"] for r in comparable],
            [r["methods"]["2d_proxy"]["volume_m3"] for r in comparable],
        ),
    }


def _case_brief(record: dict) -> dict:
    return {
        "stone_id": record["stone_id"],
        "volume_2d5_m3": record["methods"]["2d5"]["volume_m3"],
        "volume_2d_proxy_m3": record["methods"]["2d_proxy"]["volume_m3"],
        "proxy_2d_to_2d5": record["ratios"]["proxy_2d_to_2d5"],
        "proxy_minus_2d5_m3": record["differences"]["proxy_minus_2d5_m3"],
        "point_count": record["point_cloud"]["point_count"],
        "z_range_m": record["point_cloud"]["z_range_m"],
        "equivalent_diameter_m": record["fusion_prior"]["equivalent_diameter_m"],
        "qc_passed": record["qc"]["passed"],
        "qc_flags": record["qc"]["flags"],
    }


def _top_cases(records: list[dict], key: str, limit: int, reverse: bool = True) -> list[dict]:
    def extract(record: dict) -> float:
        current = record
        for part in key.split("."):
            current = current[part]
        return float(current)

    ranked = sorted(records, key=extract, reverse=reverse)
    return [_case_brief(record) for record in ranked[:limit]]


def _evaluate_stone(
    stone: dict,
    stone_points: np.ndarray,
    crop_info: dict,
    ground_dem: GroundDEM,
    config: dict,
) -> dict:
    selected_qc_profile, qc_profiles = _profile_specs(config)
    grid_res = float(config.get("grid", {}).get("resolution_m", 0.05))
    proxy_cfg = config.get("proxy_2d", {})

    projected_shape = estimate_projected_footprint(stone_points)
    volume_2d5 = estimate_2d5_with_ground(
        stone_points,
        ground_dem,
        grid_resolution=grid_res,
    )
    diameter_source = str(proxy_cfg.get("diameter_source", "fusion_prior_equivalent_diameter_m"))
    if diameter_source == "projected_shape_equivalent_diameter_m":
        proxy_diameter_m = float(projected_shape.get("equivalent_diameter_m", 0.0))
        proxy_area_m2 = float(projected_shape.get("convex_hull_area_m2", 0.0))
    else:
        proxy_diameter_m = float(stone.get("equivalent_diameter_m", 0.0))
        proxy_area_m2 = float(stone.get("area_m2", 0.0))
    volume_2d_proxy = estimate_2d_proxy_from_diameter(
        equivalent_diameter_m=proxy_diameter_m,
        area_m2=proxy_area_m2,
    )

    point_count = int(len(stone_points))
    z_range = float(np.ptp(stone_points[:, 2])) if point_count > 0 else 0.0
    main_volume = float(volume_2d5.get("volume_m3", 0.0))
    proxy_volume = float(volume_2d_proxy.get("volume_m3", 0.0))
    proxy_ratio = _safe_ratio(proxy_volume, main_volume)

    profile_results = {
        profile_name: _evaluate_qc_profile(
            profile_name=profile_name,
            profile_spec=profile_spec,
            point_count=point_count,
            z_range_m=z_range,
            volume_2d5=volume_2d5,
        )
        for profile_name, profile_spec in qc_profiles.items()
    }
    selected_qc = dict(profile_results[selected_qc_profile])

    return {
        "stone_id": stone["stone_id"],
        "geometry": {
            "bbox_world": stone.get("bbox_world", []),
            "centroid_world": stone.get("centroid_world", []),
        },
        "fusion_prior": {
            "source_detection_count": int(stone.get("source_detection_count", 0)),
            "source_patches_span": int(stone.get("source_patches_span", stone.get("source_detection_count", 0))),
            "equivalent_diameter_m": round(float(stone.get("equivalent_diameter_m", 0.0)), 4),
            "area_m2": round(float(stone.get("area_m2", 0.0)), 4),
            "bbox_area_m2": round(float(stone.get("bbox_area_m2", 0.0)), 4),
            "score_mean": round(float(stone.get("score_mean", 0.0)), 4),
            "validation_3d_passed": bool(stone.get("validation_3d", {}).get("passed", False)),
        },
        "crop": crop_info,
        "point_cloud": {
            "point_count": point_count,
            "z_range_m": round(z_range, 4),
        },
        "projected_shape": projected_shape,
        "methods": {
            "selected_method": "2d5_with_ground",
            "2d5": volume_2d5,
            "2d_proxy": volume_2d_proxy,
            "convex_hull": {
                "method": "convex_hull",
                "status": "skipped",
                "reason": "not_used_in_volume_main_pipeline",
                "volume_m3": 0.0,
                "note": "Convex hull is retained only for separate case-level diagnostics.",
            },
        },
        "ratios": {
            "proxy_2d_to_2d5": round(float(proxy_ratio), 4),
        },
        "differences": {
            "proxy_minus_2d5_m3": round(float(proxy_volume - main_volume), 4),
        },
        "qc_profile_selected": selected_qc_profile,
        "qc_profiles": profile_results,
        "qc": selected_qc,
    }


def _summarize_run(
    source: str,
    method: str,
    config: dict,
    fusion_stats: dict,
    ground_dem: GroundDEM,
    stones_total: int,
    stones_used: int,
    processed_records: list[dict],
    crop_empty: int,
) -> dict:
    top_k = int(config.get("output", {}).get("top_k_cases", 10))
    selected_qc_profile, qc_profiles = _profile_specs(config)
    output_cfg = config.get("output", {})
    diameter_bins = [float(v) for v in output_cfg.get("diameter_bins_m", [0.5, 0.75, 1.0, 1.5, 10.0])]
    if len(diameter_bins) < 2:
        diameter_bins = [0.5, 0.75, 1.0, 1.5, 10.0]

    passed = [r for r in processed_records if r["qc"]["passed"]]
    failed = [r for r in processed_records if not r["qc"]["passed"]]
    qc_flags = [flag for record in processed_records for flag in record["qc"]["flags"]]
    all_proxy_valid = [r for r in processed_records if r["methods"]["2d_proxy"]["status"] == "ok"]
    comparison_selected = _comparison_summary(passed)

    qc_profile_summary: dict[str, dict] = {}
    for profile_name, profile_spec in qc_profiles.items():
        profile_passed = [r for r in processed_records if r["qc_profiles"][profile_name]["passed"]]
        profile_failed = [r for r in processed_records if not r["qc_profiles"][profile_name]["passed"]]
        profile_flags = [
            flag
            for record in processed_records
            for flag in record["qc_profiles"][profile_name]["flags"]
        ]
        qc_profile_summary[profile_name] = {
            "description": profile_spec.get("description", ""),
            "rules": profile_spec.get("rules", {}),
            "passed_count": int(len(profile_passed)),
            "failed_count": int(len(profile_failed)),
            "pass_ratio": round(_safe_ratio(len(profile_passed), len(processed_records)), 4),
            "flag_counts": _count_values(profile_flags),
            "volume_2d5": _stats([r["methods"]["2d5"]["volume_m3"] for r in profile_passed]),
            "volume_2d_proxy": _stats([r["methods"]["2d_proxy"]["volume_m3"] for r in profile_passed]),
            "comparison_2d_proxy_vs_2d5": _comparison_summary(profile_passed),
            "diameter_bins": _diameter_bin_rows(processed_records, profile_name, diameter_bins),
        }

    summary = {
        "analysis_version": "volume_v3",
        "source": source,
        "method": method,
        "selected_volume_method": "2d5_with_ground",
        "selected_qc_profile": selected_qc_profile,
        "config": {
            "ground_dem": config.get("ground_dem", {}),
            "grid": config.get("grid", {}),
            "crop": config.get("crop", {}),
            "proxy_2d": config.get("proxy_2d", {}),
            "filter_legacy": config.get("filter", {}),
            "qc_profiles": config.get("qc_profiles", {}),
            "output": config.get("output", {}),
        },
        "scene": {
            "total_fused_stones": int(stones_total),
            "stones_with_masks": int(stones_used),
            "processed_stones": int(len(processed_records)),
            "qc_passed": int(len(passed)),
            "qc_failed": int(len(failed)),
            "qc_pass_ratio": round(_safe_ratio(len(passed), len(processed_records)), 4),
            "empty_crops": int(crop_empty),
            "qc_profile": selected_qc_profile,
        },
        "fusion": {
            "selected_stones": int(len(processed_records)),
            "candidate_stones": int(fusion_stats.get("candidate_stones", len(processed_records))),
            "accepted_stones": int(fusion_stats.get("output_stones", len(processed_records))),
            "rejected_stones": int(fusion_stats.get("rejected_stones", 0)),
            "validation_enabled": bool(fusion_stats.get("validation_3d", {}).get("enabled", False)),
        },
        "ground_dem": ground_dem.to_dict(),
        "qc": {
            "flag_counts": _count_values(qc_flags),
            "passed_count": int(len(passed)),
            "failed_count": int(len(failed)),
        },
        "qc_profiles": qc_profile_summary,
        "data_quality": {
            "point_count": _stats([r["point_cloud"]["point_count"] for r in processed_records]),
            "z_range_m": _stats([r["point_cloud"]["z_range_m"] for r in processed_records]),
            "equivalent_diameter_m": _stats([r["fusion_prior"]["equivalent_diameter_m"] for r in processed_records]),
            "projected_hull_area_m2": _stats([r["projected_shape"]["convex_hull_area_m2"] for r in processed_records]),
            "crop_retention_ratio": _stats([r["crop"].get("retention_ratio", 0.0) for r in processed_records]),
        },
        "volume_2d5": {
            "all_processed": _stats([r["methods"]["2d5"]["volume_m3"] for r in processed_records]),
            "qc_passed_only": _stats([r["methods"]["2d5"]["volume_m3"] for r in passed]),
            "surface_area_m2": _stats([r["methods"]["2d5"]["surface_area_m2"] for r in processed_records]),
        },
        "volume_2d_proxy": {
            "valid_count": int(len(all_proxy_valid)),
            "invalid_count": int(len(processed_records) - len(all_proxy_valid)),
            "all_processed": _stats([r["methods"]["2d_proxy"]["volume_m3"] for r in all_proxy_valid]),
            "qc_passed_only": _stats([r["methods"]["2d_proxy"]["volume_m3"] for r in passed if r["methods"]["2d_proxy"]["status"] == "ok"]),
        },
        "comparison_2d_proxy_vs_2d5": comparison_selected,
        "diameter_bins": {
            "profile": selected_qc_profile,
            "bin_edges_m": diameter_bins,
            "rows": _diameter_bin_rows(processed_records, selected_qc_profile, diameter_bins),
        },
        "representative_cases": {
            "largest_2d5": _top_cases(processed_records, "methods.2d5.volume_m3", top_k),
            "largest_proxy_gap": _top_cases(processed_records, "differences.proxy_minus_2d5_m3", top_k),
            "qc_failed_examples": [_case_brief(record) for record in failed[:top_k]],
        },
    }
    return summary


def _run_single_case(
    source: str,
    method: str,
    config: dict,
    debug_dem: bool = False,
    max_stones: int | None = None,
    progress_every: int = 200,
) -> dict:
    print(f"\n{'=' * 72}")
    print(f"Volume run: {source} / {method}")
    print(f"{'=' * 72}")

    gt = CURRENT_SCENE.load_gt()
    print("Loading scene point clouds...")
    pc = _load_point_cloud()

    print("Loading fused stone groups and detections...")
    fusion_stats, stones, detections = _load_fusion(source, method)
    if max_stones is not None and max_stones > 0:
        stones = stones[:max_stones]
    print(f"  fused stones: {len(stones)}")
    print(f"  detections:    {len(detections)}")

    print("Building ground DEM...")
    dem_cfg = config.get("ground_dem", {})
    ground_dem = GroundDEM(
        pc,
        resolution=float(dem_cfg.get("resolution_m", 0.5)),
        percentile=int(dem_cfg.get("percentile", 5)),
        subsample_step=int(dem_cfg.get("subsample_step", 100)),
        min_points_per_cell=int(dem_cfg.get("min_points_per_cell", 3)),
    )

    if debug_dem:
        debug_info = {
            "status": "debug_dem_only",
            "ground_dem": ground_dem.to_dict(),
        }
        print(json.dumps(debug_info, indent=2, ensure_ascii=False))
        return debug_info

    results: list[dict] = []
    empty_crops = 0
    used_stones = 0
    crop_cfg = config.get("crop", {})
    crop_index = _load_point_cloud_index(
        pc,
        cell_size=float(crop_cfg.get("index_cell_size_m", 1.0)),
    )

    print("Estimating stone volumes...")
    for stone_idx, stone in enumerate(stones, start=1):
        if int(stone.get("source_detection_count", 0)) == 0:
            continue

        used_stones += 1
        stone_points, crop_info = crop_stone_point_cloud(
            pc,
            stone,
            detections,
            gt,
            CURRENT_SCENE.xy_transform,
            bbox_pad_m=float(crop_cfg.get("bbox_pad_m", 0.5)),
            pc_index=crop_index,
        )
        crop_info = dict(crop_info)
        bbox_candidates = int(crop_info.get("bbox_candidate_count", 0))
        kept_points = int(crop_info.get("kept_point_count", len(stone_points)))
        crop_info["retention_ratio"] = round(_safe_ratio(kept_points, bbox_candidates), 6)

        if len(stone_points) == 0:
            empty_crops += 1
            continue

        results.append(
            _evaluate_stone(
                stone=stone,
                stone_points=stone_points,
                crop_info=crop_info,
                ground_dem=ground_dem,
                config=config,
            )
        )
        if progress_every > 0 and (stone_idx % progress_every == 0 or stone_idx == len(stones)):
            print(
                f"  progress: {stone_idx}/{len(stones)} stones, "
                f"kept {len(results)}, empty crops {empty_crops}"
            )

    summary = _summarize_run(
        source=source,
        method=method,
        config=config,
        fusion_stats=fusion_stats,
        ground_dem=ground_dem,
        stones_total=len(stones),
        stones_used=used_stones,
        processed_records=results,
        crop_empty=empty_crops,
    )
    summary["scene"]["subset_mode"] = bool(max_stones is not None and max_stones > 0)
    summary["scene"]["max_stones_requested"] = int(max_stones) if max_stones is not None else None

    output_cfg = config.get("output", {})
    if max_stones is not None and max_stones > 0:
        run_dir = SELF_DIR / "outputs" / source / method / "_debug" / f"max_stones_{max_stones}"
    else:
        run_dir = SELF_DIR / "outputs" / source / method
    run_dir.mkdir(parents=True, exist_ok=True)

    volume_stats_path = run_dir / "volume_stats.json"
    stone_volumes_path = run_dir / "stone_volumes.json"
    qc_summary_path = run_dir / "volume_qc_summary.json"
    comparison_path = run_dir / "volume_methods_comparison.json"
    top_cases_path = run_dir / "volume_top_cases.json"

    summary["outputs"] = {
        "volume_stats_json": volume_stats_path,
        "stone_volumes_json": stone_volumes_path,
        "volume_qc_summary_json": qc_summary_path,
        "volume_methods_comparison_json": comparison_path,
        "volume_top_cases_json": top_cases_path,
    }

    main_output = dict(summary)
    if bool(output_cfg.get("embed_stones_in_main_json", False)):
        main_output["stones"] = results

    _write_json(volume_stats_path, main_output)
    _write_json(stone_volumes_path, {"stones": results})
    _write_json(qc_summary_path, summary["qc"])
    _write_json(
        comparison_path,
        {
            "selected_volume_method": summary["selected_volume_method"],
            "selected_qc_profile": summary["selected_qc_profile"],
            "volume_2d5": summary["volume_2d5"],
            "volume_2d_proxy": summary["volume_2d_proxy"],
            "comparison_2d_proxy_vs_2d5": summary["comparison_2d_proxy_vs_2d5"],
            "diameter_bins": summary["diameter_bins"],
        },
    )
    _write_json(top_cases_path, summary["representative_cases"])

    _write_json(
        run_dir / "run_manifest.json",
        {
            "source": source,
            "method": method,
            "inputs": {
                "scene": CURRENT_SCENE.to_dict(),
                "tfw": CURRENT_SCENE.tfw_path,
                "point_clouds": list(CURRENT_SCENE.pointcloud_paths),
                "fusion": FUSION_ROOT / source / method / "fusion_stats.json",
                "detections": DETECTION_ROOT / source / "detections.json",
            },
            "outputs": summary["outputs"],
            "config": config,
        },
    )

    if bool(output_cfg.get("save_per_stone", False)):
        stones_dir = run_dir / "stones"
        stones_dir.mkdir(parents=True, exist_ok=True)
        for record in results:
            _write_json(stones_dir / f"{record['stone_id']}.json", record)

    print("\nRun summary")
    if summary["scene"]["subset_mode"]:
        print(f"  subset mode: first {summary['scene']['max_stones_requested']} stones")
    print(f"  qc profile: {summary['selected_qc_profile']}")
    print(f"  qc passed: {summary['scene']['qc_passed']} / {summary['scene']['processed_stones']}")
    print(f"  2.5D mean: {summary['volume_2d5']['qc_passed_only']['mean']:.4f} m3")
    print(f"  2D proxy mean: {summary['volume_2d_proxy']['qc_passed_only']['mean']:.4f} m3")
    print(f"  2D vs 2.5D comparable: {summary['comparison_2d_proxy_vs_2d5']['comparable_stones']}")
    print(f"  output: {volume_stats_path}")
    return summary


def _resolve_runs(source: str, method: str, run_all: bool) -> list[tuple[str, str]]:
    if run_all or source == ALL_TOKEN or method == ALL_TOKEN:
        return [(s, m) for s in SOURCES for m in METHODS]
    return [(source, method)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Volume experiment runner")
    parser.add_argument("--source", choices=[ALL_TOKEN, *SOURCES], default="sahi")
    parser.add_argument("--method", choices=[ALL_TOKEN, *METHODS], default="correlation_clustering")
    parser.add_argument("--all", action="store_true", help="Run every source/method pair")
    parser.add_argument("--debug-dem", action="store_true", help="Build the ground DEM and exit")
    parser.add_argument("--max-stones", type=int, default=None, help="Process only the first N fused stones")
    parser.add_argument("--progress-every", type=int, default=200, help="Print progress every N stones")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = json.loads((SELF_DIR / "config.json").read_text(encoding="utf-8"))
    runs = _resolve_runs(args.source, args.method, args.all)

    manifests = []
    for source, method in runs:
        manifests.append(
            _run_single_case(
                source,
                method,
                config,
                debug_dem=args.debug_dem,
                max_stones=args.max_stones,
                progress_every=args.progress_every,
            )
        )
        if args.debug_dem:
            break

    manifest_path = SELF_DIR / "outputs" / "volume_manifest.json"
    _write_json(
        manifest_path,
        {
            "runs": manifests,
            "selected_source": args.source,
            "selected_method": args.method,
            "run_all": bool(args.all),
        },
    )


if __name__ == "__main__":
    main()

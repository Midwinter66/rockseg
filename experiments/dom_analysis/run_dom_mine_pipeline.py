from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "experiments" / "runs"
DEFAULT_ULTRALYTICS_DIR = PROJECT_ROOT / "Ultralytics"


def _parse_tfw(tfw_path: Path) -> tuple[float, float, float, float, float, float]:
    lines = [float(line.strip()) for line in tfw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 6:
        raise ValueError(f"Invalid TFW file: {tfw_path}")
    return (lines[4], lines[0], lines[2], lines[5], lines[1], lines[3])


def _pixel_to_world(gt: tuple[float, float, float, float, float, float], px: float, py: float) -> tuple[float, float]:
    return float(gt[0] + px * gt[1] + py * gt[2]), float(gt[3] + px * gt[4] + py * gt[5])


def _load_dom_summary(dom_path: Path, tfw_path: Path) -> dict:
    dom = Image.open(dom_path)
    width_px, height_px = dom.size
    gt = _parse_tfw(tfw_path)
    xmin, ymax = _pixel_to_world(gt, 0, 0)
    xmax, ymin = _pixel_to_world(gt, width_px, height_px)
    resolution_m = abs(gt[1])
    return {
        "path": str(dom_path),
        "width_px": width_px,
        "height_px": height_px,
        "resolution_m": resolution_m,
        "resolution_x_m": gt[1],
        "resolution_y_m": gt[5],
        "bounds_world": [xmin, ymin, xmax, ymax],
        "area_m2": round((xmax - xmin) * (ymax - ymin), 6),
    }


def _safe_json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _band_table(diameters: list[float]) -> list[dict]:
    bins = [
        (0.05, 0.10, "0.05-0.10 m"),
        (0.10, 0.20, "0.10-0.20 m"),
        (0.20, 0.50, "0.20-0.50 m"),
        (0.50, 0.75, "0.50-0.75 m"),
        (0.75, 1.00, "0.75-1.00 m"),
        (1.00, 1.50, "1.00-1.50 m"),
        (1.50, 2.00, "1.50-2.00 m"),
        (2.00, math.inf, ">= 2.00 m"),
    ]
    total = max(len(diameters), 1)
    out = []
    for lo, hi, label in bins:
        count = sum(1 for d in diameters if lo <= d < hi)
        out.append({
            "range": label,
            "lower_m": lo,
            "upper_m": None if math.isinf(hi) else hi,
            "count": count,
            "percent": count / total * 100.0,
        })
    return out


def _oversize_counts(diameters: list[float], thresholds=(1.2, 1.5, 2.0)) -> dict[str, dict[str, float]]:
    total = max(len(diameters), 1)
    out: dict[str, dict[str, float]] = {}
    for threshold in thresholds:
        count = sum(1 for d in diameters if d > threshold)
        out[f"{threshold:.1f}"] = {
            "count": count,
            "percent": count / total * 100.0,
        }
    return out


def _select_representative_tiles(tile_stats: dict, detections: list[dict], n: int = 6) -> list[str]:
    kept = [t for t in tile_stats.get("tiles", []) if not t.get("skipped", False)]
    if not kept:
        return []
    counts = Counter(det.get("source_patch_id", "") for det in detections if det.get("source_patch_id"))
    ranked = sorted(
        ((counts.get(t.get("tile_id", ""), 0), t.get("tile_id", "")) for t in kept),
        key=lambda x: (x[0], x[1]),
    )
    if not ranked:
        return []
    quantiles = np.linspace(0.1, 0.95, n)
    chosen: list[str] = []
    for q in quantiles:
        idx = int(round(q * (len(ranked) - 1)))
        tile_id = ranked[idx][1]
        if tile_id and tile_id not in chosen:
            chosen.append(tile_id)
    for _, tile_id in ranked:
        if len(chosen) >= n:
            break
        if tile_id not in chosen:
            chosen.append(tile_id)
    return chosen[:n]


def _import_modules():
    import experiments.slicing.run_slicing_experiment as slicing
    import experiments.detection.run_detection_experiment as detection
    import experiments.fusion.run_fusion_experiment as fusion
    import experiments.fusion.visualize_fusion as visualize_fusion

    return slicing, detection, fusion, visualize_fusion


def _configure_modules(
    dataset_path: Path,
    run_root: Path,
):
    slicing, detection, fusion, visualize_fusion = _import_modules()

    dom_path = dataset_path / "DOM.tif"
    tfw_path = dataset_path / "DOM.tfw"

    slicing.DOM_PATH = dom_path
    slicing.DOM_WORLD_PATH = tfw_path
    slicing.SELF_DIR = run_root / "slicing"

    detection.DOM_PATH = dom_path
    detection.DOM_WORLD_PATH = tfw_path
    detection.SLICING_OUTPUTS = run_root / "slicing" / "outputs"
    detection.SELF_DIR = run_root / "detection"

    fusion.DETECTION_OUTPUTS = run_root / "detection" / "outputs"
    fusion.SELF_DIR = run_root / "fusion"

    visualize_fusion.DOM_PATH = dom_path
    visualize_fusion.TFW_PATH = tfw_path
    visualize_fusion.DETECTION_OUTPUTS = run_root / "detection" / "outputs"
    visualize_fusion.SLICING_OUTPUTS = run_root / "slicing" / "outputs"
    visualize_fusion.FUSION_OUTPUTS = run_root / "fusion" / "outputs"

    return slicing, detection, fusion, visualize_fusion


def _build_summary(
    dataset_name: str,
    dataset_path: Path,
    run_root: Path,
    slicing_stats: dict,
    detection_stats: dict,
    fusion_stats: dict,
) -> dict:
    dom_summary = _load_dom_summary(dataset_path / "DOM.tif", dataset_path / "DOM.tfw")
    detections = _safe_json_load(run_root / "detection" / "outputs" / "quadtree_dom" / "detections.json")
    stones = fusion_stats.get("stones", [])
    diameters = [float(s.get("equivalent_diameter_m", 0)) for s in stones if float(s.get("equivalent_diameter_m", 0)) > 0]
    areas = [float(s.get("area_m2", 0)) for s in stones if float(s.get("area_m2", 0)) >= 0]
    scores = [float(s.get("score_max", s.get("score_mean", 0))) for s in stones]

    largest_raw = sorted(stones, key=lambda s: float(s.get("equivalent_diameter_m", 0)), reverse=True)[:20]
    top_counts = Counter(det.get("source_patch_id", "") for det in detections if det.get("source_patch_id"))
    tile_stats = _safe_json_load(run_root / "slicing" / "outputs" / "quadtree_dom" / "tile_stats.json")
    selected_tiles = _select_representative_tiles(tile_stats, detections, n=6)

    summary = {
        "generated_date": datetime.now().strftime("%Y-%m-%d"),
        "dataset": {
            "name": dataset_name,
            "dom": dom_summary,
        },
        "slicing": {
            "method": slicing_stats.get("method", "quadtree_dom"),
            "config": slicing_stats.get("config", {}),
            "total_tiles": int(slicing_stats.get("total_tiles", 0)),
            "kept_tiles": int(slicing_stats.get("kept_tiles", 0)),
            "skipped_tiles": int(slicing_stats.get("skipped_tiles", 0)),
            "kept_percent": float(slicing_stats.get("kept_tiles", 0)) / max(int(slicing_stats.get("total_tiles", 1)), 1) * 100.0,
            "coverage_ratio": float(slicing_stats.get("coverage_ratio", 0)),
            "tile_area_m2": slicing_stats.get("tile_size_distribution_m2", {}),
            "elapsed_seconds": float(slicing_stats.get("elapsed_seconds", 0)),
        },
        "detection": {
            "mode": "single_scale",
            "config": detection_stats.get("config", {}),
            "processed_tiles": int(detection_stats.get("processed_tiles", 0)),
            "detections": int(detection_stats.get("detection_count", 0)),
            "detections_per_tile": detection_stats.get("detections_per_tile", {}),
            "detections_per_dom_m2": float(detection_stats.get("detection_count", 0)) / max(dom_summary["area_m2"], 1),
            "elapsed_seconds": float(detection_stats.get("elapsed_seconds", 0)),
            "seconds_per_tile": float(detection_stats.get("elapsed_seconds", 0)) / max(int(detection_stats.get("processed_tiles", 1)), 1),
        },
        "fusion": {
            "method": fusion_stats.get("method", "correlation_clustering"),
            "input_detections": int(fusion_stats.get("input_detections", 0)),
            "output_stones": int(fusion_stats.get("output_stones", 0)),
            "removed_duplicate_detections": int(fusion_stats.get("input_detections", 0)) - int(fusion_stats.get("output_stones", 0)),
            "merge_ratio": float(fusion_stats.get("merge_ratio", 0)),
            "group_size_counts": fusion_stats.get("group_size_counts", {}),
            "diagnostics": fusion_stats.get("fusion_diagnostics", {}),
            "stones_per_dom_m2": float(fusion_stats.get("output_stones", 0)) / max(dom_summary["area_m2"], 1),
            "elapsed_seconds": float(fusion_stats.get("elapsed_seconds", 0)),
        },
        "stone_statistics": {
            "diameter_m": {
                "mean": float(np.mean(diameters)) if diameters else 0.0,
                "std": float(np.std(diameters)) if diameters else 0.0,
                "quantiles": {
                    "0": _quantile(diameters, 0.0),
                    "0.01": _quantile(diameters, 0.01),
                    "0.05": _quantile(diameters, 0.05),
                    "0.1": _quantile(diameters, 0.1),
                    "0.25": _quantile(diameters, 0.25),
                    "0.5": _quantile(diameters, 0.5),
                    "0.75": _quantile(diameters, 0.75),
                    "0.9": _quantile(diameters, 0.9),
                    "0.95": _quantile(diameters, 0.95),
                    "0.99": _quantile(diameters, 0.99),
                    "1": max(diameters) if diameters else 0.0,
                },
                "bands": _band_table(diameters),
            },
            "area_m2": {
                "sum": float(np.sum(areas)) if areas else 0.0,
                "mean": float(np.mean(areas)) if areas else 0.0,
                "std": float(np.std(areas)) if areas else 0.0,
                "quantiles": {
                    "0": _quantile(areas, 0.0),
                    "0.5": _quantile(areas, 0.5),
                    "0.9": _quantile(areas, 0.9),
                    "1": max(areas) if areas else 0.0,
                },
            },
            "score": {
                "mean": float(np.mean(scores)) if scores else 0.0,
                "std": float(np.std(scores)) if scores else 0.0,
            },
        },
        "oversize": {
            "count": _oversize_counts(diameters, thresholds=(1.2, 1.5, 2.0)),
            "count_bands": _band_table(diameters),
        },
        "quality_audit": {
            "largest_raw": [
                {
                    "stone_id": s.get("stone_id"),
                    "diameter_m": float(s.get("equivalent_diameter_m", 0)),
                    "area_m2": float(s.get("area_m2", 0)),
                    "score_mean": float(s.get("score_mean", 0)),
                    "group_size": int(s.get("source_detection_count", 1)),
                }
                for s in largest_raw
            ],
            "largest_raw_count": len(largest_raw),
            "selected_tiles": selected_tiles,
            "selected_tiles_detection_counts": {tile_id: int(top_counts.get(tile_id, 0)) for tile_id in selected_tiles},
            "limitation": "本次报告仅使用 DOM 与 2D 分割结果，不引入点云三维复核。",
        },
        "runtime": {
            "slicing_seconds": float(slicing_stats.get("elapsed_seconds", 0)),
            "detection_seconds": float(detection_stats.get("elapsed_seconds", 0)),
            "fusion_seconds": float(fusion_stats.get("elapsed_seconds", 0)),
            "total_seconds": float(slicing_stats.get("elapsed_seconds", 0))
            + float(detection_stats.get("elapsed_seconds", 0))
            + float(fusion_stats.get("elapsed_seconds", 0)),
        },
        "paths": {
            "dataset_path": str(dataset_path),
            "run_root": str(run_root),
            "slicing_output": str(run_root / "slicing" / "outputs"),
            "detection_output": str(run_root / "detection" / "outputs"),
            "fusion_output": str(run_root / "fusion" / "outputs"),
            "tile_mask_manifest": str(run_root / "fusion" / "outputs" / "quadtree_dom" / "correlation_clustering" / "tile_mask_groups" / "manifest.json"),
            "report_tile_masks": str(run_root / "report_assets" / "tile_masks" / "manifest.json"),
        },
    }
    return summary


def run_dataset(dataset_name: str, dataset_path: Path, run_root: Path) -> Path:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(DEFAULT_ULTRALYTICS_DIR))
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "slicing").mkdir(parents=True, exist_ok=True)
    (run_root / "detection").mkdir(parents=True, exist_ok=True)
    (run_root / "fusion").mkdir(parents=True, exist_ok=True)

    slicing, detection, fusion, visualize_fusion = _configure_modules(dataset_path, run_root)

    print(f"\n{'=' * 70}")
    print(f"  Dataset: {dataset_name}")
    print(f"  DOM: {dataset_path / 'DOM.tif'}")
    print(f"  Run root: {run_root}")
    print(f"{'=' * 70}")

    slicing_cfg = slicing._load_config("quadtree_dom")
    slicing_out = slicing._resolve_output_dir("quadtree_dom")
    slicing_stats = slicing.RUNNERS["quadtree_dom"](slicing_cfg, slicing_out)

    detection_stats = detection._run_detection("quadtree_dom", limit=None, multi_scale=False, scales=None)

    fusion_stats = fusion._run_fusion("quadtree_dom", "correlation_clustering")

    summary = _build_summary(dataset_name, dataset_path, run_root, slicing_stats, detection_stats, fusion_stats)
    summary_path = run_root / "summary" / "report_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    tile_stats = _safe_json_load(run_root / "slicing" / "outputs" / "quadtree_dom" / "tile_stats.json")
    detections = _safe_json_load(run_root / "detection" / "outputs" / "quadtree_dom" / "detections.json")
    selected_tiles = summary["quality_audit"]["selected_tiles"]
    if selected_tiles:
        bins = [0.05, 0.10, 0.20, 0.50]
        tile_mask_root = run_root / "fusion" / "outputs" / "quadtree_dom" / "correlation_clustering" / "tile_mask_groups"
        report_tile_root = run_root / "report_assets" / "tile_masks"
        report_tile_root.mkdir(parents=True, exist_ok=True)
        copied_tiles: list[dict] = []
        if tile_mask_root.exists():
            # keep the selection fresh for each run
            import shutil

            shutil.rmtree(tile_mask_root)
        for tile_id in selected_tiles:
            manifest_path = visualize_fusion.write_tile_mask_groups(
                source="quadtree_dom",
                method="correlation_clustering",
                bins=bins,
                alpha=0.42,
                max_tiles=None,
                tile_id=tile_id,
                only_merged=False,
                draw_labels=True,
            )
            manifest = _safe_json_load(manifest_path)
            tile_entry = next((t for t in manifest.get("tiles", []) if t.get("tile_id") == tile_id), None)
            if tile_entry and tile_entry.get("all_groups_image"):
                import shutil

                copied_path = report_tile_root / f"{tile_id}.png"
                shutil.copy2(tile_entry["all_groups_image"], copied_path)
                copied_tiles.append({
                    "tile_id": tile_id,
                    "mask_count": int(tile_entry.get("mask_count", 0)),
                    "image_path": str(copied_path),
                })

        (report_tile_root / "manifest.json").write_text(
            json.dumps({"dataset": dataset_name, "tiles": copied_tiles}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Store a short manifest to make report assembly easier.
    manifest = {
        "dataset": dataset_name,
        "summary": str(summary_path),
        "tile_mask_manifest": str(run_root / "fusion" / "outputs" / "quadtree_dom" / "correlation_clustering" / "tile_mask_groups" / "manifest.json"),
        "report_tile_masks": str(run_root / "report_assets" / "tile_masks" / "manifest.json"),
        "tile_stats": str(run_root / "slicing" / "outputs" / "quadtree_dom" / "tile_stats.json"),
        "detection_stats": str(run_root / "detection" / "outputs" / "quadtree_dom" / "detection_stats.json"),
        "fusion_stats": str(run_root / "fusion" / "outputs" / "quadtree_dom" / "correlation_clustering" / "fusion_stats.json"),
    }
    (run_root / "summary" / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  Summary: {summary_path}")
    print(f"  Manifest: {run_root / 'summary' / 'run_manifest.json'}")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DOM-only mine analysis pipeline")
    parser.add_argument("--name", required=True, help="Dataset name, e.g. dom1 or dom2")
    parser.add_argument("--dom-path", required=True, help="Path to DOM.tif")
    parser.add_argument("--dom-world-path", required=True, help="Path to DOM.tfw")
    parser.add_argument("--run-root", default=None, help="Output root for this run")
    args = parser.parse_args()

    dataset_path = Path(args.dom_world_path).parent
    if args.run_root:
        run_root = Path(args.run_root)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = DEFAULT_RUNS_ROOT / f"{args.name}_{stamp}"

    # keep the TFW path explicit by validation
    if not Path(args.dom_path).exists():
        raise FileNotFoundError(args.dom_path)
    if not Path(args.dom_world_path).exists():
        raise FileNotFoundError(args.dom_world_path)

    if Path(args.dom_path).parent != dataset_path:
        raise ValueError("DOM.tif and DOM.tfw must live in the same folder for this pipeline")

    run_dataset(args.name, dataset_path, run_root)


if __name__ == "__main__":
    main()

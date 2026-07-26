from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import rasterio


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common.scene_reference import CURRENT_SCENE

OUT_DIR = ROOT / "docs" / "results"
TABLE_DIR = OUT_DIR / "tables"

SLICING_SAHI = ROOT / "experiments" / "slicing" / "outputs" / "sahi" / "slicing_summary.json"
SLICING_QUADTREE = ROOT / "experiments" / "slicing" / "outputs" / "quadtree_dom" / "slicing_summary.json"
DETECTION = ROOT / "experiments" / "detection" / "outputs" / "quadtree_dom" / "detection_stats.json"
FUSION = ROOT / "experiments" / "fusion" / "outputs" / "quadtree_dom" / "correlation_clustering" / "fusion_summary.json"
VOLUME = ROOT / "experiments" / "volume" / "outputs" / "quadtree_dom" / "correlation_clustering" / "volume_stats.json"
STONE_VOLUMES = ROOT / "experiments" / "volume" / "outputs" / "quadtree_dom" / "correlation_clustering" / "stone_volumes.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def write_md_table(path: Path, title: str, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table(name: str, title: str, headers: list[str], rows: list[list[object]]) -> None:
    write_csv(TABLE_DIR / f"{name}.csv", headers, rows)
    write_md_table(TABLE_DIR / f"{name}.md", title, headers, rows)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def volume_bin_rows(stones: list[dict]) -> list[list[object]]:
    passed = [s for s in stones if s["qc"]["passed"]]
    total_n = len(passed)
    total_v_2d5 = sum(s["methods"]["2d5"]["volume_m3"] for s in passed)
    total_v_2d = sum(s["methods"]["2d_proxy"]["volume_m3"] for s in passed)
    bins = [
        (0.5, 0.75),
        (0.75, 1.0),
        (1.0, 1.5),
        (1.5, 10.0),
    ]
    rows: list[list[object]] = []
    for lo, hi in bins:
        subset = [s for s in passed if lo <= s["fusion_prior"]["equivalent_diameter_m"] < hi]
        volume_2d5_sum = sum(s["methods"]["2d5"]["volume_m3"] for s in subset)
        volume_2d_sum = sum(s["methods"]["2d_proxy"]["volume_m3"] for s in subset)
        volume_2d5_mean = volume_2d5_sum / len(subset) if subset else 0.0
        volume_2d_mean = volume_2d_sum / len(subset) if subset else 0.0
        rows.append([
            f"{lo:.2f}-{hi:.2f}",
            len(subset),
            round(safe_ratio(len(subset), total_n), 4),
            round(volume_2d5_sum, 4),
            round(safe_ratio(volume_2d5_sum, total_v_2d5), 4),
            round(volume_2d5_mean, 4),
            round(volume_2d_sum, 4),
            round(safe_ratio(volume_2d_sum, total_v_2d), 4),
            round(volume_2d_mean, 4),
        ])
    return rows


def representative_rows(stones: list[dict]) -> list[list[object]]:
    passed = [s for s in stones if s["qc"]["passed"]]
    sorted_passed = sorted(passed, key=lambda s: s["methods"]["2d5"]["volume_m3"])
    chosen = [
        ("small", sorted_passed[0]),
        ("median", sorted_passed[len(sorted_passed) // 2]),
        ("large", sorted(passed, key=lambda s: s["methods"]["2d5"]["volume_m3"], reverse=True)[0]),
    ]
    rows: list[list[object]] = []
    for label, stone in chosen:
        rows.append([
            label,
            stone["stone_id"],
            stone["methods"]["2d5"]["volume_m3"],
            stone["methods"]["2d_proxy"]["volume_m3"],
            stone["ratios"]["proxy_2d_to_2d5"],
            stone["fusion_prior"]["equivalent_diameter_m"],
            stone["projected_shape"]["convex_hull_area_m2"],
            stone["point_cloud"]["z_range_m"],
            stone["point_cloud"]["point_count"],
        ])
    return rows


def scene_snapshot() -> dict:
    gt = CURRENT_SCENE.load_gt()
    with rasterio.open(CURRENT_SCENE.dom_path) as dataset:
        width_px = int(dataset.width)
        height_px = int(dataset.height)

    point_count = 0
    point_files: list[dict] = []
    try:
        import laspy

        for path in CURRENT_SCENE.pointcloud_paths:
            with laspy.open(path) as reader:
                count = int(reader.header.point_count)
            point_count += count
            point_files.append({"name": path.name, "point_count": count})
    except (ImportError, FileNotFoundError):
        point_files = [{"name": path.name, "point_count": None} for path in CURRENT_SCENE.pointcloud_paths]

    resolution_x_m = abs(float(gt[1]))
    resolution_y_m = abs(float(gt[5]))
    return {
        "name": CURRENT_SCENE.name,
        "data_origin": "OSGB-derived photogrammetric products (not LiDAR)",
        "dom": {
            "file": "data/dom2/DOM.tif",
            "width_px": int(width_px),
            "height_px": int(height_px),
            "resolution_x_m": resolution_x_m,
            "resolution_y_m": resolution_y_m,
            "area_m2": round(width_px * height_px * resolution_x_m * resolution_y_m, 4),
            "crs": "EPSG:4536",
        },
        "point_cloud": {
            "files": point_files,
            "total_point_count": point_count or None,
            "coordinate_mode": CURRENT_SCENE.xy_transform.mode,
            "x_shift": CURRENT_SCENE.xy_transform.x_shift,
            "y_shift": CURRENT_SCENE.xy_transform.y_shift,
        },
    }


def build_current_results(
    slicing_sahi: dict,
    slicing_quadtree: dict,
    detection: dict,
    fusion: dict,
    volume: dict,
) -> dict:
    return {
        "snapshot_version": "2026-07-main-scene-v1",
        "scope": {
            "source": "quadtree_dom",
            "fusion_method": "correlation_clustering",
            "minimum_equivalent_diameter_m": detection["config"]["inference"]["min_stone_diameter_m"],
            "volume_method": volume["selected_volume_method"],
            "qc_profile": volume["selected_qc_profile"],
        },
        "scene": scene_snapshot(),
        "slicing": {
            "sahi": {
                "total_tiles": slicing_sahi["total_tiles"],
                "kept_tiles": slicing_sahi["kept_tiles"],
                "coverage_ratio": slicing_sahi["coverage_ratio"],
            },
            "quadtree_dom": {
                "total_tiles": slicing_quadtree["total_tiles"],
                "kept_tiles": slicing_quadtree["kept_tiles"],
                "coverage_ratio": slicing_quadtree["coverage_ratio"],
            },
        },
        "detection": {
            "raw_mask_candidates": detection["mask_candidates_total"],
            "diameter_filtered_detections": detection["detection_count"],
            "processed_tiles": detection["processed_tiles"],
            "failed_tiles": detection["failed_tiles"],
            "diameter_m": detection["diameter_m"],
            "area_m2": detection["area_m2"],
            "confidence": detection["confidence"],
        },
        "fusion": {
            "input_detections": fusion["input_detections"],
            "candidate_stones": fusion["candidate_stones"],
            "accepted_stones": fusion["output_stones"],
            "rejected_stones": fusion["rejected_stones"],
            "candidate_acceptance_ratio": fusion["candidate_acceptance_ratio"],
            "merge_ratio": fusion["merge_ratio"],
            "accepted_diameter_m": fusion["accepted"]["equivalent_diameter_m"],
            "validation_3d": fusion["validation_3d"],
        },
        "volume": {
            "processed_stones": volume["scene"]["processed_stones"],
            "qc_passed": volume["scene"]["qc_passed"],
            "qc_failed": volume["scene"]["qc_failed"],
            "qc_pass_ratio": volume["scene"]["qc_pass_ratio"],
            "volume_2d5_m3": volume["volume_2d5"]["qc_passed_only"],
            "volume_2d_proxy_m3": volume["volume_2d_proxy"]["qc_passed_only"],
            "comparison": volume["comparison_2d_proxy_vs_2d5"],
            "diameter_bins": volume["diameter_bins"],
        },
        "validation_status": {
            "current_scene_manual_detection_ground_truth": "not_completed",
            "per_stone_volume_ground_truth": "not_available",
            "note": "Do not interpret the 3D validation acceptance ratio or volume QC pass ratio as detection accuracy.",
        },
    }


def write_current_results_markdown(path: Path, snapshot: dict) -> None:
    scene = snapshot["scene"]
    detection = snapshot["detection"]
    fusion = snapshot["fusion"]
    volume = snapshot["volume"]
    lines = [
        "# Current Main-Scene Results",
        "",
        "> Scope: `dom2 + pointcloud2`, minimum equivalent diameter `0.5 m`, quadtree tiling, correlation-clustering fusion, and GroundDEM-based 2.5D volume.",
        "",
        "## Dataset",
        "",
        f"- Data origin: {scene['data_origin']}.",
        f"- DOM: {scene['dom']['width_px']} x {scene['dom']['height_px']} pixels at {scene['dom']['resolution_x_m']:.2f} m/pixel, EPSG:4536.",
        f"- Scene area: {scene['dom']['area_m2']:.2f} m2.",
        f"- Point-cloud points: {scene['point_cloud']['total_point_count']:,}.",
        "- Coordinate mapping: absolute world coordinates with zero XY shift.",
        "",
        "## Pipeline Counts",
        "",
        "| Stage | Count |",
        "|---|---:|",
        f"| Quadtree tiles generated | {snapshot['slicing']['quadtree_dom']['total_tiles']} |",
        f"| Quadtree tiles retained | {snapshot['slicing']['quadtree_dom']['kept_tiles']} |",
        f"| Raw mask candidates | {detection['raw_mask_candidates']} |",
        f"| Detections after 0.5 m diameter filter | {detection['diameter_filtered_detections']} |",
        f"| Fusion candidates | {fusion['candidate_stones']} |",
        f"| 3D-accepted fused stones | {fusion['accepted_stones']} |",
        f"| 3D-rejected candidates | {fusion['rejected_stones']} |",
        f"| Volume QC passed | {volume['qc_passed']} |",
        "",
        "## Main Measurements",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accepted median equivalent diameter | {fusion['accepted_diameter_m']['median']:.4f} m |",
        f"| Accepted mean equivalent diameter | {fusion['accepted_diameter_m']['mean']:.4f} m |",
        f"| 2.5D total volume | {volume['volume_2d5_m3']['sum']:.4f} m3 |",
        f"| 2.5D mean volume | {volume['volume_2d5_m3']['mean']:.4f} m3 |",
        f"| 2.5D median volume | {volume['volume_2d5_m3']['median']:.4f} m3 |",
        f"| 2D proxy total volume | {volume['volume_2d_proxy_m3']['sum']:.4f} m3 |",
        f"| Pearson correlation, 2D proxy vs. 2.5D | {volume['comparison']['pearson_r']:.4f} |",
        f"| Median 2D proxy / 2.5D ratio | {volume['comparison']['proxy_2d_to_2d5']['median']:.4f} |",
        "",
        "## Interpretation Boundary",
        "",
        "The current full-scene pipeline is operational, but the current scene does not yet have a complete manual detection ground truth or per-stone volume ground truth. The 3D acceptance ratio and volume QC pass ratio describe pipeline filtering and numerical validity; they are not precision, recall, or absolute volume accuracy.",
        "",
        "Machine-readable values are available in `current_results.json`; manuscript tables are under `tables/`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    slicing_sahi = load_json(SLICING_SAHI)
    slicing_quadtree = load_json(SLICING_QUADTREE)
    detection = load_json(DETECTION)
    fusion = load_json(FUSION)
    volume = load_json(VOLUME)
    stone_volumes = load_json(STONE_VOLUMES)["stones"]

    write_table(
        "table_01_pipeline_overview",
        "Pipeline Overview",
        ["Stage", "Input count", "Output count", "Keep ratio", "Notes"],
        [
            ["Mask candidates", detection["mask_candidates_total"], detection["mask_candidates_total"], 1.0, "Raw segmentation candidates before diameter filtering"],
            ["Detection", detection["mask_candidates_total"], detection["detection_count"], round(safe_ratio(detection["detection_count"], detection["mask_candidates_total"]), 4), "After min_stone_diameter_m = 0.5 m filtering"],
            ["Fusion candidates", detection["detection_count"], fusion["candidate_stones"], round(safe_ratio(fusion["candidate_stones"], detection["detection_count"]), 4), "Cross-tile duplicate grouping candidates"],
            ["Accepted fused stones", fusion["candidate_stones"], fusion["output_stones"], round(safe_ratio(fusion["output_stones"], fusion["candidate_stones"]), 4), "After 3D validation"],
            ["Volume QC passed", volume["scene"]["processed_stones"], volume["scene"]["qc_passed"], round(volume["scene"]["qc_pass_ratio"], 4), "2.5D volume retained for final statistics"],
        ],
    )

    write_table(
        "table_02_slicing_comparison",
        "Slicing Comparison",
        ["Method", "Total tiles", "Kept tiles", "Skipped tiles", "Coverage ratio", "Key parameters"],
        [
            [
                "SAHI",
                slicing_sahi["total_tiles"],
                slicing_sahi["kept_tiles"],
                slicing_sahi["skipped_tiles"],
                slicing_sahi["coverage_ratio"],
                f"patch_size={slicing_sahi['config']['patch_size']}, overlap={slicing_sahi['config']['overlap']}, min_content_ratio={slicing_sahi['config']['min_content_ratio']}",
            ],
            [
                "Quadtree DOM",
                slicing_quadtree["total_tiles"],
                slicing_quadtree["kept_tiles"],
                slicing_quadtree["skipped_tiles"],
                slicing_quadtree["coverage_ratio"],
                (
                    f"base_tile_size_m={slicing_quadtree['config']['base_tile_size_m']}, "
                    f"min_tile_size_m={slicing_quadtree['config']['min_tile_size_m']}, "
                    f"min_edge_density={slicing_quadtree['config']['min_edge_density']}, "
                    f"tile_overlap_m={slicing_quadtree['config']['tile_overlap_m']}, "
                    f"min_content_ratio={slicing_quadtree['config']['min_content_ratio']}"
                ),
            ],
        ],
    )

    write_table(
        "table_03_detection_summary",
        "Detection Summary",
        ["Metric", "Value"],
        [
            ["Processed tiles", detection["processed_tiles"]],
            ["Successful tiles", detection["successful_tiles"]],
            ["Tiles with detections", detection["tiles_with_detections"]],
            ["Detection count", detection["detection_count"]],
            ["Mask candidates total", detection["mask_candidates_total"]],
            ["Removed by min diameter", detection["filtered_candidates"]["below_min_diameter"]],
            ["Detection keep ratio from mask candidates", round(safe_ratio(detection["detection_count"], detection["mask_candidates_total"]), 4)],
            ["Mean detections per successful tile", detection["detections_per_tile"]["mean_per_successful_tile"]],
            ["Median equivalent diameter (m)", detection["diameter_m"]["median"]],
            ["Mean equivalent diameter (m)", detection["diameter_m"]["mean"]],
            ["Median area (m2)", detection["area_m2"]["median"]],
            ["Mean confidence", detection["confidence"]["mean"]],
            ["Elapsed seconds", detection["elapsed_seconds"]],
        ],
    )

    write_table(
        "table_04_fusion_summary",
        "Fusion Summary",
        ["Metric", "Value"],
        [
            ["Input detections", fusion["input_detections"]],
            ["Candidate stones", fusion["candidate_stones"]],
            ["Accepted stones", fusion["output_stones"]],
            ["Rejected stones", fusion["rejected_stones"]],
            ["Candidate acceptance ratio", fusion["candidate_acceptance_ratio"]],
            ["Merge ratio", fusion["merge_ratio"]],
            ["Accepted median diameter (m)", fusion["accepted"]["equivalent_diameter_m"]["median"]],
            ["Accepted mean diameter (m)", fusion["accepted"]["equivalent_diameter_m"]["mean"]],
            ["Accepted median area (m2)", fusion["accepted"]["area_m2"]["median"]],
            ["Accepted mean area (m2)", fusion["accepted"]["area_m2"]["mean"]],
            ["Rejected by insufficient_p90_height", fusion["validation_3d"]["rejection_reasons"]["insufficient_p90_height"]],
            ["Rejected by insufficient_elevated_ratio", fusion["validation_3d"]["rejection_reasons"]["insufficient_elevated_ratio"]],
            ["Rejected by insufficient_z_range", fusion["validation_3d"]["rejection_reasons"]["insufficient_z_range"]],
            ["Rejected by too_few_points", fusion["validation_3d"]["rejection_reasons"]["too_few_points"]],
        ],
    )

    write_table(
        "table_05_volume_summary",
        "Volume Summary",
        ["Metric", "Value"],
        [
            ["Processed stones", volume["scene"]["processed_stones"]],
            ["Selected QC profile", volume["selected_qc_profile"]],
            ["QC passed stones", volume["scene"]["qc_passed"]],
            ["QC failed stones", volume["scene"]["qc_failed"]],
            ["QC pass ratio", volume["scene"]["qc_pass_ratio"]],
            ["2.5D mean volume (m3)", volume["volume_2d5"]["qc_passed_only"]["mean"]],
            ["2.5D median volume (m3)", volume["volume_2d5"]["qc_passed_only"]["median"]],
            ["2.5D p25 volume (m3)", volume["volume_2d5"]["qc_passed_only"]["p25"]],
            ["2.5D p75 volume (m3)", volume["volume_2d5"]["qc_passed_only"]["p75"]],
            ["2.5D total volume (m3)", volume["volume_2d5"]["qc_passed_only"]["sum"]],
            ["2D proxy mean volume (m3)", volume["volume_2d_proxy"]["qc_passed_only"]["mean"]],
            ["2D proxy median volume (m3)", volume["volume_2d_proxy"]["qc_passed_only"]["median"]],
            ["2D proxy total volume (m3)", volume["volume_2d_proxy"]["qc_passed_only"]["sum"]],
            ["Ground DEM raw coverage ratio", volume["ground_dem"]["raw_coverage_ratio"]],
            ["Ground DEM filled coverage ratio", volume["ground_dem"]["coverage_ratio"]],
            ["2D proxy / 2.5D median ratio", volume["comparison_2d_proxy_vs_2d5"]["proxy_2d_to_2d5"]["median"]],
            ["2D proxy / 2.5D mean ratio", volume["comparison_2d_proxy_vs_2d5"]["proxy_2d_to_2d5"]["mean"]],
            ["2D proxy minus 2.5D mean (m3)", volume["comparison_2d_proxy_vs_2d5"]["proxy_minus_2d5_m3"]["mean"]],
            ["Method comparison Pearson r", volume["comparison_2d_proxy_vs_2d5"]["pearson_r"]],
        ],
    )

    write_table(
        "table_06_diameter_volume_bins",
        "Diameter-Volume Bins",
        [
            "Diameter bin (m)",
            "Stone count",
            "Count ratio",
            "2.5D volume sum (m3)",
            "2.5D volume ratio",
            "2.5D mean volume (m3)",
            "2D proxy volume sum (m3)",
            "2D proxy volume ratio",
            "2D proxy mean volume (m3)",
        ],
        volume_bin_rows(stone_volumes),
    )

    write_table(
        "table_07_representative_cases",
        "Representative Cases",
        ["Class", "Stone ID", "2.5D volume (m3)", "2D proxy volume (m3)", "2D/2.5D", "Equivalent diameter (m)", "Projected hull area (m2)", "Z range (m)", "Point count"],
        representative_rows(stone_volumes),
    )

    notes = {
        "template_revision_notes": [
            "The current template can be kept as the main skeleton for a Measurement manuscript.",
            "Section 1.2 should avoid overclaiming absolute error evaluation because no manual ground-truth volume is available.",
            "Section 3.3.2 should be updated to the current quadtree configuration: base_tile_size_m=10, min_tile_size_m=5, min_edge_density=0.10, tile_overlap_m=0.5, min_content_ratio=0.05.",
            "Section 3.4.1 should state that the reported experiment uses single-scale inference at imgsz=1024; multi-scale remains an optional sensitivity setting rather than the main reported result.",
            "Section 3.7 should explicitly state that evaluation is based on pipeline consistency, filtering behavior, fusion validation, and relative volume-method comparison, not manual stone-by-stone ground truth.",
            "Section 4.6 Sensitivity Analysis can remain in the outline, but if no stable parameter study is finalized it should be shortened or moved to supplementary material.",
        ]
    }
    (OUT_DIR / "manuscript_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    snapshot = build_current_results(
        slicing_sahi=slicing_sahi,
        slicing_quadtree=slicing_quadtree,
        detection=detection,
        fusion=fusion,
        volume=volume,
    )
    (OUT_DIR / "current_results.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_current_results_markdown(OUT_DIR / "current_results.md", snapshot)

    print(json.dumps({
        "output_dir": str(OUT_DIR),
        "generated_files": sorted(str(p.relative_to(OUT_DIR)) for p in OUT_DIR.rglob("*") if p.is_file()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value: float) -> float:
    return round(value * 100.0, 2)


def main() -> None:
    manifest = load_json(ROOT / "site_b_run_manifest.json")
    slicing = load_json(OUT / "slicing" / "quadtree_dom" / "slicing_summary.json")
    detection = load_json(OUT / "detection" / "quadtree_dom" / "detection_stats.json")
    p0 = load_json(OUT / "p0" / "fusion" / "quadtree_dom" / "correlation_clustering" / "fusion_stats.json")
    p1 = load_json(OUT / "p1" / "fusion" / "quadtree_dom" / "correlation_clustering" / "fusion_stats.json")
    volume = load_json(
        OUT
        / "p1"
        / "volume"
        / "outputs"
        / "quadtree_dom"
        / "correlation_clustering"
        / "volume_stats.json"
    )

    stage_status = {}
    for event in manifest.get("events", []):
        if event.get("stage") in {"slicing", "detection", "p0", "p1", "volume"}:
            stage_status[event["stage"]] = event.get("status")

    checks = {
        "all_required_stages_complete": all(
            stage_status.get(stage) == "complete"
            for stage in ["slicing", "detection", "p0", "p1", "volume"]
        ),
        "detection_tiles_match_slicing_kept": detection["processed_tiles"] == slicing["kept_tiles"],
        "p0_p1_input_detections_match": p0["input_detections"] == p1["input_detections"] == detection["detection_count"],
        "p0_p1_candidate_stones_match": p0["candidate_stones"] == p1["candidate_stones"],
        "p1_accept_reject_closes": p1["output_stones"] + p1["rejected_stones"] == p1["candidate_stones"],
        "volume_processed_matches_p1_accepted": volume["scene"]["processed_stones"] == p1["output_stones"],
        "volume_qc_pass_fail_closes": volume["scene"]["qc_passed"] + volume["scene"]["qc_failed"]
        == volume["scene"]["processed_stones"],
        "scene_paths_are_site_b": "data\\dom3\\DOM.tif" in manifest["scene"]["dom_path"]
        and all("data\\pointcloud3" in p for p in manifest["scene"]["pointcloud_paths"]),
    }

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "site": manifest["scene"]["name"],
        "coordinate_qc_status": manifest.get("scene_coordinate_qc", {}).get("status"),
        "stage_status": stage_status,
        "checks": checks,
        "key_results": {
            "slicing_total_tiles": slicing["total_tiles"],
            "slicing_kept_tiles": slicing["kept_tiles"],
            "slicing_skipped_tiles": slicing["skipped_tiles"],
            "dom_area_m2": slicing["dom_area_m2"],
            "detection_count": detection["detection_count"],
            "raw_mask_candidates": detection["mask_candidates_total"],
            "diameter_filtered_removed": detection["filtered_candidates"]["total_removed"],
            "p0_candidate_stones": p0["candidate_stones"],
            "p0_accepted_stones": p0["output_stones"],
            "p1_candidate_stones": p1["candidate_stones"],
            "p1_accepted_stones": p1["output_stones"],
            "p1_rejected_stones": p1["rejected_stones"],
            "p1_acceptance_ratio": p1["validation_3d"]["acceptance_ratio"],
            "p1_rejection_ratio": p1["validation_3d"]["rejection_ratio"],
            "p1_rejection_reasons": p1["validation_3d"]["rejection_reasons"],
            "volume_processed_stones": volume["scene"]["processed_stones"],
            "volume_qc_passed": volume["scene"]["qc_passed"],
            "volume_qc_failed": volume["scene"]["qc_failed"],
            "volume_2d5_sum_m3": volume["volume_2d5"]["qc_passed_only"]["sum"],
            "volume_2d5_mean_m3": volume["volume_2d5"]["qc_passed_only"]["mean"],
            "volume_2d5_median_m3": volume["volume_2d5"]["qc_passed_only"]["median"],
            "volume_2d_proxy_sum_m3": volume["volume_2d_proxy"]["qc_passed_only"]["sum"],
            "volume_2d_proxy_mean_m3": volume["volume_2d_proxy"]["qc_passed_only"]["mean"],
            "proxy_minus_2d5_sum_m3": volume["comparison_2d_proxy_vs_2d5"]["proxy_minus_2d5_m3"]["sum"],
            "proxy_to_2d5_mean_ratio": volume["comparison_2d_proxy_vs_2d5"]["proxy_2d_to_2d5"]["mean"],
            "proxy_2d5_pearson_r": volume["comparison_2d_proxy_vs_2d5"]["pearson_r"],
            "diameter_bins": volume["diameter_bins"]["rows"],
        },
        "paper_use_notes": [
            "The P1 acceptance ratio is a geometric screening retention ratio, not detection accuracy.",
            "The 2D proxy comparison is an internal methodological contrast, not ground-truth volume validation.",
            "Coordinate QC remains provisional until manual feature residuals are checked.",
        ],
    }

    report_json = OUT / "site_b_quality_check_summary.json"
    report_md = OUT / "site_b_quality_check_report.md"
    report_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    kr = summary["key_results"]
    md = f"""# Mine Site B Output Quality Check

Created UTC: {summary['created_utc']}

## Completion

Required stages complete: {checks['all_required_stages_complete']}

| Stage | Status |
|---|---|
| Slicing | {stage_status.get('slicing')} |
| Detection | {stage_status.get('detection')} |
| P0 fusion | {stage_status.get('p0')} |
| P1 fusion | {stage_status.get('p1')} |
| Volume | {stage_status.get('volume')} |

## Consistency Checks

| Check | Result |
|---|---|
| Detection processed tiles equal kept tiles | {checks['detection_tiles_match_slicing_kept']} |
| P0/P1 input detections equal detection count | {checks['p0_p1_input_detections_match']} |
| P0/P1 candidate stones equal | {checks['p0_p1_candidate_stones_match']} |
| P1 accepted + rejected equals candidate | {checks['p1_accept_reject_closes']} |
| Volume processed stones equal P1 accepted stones | {checks['volume_processed_matches_p1_accepted']} |
| Volume QC passed + failed equals processed | {checks['volume_qc_pass_fail_closes']} |
| Manifest paths point to Site B inputs | {checks['scene_paths_are_site_b']} |

## Key Results

| Item | Value |
|---|---:|
| Total / kept / skipped tiles | {kr['slicing_total_tiles']} / {kr['slicing_kept_tiles']} / {kr['slicing_skipped_tiles']} |
| DOM area (m2) | {kr['dom_area_m2']} |
| Raw mask candidates | {kr['raw_mask_candidates']} |
| Diameter-filtered detections | {kr['detection_count']} |
| P0 candidate / accepted stones | {kr['p0_candidate_stones']} / {kr['p0_accepted_stones']} |
| P1 candidate / accepted / rejected stones | {kr['p1_candidate_stones']} / {kr['p1_accepted_stones']} / {kr['p1_rejected_stones']} |
| P1 geometric screening retention (%) | {pct(kr['p1_acceptance_ratio'])} |
| Volume processed / QC passed / QC failed | {kr['volume_processed_stones']} / {kr['volume_qc_passed']} / {kr['volume_qc_failed']} |
| 2.5D total volume (m3) | {kr['volume_2d5_sum_m3']} |
| 2D proxy total volume (m3) | {kr['volume_2d_proxy_sum_m3']} |
| 2D proxy minus 2.5D total (m3) | {kr['proxy_minus_2d5_sum_m3']} |
| Mean proxy-to-2.5D ratio | {kr['proxy_to_2d5_mean_ratio']} |
| Pearson r between proxy and 2.5D volume | {kr['proxy_2d5_pearson_r']} |

## Paper Notes

- P1 retention is a three-dimensional geometric screening result, not precision, recall, or F1.
- The 2D proxy comparison can support why point-cloud height information is needed, but it cannot prove absolute volume accuracy without ground truth.
- The coordinate check remains provisional until manual feature residuals are completed.
"""
    report_md.write_text(md, encoding="utf-8")

    print(report_json)
    print(report_md)


if __name__ == "__main__":
    main()

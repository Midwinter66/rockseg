"""
Detection stage experiment.

This script reads slicing outputs, runs YOLO instance segmentation on each kept
tile, converts detections into world-coordinate records, and writes both:
1. a full detection list for fusion
2. an analysis-friendly summary for paper/report use

Examples:
  python experiments/detection/run_detection_experiment.py --source quadtree_dom
  python experiments/detection/run_detection_experiment.py --source sahi --limit 10
  python experiments/detection/run_detection_experiment.py --source quadtree_dom --multi-scale
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000

DOM_PATH = PROJECT_ROOT / "data" / "dom2" / "DOM.tif"
DOM_WORLD_PATH = PROJECT_ROOT / "data" / "dom2" / "DOM.tfw"
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
SLICING_OUTPUTS = PROJECT_ROOT / "experiments" / "slicing" / "outputs"
SELF_DIR = Path(__file__).resolve().parent
DETECTION_CONFIG_PATH = PROJECT_ROOT / "experiments" / "configs" / "detection" / "default.json"

SOURCES = ["sahi", "quadtree_dom"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_tfw(tfw_path: Path) -> tuple[float, float, float, float, float, float]:
    lines = [float(line.strip()) for line in tfw_path.read_text("utf-8").splitlines() if line.strip()]
    if len(lines) != 6:
        raise ValueError(f"Invalid TFW file: {tfw_path}")
    return (lines[4], lines[0], lines[2], lines[5], lines[1], lines[3])


def _pixel_to_world(gt: tuple[float, float, float, float, float, float], px: float, py: float) -> tuple[float, float]:
    return float(gt[0] + px * gt[1] + py * gt[2]), float(gt[3] + px * gt[4] + py * gt[5])


def _resolve_output_dir(method: str) -> Path:
    path = SELF_DIR / "outputs" / method
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_device(device_cfg: Any) -> Any:
    if device_cfg in (None, "", "auto"):
        return None
    return device_cfg


def _device_label(device: Any) -> str:
    return "auto" if device is None else str(device)


def _summarize_numeric(values: list[float], digits: int = 4) -> dict | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(np.min(arr)), digits),
        "max": round(float(np.max(arr)), digits),
        "mean": round(float(np.mean(arr)), digits),
        "median": round(float(np.median(arr)), digits),
        "p25": round(float(np.percentile(arr, 25)), digits),
        "p75": round(float(np.percentile(arr, 75)), digits),
    }


def _rle_encode(mask: np.ndarray) -> dict:
    flat = mask.flatten()
    counts: list[int] = []
    if len(flat) == 0:
        return {"size": list(mask.shape), "counts": [], "starts_with": 0}

    prev = flat[0]
    starts_with = 1 if prev > 0 else 0
    count = 1
    for value in flat[1:]:
        if value == prev:
            count += 1
        else:
            counts.append(count)
            prev = value
            count = 1
    counts.append(count)
    return {"size": list(mask.shape), "counts": counts, "starts_with": starts_with}


def _empty_filter_counts() -> dict[str, int]:
    return {
        "mask_candidates": 0,
        "filtered_zero_area": 0,
        "filtered_min_diameter": 0,
        "filtered_zero_moments": 0,
        "kept_detections": 0,
    }


def _merge_filter_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = int(target.get(key, 0)) + int(value)


def _predict_result(model, image: np.ndarray, imgsz: int, conf: float, max_det: int, device: Any):
    if hasattr(model, "predictor") and model.predictor is not None:
        model.predictor.args.retina_masks = False
    return model.predict(
        image,
        imgsz=imgsz,
        conf=conf,
        max_det=max_det,
        device=device,
        retina_masks=False,
        verbose=False,
    )[0]


def _extract_detections_from_result(
    result,
    crop_shape: tuple[int, int],
    gt: tuple[float, float, float, float, float, float],
    offset_x: int,
    offset_y: int,
    min_diameter_m: float,
    inference_imgsz: int,
) -> tuple[list[dict], dict[str, int]]:
    counts = _empty_filter_counts()
    detections: list[dict] = []

    if result.masks is None or len(result.masks.data) == 0:
        return detections, counts

    crop_h, crop_w = crop_shape
    resolution = abs(float(gt[1]))
    counts["mask_candidates"] = int(len(result.masks.data))

    for idx, mask_tensor in enumerate(result.masks.data):
        mask = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
        if mask.shape[:2] != (crop_h, crop_w):
            mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

        area_px = int(np.count_nonzero(mask > 0))
        if area_px <= 0:
            counts["filtered_zero_area"] += 1
            continue

        area_m2 = float(area_px * resolution**2)
        eq_diameter_m = float(math.sqrt(4.0 * area_m2 / math.pi))
        if eq_diameter_m < min_diameter_m:
            counts["filtered_min_diameter"] += 1
            continue

        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            counts["filtered_zero_moments"] += 1
            continue

        cx = int(moments["m10"] / moments["m00"]) + offset_x
        cy = int(moments["m01"] / moments["m00"]) + offset_y
        wx, wy = _pixel_to_world(gt, cx, cy)

        bx, by, bw, bh = cv2.boundingRect(mask)
        gx0, gy0 = bx + offset_x, by + offset_y
        gx1, gy1 = gx0 + bw, gy0 + bh
        wx0, wy0 = _pixel_to_world(gt, gx0, gy0)
        wx1, wy1 = _pixel_to_world(gt, gx1, gy1)

        score = 0.0
        if result.boxes is not None and len(result.boxes) > idx:
            score = float(result.boxes.conf[idx].item())

        detections.append(
            {
                "score": round(score, 4),
                "area_m2": round(area_m2, 4),
                "equivalent_diameter_m": round(eq_diameter_m, 4),
                "centroid_world": [round(wx, 4), round(wy, 4)],
                "bbox_world": [
                    round(min(wx0, wx1), 4),
                    round(min(wy0, wy1), 4),
                    round(max(wx0, wx1), 4),
                    round(max(wy0, wy1), 4),
                ],
                "pixel_origin": [offset_x, offset_y],
                "rle_mask": _rle_encode(mask),
                "inference_imgsz": int(inference_imgsz),
            }
        )

    counts["kept_detections"] = len(detections)
    return detections, counts


def _boxes_iou_world(a: dict, b: dict) -> float:
    ax0, ay0, ax1, ay1 = a["bbox_world"]
    bx0, by0, bx1, by1 = b["bbox_world"]
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _fuse_multi_scale_dets(all_dets: list[dict], iou_thresh: float = 0.5) -> tuple[list[dict], int]:
    if not all_dets:
        return [], 0

    sorted_dets = sorted(all_dets, key=lambda det: float(det.get("score", 0.0)), reverse=True)
    kept: list[dict] = []
    duplicates_removed = 0
    for det in sorted_dets:
        is_duplicate = any(_boxes_iou_world(det, kept_det) >= iou_thresh for kept_det in kept)
        if is_duplicate:
            duplicates_removed += 1
            continue
        kept.append(det)
    return kept, duplicates_removed


def _base_tile_report(
    source_patch_id: str,
    tile_seq: int,
    crop: np.ndarray,
    offset_x: int,
    offset_y: int,
    mode: str,
) -> dict:
    height, width = crop.shape[:2]
    return {
        "source_patch_id": source_patch_id,
        "tile_seq": tile_seq,
        "mode": mode,
        "status": "pending",
        "failure_reason": "",
        "crop_origin_px": [int(offset_x), int(offset_y)],
        "crop_size_px": [int(width), int(height)],
        "predict_seconds": 0.0,
        "runtime_error_count": 0,
        "mask_candidates": 0,
        "filtered_zero_area": 0,
        "filtered_min_diameter": 0,
        "filtered_zero_moments": 0,
        "duplicates_removed": 0,
        "detection_count": 0,
    }


def _infer_tile(
    crop: np.ndarray,
    model,
    inference_cfg: dict,
    gt: tuple[float, float, float, float, float, float],
    offset_x: int,
    offset_y: int,
    source_patch_id: str,
    tile_seq: int,
) -> tuple[list[dict], dict]:
    device = _normalize_device(inference_cfg.get("device"))
    report = _base_tile_report(source_patch_id, tile_seq, crop, offset_x, offset_y, mode="single_scale")

    start = time.perf_counter()
    try:
        result = _predict_result(
            model=model,
            image=crop,
            imgsz=int(inference_cfg["imgsz"]),
            conf=float(inference_cfg["conf"]),
            max_det=int(inference_cfg.get("max_det", 1000)),
            device=device,
        )
    except RuntimeError as exc:
        report["status"] = "runtime_error"
        report["failure_reason"] = str(exc)[:240]
        report["runtime_error_count"] = 1
        report["predict_seconds"] = round(time.perf_counter() - start, 4)
        return [], report

    detections, counts = _extract_detections_from_result(
        result=result,
        crop_shape=crop.shape[:2],
        gt=gt,
        offset_x=offset_x,
        offset_y=offset_y,
        min_diameter_m=float(inference_cfg.get("min_stone_diameter_m", 0.0)),
        inference_imgsz=int(inference_cfg["imgsz"]),
    )
    _merge_filter_counts(report, counts)
    report["detection_count"] = len(detections)
    report["predict_seconds"] = round(time.perf_counter() - start, 4)

    if counts["mask_candidates"] == 0:
        report["status"] = "no_masks"
    elif detections:
        report["status"] = "ok"
    else:
        report["status"] = "filtered_out"

    return detections, report


def _infer_tile_multi_scale(
    crop: np.ndarray,
    model,
    inference_cfg: dict,
    gt: tuple[float, float, float, float, float, float],
    offset_x: int,
    offset_y: int,
    scales: list[int],
    source_patch_id: str,
    tile_seq: int,
) -> tuple[list[dict], dict]:
    device = _normalize_device(inference_cfg.get("device"))
    report = _base_tile_report(source_patch_id, tile_seq, crop, offset_x, offset_y, mode="multi_scale")
    report["scale_details"] = []

    all_dets: list[dict] = []
    total_start = time.perf_counter()
    iou_thresh = float(inference_cfg.get("multi_scale_iou_threshold", 0.5))

    for imgsz in scales:
        scale_start = time.perf_counter()
        try:
            result = _predict_result(
                model=model,
                image=crop,
                imgsz=int(imgsz),
                conf=float(inference_cfg["conf"]),
                max_det=int(inference_cfg.get("max_det", 1000)),
                device=device,
            )
        except RuntimeError as exc:
            report["runtime_error_count"] += 1
            report["scale_details"].append(
                {
                    "imgsz": int(imgsz),
                    "status": "runtime_error",
                    "predict_seconds": round(time.perf_counter() - scale_start, 4),
                    "failure_reason": str(exc)[:160],
                }
            )
            continue

        detections, counts = _extract_detections_from_result(
            result=result,
            crop_shape=crop.shape[:2],
            gt=gt,
            offset_x=offset_x,
            offset_y=offset_y,
            min_diameter_m=float(inference_cfg.get("min_stone_diameter_m", 0.0)),
            inference_imgsz=int(imgsz),
        )
        _merge_filter_counts(report, counts)
        all_dets.extend(detections)
        report["scale_details"].append(
            {
                "imgsz": int(imgsz),
                "status": "ok" if detections else ("no_masks" if counts["mask_candidates"] == 0 else "filtered_out"),
                "predict_seconds": round(time.perf_counter() - scale_start, 4),
                "mask_candidates": int(counts["mask_candidates"]),
                "kept_detections": int(counts["kept_detections"]),
                "filtered_min_diameter": int(counts["filtered_min_diameter"]),
            }
        )

    fused_dets, duplicates_removed = _fuse_multi_scale_dets(all_dets, iou_thresh=iou_thresh)
    report["duplicates_removed"] = int(duplicates_removed)
    report["detection_count"] = len(fused_dets)
    report["predict_seconds"] = round(time.perf_counter() - total_start, 4)

    if report["mask_candidates"] == 0 and report["runtime_error_count"] == len(scales):
        report["status"] = "runtime_error"
        report["failure_reason"] = "all scales failed with runtime errors"
    elif report["mask_candidates"] == 0:
        report["status"] = "no_masks"
    elif fused_dets:
        report["status"] = "ok"
    else:
        report["status"] = "filtered_out"

    return fused_dets, report


def _collect_crop(
    item: dict,
    is_sahi: bool,
    slice_stats: dict,
    dom_img: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    if is_sahi:
        x, y = item["pixel_origin"]
        size = item["pixel_size"]
        return dom_img[y : y + size, x : x + size], int(x), int(y)

    bounds = item["bounds_m"]
    dom_w = int(slice_stats.get("dom_dims", {}).get("width_px") or dom_img.shape[1])
    dom_h = int(slice_stats.get("dom_dims", {}).get("height_px") or dom_img.shape[0])
    dom_bounds = slice_stats.get("dom_bounds_world", [0, 0, 0, 0])
    world_w = dom_bounds[2] - dom_bounds[0]
    world_h = dom_bounds[3] - dom_bounds[1]

    px0 = int((bounds[0] - dom_bounds[0]) / world_w * dom_w)
    px1 = int((bounds[2] - dom_bounds[0]) / world_w * dom_w)
    py0 = int((dom_bounds[3] - bounds[3]) / world_h * dom_h)
    py1 = int((dom_bounds[3] - bounds[1]) / world_h * dom_h)

    px0, px1 = sorted([max(0, px0), min(dom_w, px1)])
    py0, py1 = sorted([max(0, py0), min(dom_h, py1)])
    return dom_img[py0:py1, px0:px1], int(px0), int(py0)


def _build_stats(
    source_method: str,
    config: dict,
    multi_scale: bool,
    scales: list[int] | None,
    elapsed: float,
    tile_reports: list[dict],
    all_dets: list[dict],
    items: list[dict],
    kept_items: list[dict],
    out_dir: Path,
) -> dict:
    status_counts = Counter(report["status"] for report in tile_reports)
    positive_tiles = [report for report in tile_reports if report["detection_count"] > 0]
    failed_statuses = {"runtime_error", "empty_crop"}
    successful_reports = [report for report in tile_reports if report["status"] not in failed_statuses]
    successful_zero_det_reports = [report for report in successful_reports if report["detection_count"] == 0]

    areas = [float(det["area_m2"]) for det in all_dets]
    diameters = [float(det["equivalent_diameter_m"]) for det in all_dets]
    scores = [float(det["score"]) for det in all_dets]
    tile_detection_counts = [int(report["detection_count"]) for report in tile_reports]

    total_mask_candidates = sum(int(report["mask_candidates"]) for report in tile_reports)
    total_filtered_zero_area = sum(int(report["filtered_zero_area"]) for report in tile_reports)
    total_filtered_min_diameter = sum(int(report["filtered_min_diameter"]) for report in tile_reports)
    total_filtered_zero_moments = sum(int(report["filtered_zero_moments"]) for report in tile_reports)
    total_duplicates_removed = sum(int(report.get("duplicates_removed", 0)) for report in tile_reports)
    total_runtime_errors = sum(int(report["runtime_error_count"]) for report in tile_reports)
    runtime_error_messages = [
        report["failure_reason"]
        for report in tile_reports
        if report["status"] == "runtime_error" and report.get("failure_reason")
    ][:5]

    processed_tile_count = len(kept_items)
    positive_tile_count = len(positive_tiles)
    successful_tile_count = len(successful_reports)
    failed_tile_count = sum(status_counts.get(status, 0) for status in failed_statuses)

    stats = {
        "analysis_version": "detection_v2",
        "source_method": source_method,
        "model_path": str(MODEL_PATH),
        "dom_path": str(DOM_PATH),
        "config": config,
        "multi_scale": bool(multi_scale),
        "scales": list(scales or []),
        "device": _device_label(_normalize_device(config["inference"].get("device"))),
        "total_tiles": len(items),
        "processed_tiles": processed_tile_count,
        "successful_tiles": successful_tile_count,
        "failed_tiles": failed_tile_count,
        "tiles_with_detections": positive_tile_count,
        "successful_zero_detection_tiles": len(successful_zero_det_reports),
        "status_breakdown": dict(status_counts),
        "detection_count": len(all_dets),
        "mask_candidates_total": total_mask_candidates,
        "filtered_candidates": {
            "zero_area": total_filtered_zero_area,
            "below_min_diameter": total_filtered_min_diameter,
            "zero_moments": total_filtered_zero_moments,
            "total_removed": total_filtered_zero_area + total_filtered_min_diameter + total_filtered_zero_moments,
        },
        "multi_scale_duplicates_removed": total_duplicates_removed,
        "runtime_errors": {
            "tile_count": int(status_counts.get("runtime_error", 0)),
            "event_count": total_runtime_errors,
            "messages_sample": runtime_error_messages,
        },
        "detections_per_tile": {
            "mean_per_processed_tile": round(float(len(all_dets) / processed_tile_count), 4) if processed_tile_count else 0.0,
            "mean_per_successful_tile": round(float(len(all_dets) / successful_tile_count), 4) if successful_tile_count else 0.0,
            "mean_per_positive_tile": round(float(len(all_dets) / positive_tile_count), 4) if positive_tile_count else 0.0,
            "max_per_tile": int(max(tile_detection_counts)) if tile_detection_counts else 0,
        },
        "empty_tile_ratio_after_success": round(float(len(successful_zero_det_reports) / successful_tile_count), 4)
        if successful_tile_count
        else None,
        "confidence": _summarize_numeric(scores, digits=4),
        "area_m2": _summarize_numeric(areas, digits=4),
        "diameter_m": _summarize_numeric(diameters, digits=4),
        "tile_detection_count": _summarize_numeric([float(v) for v in tile_detection_counts], digits=4),
        "elapsed_seconds": round(elapsed, 2),
        "outputs": {
            "detection_stats_json": str(out_dir / "detection_stats.json"),
            "detections_json": str(out_dir / "detections.json"),
            "tile_detection_summary_json": str(out_dir / "tile_detection_summary.json"),
        },
    }
    return stats


def _run_detection(
    source_method: str,
    limit: int | None = None,
    multi_scale: bool = False,
    scales: list[int] | None = None,
) -> dict:
    from ultralytics import YOLO

    start = time.perf_counter()

    config = _load_json(DETECTION_CONFIG_PATH)
    inference_cfg = config["inference"]
    gt = _parse_tfw(DOM_WORLD_PATH)
    slice_stats = _load_json(SLICING_OUTPUTS / source_method / "tile_stats.json")

    model = YOLO(str(MODEL_PATH))
    dom_img = np.array(Image.open(DOM_PATH))
    if dom_img.ndim == 3 and dom_img.shape[2] == 3:
        dom_img = cv2.cvtColor(dom_img, cv2.COLOR_RGB2BGR)

    is_sahi = "patches" in slice_stats
    items = slice_stats["patches"] if is_sahi else slice_stats["tiles"]
    kept_items = [
        item
        for item in items
        if item.get("status", "kept") == "kept" and not item.get("skipped", False)
    ]
    if limit is not None:
        kept_items = kept_items[:limit]

    device = _device_label(_normalize_device(inference_cfg.get("device")))
    scale_label = list(scales or []) if multi_scale else [int(inference_cfg["imgsz"])]
    print(f"  source={source_method}  kept_tiles={len(kept_items)}/{len(items)}")
    print(
        f"  conf={float(inference_cfg['conf']):.2f}  "
        f"min_diameter={float(inference_cfg.get('min_stone_diameter_m', 0.0)):.2f}m  "
        f"device={device}  scales={scale_label}"
    )

    all_dets: list[dict] = []
    tile_reports: list[dict] = []
    total = len(kept_items)

    for tile_seq, item in enumerate(kept_items, start=1):
        source_patch_id = item.get("patch_id", item.get("tile_id", f"tile_{tile_seq:06d}"))
        crop, offset_x, offset_y = _collect_crop(item, is_sahi=is_sahi, slice_stats=slice_stats, dom_img=dom_img)

        if crop.size == 0:
            tile_reports.append(
                {
                    "source_patch_id": source_patch_id,
                    "tile_seq": tile_seq,
                    "mode": "crop",
                    "status": "empty_crop",
                    "failure_reason": "empty crop after coordinate conversion",
                    "crop_origin_px": [int(offset_x), int(offset_y)],
                    "crop_size_px": [0, 0],
                    "predict_seconds": 0.0,
                    "runtime_error_count": 0,
                    "mask_candidates": 0,
                    "filtered_zero_area": 0,
                    "filtered_min_diameter": 0,
                    "filtered_zero_moments": 0,
                    "duplicates_removed": 0,
                    "detection_count": 0,
                }
            )
            continue

        if multi_scale and scales:
            detections, report = _infer_tile_multi_scale(
                crop=crop,
                model=model,
                inference_cfg=inference_cfg,
                gt=gt,
                offset_x=offset_x,
                offset_y=offset_y,
                scales=scales,
                source_patch_id=source_patch_id,
                tile_seq=tile_seq,
            )
        else:
            detections, report = _infer_tile(
                crop=crop,
                model=model,
                inference_cfg=inference_cfg,
                gt=gt,
                offset_x=offset_x,
                offset_y=offset_y,
                source_patch_id=source_patch_id,
                tile_seq=tile_seq,
            )

        for det_idx, det in enumerate(detections):
            det["source_patch_id"] = source_patch_id
            det["source_method"] = source_method
            det["source_tile_seq"] = tile_seq
            det["detection_id"] = f"{source_method}_{tile_seq:06d}_{det_idx:03d}"

        all_dets.extend(detections)
        tile_reports.append(report)

        if total > 0:
            filled = int(28 * tile_seq / total)
            bar = "#" * filled + "-" * (28 - filled)
            print(
                f"  [{bar}] {tile_seq:>4}/{total}  "
                f"tile_dets={len(detections):>3}  total_dets={len(all_dets):>5}",
                end="\r",
            )

    elapsed = time.perf_counter() - start
    print()

    out_dir = _resolve_output_dir(source_method)
    stats = _build_stats(
        source_method=source_method,
        config=config,
        multi_scale=multi_scale,
        scales=scales if multi_scale else [int(inference_cfg["imgsz"])],
        elapsed=elapsed,
        tile_reports=tile_reports,
        all_dets=all_dets,
        items=items,
        kept_items=kept_items,
        out_dir=out_dir,
    )

    print("  Detection summary")
    print(f"    detections: {stats['detection_count']}")
    print(f"    positive tiles: {stats['tiles_with_detections']} / {stats['processed_tiles']}")
    print(f"    failed tiles: {stats['failed_tiles']}")
    print(f"    filtered by min diameter: {stats['filtered_candidates']['below_min_diameter']}")
    print(f"    runtime error events: {stats['runtime_errors']['event_count']}")
    print(f"    elapsed: {stats['elapsed_seconds']:.2f}s")

    (out_dir / "detection_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "detections.json").write_text(
        json.dumps(all_dets, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "tile_detection_summary.json").write_text(
        json.dumps(tile_reports, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO detection on slicing results")
    parser.add_argument(
        "--source",
        choices=["all"] + SOURCES,
        default="all",
        help="Slicing source to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N kept tiles for quick testing",
    )
    parser.add_argument(
        "--multi-scale",
        action="store_true",
        help="Run each tile with multiple imgsz values and fuse duplicate detections",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="640,1024,1280",
        help="Comma-separated imgsz list used when --multi-scale is enabled",
    )
    args = parser.parse_args()

    scales = [int(value) for value in args.scales.split(",")] if args.multi_scale else None

    if not MODEL_PATH.exists():
        print(f"ERROR: model not found: {MODEL_PATH}")
        sys.exit(1)

    methods = SOURCES if args.source == "all" else [args.source]

    print(f"\n{'=' * 64}")
    print("  Detection Experiment")
    print(f"  Model: {MODEL_PATH.name}")
    print(f"  Source(s): {', '.join(methods)}")
    if args.multi_scale:
        print(f"  Multi-scale: enabled ({scales})")
    else:
        default_cfg = _load_json(DETECTION_CONFIG_PATH)["inference"]
        print(f"  Multi-scale: disabled (imgsz={default_cfg['imgsz']})")
    print(f"{'=' * 64}\n")

    results = {}
    for method in methods:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        results[method] = _run_detection(
            source_method=method,
            limit=args.limit,
            multi_scale=args.multi_scale,
            scales=scales,
        )

    manifest_path = SELF_DIR / "outputs" / "detection_manifest.json"
    manifest_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nDetection manifest: {manifest_path}")


if __name__ == "__main__":
    main()

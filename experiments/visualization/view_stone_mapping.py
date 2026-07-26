"""
Inspect whether one fused stone is spatially aligned on the DOM and point cloud.

Examples:
  python experiments/visualization/view_stone_mapping.py --source quadtree_dom --method correlation_clustering --list-stones 20
  python experiments/visualization/view_stone_mapping.py --source quadtree_dom --method correlation_clustering --stone-id stone_000000
  python experiments/visualization/view_stone_mapping.py --source quadtree_dom --method correlation_clustering --stone-rank 0 --no-viewer
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.scene_reference import CURRENT_SCENE
from experiments.common.stone_region import crop_stone_point_cloud

Image.MAX_IMAGE_PIXELS = 500_000_000

FUSION_ROOT = PROJECT_ROOT / "experiments" / "fusion" / "outputs"
DETECTION_ROOT = PROJECT_ROOT / "experiments" / "detection" / "outputs"
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "visualization" / "outputs" / "stone_mapping"

SOURCES = ["sahi", "quadtree_dom"]
METHODS = ["heuristic", "correlation_clustering"]
STONE_MODES = ["accepted", "rejected", "all"]

DOM_BOX_COLOR = (30, 180, 40)
DOM_FUSED_COLOR = (30, 180, 40)
DOM_CENTROID_COLOR = (30, 30, 220)
PC_CONTEXT_COLOR = np.asarray([0.68, 0.68, 0.70], dtype=np.float64)
PC_STONE_COLOR = np.asarray([0.92, 0.28, 0.18], dtype=np.float64)
PC_BBOX_COLOR = (0.05, 0.65, 0.12)
PC_BACKGROUND = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
DETECTION_COLORS = [
    (0, 170, 255),
    (255, 120, 0),
    (180, 60, 255),
    (70, 210, 140),
    (255, 60, 160),
    (0, 220, 220),
]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_data(source: str, method: str) -> tuple[dict, list[dict]]:
    fusion_path = FUSION_ROOT / source / method / "fusion_stats.json"
    detections_path = DETECTION_ROOT / source / "detections.json"
    if not fusion_path.exists():
        raise FileNotFoundError(f"Missing fusion result: {fusion_path}")
    if not detections_path.exists():
        raise FileNotFoundError(f"Missing detection result: {detections_path}")
    return _load_json(fusion_path), _load_json(detections_path)


def _select_stones(fusion: dict, mode: str) -> list[dict]:
    accepted = list(fusion.get("stones", []))
    rejected = list(fusion.get("rejected_stones_detail", []))
    if mode == "accepted":
        return accepted
    if mode == "rejected":
        return rejected
    return accepted + rejected


def _bbox_area_m2(bbox_world: list[float] | tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = [float(v) for v in bbox_world[:4]]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _sorted_stones(stones: list[dict]) -> list[dict]:
    def key_fn(stone: dict) -> tuple[float, float]:
        area = _bbox_area_m2(stone.get("bbox_world", [0.0, 0.0, 0.0, 0.0]))
        score = float(stone.get("score_mean", stone.get("score_max", 0.0)))
        return area, score

    return sorted(stones, key=key_fn, reverse=True)


def _find_stone(stones: list[dict], stone_id: str | None, stone_rank: int | None) -> tuple[dict, int]:
    if stone_id:
        for rank, stone in enumerate(stones):
            if stone.get("stone_id") == stone_id:
                return stone, rank
        raise KeyError(f"stone_id not found: {stone_id}")
    if stone_rank is None:
        raise ValueError("Either --stone-id or --stone-rank must be provided.")
    if stone_rank < 0 or stone_rank >= len(stones):
        raise IndexError(f"stone-rank out of range: 0..{len(stones) - 1}")
    return stones[stone_rank], stone_rank


def _world_to_pixel(gt: tuple[float, float, float, float, float, float], x_world: float, y_world: float) -> tuple[float, float]:
    origin_x, res_x, _, origin_y, _, res_y = gt
    px = (float(x_world) - origin_x) / res_x
    py = (float(y_world) - origin_y) / res_y
    return float(px), float(py)


def _world_bbox_to_pixel_box(
    bbox_world: list[float] | tuple[float, float, float, float],
    gt: tuple[float, float, float, float, float, float],
    *,
    pad_m: float = 0.0,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in bbox_world[:4]]
    x0 -= pad_m
    y0 -= pad_m
    x1 += pad_m
    y1 += pad_m
    px0, py0 = _world_to_pixel(gt, x0, y0)
    px1, py1 = _world_to_pixel(gt, x1, y1)
    return (
        int(math.floor(min(px0, px1))),
        int(math.floor(min(py0, py1))),
        int(math.ceil(max(px0, px1))),
        int(math.ceil(max(py0, py1))),
    )


def _clip_box(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    return x0, y0, x1, y1


def _rle_decode(rle: dict, expected_area_px: float | None = None) -> np.ndarray:
    h, w = [int(v) for v in rle["size"]]
    mask = np.zeros(h * w, dtype=np.uint8)
    counts = [int(v) for v in rle.get("counts", [])]
    starts_with = rle.get("starts_with")
    if starts_with is None:
        odd_area = sum(counts[1::2])
        even_area = sum(counts[0::2])
        if expected_area_px is not None:
            starts_with = 0 if abs(odd_area - expected_area_px) <= abs(even_area - expected_area_px) else 1
        else:
            starts_with = 0 if odd_area <= even_area else 1
    pos = 0
    for idx, count in enumerate(counts):
        if (int(starts_with) + idx) % 2 == 1:
            mask[pos : pos + count] = 255
        pos += count
    return mask.reshape(h, w)


def _local_label_box(
    bbox_world: list[float],
    gt: tuple[float, float, float, float, float, float],
    crop_box_px: tuple[int, int, int, int],
    scale: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _world_bbox_to_pixel_box(bbox_world, gt)
    crop_x0, crop_y0, _, _ = crop_box_px
    return (
        int(round((x0 - crop_x0) * scale)),
        int(round((y0 - crop_y0) * scale)),
        int(round((x1 - crop_x0) * scale)),
        int(round((y1 - crop_y0) * scale)),
    )


def _draw_detection_mask(
    canvas: np.ndarray,
    detection: dict,
    *,
    crop_box_px: tuple[int, int, int, int],
    scale: float,
    color: tuple[int, int, int],
    gt: tuple[float, float, float, float, float, float],
) -> None:
    import cv2

    crop_x0, crop_y0, crop_x1, crop_y1 = crop_box_px
    mask = _rle_decode(detection["rle_mask"])
    det_x0, det_y0 = [int(v) for v in detection["pixel_origin"]]
    det_h, det_w = mask.shape[:2]
    det_x1 = det_x0 + det_w
    det_y1 = det_y0 + det_h

    ix0 = max(crop_x0, det_x0)
    iy0 = max(crop_y0, det_y0)
    ix1 = min(crop_x1, det_x1)
    iy1 = min(crop_y1, det_y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return

    mask_crop = mask[iy0 - det_y0 : iy1 - det_y0, ix0 - det_x0 : ix1 - det_x0]
    target_w = max(1, int(round(mask_crop.shape[1] * scale)))
    target_h = max(1, int(round(mask_crop.shape[0] * scale)))
    if scale != 1.0:
        mask_crop = cv2.resize(mask_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    dst_x0 = int(round((ix0 - crop_x0) * scale))
    dst_y0 = int(round((iy0 - crop_y0) * scale))
    dst_x1 = min(canvas.shape[1], dst_x0 + mask_crop.shape[1])
    dst_y1 = min(canvas.shape[0], dst_y0 + mask_crop.shape[0])
    mask_crop = mask_crop[: dst_y1 - dst_y0, : dst_x1 - dst_x0]
    if mask_crop.size == 0:
        return

    region = canvas[dst_y0:dst_y1, dst_x0:dst_x1]
    selected = mask_crop > 0
    if np.any(selected):
        region[selected] = np.clip(
            region[selected].astype(np.float32) * 0.48 + np.asarray(color, dtype=np.float32) * 0.52,
            0,
            255,
        ).astype(np.uint8)

    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(region, contours, -1, color, 2, cv2.LINE_AA)

    bx0, by0, bx1, by1 = _local_label_box(detection.get("bbox_world", [0, 0, 0, 0]), gt, crop_box_px, scale)
    cv2.rectangle(canvas, (bx0, by0), (bx1, by1), color, 1, cv2.LINE_AA)


def _render_dom_preview(
    stone: dict,
    detections: list[dict],
    gt: tuple[float, float, float, float, float, float],
    *,
    crop_pad_m: float,
    min_preview_side: int,
    output_path: Path,
) -> dict:
    import cv2

    dom = Image.open(CURRENT_SCENE.dom_path).convert("RGB")
    crop_box_px = _clip_box(
        *_world_bbox_to_pixel_box(stone.get("bbox_world", [0, 0, 0, 0]), gt, pad_m=crop_pad_m),
        dom.width,
        dom.height,
    )
    crop_rgb = np.array(dom.crop(crop_box_px))
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    crop_h, crop_w = crop_bgr.shape[:2]

    scale = 1.0
    max_side = max(crop_w, crop_h)
    if max_side > 0 and max_side < min_preview_side:
        scale = min_preview_side / max_side
        crop_bgr = cv2.resize(
            crop_bgr,
            (max(1, int(round(crop_w * scale))), max(1, int(round(crop_h * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )

    overlay = crop_bgr.copy()
    used_detections: list[dict] = []
    for det_rank, det_idx in enumerate(stone.get("detection_indices", [])):
        if not (0 <= det_idx < len(detections)):
            continue
        det = detections[det_idx]
        color = DETECTION_COLORS[det_rank % len(DETECTION_COLORS)]
        _draw_detection_mask(
            overlay,
            det,
            crop_box_px=crop_box_px,
            scale=scale,
            color=color,
            gt=gt,
        )
        used_detections.append(
            {
                "detection_index": int(det_idx),
                "source_patch_id": det.get("source_patch_id"),
                "score": float(det.get("score", 0.0)),
                "area_m2": float(det.get("area_m2", 0.0)),
                "equivalent_diameter_m": float(det.get("equivalent_diameter_m", 0.0)),
                "bbox_world": det.get("bbox_world"),
                "centroid_world": det.get("centroid_world"),
            }
        )

    fused_box = _local_label_box(stone.get("bbox_world", [0, 0, 0, 0]), gt, crop_box_px, scale)
    cv2.rectangle(overlay, (fused_box[0], fused_box[1]), (fused_box[2], fused_box[3]), DOM_FUSED_COLOR, 2, cv2.LINE_AA)

    centroid = stone.get("centroid_world")
    if centroid and len(centroid) >= 2:
        cx, cy = _world_to_pixel(gt, float(centroid[0]), float(centroid[1]))
        lx = int(round((cx - crop_box_px[0]) * scale))
        ly = int(round((cy - crop_box_px[1]) * scale))
        cv2.circle(overlay, (lx, ly), 4, DOM_CENTROID_COLOR, -1, cv2.LINE_AA)

    header_h = 90
    header = np.full((header_h, overlay.shape[1], 3), 255, dtype=np.uint8)
    title = f"{stone.get('stone_id')} | dets={len(used_detections)} | bbox_area={_bbox_area_m2(stone.get('bbox_world', [0,0,0,0])):.3f} m2"
    cv2.putText(header, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 25, 25), 2, cv2.LINE_AA)
    bbox_text = "bbox_world=" + ", ".join(f"{float(v):.3f}" for v in stone.get("bbox_world", [0, 0, 0, 0]))
    cv2.putText(header, bbox_text, (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (50, 50, 50), 1, cv2.LINE_AA)

    legend_x = 16
    legend_y = 74
    cv2.rectangle(header, (legend_x, legend_y - 10), (legend_x + 18, legend_y + 2), DOM_FUSED_COLOR, -1)
    cv2.putText(header, "fused bbox", (legend_x + 24, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
    cursor_x = legend_x + 130
    for det_rank, info in enumerate(used_detections[:4]):
        color = DETECTION_COLORS[det_rank % len(DETECTION_COLORS)]
        cv2.rectangle(header, (cursor_x, legend_y - 10), (cursor_x + 18, legend_y + 2), color, -1)
        label = f"d{info['detection_index']} {info.get('source_patch_id', '')}"
        cv2.putText(header, label[:24], (cursor_x + 24, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (40, 40, 40), 1, cv2.LINE_AA)
        cursor_x += 220
        if cursor_x > header.shape[1] - 220:
            break

    preview = np.vstack([header, overlay])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)
    return {
        "dom_crop_pixel_box": [int(v) for v in crop_box_px],
        "dom_crop_size_px": [int(crop_w), int(crop_h)],
        "preview_scale": float(scale),
        "preview_image": str(output_path),
        "used_detections": used_detections,
    }


def _load_local_scene_points(
    point_bbox: tuple[float, float, float, float],
    *,
    chunk_size: int,
) -> tuple[np.ndarray, dict]:
    import laspy

    x0, y0, x1, y1 = [float(v) for v in point_bbox]
    point_parts: list[np.ndarray] = []
    file_stats: list[dict] = []

    for path in CURRENT_SCENE.pointcloud_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing point cloud file: {path}")

        kept_total = 0
        chunk_count = 0
        with laspy.open(path) as reader:
            total_points = int(reader.header.point_count)
            for points in reader.chunk_iterator(chunk_size):
                chunk_count += 1
                xs = np.asarray(points.x)
                ys = np.asarray(points.y)
                keep = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
                if not np.any(keep):
                    continue
                xyz = np.column_stack(
                    (
                        xs[keep],
                        ys[keep],
                        np.asarray(points.z)[keep],
                    )
                ).astype(np.float64, copy=False)
                point_parts.append(xyz)
                kept_total += len(xyz)

        file_stats.append(
            {
                "file": str(path),
                "total_points": int(total_points),
                "kept_points": int(kept_total),
                "chunks_read": int(chunk_count),
            }
        )

    if point_parts:
        points = np.concatenate(point_parts, axis=0)
    else:
        points = np.empty((0, 3), dtype=np.float64)

    return points, {
        "point_bbox": [float(v) for v in point_bbox],
        "file_stats": file_stats,
        "local_point_count": int(len(points)),
    }


def _match_masked_points(candidate_points: np.ndarray, masked_points: np.ndarray) -> np.ndarray:
    matched = np.zeros(len(candidate_points), dtype=bool)
    if len(candidate_points) == 0 or len(masked_points) == 0:
        return matched

    lookup: dict[tuple[float, float, float], list[int]] = {}
    for idx, row in enumerate(candidate_points):
        key = tuple(np.round(row, 6))
        lookup.setdefault(key, []).append(idx)

    for row in masked_points:
        key = tuple(np.round(row, 6))
        positions = lookup.get(key)
        if not positions:
            continue
        matched[positions.pop()] = True
    return matched


def _sample_context_positions(matched_mask: np.ndarray, *, max_points: int, seed: int) -> np.ndarray:
    count = len(matched_mask)
    if max_points <= 0 or count <= max_points:
        return np.arange(count, dtype=np.int64)

    rng = np.random.default_rng(seed)
    matched_pos = np.flatnonzero(matched_mask)
    if len(matched_pos) >= max_points:
        return np.sort(rng.choice(matched_pos, size=max_points, replace=False)).astype(np.int64, copy=False)

    keep = set(int(v) for v in matched_pos.tolist())
    remaining = max_points - len(matched_pos)
    other_pos = np.flatnonzero(~matched_mask)
    if remaining > 0 and len(other_pos) > 0:
        sampled = rng.choice(other_pos, size=min(remaining, len(other_pos)), replace=False)
        keep.update(int(v) for v in sampled.tolist())
    return np.asarray(sorted(keep), dtype=np.int64)


def _build_pc_summary(points: np.ndarray, masked_points: np.ndarray, crop_info: dict, local_load_info: dict) -> dict:
    summary = {
        "local_point_count": int(len(points)),
        "masked_point_count": int(len(masked_points)),
        "crop_info": crop_info,
        "load_info": local_load_info,
    }
    if len(points) > 0:
        summary["local_xyz_min"] = [float(v) for v in points.min(axis=0)]
        summary["local_xyz_max"] = [float(v) for v in points.max(axis=0)]
    if len(masked_points) > 0:
        summary["stone_xyz_min"] = [float(v) for v in masked_points.min(axis=0)]
        summary["stone_xyz_max"] = [float(v) for v in masked_points.max(axis=0)]
        summary["stone_size_m"] = [float(v) for v in np.ptp(masked_points, axis=0)]
        summary["stone_z_range_m"] = float(np.ptp(masked_points[:, 2]))
    return summary


def _open_pointcloud_viewer(
    points: np.ndarray,
    masked_points: np.ndarray,
    *,
    stone: dict,
    point_size: float,
    max_context_points: int,
    seed: int,
) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D is not installed. DOM preview and summary were saved, but 3D viewer was skipped.")
        return

    if len(points) == 0:
        print("No local point-cloud points were loaded, skipping 3D viewer.")
        return

    matched_mask = _match_masked_points(points, masked_points)
    selected_pos = _sample_context_positions(matched_mask, max_points=max_context_points, seed=seed)
    display_points = points[selected_pos]
    display_mask = matched_mask[selected_pos]

    display_origin = display_points.mean(axis=0)
    colors = np.tile(PC_CONTEXT_COLOR, (len(display_points), 1))
    colors[display_mask] = PC_STONE_COLOR

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(display_points - display_origin)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    bbox_geom = None
    if len(masked_points) > 0:
        mask_cloud = o3d.geometry.PointCloud()
        mask_cloud.points = o3d.utility.Vector3dVector(masked_points - display_origin)
        bbox_geom = mask_cloud.get_axis_aligned_bounding_box()
    else:
        bbox_world = stone.get("bbox_world", [0, 0, 0, 0])
        px0, py0, px1, py1 = CURRENT_SCENE.xy_transform.world_bbox_to_point_bbox(bbox_world, pad_m=0.0)
        z_min = float(display_points[:, 2].min())
        z_max = float(display_points[:, 2].max())
        bbox_geom = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.asarray([px0, py0, z_min]) - display_origin,
            max_bound=np.asarray([px1, py1, z_max]) - display_origin,
        )
    bbox_geom.color = PC_BBOX_COLOR

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name=f"Stone Mapping - {stone.get('stone_id')}", width=1480, height=920)
    visualizer.add_geometry(cloud)
    visualizer.add_geometry(bbox_geom)
    render_option = visualizer.get_render_option()
    render_option.background_color = PC_BACKGROUND
    render_option.point_size = float(point_size)
    visualizer.run()
    visualizer.destroy_window()


def _stone_summary(stone: dict, rank: int, dom_info: dict, pc_info: dict) -> dict:
    validation = stone.get("validation_3d", {})
    return {
        "stone_id": stone.get("stone_id"),
        "stone_rank": int(rank),
        "source_detection_count": int(stone.get("source_detection_count", len(stone.get("detection_indices", [])))),
        "detection_indices": [int(v) for v in stone.get("detection_indices", [])],
        "bbox_world": stone.get("bbox_world"),
        "centroid_world": stone.get("centroid_world"),
        "bbox_area_m2": float(_bbox_area_m2(stone.get("bbox_world", [0, 0, 0, 0]))),
        "score_mean": float(stone.get("score_mean", stone.get("score_max", 0.0))),
        "validation_3d": validation,
        "dom": dom_info,
        "pointcloud": pc_info,
    }


def _print_summary(summary: dict) -> None:
    print("")
    print("=" * 72)
    print(f"Stone mapping check: {summary['stone_id']}  (rank={summary['stone_rank']})")
    print("=" * 72)
    print(f"  source detections : {summary['source_detection_count']}")
    print(f"  detection indices : {summary['detection_indices']}")
    print(f"  bbox_world        : {summary['bbox_world']}")
    print(f"  centroid_world    : {summary['centroid_world']}")
    print(f"  bbox_area_m2      : {summary['bbox_area_m2']:.4f}")
    print(f"  score_mean        : {summary['score_mean']:.4f}")
    print(f"  dom preview       : {summary['dom']['preview_image']}")
    print(f"  dom crop px       : {summary['dom']['dom_crop_pixel_box']}")
    print(f"  preview scale     : {summary['dom']['preview_scale']:.3f}")
    print(f"  local points      : {summary['pointcloud']['local_point_count']:,}")
    print(f"  masked points     : {summary['pointcloud']['masked_point_count']:,}")
    print(f"  crop query mode   : {summary['pointcloud']['crop_info'].get('query_mode')}")
    if "stone_z_range_m" in summary["pointcloud"]:
        print(f"  stone z_range_m   : {summary['pointcloud']['stone_z_range_m']:.4f}")
    validation = summary.get("validation_3d", {})
    if validation:
        print(f"  validation passed : {validation.get('passed')}")
        if validation.get("reasons"):
            print(f"  validation reason : {', '.join(validation['reasons'])}")
    print("")
    print("  detections:")
    for item in summary["dom"].get("used_detections", []):
        print(
            "   - "
            f"idx={item['detection_index']}  "
            f"tile={item.get('source_patch_id')}  "
            f"score={item['score']:.4f}  "
            f"diameter={item['equivalent_diameter_m']:.4f} m"
        )


def _write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check one fused stone on DOM and point cloud.")
    parser.add_argument("--source", choices=SOURCES, default="quadtree_dom")
    parser.add_argument("--method", choices=METHODS, default="correlation_clustering")
    parser.add_argument("--mode", choices=STONE_MODES, default="accepted")
    parser.add_argument("--stone-id", default=None, help="Target stone_id, for example stone_000000")
    parser.add_argument("--stone-rank", type=int, default=None, help="Rank after sorting by bbox area")
    parser.add_argument("--list-stones", type=int, nargs="?", const=20, default=None, help="List top-N stones and exit")
    parser.add_argument("--bbox-pad-m", type=float, default=1.0, help="Context padding around fused bbox in metres")
    parser.add_argument("--min-dom-preview-side", type=int, default=900, help="Upscale small DOM crops for clearer review")
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--max-context-points", type=int, default=180_000, help="Maximum displayed local points")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-viewer", action="store_true", help="Only export DOM preview and JSON summary")
    args = parser.parse_args()

    fusion, detections = _load_data(args.source, args.method)
    stones = _sorted_stones(_select_stones(fusion, args.mode))
    if not stones:
        print(f"No stones found for mode={args.mode}")
        return

    if args.list_stones is not None:
        limit = min(args.list_stones, len(stones))
        print(f"mode={args.mode}  listing top {limit} stones")
        for rank, stone in enumerate(stones[:limit]):
            validation = stone.get("validation_3d", {})
            print(
                f"[{rank:03d}] {stone.get('stone_id')}  "
                f"bbox_area={_bbox_area_m2(stone.get('bbox_world', [0, 0, 0, 0])):.4f} m2  "
                f"dets={stone.get('source_detection_count', len(stone.get('detection_indices', [])))}  "
                f"score={float(stone.get('score_mean', stone.get('score_max', 0.0))):.4f}  "
                f"passed={validation.get('passed', 'n/a')}"
            )
        return

    stone, rank = _find_stone(stones, args.stone_id, args.stone_rank)
    gt = CURRENT_SCENE.load_gt()

    output_dir = OUTPUT_ROOT / args.source / args.method / stone.get("stone_id", f"rank_{rank:04d}")
    dom_preview_path = output_dir / "dom_mapping.png"
    summary_path = output_dir / "summary.json"

    dom_info = _render_dom_preview(
        stone,
        detections,
        gt,
        crop_pad_m=args.bbox_pad_m,
        min_preview_side=args.min_dom_preview_side,
        output_path=dom_preview_path,
    )

    point_bbox = CURRENT_SCENE.xy_transform.world_bbox_to_point_bbox(
        stone.get("bbox_world", [0, 0, 0, 0]),
        pad_m=args.bbox_pad_m,
    )
    local_points, local_load_info = _load_local_scene_points(point_bbox, chunk_size=args.chunk_size)
    masked_points, crop_info = crop_stone_point_cloud(
        local_points,
        stone,
        detections,
        gt,
        CURRENT_SCENE.xy_transform,
        bbox_pad_m=args.bbox_pad_m,
        pc_index=None,
    )
    pc_info = _build_pc_summary(local_points, masked_points, crop_info, local_load_info)

    summary = _stone_summary(stone, rank, dom_info, pc_info)
    _write_summary(summary_path, summary)
    _print_summary(summary)
    print(f"  summary json      : {summary_path}")

    if not args.no_viewer:
        _open_pointcloud_viewer(
            local_points,
            masked_points,
            stone=stone,
            point_size=args.point_size,
            max_context_points=args.max_context_points,
            seed=args.seed + rank,
        )


if __name__ == "__main__":
    main()

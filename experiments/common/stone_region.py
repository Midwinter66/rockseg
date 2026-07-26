from __future__ import annotations

import numpy as np

from experiments.common.pointcloud_index import PointCloudXYGridIndex
from experiments.common.scene_reference import XYCoordinateTransform


def pixel_to_world(
    gt: tuple[float, float, float, float, float, float],
    px: float,
    py: float,
) -> tuple[float, float]:
    return (
        float(gt[0] + px * gt[1] + py * gt[2]),
        float(gt[3] + px * gt[4] + py * gt[5]),
    )


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    mask = np.zeros(h * w, dtype=np.uint8)
    counts = [int(v) for v in rle.get("counts", [])]
    starts_with = rle.get("starts_with")
    if starts_with is None:
        starts_with = 0 if sum(counts[1::2]) <= sum(counts[0::2]) else 1
    pos = 0
    for i, count in enumerate(counts):
        if (int(starts_with) + i) % 2 == 1:
            mask[pos : pos + count] = 255
        pos += count
    return mask.reshape(h, w)


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    fg = mask > 0
    if not fg.any():
        return fg

    interior = fg.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(fg)
            src_y0 = max(0, -dy)
            src_y1 = fg.shape[0] - max(0, dy)
            src_x0 = max(0, -dx)
            src_x1 = fg.shape[1] - max(0, dx)
            dst_y0 = max(0, dy)
            dst_y1 = fg.shape[0] - max(0, -dy)
            dst_x0 = max(0, dx)
            dst_x1 = fg.shape[1] - max(0, -dx)
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = fg[src_y0:src_y1, src_x0:src_x1]
            interior &= shifted
    return fg & ~interior


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)

    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        return np.unique(hull, axis=0)
    return hull


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    if len(points) == 0 or len(polygon) < 3:
        return np.zeros(len(points), dtype=bool)

    x = points[:, 0]
    y = points[:, 1]
    px = polygon[:, 0]
    py = polygon[:, 1]
    x2 = np.roll(px, -1)
    y2 = np.roll(py, -1)

    inside = np.zeros(len(points), dtype=bool)
    eps = 1e-12
    for i in range(len(polygon)):
        y_cond = ((py[i] > y) != (y2[i] > y))
        x_intersect = (x2[i] - px[i]) * (y - py[i]) / (y2[i] - py[i] + eps) + px[i]
        inside ^= y_cond & (x <= x_intersect)
    return inside


def mask_to_laz_polygon(
    mask: np.ndarray,
    pixel_origin: list[int],
    gt: tuple[float, float, float, float, float, float],
    xy_transform: XYCoordinateTransform,
) -> np.ndarray | None:
    boundary = _mask_boundary(mask)
    coords = np.argwhere(boundary)
    if len(coords) == 0:
        return None

    hull = _convex_hull_2d(coords[:, ::-1].astype(np.float64))
    if len(hull) < 3:
        return None

    ox, oy = pixel_origin
    polygon = []
    for px, py in hull:
        wx, wy = pixel_to_world(gt, px + ox, py + oy)
        px_pc, py_pc = xy_transform.world_to_point_xy(wx, wy)
        polygon.append([px_pc, py_pc])

    if len(polygon) < 3:
        return None
    return np.asarray([polygon], dtype=np.float32)


def crop_stone_point_cloud(
    pc: np.ndarray,
    stone: dict,
    detections: list[dict],
    gt: tuple[float, float, float, float, float, float],
    xy_transform: XYCoordinateTransform,
    bbox_pad_m: float = 0.5,
    pc_index: PointCloudXYGridIndex | None = None,
) -> tuple[np.ndarray, dict]:
    polygon_list = []
    for idx in stone.get("detection_indices", []):
        if idx >= len(detections):
            continue
        det = detections[idx]
        mask = rle_decode(det["rle_mask"])
        poly = mask_to_laz_polygon(mask, det["pixel_origin"], gt, xy_transform)
        if poly is not None:
            polygon_list.append(poly)

    bbox = stone["bbox_world"]
    x0, y0, x1, y1 = xy_transform.world_bbox_to_point_bbox(bbox, pad_m=bbox_pad_m)

    if pc_index is not None:
        candidate_indices = pc_index.query_bbox_indices(x0, y0, x1, y1)
        candidates = pc[candidate_indices].copy() if len(candidate_indices) > 0 else np.empty((0, 3), dtype=pc.dtype)
        index_mode = "xy_grid"
    else:
        bbox_mask = (
            (pc[:, 0] >= x0)
            & (pc[:, 0] <= x1)
            & (pc[:, 1] >= y0)
            & (pc[:, 1] <= y1)
        )
        candidates = pc[bbox_mask].copy()
        index_mode = "full_scan"

    crop_info = {
        "bbox_candidate_count": int(len(candidates)),
        "polygon_count": int(len(polygon_list)),
        "bbox_pad_m": float(bbox_pad_m),
        "query_mode": index_mode,
    }

    if len(candidates) == 0 or not polygon_list:
        return np.empty((0, 3), dtype=np.float32), crop_info

    keep = np.zeros(len(candidates), dtype=bool)
    for poly in polygon_list:
        inside = _points_in_polygon(candidates[:, :2], poly[0])
        keep |= inside

    stone_pts = candidates[keep]
    crop_info["kept_point_count"] = int(len(stone_pts))
    return stone_pts, crop_info

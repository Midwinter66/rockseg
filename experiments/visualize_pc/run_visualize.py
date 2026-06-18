"""
全景点云可视化 — 非石头区域灰色，每个石头不同颜色循环显示

用法:
  cd D:\github_project\image_segment\DOM_Space_message_val

  # 显示全景点云 + 所有石头
  python experiments/visualize_pc/run_visualize.py --overview

  # 配合过滤
  python experiments/visualize_pc/run_visualize.py --overview --min-z-range 0.3
"""

from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ── 数据路径 ────────────────────────────────────────────────────────
LAZ_PATHS = [
    PROJECT_ROOT / "data" / "pointcloud3" / "Data" / "BlockB.laz",
    PROJECT_ROOT / "data" / "pointcloud3" / "Data" / "BlockY.laz",
]
TFW_PATH = PROJECT_ROOT / "data" / "dom3" / "DOM.tfw"
FUSION_ROOT = PROJECT_ROOT / "experiments" / "fusion" / "outputs"
DETECTION_ROOT = PROJECT_ROOT / "experiments" / "detection" / "outputs"

SOURCES = ["sahi", "quadtree_dom"]
METHODS = ["heuristic", "correlation_clustering"]

# LAZ 偏移（DOM → LAZ 逆变换用）
LAZ_OFFSET_X = 623499.106100
LAZ_OFFSET_Y = 4678587.301000

# 颜色循环（12 种容易区分的颜色）
COLOR_CYCLE = [
    [1.0, 0.0, 0.0],  # 红
    [0.0, 0.6, 0.0],  # 绿
    [0.0, 0.0, 1.0],  # 蓝
    [1.0, 0.7, 0.0],  # 橙
    [0.6, 0.0, 1.0],  # 紫
    [0.0, 0.8, 0.8],  # 青
    [1.0, 0.0, 0.6],  # 粉
    [0.6, 0.4, 0.0],  # 棕
    [0.0, 0.5, 0.3],  # 深绿
    [0.8, 0.2, 0.2],  # 暗红
    [0.3, 0.3, 0.7],  # 灰蓝
    [0.7, 0.5, 0.3],  # 土黄
]

_CACHED_PC: np.ndarray | None = None


# ══════════════════════════════════════════════════════════════════════
#  工具
# ══════════════════════════════════════════════════════════════════════

def _parse_tfw(path: Path) -> tuple:
    lines = [float(l.strip()) for l in path.read_text("utf-8").splitlines() if l.strip()]
    return (lines[4], lines[0], lines[2], lines[5], lines[1], lines[3])


def _pixel_to_world(gt: tuple, px: float, py: float) -> tuple[float, float]:
    return float(gt[0] + px * gt[1] + py * gt[2]), float(gt[3] + px * gt[4] + py * gt[5])


def _rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, count in enumerate(rle["counts"]):
        if i % 2 == 1:
            mask[pos:pos + count] = 255
        pos += count
    return mask.reshape(h, w)


def _mask_to_laz_polygon(mask: np.ndarray, pixel_origin: list[int],
                          gt: tuple) -> np.ndarray | None:
    """将检测 mask 转为 LAZ 局部坐标多边形"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None

    eps = max(1.0, 0.01 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, eps, True)

    ox, oy = pixel_origin
    poly = []
    for px, py in approx.reshape(-1, 2):
        wx, wy = _pixel_to_world(gt, px + ox, py + oy)
        poly.append([wx - LAZ_OFFSET_X, wy - LAZ_OFFSET_Y])  # DOM → LAZ
    return np.array([poly], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════
#  加载
# ══════════════════════════════════════════════════════════════════════

def _load_point_cloud() -> np.ndarray:
    """加载 LAZ，不偏移（保持原始局部坐标）"""
    global _CACHED_PC
    if _CACHED_PC is not None:
        return _CACHED_PC
    import laspy
    all_pts = []
    for path in LAZ_PATHS:
        if not path.exists():
            print(f"  [WARN] 点云不存在: {path}")
            continue
        las = laspy.read(str(path))
        pts = np.column_stack([las.x, las.y, las.z]).astype(np.float32)
        all_pts.append(pts)
        print(f"  Loaded {path.name}: {len(pts)} points")
    _CACHED_PC = np.vstack(all_pts)
    print(f"  点云总计: {len(_CACHED_PC)} points (原始 LAZ 坐标)")
    return _CACHED_PC


def _load_data(source: str, method: str) -> tuple[list[dict], list[dict]]:
    fusion_path = FUSION_ROOT / source / method / "fusion_stats.json"
    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    stones = fusion["stones"]
    det_path = DETECTION_ROOT / source / "detections.json"
    detections = json.loads(det_path.read_text(encoding="utf-8"))
    print(f"  融合: {len(stones)} stones, 检测: {len(detections)} detections")
    return stones, detections


# ══════════════════════════════════════════════════════════════════════
#  主流程：为每个点着色
# ══════════════════════════════════════════════════════════════════════

def _overview(pc: np.ndarray, stones: list[dict], detections: list[dict],
              gt: tuple, min_z_range: float = 0.0) -> None:
    import open3d as o3d

    # 降采样到可显示的数量
    target = 3_000_000
    step = max(1, len(pc) // target)
    pc_sub = pc[::step].copy()
    print(f"  降采样: {len(pc)} → {len(pc_sub)} (1/{step})")

    # 初始化颜色：默认灰色
    colors = np.full((len(pc_sub), 3), 0.6, dtype=np.float32)

    # 处理每个石头
    stone_polys = []
    for s in stones:
        if s["source_detection_count"] == 0:
            continue
        # 收集该石头的所有检测的多边形
        poly_list = []
        for idx in s.get("detection_indices", []):
            if idx >= len(detections):
                continue
            det = detections[idx]
            mask = _rle_decode(det["rle_mask"])
            poly = _mask_to_laz_polygon(mask, det["pixel_origin"], gt)
            if poly is not None:
                poly_list.append(poly)
        if poly_list:
            stone_polys.append((s, poly_list))

    print(f"  处理 {len(stone_polys)} 个石头...")

    # 用 bbox 粗筛 + 多边形精筛
    stone_id_map = {}  # point_index → stone_id
    for si, (s, polys) in enumerate(stone_polys):
        b = s["bbox_world"]
        # bbox 转 LAZ 坐标
        bx0 = b[0] - LAZ_OFFSET_X - 0.5
        by0 = b[1] - LAZ_OFFSET_Y - 0.5
        bx1 = b[2] - LAZ_OFFSET_X + 0.5
        by1 = b[3] - LAZ_OFFSET_Y + 0.5

        mask_bbox = ((pc_sub[:, 0] >= bx0) & (pc_sub[:, 0] <= bx1) &
                     (pc_sub[:, 1] >= by0) & (pc_sub[:, 1] <= by1))
        candidates_idx = np.where(mask_bbox)[0]
        if len(candidates_idx) == 0:
            continue

        candidates = pc_sub[candidates_idx]
        keep = np.zeros(len(candidates), dtype=bool)
        for poly in polys:
            inside = np.array([
                cv2.pointPolygonTest(poly, (p[0], p[1]), False) >= 0
                for p in candidates
            ])
            keep |= inside

        matched = candidates_idx[keep]
        for pi in matched:
            if pi not in stone_id_map:  # 先到先得
                stone_id_map[pi] = si

    # 着色
    for pi, si in stone_id_map.items():
        colors[pi] = COLOR_CYCLE[si % len(COLOR_CYCLE)]

    colored = len(stone_id_map)
    print(f"  已着色: {colored} 点（属于 {len(stone_polys)} 个石头）")
    print(f"  灰色区域: {len(pc_sub) - colored} 点（非石头区）")

    # 显示
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_sub)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"全景点云 — {colored} 个彩色点属于石头",
        width=1280, height=800,
    )


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="全景点云 + 石头着色")
    parser.add_argument("--source", choices=SOURCES, default="sahi")
    parser.add_argument("--method", choices=METHODS, default="correlation_clustering")
    parser.add_argument("--min-z-range", type=float, default=0.0,
                        help="最小高度差过滤 (默认 0=不过滤)")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  全景点云 — 石头区域彩色显示")
    print(f"  {args.source}/{args.method}")
    print(f"{'='*55}\n")

    gt = _parse_tfw(TFW_PATH)
    pc = _load_point_cloud()
    stones, detections = _load_data(args.source, args.method)
    _overview(pc, stones, detections, gt, args.min_z_range)


if __name__ == "__main__":
    main()

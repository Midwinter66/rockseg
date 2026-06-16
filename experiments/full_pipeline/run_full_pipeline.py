"""
完整端到端管线 — 切片 → 检测 → 融合 → 3D提取 → 体积计算 → 统计图表

用法:
  cd D:\github_project\image_segment\DOM_Space_message_val
  python experiments/full_pipeline/run_full_pipeline.py

可选参数:
  --config PATH        自定义配置
  --run-id ID          运行标识
  --method METHOD      切片方式 (quadtree_dom / sahi_dense)
  --limit N            限制检测的 tile 数(快速测试)
  --skip-slicing       跳过切片阶段
  --skip-detection     跳过检测阶段
  --skip-fusion        跳过融合阶段
  --skip-3d            跳过 3D 提取(仅统计分析检测结果)
  --stone-id ID        指定某个石块打开 3D 查看器
  --volume-method M    体积计算方法 (alpha_shape/convex_hull/grid_2d5)
  --open3d             运行结束后打开 Open3D 窗口查看石块
"""

from __future__ import annotations
import argparse, json, sys, time, math, base64
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "v2" / "src"))
print(f"Project root: {PROJECT_ROOT}")

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
Image.MAX_IMAGE_PIXELS = 500_000_000

# ── 导入实验模块 ─────────────────────────────────────────────────
from experiments.full_pipeline.volume import compute_volume
from experiments.full_pipeline.stats import generate_all_charts, compute_statistics
from experiments.full_pipeline.visualize_3d import visualize_stone


# ══════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════

def _parse_tfw(path: Path) -> tuple:
    lines = [float(l.strip()) for l in path.read_text("utf-8").splitlines() if l.strip()]
    return (lines[4], lines[0], lines[2], lines[5], lines[1], lines[3])

def _pixel_to_world(gt, px, py):
    return float(gt[0] + px * gt[1] + py * gt[2]), float(gt[3] + px * gt[4] + py * gt[5])

def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def _ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _bbox_intersects(a, b, pad=0.0):
    return not (a[2]+pad<b[0]-pad or b[2]+pad<a[0]-pad or a[3]+pad<b[1]-pad or b[3]+pad<a[1]-pad)

def _bbox_iou(a, b):
    ix0, iy0 = max(a[0],b[0]), max(a[1],b[1])
    ix1, iy1 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0.0, ix1-ix0)*max(0.0, iy1-iy0)
    if inter == 0: return 0.0
    aa = (a[2]-a[0])*(a[3]-a[1])
    ab = (b[2]-b[0])*(b[3]-b[1])
    return inter/(aa+ab-inter)


# ══════════════════════════════════════════════════════════════════
#  阶段 1: 切片
# ══════════════════════════════════════════════════════════════════

def stage_slicing(config: dict, run_root: Path, method: str,
                  dom_bgr: np.ndarray, gt: tuple | None,
                  pc_points: np.ndarray | None = None) -> list[dict]:
    """根据配置切片，返回 tile 列表"""
    print(f"\n{'='*60}")
    print(f"  Stage 1: Slicing ({method})")
    print(f"{'='*60}")

    dom_bounds = _get_dom_bounds(gt, dom_bgr.shape[1], dom_bgr.shape[0])

    if method == "sahi_dense":
        sc = config["slicing"]["sahi_dense"] if "sahi_dense" in config["slicing"] else config["slicing"].get("cover", {})
        # 使用 SAHI dense 默认参数
        pc = config["slicing"].get("sahi_dense", {
            "patch_size": 1024, "overlap": 0.35,
            "black_pixel_threshold": 5, "min_content_ratio": 0.25,
            "include_edge_patches": True,
        })
        return _sahi_slice(dom_bgr, gt, pc)
    else:
        # quadtree_dom
        cc = config["slicing"]["cover"]
        return _quadtree_dom_slice(dom_bgr, dom_bounds, cc)


def _get_dom_bounds(gt, w, h):
    if gt is not None:
        xmin, ymax = _pixel_to_world(gt, 0, 0)
        xmax, ymin = _pixel_to_world(gt, w, h)
        return [min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)]
    return [0.0, 0.0, float(w), float(h)]


def _build_positions(limit, size, stride, include_edge=True):
    if limit <= size:
        return [0]
    pos = list(range(0, limit - size + 1, stride))
    if include_edge and pos[-1] != limit - size:
        pos.append(limit - size)
    return sorted(set(pos))


def _compute_content_ratio(patch, black_th):
    if patch.ndim == 3:
        gray = patch.mean(axis=2)
    else:
        gray = patch
    return float(np.count_nonzero(gray > black_th) / max(gray.size, 1))


def _sahi_slice(dom_bgr, gt, pc):
    """SAHI 固定滑窗切片"""
    t0 = time.perf_counter()
    h, w = dom_bgr.shape[:2]
    patch_sz = int(pc.get("patch_size", 1024))
    overlap = float(pc.get("overlap", 0.35))
    stride = max(1, int(round(patch_sz * (1 - overlap))))
    black_th = int(pc.get("black_pixel_threshold", 5))
    min_content = float(pc.get("min_content_ratio", 0.25))
    include_edge = bool(pc.get("include_edge_patches", True))

    xs = _build_positions(w, patch_sz, stride, include_edge)
    ys = _build_positions(h, patch_sz, stride, include_edge)

    tiles = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            patch = dom_bgr[y:y+patch_sz, x:x+patch_sz]
            cr = _compute_content_ratio(patch, black_th)
            kept = cr >= min_content
            wx0, wy0 = _pixel_to_world(gt, x, y)
            wx1, wy1 = _pixel_to_world(gt, x+patch_sz, y+patch_sz)
            tiles.append({
                "tile_id": f"patch_{len(tiles):06d}",
                "pixel_origin": [x, y],
                "pixel_size": patch_sz,
                "bounds_m": [min(wx0,wx1), min(wy0,wy1), max(wx0,wx1), max(wy0,wy1)],
                "content_ratio": round(cr, 4),
                "skipped": not kept,
            })

    elapsed = time.perf_counter() - t0
    kept = [t for t in tiles if not t["skipped"]]
    print(f"  SAHI: {len(kept)}/{len(tiles)} tiles kept in {elapsed:.1f}s")
    return tiles


def _quadtree_dom_slice(dom_bgr, dom_bounds, cc):
    """四叉树 DOM 纹理切片（内联实现，不依赖实验模块）"""
    t0 = time.perf_counter()
    from experiments.slicing.run_slicing_experiment import DOMTextureQuadTree
    qt = DOMTextureQuadTree(
        dom_bounds,
        float(cc["base_tile_size_m"]),
        float(cc["min_tile_size_m"]),
        float(cc["max_tile_size_m"]),
    )
    tiles = qt.generate(
        dom_bgr,
        float(cc["min_edge_density"]),
        int(cc.get("canny_low", 30)),
        int(cc.get("canny_high", 90)),
        black_threshold=int(cc.get("black_pixel_threshold", 5)),
        min_content_ratio=float(cc.get("min_content_ratio", 0.35)),
    )
    elapsed = time.perf_counter() - t0
    kept = [t for t in tiles if not t.get("skipped", False)]
    print(f"  Quadtree-DOM: {len(kept)}/{len(tiles)} tiles in {elapsed:.1f}s")
    return tiles


# ══════════════════════════════════════════════════════════════════
#  阶段 2: 检测
# ══════════════════════════════════════════════════════════════════

def stage_detection(config: dict, run_root: Path, tiles: list[dict],
                    dom_bgr: np.ndarray, gt: tuple) -> list[dict]:
    """对每个 tile 运行 YOLO 推理"""
    print(f"\n{'='*60}")
    print(f"  Stage 2: YOLO Detection")
    print(f"{'='*60}")
    from ultralytics import YOLO

    ic = config["inference"]
    model_path = Path(config["paths"]["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path.resolve()))
    # 强制关闭 retina_masks
    if hasattr(model, "predictor") and model.predictor is not None:
        model.predictor.args.retina_masks = False

    kept_tiles = [t for t in tiles if not t.get("skipped", False)]
    print(f"  Model: {model_path.name}, tiles: {len(kept_tiles)}")

    all_dets = []
    for tile in kept_tiles:
        if "pixel_origin" in tile:
            # SAHI 风格
            x, y = tile["pixel_origin"]
            sz = tile["pixel_size"]
            crop = dom_bgr[y:y+sz, x:x+sz]
            offset_x, offset_y = x, y
        else:
            # 四叉树风格: 世界坐标 → 像素
            b = tile["bounds_m"]
            gb = _get_dom_bounds(gt, dom_bgr.shape[1], dom_bgr.shape[0])
            gw, gh = gb[2]-gb[0], gb[3]-gb[1]
            px0 = int((b[0]-gb[0])/gw * dom_bgr.shape[1])
            px1 = int((b[2]-gb[0])/gw * dom_bgr.shape[1])
            py0 = int((gb[3]-b[3])/gh * dom_bgr.shape[0])
            py1 = int((gb[3]-b[1])/gh * dom_bgr.shape[0])
            px0, px1 = sorted([max(0,px0), min(dom_bgr.shape[1],px1)])
            py0, py1 = sorted([max(0,py0), min(dom_bgr.shape[0],py1)])
            if (px1-px0) < 10 or (py1-py0) < 10:
                continue
            crop = dom_bgr[py0:py1, px0:px1]
            offset_x, offset_y = px0, py0

        dets = _infer_crop(crop, model, ic, gt, offset_x, offset_y)
        for d in dets:
            d["source_tile_id"] = tile.get("tile_id", "")
        all_dets.extend(dets)

    print(f"  Total detections: {len(all_dets)}")

    # 保存检测结果
    det_path = run_root / "detections.json"
    _write_json(det_path, {
        "count": len(all_dets),
        "detections": all_dets,
    })
    print(f"  Saved: {det_path}")
    return all_dets


def _infer_crop(crop, model, ic, gt, offset_x, offset_y):
    """在 crop 上运行 YOLO 并返回检测列表"""
    # 大图拆子块
    h, w = crop.shape[:2]
    MAX_DIRECT = 1280
    if max(w, h) <= MAX_DIRECT:
        crops = [(0, 0, w, h)]
    else:
        SUB = 1024
        OLAP = 64
        stride = SUB - OLAP
        crops = []
        for y0 in range(0, max(1, h - SUB + 1), stride):
            for x0 in range(0, max(1, w - SUB + 1), stride):
                y1 = min(y0 + SUB, h)
                x1 = min(x0 + SUB, w)
                if (y1-y0) >= 256 and (x1-x0) >= 256:
                    crops.append((x0, y0, x1, y1))

    detections = []
    res_m = None
    for (x0, y0, x1, y1) in crops:
        sub = crop[y0:y1, x0:x1]
        try:
            res = model.predict(sub, imgsz=ic["imgsz"], conf=ic["conf"],
                               max_det=ic.get("max_det", 1000),
                               retina_masks=False, verbose=False)[0]
        except Exception as e:
            continue
        if res.masks is None or len(res.masks.data) == 0:
            continue
        res_m = res

        resolution = abs(gt[1])
        min_diam = float(ic.get("min_stone_diameter_m", 0))

        for idx, mask_t in enumerate(res.masks.data):
            mask = (mask_t.cpu().numpy() * 255).astype(np.uint8)
            if mask.shape[:2] != sub.shape[:2]:
                mask = cv2.resize(mask, (sub.shape[1], sub.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

            area_px = int(np.count_nonzero(mask > 0))
            if area_px == 0:
                continue
            area_m2 = float(area_px * resolution**2)
            eq_diam = float(math.sqrt(4 * area_m2 / math.pi))
            if eq_diam < min_diam:
                continue

            moments = cv2.moments(mask)
            if moments["m00"] == 0:
                continue
            cx = int(moments["m10"] / moments["m00"]) + x0 + offset_x
            cy = int(moments["m01"] / moments["m00"]) + y0 + offset_y
            wx, wy = _pixel_to_world(gt, cx, cy)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 3:
                continue
            eps = max(1.0, 0.01 * cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, eps, True)
            poly = [[float(px + x0 + offset_x), float(py + y0 + offset_y)]
                    for px, py in approx.reshape(-1, 2)]

            bx, by, bw, bh = cv2.boundingRect(mask)
            gbx0 = bx + x0 + offset_x
            gby0 = by + y0 + offset_y
            gbx1 = gbx0 + bw
            gby1 = gby0 + bh
            w0, w1 = _pixel_to_world(gt, gbx0, gby0)
            w2, w3 = _pixel_to_world(gt, gbx1, gby1)

            score = float(res.boxes.conf[idx].item()) if res.boxes and len(res.boxes) > idx else 0.0

            detections.append({
                "detection_id": f"det_{len(detections):06d}",
                "score": round(score, 4),
                "area_m2": round(area_m2, 4),
                "equivalent_diameter_m": round(eq_diam, 4),
                "centroid_world": [round(wx, 4), round(wy, 4)],
                "bbox_world": [round(min(w0,w2),4), round(min(w1,w3),4),
                              round(max(w0,w2),4), round(max(w1,w3),4)],
                "polygon_pixel": poly,
            })

    return detections


# ══════════════════════════════════════════════════════════════════
#  阶段 3: 融合 (相关性聚类)
# ══════════════════════════════════════════════════════════════════

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False
        if self.rank[ra] < self.rank[rb]: ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]: self.rank[ra] += 1
        return True
    def groups(self):
        g = defaultdict(list)
        for i in range(len(self.parent)):
            g[self.find(i)].append(i)
        return g


def stage_fusion(config: dict, run_root: Path,
                 detections: list[dict]) -> list[dict]:
    """相关性聚类融合"""
    print(f"\n{'='*60}")
    print(f"  Stage 3: Fusion (correlation clustering)")
    print(f"{'='*60}")
    t0 = time.perf_counter()

    fc = config["fusion"]["correlation"]
    sigma = float(fc["distance_sigma"])
    pos_th = float(fc["positive_weight_threshold"])
    iou_w = float(fc.get("iou_weight", 0.3))
    tile_boost = float(fc.get("same_tile_boost", 1.5))
    max_dist = float(fc.get("max_distance_m", 3.0))
    require_bbox = bool(fc.get("require_bbox_intersect", True))

    n = len(detections)
    if n == 0:
        return []
    uf = UnionFind(n)

    for i in range(n):
        c1 = detections[i].get("centroid_world", [0,0])
        bbox1 = detections[i].get("bbox_world", [0,0,0,0])
        for j in range(i+1, n):
            c2 = detections[j].get("centroid_world", [0,0])
            dist = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
            if dist > max_dist:
                continue
            if require_bbox:
                bbox2 = detections[j].get("bbox_world", [0,0,0,0])
                if not _bbox_intersects(bbox1, bbox2):
                    continue
            w_dist = math.exp(-dist**2 / (2*sigma**2))
            iou = 0.0
            if iou_w > 0:
                bbox2 = detections[j].get("bbox_world", [0,0,0,0])
                if _bbox_intersects(bbox1, bbox2):
                    iou = _bbox_iou(bbox1, bbox2)
            boost = tile_boost if detections[i].get("source_tile_id") == detections[j].get("source_tile_id") else 1.0
            weight = w_dist * (1 + iou_w * iou) * boost
            if weight >= pos_th:
                uf.union(i, j)

    groups = list(uf.groups().values())
    stones = []
    for gi, indices in enumerate(groups):
        members = [detections[i] for i in indices]
        stone_id = f"stone_{gi:06d}"
        bboxes = [m["bbox_world"] for m in members]
        centroids = [m["centroid_world"] for m in members]

        stone = {
            "stone_id": stone_id,
            "source_detection_ids": [m["detection_id"] for m in members],
            "source_tile_ids": list({m.get("source_tile_id", "") for m in members}),
            "source_detection_count": len(members),
            "centroid_world": [
                round(sum(c[0] for c in centroids)/len(centroids), 4),
                round(sum(c[1] for c in centroids)/len(centroids), 4),
            ],
            "bbox_world": [
                round(min(b[0] for b in bboxes), 4),
                round(min(b[1] for b in bboxes), 4),
                round(max(b[2] for b in bboxes), 4),
                round(max(b[3] for b in bboxes), 4),
            ],
            "area_m2": round(sum(m["area_m2"] for m in members), 4),
            "equivalent_diameter_m": round(
                math.sqrt(4 * sum(m["area_m2"] for m in members) / math.pi), 4),
            "score_avg": round(sum(m["score"] for m in members)/len(members), 4),
            "score_max": round(max(m["score"] for m in members), 4),
            "polygon_pixel": members[0].get("polygon_pixel", []),
            "detections": members,
        }
        stones.append(stone)

    elapsed = time.perf_counter() - t0
    print(f"  {len(detections)} detections → {len(stones)} stones in {elapsed:.2f}s")
    merge_rate = (1 - len(stones)/max(len(detections),1))*100
    print(f"  Merge rate: {merge_rate:.1f}%")

    # 保存
    fus_path = run_root / "stones.json"
    _write_json(fus_path, {
        "input_detections": len(detections),
        "output_stones": len(stones),
        "merge_rate": round(merge_rate, 2),
        "stones": stones,
    })
    print(f"  Saved: {fus_path}")
    return stones


# ══════════════════════════════════════════════════════════════════
#  阶段 4: 3D 提取 + 体积计算
# ══════════════════════════════════════════════════════════════════

def stage_extract_3d(config: dict, run_root: Path, stones: list[dict],
                     pc_points: np.ndarray, gt: tuple,
                     dom_w: int, dom_h: int) -> list[dict]:
    """从点云中提取每个石块的 3D 点，计算体积"""
    print(f"\n{'='*60}")
    print(f"  Stage 4: 3D Extraction + Volume Calculation")
    print(f"{'='*60}")
    t0 = time.perf_counter()

    voxel_size = float(config["extraction"]["voxel_size_m"])
    vol_method = config["volume"]["method"]
    vol_alpha = float(config["volume"].get("alpha", 0.5))
    vol_grid_res = float(config["volume"].get("grid_resolution", 0.1))
    dom_bounds = _get_dom_bounds(gt, dom_w, dom_h)

    stone_dir = _ensure_dir(run_root / "stones")
    resolution = abs(gt[1])

    updated = []
    vol_data = []

    for idx, s in enumerate(stones):
        if idx % 50 == 0 and idx > 0:
            print(f"  Progress: {idx}/{len(stones)}")

        # 用第一个检测的 polygon 提取 3D 点
        # 简化为用 bbox 裁剪 + polygon 过滤
        bx0, by0, bx1, by1 = s["bbox_world"]
        # 在点云中裁剪候选区域
        margin = float(config["extraction"]["padding_m"])
        mask = (
            (pc_points[:, 0] >= bx0 - margin) &
            (pc_points[:, 0] <= bx1 + margin) &
            (pc_points[:, 1] >= by0 - margin) &
            (pc_points[:, 1] <= by1 + margin)
        )
        candidates = pc_points[mask]
        if len(candidates) < 4:
            continue

        # 世界坐标 → 像素坐标
        origin_x, origin_y = dom_bounds[0], dom_bounds[3]
        pxs = np.round((candidates[:, 0] - origin_x) / resolution).astype(np.int32)
        pys = np.round((origin_y - candidates[:, 1]) / resolution).astype(np.int32)

        # 收集所有关联检测的多边形，点落在任何一个多边形内即保留
        keep_any = np.zeros(len(candidates), dtype=bool)
        for det in s["detections"]:
            poly = det.get("polygon_pixel", [])
            if len(poly) < 3:
                continue
            contour = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            valid = (pxs >= 0) & (pxs < dom_w) & (pys >= 0) & (pys < dom_h)
            if not np.any(valid):
                continue
            for vi in np.where(valid)[0]:
                if cv2.pointPolygonTest(contour, (int(pxs[vi]), int(pys[vi])), False) >= 0:
                    keep_any[vi] = True

        stone_pts = candidates[keep_any]
        if len(stone_pts) < 4:
            continue

        # 滤除地面点：2D mask 包含了石块下方的地面区域，
        # 去掉底部 10% 的 Z 值点（地面），只保留石块本体
        z_vals = stone_pts[:, 2]
        z_cutoff = np.percentile(z_vals, 10)
        stone_pts = stone_pts[z_vals > z_cutoff]
        if len(stone_pts) < 4:
            continue

        # 降采样
        if voxel_size > 0 and len(stone_pts) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(stone_pts)
            pcd = pcd.voxel_down_sample(voxel_size)
            stone_pts = np.asarray(pcd.points, dtype=np.float32)

        # 保存点云
        ply_path = stone_dir / f"{s['stone_id']}.ply"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(stone_pts)
        pcd.paint_uniform_color([0.72, 0.72, 0.72])
        o3d.io.write_point_cloud(str(ply_path), pcd)

        # 计算体积
        vol_result = compute_volume(stone_pts, method=vol_method, alpha=vol_alpha,
                                     voxel_size=voxel_size,
                                     grid_resolution=vol_grid_res)

        # 更新石块信息
        s["point_count"] = len(stone_pts)
        s["pointcloud_path"] = str(ply_path)
        s["volume_m3"] = vol_result["volume_m3"]
        s["surface_area_m2"] = vol_result.get("surface_area_m2", 0.0)
        s["volume_method"] = vol_result["method"]
        s["bbox_3d"] = [
            float(stone_pts[:, 0].min()), float(stone_pts[:, 1].min()), float(stone_pts[:, 2].min()),
            float(stone_pts[:, 0].max()), float(stone_pts[:, 1].max()), float(stone_pts[:, 2].max()),
        ]

        vol_data.append(vol_result)
        updated.append(s)

        # 保存单石块 JSON
        _write_json(ply_path.with_suffix(".json"), s)

    elapsed = time.perf_counter() - t0
    n_vol = sum(1 for v in vol_data if v["volume_m3"] > 0)
    total_vol = sum(v["volume_m3"] for v in vol_data)
    print(f"  3D extracted: {len(updated)}/{len(stones)} stones with point clouds")
    print(f"  Volume calculated: {n_vol} stones, total {total_vol:.3f}m³")
    print(f"  Time: {elapsed:.1f}s")

    return updated


# ══════════════════════════════════════════════════════════════════
#  阶段 5: 统计 + 图表
# ══════════════════════════════════════════════════════════════════

def stage_statistics(config: dict, run_root: Path, stones: list[dict]) -> dict:
    """生成统计报告和图表"""
    print(f"\n{'='*60}")
    print(f"  Stage 5: Statistics & Charts")
    print(f"{'='*60}")

    chart_dir = _ensure_dir(run_root / "charts")

    # 生成图表
    generate_all_charts(stones, chart_dir, config)

    # 计算统计摘要
    summary = compute_statistics(stones)

    # 写入报告
    report_path = run_root / "report.json"
    report = {
        "run_id": run_root.name,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "statistics": summary,
        "stones": stones,
    }
    _write_json(report_path, report)

    # 文本摘要
    print(f"\n  {'='*50}")
    print(f"  📊 统计摘要")
    print(f"  {'='*50}")
    d = summary["diameter"]
    v = summary["volume"]
    print(f"  石块总数: {summary['total_stones']}")
    print(f"  直径: {d['min_m']:.2f} ~ {d['max_m']:.2f} m  (均值 {d['mean_m']:.2f} m)")
    print(f"  体积: {v['min_m3']:.4f} ~ {v['max_m3']:.4f} m³  (均值 {v['mean_m3']:.4f} m³)")
    print(f"  总体积: {v['total_m3']:.3f} m³")
    print(f"  总面积: {summary['area']['total_m2']:.1f} m²")
    print(f"  {'='*50}")
    for bucket, info in summary["size_buckets"].items():
        bar = "█" * int(info["pct"] / 2)
        print(f"  {bucket}: {info['count']:4d} ({info['pct']:.1f}%) {bar}")

    print(f"\n  Charts saved to: {chart_dir}")
    print(f"  Report saved to: {report_path}")
    return summary


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Full Pipeline: Slice → Detect → Fuse → 3D → Stats")
    parser.add_argument("--config", default=None, help="Config path")
    parser.add_argument("--run-id", default=None, help="Run ID")
    parser.add_argument("--method", default=None, choices=["quadtree_dom", "sahi_dense"],
                        help="Slicing method override")
    parser.add_argument("--limit", type=int, default=None, help="Limit tiles for quick test")
    parser.add_argument("--skip-slicing", action="store_true")
    parser.add_argument("--skip-detection", action="store_true")
    parser.add_argument("--skip-fusion", action="store_true")
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument("--stone-id", default=None, help="Open 3D viewer for a specific stone")
    parser.add_argument("--volume-method", default=None,
                        choices=["alpha_shape", "convex_hull", "grid_2d5"])
    parser.add_argument("--open3d", action="store_true", help="Open 3D viewer after pipeline")
    args = parser.parse_args()

    # ── 加载配置 ──
    config_path = args.config or str(Path(__file__).parent / "config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    if args.method:
        config["slicing"]["method"] = args.method
    if args.volume_method:
        config["volume"]["method"] = args.volume_method

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    workspace = Path(config["paths"]["workspace_dir"])
    run_root = _ensure_dir(workspace / "runs" / run_id)
    print(f"\n{'='*60}")
    print(f"  FULL PIPELINE — Run ID: {run_id}")
    print(f"  Slicing: {config['slicing']['method']}")
    print(f"  Fusion: {config['fusion']['method']} (σ={config['fusion']['correlation']['distance_sigma']})")
    print(f"  Volume: {config['volume']['method']}")
    print(f"{'='*60}")

    # ── 加载数据 ──

    dom_path = Path(config["paths"]["dom_path"])
    if not dom_path.exists():
        dom_path = PROJECT_ROOT / config["paths"]["dom_path"]
    print(f"\n  Loading DOM: {dom_path.name}")
    dom = Image.open(str(dom_path))
    dom_bgr = np.array(dom)
    if dom_bgr.ndim == 3 and dom_bgr.shape[2] == 3:
        dom_bgr = cv2.cvtColor(dom_bgr, cv2.COLOR_RGB2BGR)
    print(f"  DOM size: {dom_bgr.shape[1]}x{dom_bgr.shape[0]}")

    gt = None
    tfw_path = Path(config["paths"].get("dom_world", ""))
    if tfw_path.exists():
        gt = _parse_tfw(tfw_path)
    elif (dom_path.parent / dom_path.stem).with_suffix(".tfw").exists():
        gt = _parse_tfw((dom_path.parent / dom_path.stem).with_suffix(".tfw"))
    print(f"  TFW: {'loaded' if gt else 'NOT FOUND (using pixel coords)'}")

    pc_points = None
    pc_paths = config["paths"].get("pointcloud_paths", [])
    if pc_paths:
        all_pts = []
        for rel in pc_paths:
            p = Path(rel)
            if not p.exists():
                p = PROJECT_ROOT / rel
            if p.exists():
                pcd = o3d.io.read_point_cloud(str(p))
                pts = np.asarray(pcd.points, dtype=np.float32)
                if len(pts) > 0:
                    all_pts.append(pts)
                    print(f"  Point cloud: {p.name} ({len(pts)} points)")
        if all_pts:
            pc_points = np.vstack(all_pts)
    print(f"  Point cloud total: {len(pc_points) if pc_points is not None else 0} points")

    # ── 阶段 1: 切片 ──
    if not args.skip_slicing:
        tiles = stage_slicing(config, run_root, config["slicing"]["method"],
                              dom_bgr, gt, pc_points)
        if not tiles:
            print("  [ERROR] No tiles generated"); return
    else:
        tiles = []
        print("  [SKIP] Slicing")

    # ── 阶段 2: 检测 ──
    if not args.skip_detection:
        detections = stage_detection(config, run_root, tiles, dom_bgr, gt)
        if not detections:
            print("  [ERROR] No detections"); return
        _write_json(run_root / "detections.json", {"count": len(detections), "detections": detections})
    else:
        det_path = run_root / "detections.json"
        if det_path.exists():
            detections = _read_json(det_path)["detections"]
            print(f"  [LOAD] {len(detections)} detections from {det_path}")
        else:
            print("  [ERROR] No detection results found"); return

    # ── 阶段 3: 融合 ──
    if not args.skip_fusion:
        stones = stage_fusion(config, run_root, detections)
        if not stones:
            print("  [ERROR] No stones after fusion"); return
    else:
        st_path = run_root / "stones.json"
        if st_path.exists():
            stones = _read_json(st_path)["stones"]
            print(f"  [LOAD] {len(stones)} stones from {st_path}")
        else:
            print("  [ERROR] No fusion results found"); return

    # ── 阶段 4: 3D 提取 + 体积 ──
    if not args.skip_3d and pc_points is not None and gt is not None:
        stones = stage_extract_3d(config, run_root, stones, pc_points, gt,
                                   dom_bgr.shape[1], dom_bgr.shape[0])
    else:
        if not pc_points:
            print("  [SKIP] 3D extraction (no point cloud)")
        if not gt:
            print("  [SKIP] 3D extraction (no TFW/geo reference)")

    # ── 阶段 5: 统计 ──
    summary = stage_statistics(config, run_root, stones)

    # ── 查看指定石块 ──
    if args.stone_id:
        for s in stones:
            if s["stone_id"] == args.stone_id:
                ply_path = run_root / "stones" / f"{s['stone_id']}.ply"
                if ply_path.exists():
                    pcd = o3d.io.read_point_cloud(str(ply_path))
                    pts = np.asarray(pcd.points)
                else:
                    pts = None
                visualize_stone(s, pts, window_name=f"Stone: {args.stone_id}")
                break
        else:
            print(f"  Stone {args.stone_id} not found")

    # ── Open3D 批量查看 ──
    if args.open3d:
        stone_dir = run_root / "stones"
        if stone_dir.exists():
            stone_plys = sorted(stone_dir.glob("stone_*.ply"))
            if stone_plys:
                from experiments.full_pipeline.visualize_3d import visualize_multiple_stones
                visualize_multiple_stones(stones, stone_dir)

    print(f"\n{'='*60}")
    print(f"  ✅ Pipeline complete!")
    print(f"  Results: {run_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

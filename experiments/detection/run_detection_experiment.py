"""
检测阶段实验 — 基于切片结果运行 YOLO 推理

三种模式:
  - 单个切片方法: 读该方法的 tile manifest + 切片图, 跑 YOLO
  - 对比模式 (all): 对每个已运行过的切片方法单独跑检测, 各自输出

用法:
  cd D:\github_project\image_segment\DOM_Space_message_val

  # 先确保切片结果已存在
  python experiments/slicing/run_slicing_experiment.py --method all

  # 对单个切片方法跑检测
  python experiments/detection/run_detection_experiment.py --source sahi_baseline
  python experiments/detection/run_detection_experiment.py --source quadtree_pointcloud
  python experiments/detection/run_detection_experiment.py --source quadtree_dom

  # 全部对比
  python experiments/detection/run_detection_experiment.py --source all

  # 限制检测数量 (快速测试)
  python experiments/detection/run_detection_experiment.py --source sahi_baseline --limit 10
"""

from __future__ import annotations
import argparse, json, sys, time, math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "v2" / "src"))

import cv2, numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000

# ── 常量 ────────────────────────────────────────────────────────────
DOM_PATH = PROJECT_ROOT / "data" / "dom3" / "DOM.tif"
DOM_WORLD_PATH = PROJECT_ROOT / "data" / "dom3" / "DOM.tfw"
POINTCLOUD_DIR = PROJECT_ROOT / "data" / "pointcloud2"
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
SLICING_OUTPUTS = PROJECT_ROOT / "experiments" / "slicing" / "outputs"
SELF_DIR = Path(__file__).resolve().parent

SOURCES = ["sahi_baseline", "sahi_dense", "quadtree_pointcloud", "quadtree_dom"]


# ══════════════════════════════════════════════════════════════════════
#  工具
# ══════════════════════════════════════════════════════════════════════

def _load_detection_config() -> dict:
    path = PROJECT_ROOT / "experiments" / "configs" / "detection" / "default.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_tfw(tfw_path: Path) -> tuple:
    lines = [float(l.strip()) for l in tfw_path.read_text("utf-8").splitlines() if l.strip()]
    return (lines[4], lines[0], lines[2], lines[5], lines[1], lines[3])


def _pixel_to_world(gt: tuple, px: float, py: float) -> tuple[float, float]:
    return float(gt[0] + px * gt[1] + py * gt[2]), float(gt[3] + px * gt[4] + py * gt[5])


def _load_slicing_manifest(method: str) -> dict:
    path = SLICING_OUTPUTS / method / "tile_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Slice stats not found: {path}. Run slicing experiment first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_output_dir(method: str) -> Path:
    d = SELF_DIR / "outputs" / method
    d.mkdir(parents=True, exist_ok=True)
    return d


# ══════════════════════════════════════════════════════════════════════
#  检测核心
# ══════════════════════════════════════════════════════════════════════

def _infer_patch_sahi(patch_record: dict, model, config: dict, dom_img: np.ndarray, gt: tuple) -> dict:
    """SAHI 风格: patch 已经是切割好的子图, 直接从 DOM 取"""
    ic = config["inference"]
    x, y = patch_record["pixel_origin"]
    sz = patch_record["pixel_size"]
    crop = dom_img[y:y + sz, x:x + sz]
    return _infer_one_crop(crop, model, ic, gt, x, y)


def _infer_one_crop(crop: np.ndarray, model, ic: dict, gt: tuple,
                    offset_x: int, offset_y: int) -> dict:
    """在单张 crop 上执行 YOLO 推理并提取检测结果

    始终使用 retina_masks=False 避免 process_mask_native 在
    原生分辨率上分配巨量显存导致 OOM。
    检测结果中的像素坐标加上 offset_x/offset_y 即为全局 DOM 坐标。
    """
    # 强制关闭 retina_masks（predict kwargs 可能传不进去）
    if hasattr(model, 'predictor') and model.predictor is not None:
        model.predictor.args.retina_masks = False

    try:
        res = model.predict(crop, imgsz=ic["imgsz"], conf=ic["conf"],
                            max_det=int(ic.get("max_det", 1000)),
                            retina_masks=False,
                            verbose=False)[0]
    except RuntimeError as e:
        print(f"    [WARN] YOLO predict OOM on {crop.shape}, skipping")
        return {"detection_count": 0, "detections": []}

    detections = []
    if res.masks is None or len(res.masks.data) == 0:
        return {"detection_count": 0, "detections": []}

    h, w = crop.shape[:2]
    resolution = abs(gt[1])
    min_diam = float(ic.get("min_stone_diameter_m", 0))

    for idx, mask_tensor in enumerate(res.masks.data):
        mask = mask_tensor.cpu().numpy().astype(np.uint8) * 255
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

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
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            continue
        epsilon = max(1.0, 0.01 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        poly_pixel = [[float(px + offset_x), float(py + offset_y)] for px, py in approx.reshape(-1, 2)]

        gx, gy = cx + offset_x, cy + offset_y
        wx, wy = _pixel_to_world(gt, gx, gy)

        bx, by, bw, bh = cv2.boundingRect(mask)
        gbx0, gby0 = bx + offset_x, by + offset_y
        gbx1, gby1 = gbx0 + bw, gby0 + bh
        w0, w1 = _pixel_to_world(gt, gbx0, gby0)
        w2, w3 = _pixel_to_world(gt, gbx1, gby1)

        score = float(res.boxes.conf[idx].item()) if res.boxes and len(res.boxes) > idx else 0.0

        detections.append({
            "instance_id": idx,
            "score": round(score, 4),
            "area_m2": round(area_m2, 4),
            "equivalent_diameter_m": round(eq_diam, 4),
            "centroid_pixel": [cx, cy],
            "centroid_world": [round(wx, 4), round(wy, 4)],
            "bbox_world": [round(min(w0, w2), 4), round(min(w1, w3), 4),
                          round(max(w0, w2), 4), round(max(w1, w3), 4)],
            "polygon_pixel": poly_pixel,
        })

    return {"detection_count": len(detections), "detections": detections}


def _infer_crop(crop: np.ndarray, model, ic: dict, gt: tuple,
                offset_x: int, offset_y: int, crop_sz: int) -> dict:
    """通用推理入口

    - SAHI 小图（≤1280px）：直接送入 YOLO
    - 四叉树大图：拆成 1024×1024 子块分别推理，**不做去重**，
      让融合阶段来处理子块间的重复检测（保留完整检测才能体现融合算法的差异）
    """
    h, w = crop.shape[:2]
    if max(w, h) <= 1280:
        return _infer_one_crop(crop, model, ic, gt, offset_x, offset_y)

    # ── 大 tile 拆子块 ─────────────────────────────────────────
    SUB = 1024
    OLAP = 64        # 子块间重叠，避免边界石块被截断
    stride = SUB - OLAP
    all_dets = []

    for y0 in range(0, max(1, h - SUB + 1), stride):
        for x0 in range(0, max(1, w - SUB + 1), stride):
            y1 = min(y0 + SUB, h)
            x1 = min(x0 + SUB, w)
            if (y1 - y0) < 256 or (x1 - x0) < 256:
                continue
            sub = crop[y0:y1, x0:x1]
            sub_res = _infer_one_crop(sub, model, ic, gt,
                                       offset_x + x0, offset_y + y0)
            all_dets.extend(sub_res["detections"])

    return {"detection_count": len(all_dets), "detections": all_dets}


def _run_detection(source_method: str, limit: int | None = None) -> dict:
    from ultralytics import YOLO

    t0 = time.perf_counter()

    config = _load_detection_config()
    slice_stats = _load_slicing_manifest(source_method)
    gt = _parse_tfw(DOM_WORLD_PATH)

    # 加载模型
    model = YOLO(str(MODEL_PATH))

    # 加载 DOM (BGR)
    dom_pil = Image.open(DOM_PATH)
    dom_img = np.array(dom_pil)
    if dom_img.ndim == 3 and dom_img.shape[2] == 3:
        dom_img = cv2.cvtColor(dom_img, cv2.COLOR_RGB2BGR)

    # 获取切片列表
    if "patches" in slice_stats:
        items = slice_stats["patches"]
        is_sahi = True
    else:
        items = slice_stats["tiles"]
        is_sahi = False

    kept = [it for it in items if it.get("status", "kept") == "kept" and not it.get("skipped", False)]
    if limit:
        kept = kept[:limit]

    print(f"  Source: {source_method}, patches/tiles: {len(kept)}/{len(items)}")

    all_dets = []
    for item in kept:
        if is_sahi:
            result = _infer_patch_sahi(item, model, config, dom_img, gt)
        else:
            # 四叉树 tile: 世界坐标 → 像素 crop
            b = item["bounds_m"]
            dom_w = slice_stats.get("dom_dims", {}).get("width_px") or dom_img.shape[1]
            dom_h = slice_stats.get("dom_dims", {}).get("height_px") or dom_img.shape[0]
            gb = slice_stats.get("dom_bounds_world", [0, 0, 0, 0])
            gw, gh = gb[2] - gb[0], gb[3] - gb[1]
            px0 = int((b[0] - gb[0]) / gw * dom_w)
            px1 = int((b[2] - gb[0]) / gw * dom_w)
            py0 = int((gb[3] - b[3]) / gh * dom_h)
            py1 = int((gb[3] - b[1]) / gh * dom_h)
            px0, px1 = sorted([max(0, px0), min(dom_w, px1)])
            py0, py1 = sorted([max(0, py0), min(dom_h, py1)])
            crop = dom_img[py0:py1, px0:px1]
            if crop.size == 0:
                continue
            result = _infer_crop(crop, model, config["inference"], gt, px0, py0, 0)
            result["tile_id"] = item.get("tile_id", "")

        for d in result["detections"]:
            d["source_patch_id"] = item.get("patch_id", item.get("tile_id", ""))
            d["source_method"] = source_method
        all_dets.extend(result["detections"])

    elapsed = time.perf_counter() - t0

    stats = {
        "source_method": source_method,
        "model": str(MODEL_PATH),
        "detection_config": config,
        "total_items": len(items),
        "processed": len(kept),
        "detection_count": len(all_dets),
        "elapsed_seconds": round(elapsed, 2),
        "detections": all_dets,
    }

    out_dir = _resolve_output_dir(source_method)
    (out_dir / "detection_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"  [{source_method}] {len(all_dets)} detections in {elapsed:.1f}s")
    return stats


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO detection experiment on slicing results")
    parser.add_argument("--source", choices=["all"] + SOURCES, default="all")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tiles")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)

    methods = SOURCES if args.source == "all" else [args.source]

    print(f"\n{'='*60}")
    print(f"  Detection Experiment")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Source(s): {methods}")
    print(f"{'='*60}\n")

    results = {}
    for method in methods:
        # 每个源开始前清理 GPU 缓存，避免显存碎片积累导致 OOM
        import torch
        torch.cuda.empty_cache()

        print(f"── {method} ──")
        try:
            stats = _run_detection(method, args.limit)
            results[method] = stats
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    manifest_path = SELF_DIR / "outputs" / "detection_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nDetection manifest: {manifest_path}")


if __name__ == "__main__":
    main()

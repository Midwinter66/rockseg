"""
检测阶段实验 — 对切片结果逐 tile 跑 YOLO 推理，统计检测数

用法:
  cd D:\github_project\image_segment\DOM_Space_message_val

  # 先跑切片
  python experiments/slicing/run_slicing_experiment.py --method all

  # 检测
  python experiments/detection/run_detection_experiment.py --source sahi
  python experiments/detection/run_detection_experiment.py --source quadtree_dom
  python experiments/detection/run_detection_experiment.py --source all
  python experiments/detection/run_detection_experiment.py --source sahi --limit 10
"""

from __future__ import annotations
import argparse, json, sys, time, math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2, numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000


# ── RLE 编码 ────────────────────────────────────────────────────────

def _rle_encode(mask: np.ndarray) -> dict:
    """将二值 mask 编码为 RLE（COCO 格式，无损）"""
    flat = mask.flatten()
    counts = []
    if len(flat) == 0:
        return {"size": list(mask.shape), "counts": []}
    prev = flat[0]
    cnt = 1
    for v in flat[1:]:
        if v == prev:
            cnt += 1
        else:
            counts.append(cnt)
            prev = v
            cnt = 1
    counts.append(cnt)
    return {"size": list(mask.shape), "counts": counts}

# ── 常量 ────────────────────────────────────────────────────────────
DOM_PATH = PROJECT_ROOT / "data" / "dom3" / "DOM.tif"
DOM_WORLD_PATH = PROJECT_ROOT / "data" / "dom3" / "DOM.tfw"
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
SLICING_OUTPUTS = PROJECT_ROOT / "experiments" / "slicing" / "outputs"
SELF_DIR = Path(__file__).resolve().parent
DETECTION_CONFIG_PATH = PROJECT_ROOT / "experiments" / "configs" / "detection" / "default.json"

SOURCES = ["sahi", "quadtree_dom"]


# ══════════════════════════════════════════════════════════════════════
#  工具
# ══════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_tfw(tfw_path: Path) -> tuple:
    lines = [float(l.strip()) for l in tfw_path.read_text("utf-8").splitlines() if l.strip()]
    return (lines[4], lines[0], lines[2], lines[5], lines[1], lines[3])


def _pixel_to_world(gt: tuple, px: float, py: float) -> tuple[float, float]:
    return float(gt[0] + px * gt[1] + py * gt[2]), float(gt[3] + px * gt[4] + py * gt[5])


def _resolve_output_dir(method: str) -> Path:
    d = SELF_DIR / "outputs" / method
    d.mkdir(parents=True, exist_ok=True)
    return d


# ══════════════════════════════════════════════════════════════════════
#  推理核心
# ══════════════════════════════════════════════════════════════════════

def _infer_tile(crop: np.ndarray, model, ic: dict, gt: tuple,
                offset_x: int, offset_y: int) -> list[dict]:
    """对单张 crop 推理并提取检测结果（整张直接送 YOLO，不拆子块）"""
    # 强制关闭 retina_masks 防 OOM
    if hasattr(model, 'predictor') and model.predictor is not None:
        model.predictor.args.retina_masks = False

    try:
        res = model.predict(crop, imgsz=ic["imgsz"], conf=ic["conf"],
                            max_det=int(ic.get("max_det", 1000)),
                            retina_masks=False, verbose=False)[0]
    except RuntimeError:
        return []

    if res.masks is None or len(res.masks.data) == 0:
        return []

    h, w = crop.shape[:2]
    resolution = abs(gt[1])
    min_diam = float(ic.get("min_stone_diameter_m", 0))
    all_dets = []

    for idx, mask_t in enumerate(res.masks.data):
        mask = (mask_t.cpu().numpy() * 255).astype(np.uint8)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        area_px = int(np.count_nonzero(mask > 0))
        if area_px == 0:
            continue
        area_m2 = float(area_px * resolution ** 2)
        eq_diam = float(math.sqrt(4 * area_m2 / math.pi))
        if eq_diam < min_diam:
            continue

        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"]) + offset_x
        cy = int(moments["m01"] / moments["m00"]) + offset_y
        wx, wy = _pixel_to_world(gt, cx, cy)

        # 检测框（像素→世界坐标）
        bx, by, bw, bh = cv2.boundingRect(mask)
        gx0, gy0 = bx + offset_x, by + offset_y
        gx1, gy1 = gx0 + bw, gy0 + bh
        wx0, wy0 = _pixel_to_world(gt, gx0, gy0)
        wx1, wy1 = _pixel_to_world(gt, gx1, gy1)

        score = float(res.boxes.conf[idx].item()) if res.boxes and len(res.boxes) > idx else 0.0

        # RLE 编码 mask（无损，比多边形精确）
        rle_mask = _rle_encode(mask)

        all_dets.append({
            "score": round(score, 4),
            "area_m2": round(area_m2, 4),
            "equivalent_diameter_m": round(eq_diam, 4),
            "centroid_world": [round(wx, 4), round(wy, 4)],
            "bbox_world": [round(min(wx0, wx1), 4), round(min(wy0, wy1), 4),
                           round(max(wx0, wx1), 4), round(max(wy0, wy1), 4)],
            "pixel_origin": [offset_x, offset_y],
            "rle_mask": rle_mask,
        })

    return all_dets


# ══════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════

def _run_detection(source_method: str, limit: int | None = None) -> dict:
    from ultralytics import YOLO

    t_start = time.perf_counter()

    # 加载配置
    config = _load_json(DETECTION_CONFIG_PATH)
    ic = config["inference"]
    slice_stats = _load_json(SLICING_OUTPUTS / source_method / "tile_stats.json")
    gt = _parse_tfw(DOM_WORLD_PATH)

    # 加载模型 + DOM
    model = YOLO(str(MODEL_PATH))
    dom_pil = Image.open(DOM_PATH)
    dom_img = np.array(dom_pil)
    if dom_img.ndim == 3 and dom_img.shape[2] == 3:
        dom_img = cv2.cvtColor(dom_img, cv2.COLOR_RGB2BGR)

    # 获取 kept tiles
    is_sahi = "patches" in slice_stats
    items = slice_stats["patches"] if is_sahi else slice_stats["tiles"]
    kept = [it for it in items if it.get("status", "kept") == "kept" and not it.get("skipped", False)]
    if limit:
        kept = kept[:limit]

    print(f"  ┌─ {source_method}  —  {len(kept)}/{len(items)} tiles kept  ─────────────────────")
    print(f"  │  conf={ic['conf']}  min_diam={ic.get('min_stone_diameter_m', '—')}m")
    print(f"  │")

    all_dets = []
    total = len(kept)

    for i, item in enumerate(kept, 1):
        # 切出 crop
        if is_sahi:
            x, y = item["pixel_origin"]
            sz = item["pixel_size"]
            crop = dom_img[y:y + sz, x:x + sz]
            ox, oy = x, y
        else:
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
            ox, oy = px0, py0

        dets = _infer_tile(crop, model, ic, gt, ox, oy)
        for d in dets:
            d["source_patch_id"] = item.get("patch_id", item.get("tile_id", ""))
        all_dets.extend(dets)

        # 终端进度
        pct = i / total * 100
        bar_len = 28
        filled = int(bar_len * i / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  │  [{bar}] {i:>4}/{total}  {len(dets):>2} dets  (累计 {len(all_dets):>3})", end="\r")

    elapsed = time.perf_counter() - t_start

    # 统计汇总
    areas = [d["area_m2"] for d in all_dets]
    diams = [d["equivalent_diameter_m"] for d in all_dets]
    print(f"\n  │")
    print(f"  ├─ 检测结果 ─────────────────────────────────")
    print(f"  │  总检测数:    {len(all_dets)}")
    if all_dets:
        print(f"  │  面积区间:    {min(areas):.2f} ~ {max(areas):.2f} m²")
        print(f"  │  直径区间:    {min(diams):.2f} ~ {max(diams):.2f} m")
        print(f"  │  平均直径:    {sum(diams)/len(diams):.2f} m")
    print(f"  │  耗时:        {elapsed:.1f}s")
    print(f"  └────────────────────────────────────────────")

    # 保存摘要（不存全量 detections，太大）
    stats = {
        "source_method": source_method,
        "config": config,
        "total_tiles": len(items),
        "processed_tiles": len(kept),
        "detection_count": len(all_dets),
        "elapsed_seconds": round(elapsed, 2),
        "area_m2": {"min": round(min(areas), 2), "max": round(max(areas), 2),
                    "mean": round(sum(areas) / len(areas), 2)} if areas else None,
        "diameter_m": {"min": round(min(diams), 2), "max": round(max(diams), 2),
                       "mean": round(sum(diams) / len(diams), 2)} if diams else None,
    }

    out_dir = _resolve_output_dir(source_method)
    (out_dir / "detection_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    # 完整检测列表（fusion 需要读取）
    (out_dir / "detections.json").write_text(
        json.dumps(all_dets, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO detection on slicing results")
    parser.add_argument("--source", choices=["all"] + SOURCES, default="all",
                        help="切片方法名")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理的 tile 数（快速测试）")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"ERROR: 模型不存在: {MODEL_PATH}")
        sys.exit(1)

    methods = SOURCES if args.source == "all" else [args.source]

    print(f"\n{'='*60}")
    print(f"  检测实验")
    print(f"  模型: {MODEL_PATH.name}")
    print(f"  方法: {', '.join(methods)}")
    print(f"{'='*60}\n")

    results = {}
    for method in methods:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        stats = _run_detection(method, args.limit)
        results[method] = stats

    manifest_path = SELF_DIR / "outputs" / "detection_manifest.json"
    manifest_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"\n结果汇总: {manifest_path}")


if __name__ == "__main__":
    main()

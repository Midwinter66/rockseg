"""
融合结果可视化 — 在 DOM 上绘制石块 bbox, 对比两种融合方法

两种模式:
  - single:  单一方法, 画所有石块 bbox (不同颜色=不同簇大小)
  - compare: 两方法并排, 标出差异区域

用法:
  cd D:\github_project\image_segment\DOM_Space_message_val

  # 查看单个融合结果
  python experiments/fusion/visualize_fusion.py --source sahi --method heuristic
  python experiments/fusion/visualize_fusion.py --source sahi --method correlation_clustering
  python experiments/fusion/visualize_fusion.py --source quadtree_dom --method heuristic

  # 对比两种方法 (并排图 + 差异标注)
  python experiments/fusion/visualize_fusion.py --source sahi_baseline --compare

  # 全部四组对比
  python experiments/fusion/visualize_fusion.py --source all --compare
"""

from __future__ import annotations
import argparse, json, sys, math, shutil
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2, numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000

DOM_PATH = PROJECT_ROOT / "data" / "dom2" / "DOM.tif"
TFW_PATH = PROJECT_ROOT / "data" / "dom2" / "DOM.tfw"
DETECTION_CONFIG_PATH = PROJECT_ROOT / "experiments" / "configs" / "detection" / "default.json"
SELF_DIR = Path(__file__).resolve().parent
FUSION_OUTPUTS = SELF_DIR / "outputs"
DETECTION_OUTPUTS = PROJECT_ROOT / "experiments" / "detection" / "outputs"
SLICING_OUTPUTS = PROJECT_ROOT / "experiments" / "slicing" / "outputs"

SOURCES = ["sahi", "quadtree_dom"]
DEFAULT_DIAMETER_BIN_MULTIPLIERS = (1.0, 2.0, 4.0, 10.0)


def _parse_tfw(tfw_path: str | Path) -> tuple:
    lines = [float(l.strip()) for l in Path(tfw_path).read_text("utf-8").splitlines() if l.strip()]
    return (lines[4], lines[0], lines[2], lines[5], lines[1], lines[3])


def load_dom_resized(max_side: int = 3000) -> tuple[np.ndarray, float, tuple]:
    """返回 (BGR缩小图, scale, gt)"""
    dom = Image.open(DOM_PATH)
    w, h = dom.size
    scale = min(max_side / w, max_side / h, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    dom_small = dom.resize((nw, nh), Image.LANCZOS)
    img = np.array(dom_small)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gt = _parse_tfw(TFW_PATH)
    return img, scale, gt


def world_to_px(box_world: list[float], gt: tuple, scale: float) -> tuple[int, int, int, int]:
    """世界坐标 bbox → 缩小图像素坐标 (x0, y0, x1, y1)"""
    origin_x, res_x, _, origin_y, _, res_y = gt
    x0 = int((box_world[0] - origin_x) / abs(res_x) * scale)
    y0 = int((origin_y - box_world[3]) / abs(res_y) * scale)  # y 翻转
    x1 = int((box_world[2] - origin_x) / abs(res_x) * scale)
    y1 = int((origin_y - box_world[1]) / abs(res_y) * scale)
    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])
    return x0, y0, x1, y1


def _load_fusion_stats(source: str, method: str) -> dict:
    path = FUSION_OUTPUTS / source / method / "fusion_stats.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_detections(source: str) -> list[dict]:
    path = DETECTION_OUTPUTS / source / "detections.json"
    if not path.exists():
        raise FileNotFoundError(f"Detection results not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_slicing_stats(source: str) -> dict:
    path = SLICING_OUTPUTS / source / "tile_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Slicing results not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_detection_config() -> dict:
    return json.loads(DETECTION_CONFIG_PATH.read_text(encoding="utf-8"))


def _rle_decode(rle: dict, expected_area_px: float | None = None) -> np.ndarray:
    """Decode the row-major RLE produced by the detection stage."""
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
    for i, count in enumerate(counts):
        if (int(starts_with) + i) % 2 == 1:
            mask[pos:pos + count] = 255
        pos += count
    return mask.reshape(h, w)


def _parse_diameter_bins(text: str) -> list[float]:
    values = sorted({float(v.strip()) for v in text.split(",") if v.strip()})
    if not values or any(v < 0 for v in values):
        raise ValueError("--diameter-bins must contain non-negative comma-separated values")
    return values


def _default_diameter_bins() -> list[float]:
    detection_cfg = _load_detection_config()
    min_diameter = float(detection_cfg.get("inference", {}).get("min_stone_diameter_m", 0.05))
    if min_diameter <= 0:
        min_diameter = 0.05
    return [round(min_diameter * ratio, 4) for ratio in DEFAULT_DIAMETER_BIN_MULTIPLIERS]


def _diameter_group(diameter: float, bins: list[float]) -> tuple[int, str, str]:
    if diameter < bins[0]:
        return 0, f"< {bins[0]:.2f} m", f"lt_{bins[0]:.2f}m"
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:]), start=1):
        if lo <= diameter < hi:
            return i, f"{lo:.2f}-{hi:.2f} m", f"{lo:.2f}_{hi:.2f}m"
    return len(bins), f">= {bins[-1]:.2f} m", f"ge_{bins[-1]:.2f}m"


DIAMETER_COLORS = [
    (180, 180, 180),  # below configured minimum
    (0, 255, 255),    # yellow: smallest retained stones
    (0, 165, 255),    # orange
    (0, 80, 255),     # red-orange
    (255, 120, 40),   # blue
    (220, 80, 220),   # magenta
]


def _stone_diameter(stone: dict, detections: list[dict]) -> float:
    if "equivalent_diameter_m" in stone:
        return float(stone["equivalent_diameter_m"])
    values = [
        float(detections[i].get("equivalent_diameter_m", 0))
        for i in stone.get("detection_indices", [])
        if 0 <= i < len(detections)
    ]
    return float(np.median(values)) if values else 0.0


def _records_by_tile(stats: dict, detections: list[dict], bins: list[float],
                     only_merged: bool = False) -> dict[str, list[dict]]:
    """Select at most one mask per fused stone per source tile."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for stone in stats.get("stones", []):
        if only_merged and int(stone.get("source_detection_count", 1)) < 2:
            continue
        diameter = _stone_diameter(stone, detections)
        group_index, group_label, group_dir = _diameter_group(diameter, bins)
        best_by_tile: dict[str, int] = {}
        for det_idx in stone.get("detection_indices", []):
            if not (0 <= det_idx < len(detections)):
                continue
            det = detections[det_idx]
            tile_id = det.get("source_patch_id", "")
            if not tile_id:
                continue
            old_idx = best_by_tile.get(tile_id)
            if old_idx is None or float(det.get("score", 0)) > float(detections[old_idx].get("score", 0)):
                best_by_tile[tile_id] = det_idx
        for tile_id, det_idx in best_by_tile.items():
            grouped[tile_id].append({
                "stone": stone,
                "detection_index": det_idx,
                "diameter_m": diameter,
                "group_index": group_index,
                "group_label": group_label,
                "group_dir": group_dir,
            })
    return grouped


def _tile_crop_box(source: str, tile_stats: dict, tile_id: str,
                   dom_w: int, dom_h: int) -> tuple[int, int, int, int]:
    if "patches" in tile_stats:
        item = next((p for p in tile_stats["patches"] if p.get("patch_id") == tile_id), None)
        if item is None:
            raise KeyError(f"Patch not found in slicing stats: {tile_id}")
        x0, y0 = [int(v) for v in item["pixel_origin"]]
        size = int(item["pixel_size"])
        return x0, y0, min(dom_w, x0 + size), min(dom_h, y0 + size)

    item = next((t for t in tile_stats.get("tiles", []) if t.get("tile_id") == tile_id), None)
    if item is None:
        raise KeyError(f"Tile not found in slicing stats: {tile_id}")
    b = item["bounds_m"]
    gb = tile_stats["dom_bounds_world"]
    gw, gh = gb[2] - gb[0], gb[3] - gb[1]
    px0 = int((b[0] - gb[0]) / gw * dom_w)
    px1 = int((b[2] - gb[0]) / gw * dom_w)
    py0 = int((gb[3] - b[3]) / gh * dom_h)
    py1 = int((gb[3] - b[1]) / gh * dom_h)
    px0, px1 = sorted([max(0, px0), min(dom_w, px1)])
    py0, py1 = sorted([max(0, py0), min(dom_h, py1)])
    return px0, py0, px1, py1


def _render_tile_masks(crop_bgr: np.ndarray, records: list[dict], detections: list[dict],
                       alpha: float, title: str, draw_labels: bool,
                       pixel_area_m2: float) -> np.ndarray:
    vis = crop_bgr.copy()
    for record in records:
        det = detections[record["detection_index"]]
        expected_area_px = float(det.get("area_m2", 0)) / pixel_area_m2 if pixel_area_m2 > 0 else None
        mask = _rle_decode(det["rle_mask"], expected_area_px=expected_area_px)
        if mask.shape[:2] != vis.shape[:2]:
            mask = cv2.resize(mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
        selected = mask > 0
        color = np.array(DIAMETER_COLORS[record["group_index"] % len(DIAMETER_COLORS)], dtype=np.float32)
        vis[selected] = np.clip(
            vis[selected].astype(np.float32) * (1.0 - alpha) + color * alpha,
            0, 255,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, tuple(int(v) for v in color), 2)
        if draw_labels and contours:
            x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cv2.putText(vis, f"{record['diameter_m']:.2f}m", (x, max(14, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, tuple(int(v) for v in color), 1,
                        cv2.LINE_AA)

    header = np.full((42, vis.shape[1], 3), (24, 24, 28), dtype=np.uint8)
    cv2.putText(header, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (240, 240, 240), 2, cv2.LINE_AA)
    return np.vstack([header, vis])


def write_tile_mask_groups(source: str, method: str, bins: list[float], alpha: float,
                           max_tiles: int | None = None, tile_id: str | None = None,
                           only_merged: bool = False, draw_labels: bool = False) -> Path:
    stats = _load_fusion_stats(source, method)
    detections = _load_detections(source)
    tile_stats = _load_slicing_stats(source)
    records_by_tile = _records_by_tile(stats, detections, bins, only_merged=only_merged)
    tile_ids = sorted(records_by_tile)
    if tile_id:
        tile_ids = [tile_id] if tile_id in records_by_tile else []
    if max_tiles is not None:
        tile_ids = tile_ids[:max_tiles]

    output_root = FUSION_OUTPUTS / source / method / "tile_mask_groups"
    if output_root.exists():
        shutil.rmtree(output_root)
    all_dir = output_root / "all_groups"
    all_dir.mkdir(parents=True, exist_ok=True)
    dom = Image.open(DOM_PATH).convert("RGB")
    gt = _parse_tfw(TFW_PATH)
    pixel_area_m2 = abs(float(gt[1]) * float(gt[5]))
    manifest = {
        "source": source,
        "method": method,
        "diameter_bins_m": bins,
        "alpha": alpha,
        "only_merged": only_merged,
        "tiles": [],
        "group_mask_instances": {},
    }

    for current_tile_id in tile_ids:
        records = records_by_tile[current_tile_id]
        x0, y0, x1, y1 = _tile_crop_box(source, tile_stats, current_tile_id, dom.width, dom.height)
        crop_rgb = np.array(dom.crop((x0, y0, x1, y1)))
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

        combined = _render_tile_masks(
            crop_bgr, records, detections, alpha,
            f"{current_tile_id} | all diameter groups | {len(records)} masks",
            draw_labels, pixel_area_m2,
        )
        combined_path = all_dir / f"{current_tile_id}.png"
        cv2.imwrite(str(combined_path), combined)

        group_files = {}
        for group_dir in sorted({r["group_dir"] for r in records}):
            group_records = [r for r in records if r["group_dir"] == group_dir]
            group_label = group_records[0]["group_label"]
            target_dir = output_root / f"diameter_{group_dir}"
            target_dir.mkdir(parents=True, exist_ok=True)
            group_image = _render_tile_masks(
                crop_bgr, group_records, detections, alpha,
                f"{current_tile_id} | diameter {group_label} | {len(group_records)} masks",
                draw_labels, pixel_area_m2,
            )
            group_path = target_dir / f"{current_tile_id}.png"
            cv2.imwrite(str(group_path), group_image)
            group_files[group_dir] = str(group_path)
            manifest["group_mask_instances"][group_dir] = (
                manifest["group_mask_instances"].get(group_dir, 0) + len(group_records)
            )

        manifest["tiles"].append({
            "tile_id": current_tile_id,
            "crop_pixel_box": [x0, y0, x1, y1],
            "mask_count": len(records),
            "all_groups_image": str(combined_path),
            "group_images": group_files,
        })

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Tile mask images: {len(tile_ids)} tiles -> {output_root}")
    print(f"  Manifest: {manifest_path}")
    return manifest_path


# ── 簇大小 → 颜色 ────────────────────────────────────────────────────
CLUSTER_COLORS = [
    (46, 204, 113),    # 绿: 单检测
    (52, 152, 219),    # 蓝: 2-合并
    (155, 89, 182),    # 紫: 3-合并
    (241, 196, 15),    # 黄: 4-合并
    (230, 126, 34),    # 橙: 5+合并
    (231, 76, 60),     # 红: 很大簇
]


def cluster_color(size: int) -> tuple[int, int, int]:
    if size == 1:
        return CLUSTER_COLORS[0]
    if size == 2:
        return CLUSTER_COLORS[1]
    if size == 3:
        return CLUSTER_COLORS[2]
    if size <= 5:
        return CLUSTER_COLORS[3]
    if size <= 10:
        return CLUSTER_COLORS[4]
    return CLUSTER_COLORS[5]


def draw_single_method(img_bgr: np.ndarray, stats: dict, gt: tuple, scale: float) -> np.ndarray:
    """在 DOM 上画所有石块的 bbox + 标签"""
    vis = img_bgr.copy()
    stones = stats.get("stones", [])

    for s in stones:
        bbox = s.get("bbox_world", [0, 0, 0, 0])
        sz = s.get("source_detection_count", 1)
        x0, y0, x1, y1 = world_to_px(bbox, gt, scale)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        color = cluster_color(sz)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)

    # 图例
    y = 28
    for size, label in [(1, "1 det"), (2, "2 dets"), (3, "3 dets"), (4, "4-5 dets"), (6, "6-10 dets"), (11, "11+")]:
        color = cluster_color(size)
        cv2.rectangle(vis, (10, y - 12), (30, y + 2), color, -1)
        cv2.putText(vis, label, (36, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22

    return vis


def draw_compare(img_bgr: np.ndarray, stats_a: dict, stats_b: dict,
                 gt: tuple, scale: float,
                 label_a: str = "A", label_b: str = "B") -> np.ndarray:
    """并排对比: 左边 A, 右边 B, 下方差异图"""
    h, w = img_bgr.shape[:2]
    vis_a = draw_single_method(img_bgr.copy(), stats_a, gt, scale)
    vis_b = draw_single_method(img_bgr.copy(), stats_b, gt, scale)

    # 标题
    bar_a = np.full((32, w, 3), (30, 30, 42), dtype=np.uint8)
    bar_b = np.full((32, w, 3), (30, 30, 42), dtype=np.uint8)
    cv2.putText(bar_a, f" {label_a} ({stats_a.get('method','')}) - {stats_a['output_stones']} stones",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (46, 204, 113), 2)
    cv2.putText(bar_b, f" {label_b} ({stats_b.get('method','')}) - {stats_b['output_stones']} stones",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (52, 152, 219), 2)

    row1 = np.hstack([np.vstack([bar_a, vis_a]), np.vstack([bar_b, vis_b])])

    # 差异图: 两个方法 bbox 中心点的差异
    diff = img_bgr.copy()
    centers_a: dict[int, tuple] = {}
    centers_b: dict[int, tuple] = {}
    for s in stats_a.get("stones", []):
        b = s.get("bbox_world", [0, 0, 0, 0])
        centers_a[id(s)] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    for s in stats_b.get("stones", []):
        b = s.get("bbox_world", [0, 0, 0, 0])
        centers_b[id(s)] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    # 简单做法: 两个方法的 bbox 都画, 不同颜色
    for s in stats_a.get("stones", []):
        bbox = s.get("bbox_world", [0, 0, 0, 0])
        x0, y0, x1, y1 = world_to_px(bbox, gt, scale)
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            cv2.rectangle(diff, (x0, y0), (x1, y1), (46, 204, 113), 1)
    for s in stats_b.get("stones", []):
        bbox = s.get("bbox_world", [0, 0, 0, 0])
        x0, y0, x1, y1 = world_to_px(bbox, gt, scale)
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            cv2.rectangle(diff, (x0, y0), (x1, y1), (52, 152, 219), 1)

    bar_d = np.full((32, w, 3), (30, 30, 42), dtype=np.uint8)
    cv2.putText(bar_d, " Diff: GREEN=heuristic, BLUE=correlation_clustering",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2)
    row2 = np.vstack([bar_d, diff])

    # diff 图只有单张, 需要 pad 到和 row1 一样宽
    if row2.shape[1] < row1.shape[1]:
        pad_w = row1.shape[1] - row2.shape[1]
        row2 = np.hstack([row2, np.full((row2.shape[0], pad_w, 3), 30, dtype=np.uint8)])

    return np.vstack([row1, np.full((4, row1.shape[1], 3), 0, dtype=np.uint8), row2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize fusion results")
    parser.add_argument("--source", choices=["all"] + SOURCES, default="all")
    parser.add_argument("--method", choices=["heuristic", "correlation_clustering"], default="heuristic")
    parser.add_argument("--compare", action="store_true", help="Compare heuristic vs correlation_clustering")
    parser.add_argument("--max-side", type=int, default=2500, help="Max image side length")
    parser.add_argument("--tile-masks", action="store_true",
                        help="Render masks on each original slicing tile and split outputs by diameter")
    parser.add_argument("--diameter-bins", default=None,
                        help="Comma-separated diameter boundaries in metres; default is derived from min_stone_diameter_m")
    parser.add_argument("--alpha", type=float, default=0.42,
                        help="Mask opacity for tile rendering (0-1)")
    parser.add_argument("--max-tiles", type=int, default=None,
                        help="Limit the number of tile mask images")
    parser.add_argument("--tile-id", default=None,
                        help="Render one source tile only")
    parser.add_argument("--only-merged", action="store_true",
                        help="Render only stones fused from at least two detections")
    parser.add_argument("--draw-labels", action="store_true",
                        help="Draw the representative diameter beside each mask")
    args = parser.parse_args()

    if not DOM_PATH.exists():
        print(f"DOM not found: {DOM_PATH}")
        sys.exit(1)

    sources = SOURCES if args.source == "all" else [args.source]

    if args.tile_masks:
        if args.compare:
            parser.error("--tile-masks cannot be combined with --compare")
        if not 0.0 <= args.alpha <= 1.0:
            parser.error("--alpha must be between 0 and 1")
        bins = _parse_diameter_bins(args.diameter_bins) if args.diameter_bins else _default_diameter_bins()
        print(f"  Diameter bins (m): {bins}")
        for source in sources:
            try:
                write_tile_mask_groups(
                    source=source,
                    method=args.method,
                    bins=bins,
                    alpha=args.alpha,
                    max_tiles=args.max_tiles,
                    tile_id=args.tile_id,
                    only_merged=args.only_merged,
                    draw_labels=args.draw_labels,
                )
            except (FileNotFoundError, KeyError) as exc:
                print(f"  SKIP {source}: {exc}")
        return

    img_bgr, scale, gt = load_dom_resized(args.max_side)

    print(f"\n{'='*60}")
    print(f"  Fusion Visualization")
    print(f"  Sources: {sources}")
    print(f"  Max side: {args.max_side}px, Scale: {scale:.3f}")
    print(f"{'='*60}\n")

    for source in sources:
        out_dir = FUSION_OUTPUTS / source

        if args.compare:
            try:
                stats_h = _load_fusion_stats(source, "heuristic")
                stats_c = _load_fusion_stats(source, "correlation_clustering")
            except FileNotFoundError as e:
                print(f"  SKIP {source}: {e}")
                continue

            result = draw_compare(img_bgr, stats_h, stats_c, gt, scale,
                                  f"{source}/heuristic", f"{source}/correlation")
            out_path = str(out_dir / "fusion_compare.png")
            cv2.imwrite(out_path, result)
            print(f"  [{source}] compare saved: {out_path}")

        else:
            try:
                stats = _load_fusion_stats(source, args.method)
            except FileNotFoundError as e:
                print(f"  SKIP {source}: {e}")
                continue

            result = draw_single_method(img_bgr, stats, gt, scale)
            out_path = str(out_dir / f"fusion_{args.method}.png")
            cv2.imwrite(out_path, result)
            print(f"  [{source}] {args.method} saved: {out_path}")

    print(f"\nDone. Open experiments/fusion/outputs/ to view images.")


if __name__ == "__main__":
    main()

"""
融合结果可视化 — 在 DOM 上绘制石块 bbox, 对比两种融合方法

两种模式:
  - single:  单一方法, 画所有石块 bbox (不同颜色=不同簇大小)
  - compare: 两方法并排, 标出差异区域

用法:
  cd D:\github_project\image_segment\DOM_Space_message_val

  # 查看单个融合结果
  python experiments/fusion/visualize_fusion.py --source sahi_baseline --method heuristic
  python experiments/fusion/visualize_fusion.py --source sahi_baseline --method correlation_clustering

  # 对比两种方法 (并排图 + 差异标注)
  python experiments/fusion/visualize_fusion.py --source sahi_baseline --compare

  # 全部四组对比
  python experiments/fusion/visualize_fusion.py --source all --compare
"""

from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2, numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000

DOM_PATH = PROJECT_ROOT / "data" / "dom3" / "DOM.tif"
SELF_DIR = Path(__file__).resolve().parent
FUSION_OUTPUTS = SELF_DIR / "outputs"

SOURCES = ["sahi_baseline", "sahi_dense", "quadtree_pointcloud", "quadtree_dom"]


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
    tfw = PROJECT_ROOT / "data" / "dom3" / "DOM.tfw"
    gt = _parse_tfw(tfw)
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
    args = parser.parse_args()

    if not DOM_PATH.exists():
        print(f"DOM not found: {DOM_PATH}")
        sys.exit(1)

    img_bgr, scale, gt = load_dom_resized(args.max_side)
    sources = SOURCES if args.source == "all" else [args.source]

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

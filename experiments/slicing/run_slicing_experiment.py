# noqa
"""
统一切片实验运行器

支持两种方法:
  - sahi         : SAHI 固定滑窗 (由配置文件控制 overlap)
  - quadtree_dom : 四叉树按 DOM 纹理边缘密度

用法:
  cd D:\github_project\image_segment\DOM_Space_message_val

  # 1) 逐个跑
  python experiments/slicing/run_slicing_experiment.py --method sahi
  python experiments/slicing/run_slicing_experiment.py --method quadtree_dom

  # 2) 一步跑全部
  python experiments/slicing/run_slicing_experiment.py --method all

  # 3) 生成对比报告 (先跑完想要的 method 再执行)
  python experiments/slicing/visualize_tiles.py
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

# 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # experiments/slicing/ -> root
sys.path.insert(0, str(PROJECT_ROOT))

import cv2, numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000

from experiments.utils.metrics import compute_sahi_stats, compute_quadtree_stats
from experiments.utils.visualization import (
    draw_sahi_overlay,
    draw_quadtree_overlay,
    draw_quadtree_split_overlay,
)

# ── 常量: 数据路径 ─────────────────────────────────────────────────
DOM_PATH = PROJECT_ROOT / "data" / "dom3" / "DOM.tif"
DOM_WORLD_PATH = PROJECT_ROOT / "data" / "dom3" / "DOM.tfw"
SELF_DIR = Path(__file__).resolve().parent  # experiments/slicing/

# 所有已注册的切片方法
ALL_METHODS = ["sahi", "quadtree_dom"]


# ══════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════

def _load_config(method: str) -> dict:
    cfg_path = PROJECT_ROOT / "experiments" / "configs" / "slicing" / f"{method}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_output_dir(method: str) -> Path:
    d = SELF_DIR / "outputs" / method
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_tfw(tfw_path: str | Path) -> tuple[float, float, float, float, float, float]:
    """解析 TFW → GeoTransform 格式: (origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height)"""
    lines = [float(line.strip()) for line in Path(tfw_path).read_text("utf-8").splitlines() if line.strip()]
    if len(lines) != 6:
        raise ValueError(f"Invalid TFW: {tfw_path}")
    return (lines[4], lines[0], lines[2], lines[5], lines[1], lines[3])


def _pixel_to_world(gt: tuple, px: float, py: float) -> tuple[float, float]:
    return float(gt[0] + px * gt[1] + py * gt[2]), float(gt[3] + px * gt[4] + py * gt[5])


def _get_dom_info() -> dict:
    """获取 DOM 的元信息 (宽/高/分辨率/世界范围)"""
    dom = Image.open(DOM_PATH)
    w, h = dom.size
    gt = _parse_tfw(DOM_WORLD_PATH)
    res = abs(gt[1])
    xmin, ymax = _pixel_to_world(gt, 0, 0)       # 左上像素 → 世界 NW 角
    xmax, ymin = _pixel_to_world(gt, w, h)        # 右下像素 → 世界 SE 角
    return {"width": w, "height": h, "resolution_m": res,
            "dom_bounds_world": [xmin, ymin, xmax, ymax],
            "tfw_gt": gt}


# ══════════════════════════════════════════════════════════════════════
#  DOM 纹理四叉树 (新实现)
# ══════════════════════════════════════════════════════════════════════

class DOMTextureQuadTree:
    """按 DOM 纹理边缘密度自适应分裂的四叉树

    与 v2 的 QuadTreeCover 结构相同, 但分裂条件用 Canny 边缘密度
    替代点云密度。这个实现完全独立, 不依赖 Open3D 和 v2 类。
    """

    def __init__(self, bounds: list[float], base_size: float,
                 min_size: float, max_size: float):
        self.bounds = bounds
        self.base_size = base_size
        self.min_size = min_size
        self.max_size = max_size

    def generate(self, dom_image: np.ndarray,
                 edge_density_threshold: float,
                 canny_low: int = 50,
                 canny_high: int = 150,
                 black_threshold: int = 5,
                 min_content_ratio: float = 0.0,
                 tile_overlap_m: float = 0.0) -> list[dict]:
        """对 DOM 影像做 Canny 检测, 按边缘密度四分

        dom_image: BGR uint8 [H, W, 3] — 已经加载的 DOM 图像
        edge_density_threshold: 边缘密度阈值 (边缘像素 / 总像素)
        black_threshold: 灰度低于此值视为黑色无效区域
        min_content_ratio: 有效内容占比低于此值的 tile 标记为 skipped
        tile_overlap_m: 相邻 tile 之间的重叠宽度（半边扩展，0=无重叠）
        """
        xmin, ymin, xmax, ymax = self.bounds
        h_img, w_img = dom_image.shape[:2]
        dom_area_m = (xmax - xmin, ymax - ymin)

        # 全局灰度图（用于内容过滤）
        gray_full = cv2.cvtColor(dom_image, cv2.COLOR_BGR2GRAY)

        # 全局 Canny（用于边缘密度）
        edges_full = cv2.Canny(gray_full, canny_low, canny_high)

        def world_to_pixel(wx: float, wy: float) -> tuple[int, int]:
            px = int((wx - xmin) / dom_area_m[0] * w_img)
            py = int((ymax - wy) / dom_area_m[1] * h_img)
            return max(0, min(px, w_img - 1)), max(0, min(py, h_img - 1))

        # 初始网格
        nx = max(1, int(np.ceil((xmax - xmin) / self.base_size)))
        ny = max(1, int(np.ceil((ymax - ymin) / self.base_size)))
        queue: list[dict] = []
        for ix in range(nx):
            for iy in range(ny):
                tx0 = xmin + ix * self.base_size
                ty0 = ymin + iy * self.base_size
                tx1 = min(xmax, tx0 + self.base_size)
                ty1 = min(ymax, ty0 + self.base_size)
                queue.append({"tile_id": f"tile_{ix}_{iy}", "bounds_m": [tx0, ty0, tx1, ty1]})

        final_tiles: list[dict] = []
        half_overlap = tile_overlap_m / 2.0

        def _expand_bounds(b: list[float]) -> list[float]:
            """将 tile 边界向外扩展 half_overlap，截断到 DOM 边界"""
            return [
                max(xmin, b[0] - half_overlap),
                max(ymin, b[1] - half_overlap),
                min(xmax, b[2] + half_overlap),
                min(ymax, b[3] + half_overlap),
            ]

        while queue:
            tile = queue.pop(0)
            b = tile["bounds_m"]
            w = b[2] - b[0]
            h = b[3] - b[1]
            px0, py0 = world_to_pixel(b[0], b[3])
            px1, py1 = world_to_pixel(b[2], b[1])
            px0, px1 = sorted([px0, px1])
            py0, py1 = sorted([py0, py1])

            crop = edges_full[py0:py1, px0:px1]
            edge_count = int(np.count_nonzero(crop))
            total_px = max(crop.size, 1)
            edge_density = float(edge_count / total_px)

            # 内容比例过滤（黑色区域检查）
            gray_crop = gray_full[py0:py1, px0:px1]
            content_count = int(np.count_nonzero(gray_crop > black_threshold))
            content_ratio = float(content_count / total_px)

            tile["edge_density"] = edge_density
            tile["content_ratio"] = round(content_ratio, 4)
            tile["pixel_region"] = [px0, py0, px1, py1]

            # 内容太少 → 跳过（黑色区域）
            if content_ratio < min_content_ratio:
                tile["skipped"] = True
                tile["skip_reason"] = "black"
                tile["bounds_m"] = _expand_bounds(tile["bounds_m"])
                final_tiles.append(tile)
                continue

            # 完全没有边缘 → 跳过（平坦区域）
            if edge_count == 0:
                tile["skipped"] = True
                tile["skip_reason"] = "no_edges"
                tile["bounds_m"] = _expand_bounds(tile["bounds_m"])
                final_tiles.append(tile)
                continue

            # 边缘密度高 + 尺寸允许 → 四分
            if edge_density >= edge_density_threshold and max(w, h) > self.min_size:
                mx, my = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                children = [
                    {"tile_id": f"{tile['tile_id']}_0", "bounds_m": [b[0], b[1], mx, my]},
                    {"tile_id": f"{tile['tile_id']}_1", "bounds_m": [mx, b[1], b[2], my]},
                    {"tile_id": f"{tile['tile_id']}_2", "bounds_m": [b[0], my, mx, b[3]]},
                    {"tile_id": f"{tile['tile_id']}_3", "bounds_m": [mx, my, b[2], b[3]]},
                ]
                queue[:0] = children   # BFS: 当前层分完再下一层
            else:
                tile["skipped"] = False
                tile["skip_reason"] = ""
                tile["bounds_m"] = _expand_bounds(tile["bounds_m"])
                final_tiles.append(tile)

        return final_tiles


# ══════════════════════════════════════════════════════════════════════
#  各方法的 runner
# ══════════════════════════════════════════════════════════════════════

def _run_sahi(config: dict, out_dir: Path) -> dict:
    """运行 SAHI 固定滑窗切片, 返回 stats dict"""
    patch_cfg = config["patching"]
    dom_info = _get_dom_info()
    gt = dom_info["tfw_gt"]

    t0 = time.perf_counter()
    dom = Image.open(DOM_PATH)
    img = np.array(dom)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img
    h, w = img.shape[:2]

    patch_sz = int(patch_cfg["patch_size"])
    overlap = float(patch_cfg["overlap"])
    stride = max(1, int(round(patch_sz * (1 - overlap))))
    black_th = int(patch_cfg["black_pixel_threshold"])
    min_content = float(patch_cfg["min_content_ratio"])
    include_edge = bool(patch_cfg.get("include_edge_patches", True))

    def build_positions(limit: int) -> list[int]:
        if limit <= patch_sz:
            return [0]
        pos = list(range(0, limit - patch_sz + 1, stride))
        if include_edge and pos[-1] != limit - patch_sz:
            last = pos[-1]
            gap = (limit - patch_sz) - last
            if gap > stride:
                while last + stride < limit - patch_sz:
                    last += stride
                    pos.append(last)
            pos.append(limit - patch_sz)
        return sorted(set(pos))

    x_positions = build_positions(w)
    y_positions = build_positions(h)

    records = []
    for row, y in enumerate(y_positions):
        for col, x in enumerate(x_positions):
            patch = img[y:y + patch_sz, x:x + patch_sz]
            if patch.ndim == 3:
                gray = patch.mean(axis=2)
            else:
                gray = patch
            content_ratio = float(np.count_nonzero(gray > black_th) / gray.size)

            wx0, wy0 = _pixel_to_world(gt, x, y)
            wx1, wy1 = _pixel_to_world(gt, x + patch_sz, y + patch_sz)

            records.append({
                "patch_id": f"patch_{len(records):06d}",
                "grid_index": [row, col],
                "pixel_origin": [x, y],
                "pixel_size": patch_sz,
                "world_origin": [wx0, wy0],
                "world_bounds": [min(wx0, wx1), min(wy0, wy1), max(wx0, wx1), max(wy0, wy1)],
                "resolution": dom_info["resolution_m"],
                "content_ratio": content_ratio,
                "black_ratio": round(1.0 - content_ratio, 4),
                "status": "kept" if content_ratio >= min_content else "skipped_black",
            })

    elapsed = time.perf_counter() - t0

    stats = compute_sahi_stats(config["method"], patch_cfg, records, w, h, elapsed)
    stats["resolution_m"] = dom_info["resolution_m"]

    # 保存
    stats_path = out_dir / "tile_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # 生成叠加图
    overlay_path = out_dir / "tile_overlay.png"
    draw_sahi_overlay(DOM_PATH, records, overlay_path, max_side=8000, line_thickness=2)
    print(f"  [SAHI] 有效切片: {stats['kept_patches']}/{stats['total_patches']}, "
          f"耗时: {elapsed:.2f}s, 图已保存: {overlay_path}")

    return stats



def _run_quadtree_dom(config: dict, out_dir: Path) -> dict:
    """运行四叉树 (DOM 纹理边缘密度)"""
    cc = config["cover"]
    dom_info = _get_dom_info()

    t0 = time.perf_counter()
    dom = Image.open(DOM_PATH)
    img = np.array(dom)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    qt = DOMTextureQuadTree(
        dom_info["dom_bounds_world"],
        float(cc["base_tile_size_m"]),
        float(cc["min_tile_size_m"]),
        float(cc["max_tile_size_m"]),
    )
    tiles = qt.generate(
        img,
        float(cc["min_edge_density"]),
        int(cc.get("canny_low", 50)),
        int(cc.get("canny_high", 150)),
        black_threshold=int(cc.get("black_pixel_threshold", 5)),
        min_content_ratio=float(cc.get("min_content_ratio", 0.0)),
        tile_overlap_m=float(cc.get("tile_overlap_m", 0.0)),
    )
    elapsed = time.perf_counter() - t0

    stats = compute_quadtree_stats(
        config["method"], cc, tiles, dom_info["dom_bounds_world"], elapsed
    )

    stats_path = out_dir / "tile_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    overlay_path = out_dir / "tile_overlay.png"
    draw_quadtree_split_overlay(
        DOM_PATH, tiles, dom_info["dom_bounds_world"],
        img_size=(w, h),
        output_path=overlay_path, max_side=8000, line_thickness=2,
    )
    kept = len([t for t in tiles if not t["skipped"]])
    print(f"  [Quadtree-DOM] 有效tile: {kept}/{len(tiles)}, "
          f"耗时: {elapsed:.2f}s, 图已保存: {overlay_path}")
    return stats


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

RUNNERS = {
    "sahi":         _run_sahi,
    "quadtree_dom": _run_quadtree_dom,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run slicing method comparison experiment")
    parser.add_argument("--method", choices=["all"] + ALL_METHODS, default="sahi",
                        help="Slicing method to run, or 'all' to run all")
    args = parser.parse_args()

    methods = ALL_METHODS if args.method == "all" else [args.method]

    # 验证数据存在
    if not DOM_PATH.exists():
        print(f"ERROR: DOM not found at {DOM_PATH}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Slicing Experiment")
    print(f"  DOM: {DOM_PATH}")
    print(f"  Method(s): {methods}")
    print(f"{'='*60}\n")

    results = {}
    for method in methods:
        print(f"── {method} ──")
        cfg = _load_config(method)
        out_dir = _resolve_output_dir(method)
        try:
            runner = RUNNERS[method]
            stats = runner(cfg, out_dir)
            results[method] = {
                "method": method,
                "stats": stats,
                "overlay_img": str(out_dir / "tile_overlay.png"),
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # 保存 results manifest
    manifest_path = SELF_DIR / "outputs" / "results_manifest.json"
    manifest_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nResults manifest: {manifest_path}")

    if len(methods) > 1:
        print(f"\nRun:  python experiments/slicing/visualize_tiles.py")
    print(f"\nRun other methods and then:  python experiments/slicing/visualize_tiles.py")


if __name__ == "__main__":
    main()

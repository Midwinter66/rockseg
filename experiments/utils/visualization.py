"""
切片可视化工具 — 在 DOM 影像上叠加绘制 patch/tile 边界框

支持:
  - SAHI: 等大矩形, 颜色区分 kept/skipped
  - 四叉树: 大小不一的矩形, 颜色按 tile 密度或跳过状态
  - 并排对比图: 两个方法上下并排

Usage:
    from experiments.utils.visualization import draw_sahi_overlay, draw_quadtree_overlay
"""

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500000000

# ── 调色板 ──────────────────────────────────────────────────────────
KEPT_COLOR = (46, 204, 113)       # 绿色 - 保留的切片
SKIPPED_COLOR = (231, 76, 60)     # 红色 - 跳过的切片
QUAD_COLORS = [
    (52, 152, 219),               # 蓝 - level 0 (40m base)
    (155, 89, 182),               # 紫 - level 1 (20m, 一次分裂)
    (241, 196, 15),               # 黄 - level 2 (10m, 二次分裂)
    (230, 126, 34),               # 橙 - level 3+
]
TEXT_COLOR = (255, 255, 255)


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """统一转为 BGR uint8 [H, W, 3]"""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if img.dtype != np.uint8:
        img = (img / np.iinfo(img.dtype).max * 255).astype(np.uint8) if img.dtype != np.float32 else img
    return img


def load_dom_array(dom_path: str | Path, max_side: int = 8000) -> np.ndarray:
    """加载 DOM 并缩放到适合可视化的尺寸"""
    dom = Image.open(dom_path)
    w, h = dom.size
    scale = min(max_side / w, max_side / h, 1.0)
    if scale < 1.0:
        dom = dom.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return _ensure_bgr(np.array(dom)), scale


# ── SAHI 叠加 ───────────────────────────────────────────────────────

def draw_sahi_overlay(
    dom_path: str | Path,
    patches: list[dict],
    output_path: str | Path,
    max_side: int = 8000,
    line_thickness: int = 2,
    alpha: float = 0.35,
) -> Path:
    """在 DOM 上叠加 SAHI 固定大小 patch 的边界框

    patches: 每个 dict 包含:
        - pixel_origin: [x, y]
        - pixel_size: int
        - status: "kept" | "skipped_black"
    """
    img, scale = load_dom_array(dom_path, max_side)
    overlay = img.copy()

    for p in patches:
        x, y = int(p["pixel_origin"][0] * scale), int(p["pixel_origin"][1] * scale)
        sz = int(p["pixel_size"] * scale)
        color = KEPT_COLOR if p.get("status") == "kept" else SKIPPED_COLOR
        cv2.rectangle(overlay, (x, y), (x + sz, y + sz), color, -1)

    vis = cv2.addWeighted(img, 1.0 - alpha, overlay, alpha, 0)

    # 只画 kept 的边框 + id
    for p in patches:
        if p.get("status") != "kept":
            continue
        x, y = int(p["pixel_origin"][0] * scale), int(p["pixel_origin"][1] * scale)
        sz = int(p["pixel_size"] * scale)
        cv2.rectangle(vis, (x, y), (x + sz, y + sz), KEPT_COLOR, line_thickness)
        # 只在足够大的 patch 上标 ID
        if sz > 60:
            pid = p.get("patch_id", "")
            cv2.putText(vis, pid[-4:], (x + 4, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, 1)

    cv2.imwrite(str(output_path), vis)
    return Path(output_path)


# ── 四叉树叠加 ───────────────────────────────────────────────────────

def _quadtree_level(tile_id: str) -> int:
    """从 tile_id 推断四叉树深度 (分裂次数)"""
    return tile_id.count("_") - 1


def draw_quadtree_overlay(
    dom_path: str | Path,
    tiles: list[dict],
    dom_bounds: list[float],       # world: [xmin, ymin, xmax, ymax]
    img_size: tuple[int, int],     # DOM 像素: (w, h)
    output_path: str | Path,
    max_side: int = 8000,
    line_thickness: int = 2,
    alpha: float = 0.30,
    show_labels: bool = True,
) -> Path:
    """在 DOM 上叠加四叉树 tile 边界框

    tiles: 每个 dict 包含:
        - tile_id: str
        - bounds_m: [xmin, ymin, xmax, ymax]
        - skipped: bool (optional)
        - source_points: int (optional)
    dom_bounds: [xmin, ymin, xmax, ymax] — DOM 的世界坐标范围
    """
    img, scale = load_dom_array(dom_path, max_side)
    scaled_w, scaled_h = img.shape[1], img.shape[0]
    dom_w, dom_h = img_size
    dom_xmin, dom_ymin, dom_xmax, dom_ymax = dom_bounds

    def world_to_scaled(wx: float, wy: float) -> tuple[int, int]:
        """世界坐标 → 缩放后像素坐标"""
        px = (wx - dom_xmin) / (dom_xmax - dom_xmin) * dom_w * scale
        py = (dom_ymax - wy) / (dom_ymax - dom_ymin) * dom_h * scale
        return int(px), int(py)

    overlay = img.copy()

    for t in tiles:
        b = t.get("bounds_m", [0, 0, 0, 0])
        x0, y0 = world_to_scaled(b[0], b[3])   # 左上
        x1, y1 = world_to_scaled(b[2], b[1])   # 右下
        x0 = max(0, min(x0, scaled_w - 1))
        y0 = max(0, min(y0, scaled_h - 1))
        x1 = max(0, min(x1, scaled_w - 1))
        y1 = max(0, min(y1, scaled_h - 1))

        if t.get("skipped", False):
            color = SKIPPED_COLOR
        else:
            lvl = min(_quadtree_level(t.get("tile_id", "")), len(QUAD_COLORS) - 1)
            color = QUAD_COLORS[lvl]

        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)

    vis = cv2.addWeighted(img, 1.0 - alpha, overlay, alpha, 0)

    for t in tiles:
        if t.get("skipped", False):
            continue
        b = t.get("bounds_m", [0, 0, 0, 0])
        x0, y0 = world_to_scaled(b[0], b[3])
        x1, y1 = world_to_scaled(b[2], b[1])
        lvl = min(_quadtree_level(t.get("tile_id", "")), len(QUAD_COLORS) - 1)
        cv2.rectangle(vis, (x0, y0), (x1, y1), QUAD_COLORS[lvl], line_thickness)

        if show_labels and (y1 - y0) > 30:
            label = f"L{lvl} {t.get('source_points', 0)}pts"
            cv2.putText(vis, label, (x0 + 3, y0 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, QUAD_COLORS[lvl], 1)

    cv2.imwrite(str(output_path), vis)
    return Path(output_path)


# ── DOM 纹理四叉树的分裂可视化 ─────────────────────────────────────

def draw_quadtree_split_overlay(
    dom_path: str | Path,
    tiles: list[dict],
    dom_bounds: list[float],
    img_size: tuple[int, int],
    output_path: str | Path,
    max_side: int = 8000,
    line_thickness: int = 2,
    alpha: float = 0.25,
) -> Path:
    """专门为 DOM 纹理四叉树设计的分裂可视化:
    - 颜色深浅表示 edge_density
    - 跳过区域标灰
    """
    img, scale = load_dom_array(dom_path, max_side)
    scaled_w, scaled_h = img.shape[1], img.shape[0]
    dom_w, dom_h = img_size
    dom_xmin, dom_ymin, dom_xmax, dom_ymax = dom_bounds

    def world_to_scaled(wx: float, wy: float) -> tuple[int, int]:
        px = (wx - dom_xmin) / (dom_xmax - dom_xmin) * dom_w * scale
        py = (dom_ymax - wy) / (dom_ymax - dom_ymin) * dom_h * scale
        return int(px), int(py)

    # 归一化 edge_density 用于颜色映射
    densities = [t.get("edge_density", 0) for t in tiles if not t.get("skipped", False)]
    d_max = max(densities) if densities else 1.0

    overlay = img.copy()

    for t in tiles:
        b = t.get("bounds_m", [0, 0, 0, 0])
        x0, y0 = world_to_scaled(b[0], b[3])
        x1, y1 = world_to_scaled(b[2], b[1])
        x0, y0 = max(0, min(x0, scaled_w - 1)), max(0, min(y0, scaled_h - 1))
        x1, y1 = max(0, min(x1, scaled_w - 1)), max(0, min(y1, scaled_h - 1))

        if t.get("skipped", False):
            cv2.rectangle(overlay, (x0, y0), (x1, y1), SKIPPED_COLOR, -1)
        else:
            intensity = int(255 * (t.get("edge_density", 0) / d_max)) if d_max > 0 else 64
            # 低密度→蓝, 高密度→红暖色
            color = (intensity, 100, 255 - intensity)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)

    vis = cv2.addWeighted(img, 1.0 - alpha, overlay, alpha, 0)

    for t in tiles:
        if t.get("skipped", False):
            continue
        b = t.get("bounds_m", [0, 0, 0, 0])
        x0, y0 = world_to_scaled(b[0], b[3])
        x1, y1 = world_to_scaled(b[2], b[1])
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), line_thickness)
        if (y1 - y0) > 30:
            cv2.putText(vis, f"{t.get('edge_density', 0):.0f}",
                        (x0 + 3, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    cv2.imwrite(str(output_path), vis)
    return Path(output_path)

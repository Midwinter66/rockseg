"""
Visualization helpers for slicing outputs.

Two output variants are supported:
1. paper: compact, manuscript-friendly figure with a small inset legend
2. audit: detailed internal check figure with summary panel and legend
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500000000

PAGE_BG = (255, 255, 255)
PANEL_BG = (250, 250, 250)
PANEL_BORDER = (218, 223, 230)
TEXT_DARK = (35, 42, 52)
TEXT_MUTED = (93, 103, 117)
LEGEND_BG = (255, 255, 255)
LEGEND_BORDER = (205, 212, 220)

KEPT_FILL = (166, 219, 160)
KEPT_OUTLINE = (27, 120, 55)
SKIPPED_FILL = (253, 174, 107)
SKIPPED_OUTLINE = (217, 95, 2)

QUAD_LEVEL_COLORS = [
    (178, 223, 238),  # coarse tiles
    (102, 194, 165),
    (252, 141, 98),
    (141, 160, 203),  # finest shown level
]


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.integer):
            img = (img / np.iinfo(img.dtype).max * 255).astype(np.uint8)
        else:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def load_dom_array(dom_path: str | Path, max_side: int = 8000) -> tuple[np.ndarray, float]:
    dom = Image.open(dom_path)
    w, h = dom.size
    scale = min(max_side / w, max_side / h, 1.0)
    if scale < 1.0:
        dom = dom.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return _ensure_bgr(np.array(dom)), scale


def _clamp_rect(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _draw_text_block(
    canvas: np.ndarray,
    lines: list[str],
    x: int,
    y: int,
    font_scale: float = 0.56,
    color: tuple[int, int, int] = TEXT_DARK,
    line_gap: int = 23,
    thickness: int = 1,
) -> int:
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_gap
    return y


def _draw_legend_items(
    canvas: np.ndarray,
    items: list[tuple[tuple[int, int, int], str]],
    x: int,
    y: int,
    box_size: int = 18,
    row_gap: int = 28,
) -> int:
    for color, label in items:
        cv2.rectangle(canvas, (x, y - box_size + 2), (x + box_size, y + 2), color, -1)
        cv2.rectangle(canvas, (x, y - box_size + 2), (x + box_size, y + 2), (110, 120, 130), 1)
        cv2.putText(
            canvas,
            label,
            (x + box_size + 12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            TEXT_DARK,
            1,
            cv2.LINE_AA,
        )
        y += row_gap
    return y


def _measure_text_width(text: str, font_scale: float, thickness: int = 1) -> int:
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]


def _append_paper_inset(
    image_bgr: np.ndarray,
    title: str,
    legend_items: list[tuple[tuple[int, int, int], str]],
    subtitle: str = "",
    margin: int = 24,
) -> np.ndarray:
    canvas = image_bgr.copy()

    title_scale = 0.76
    title_thickness = 2
    body_scale = 0.5
    body_thickness = 1
    line_gap = 26
    box_size = 18
    legend_gap = 28

    text_widths = [_measure_text_width(title, title_scale, title_thickness)]
    if subtitle:
        text_widths.append(_measure_text_width(subtitle, 0.46, 1))
    for _, label in legend_items:
        text_widths.append(box_size + 12 + _measure_text_width(label, body_scale, body_thickness))

    inset_width = max(280, max(text_widths) + 48)
    inset_height = 52 + len(legend_items) * legend_gap + 24
    if subtitle:
        inset_height += 24

    x0 = margin
    y0 = margin
    x1 = min(canvas.shape[1] - margin, x0 + inset_width)
    y1 = min(canvas.shape[0] - margin, y0 + inset_height)

    cv2.rectangle(canvas, (x0, y0), (x1, y1), LEGEND_BG, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), LEGEND_BORDER, 1)

    y = y0 + 28
    y = _draw_text_block(
        canvas,
        [title],
        x0 + 18,
        y,
        font_scale=title_scale,
        thickness=title_thickness,
        line_gap=30,
    )
    if subtitle:
        y = _draw_text_block(
            canvas,
            [subtitle],
            x0 + 18,
            y,
            font_scale=0.46,
            color=TEXT_MUTED,
            line_gap=22,
        )
    _draw_legend_items(canvas, legend_items, x0 + 18, y + 4, box_size=box_size, row_gap=legend_gap)
    return canvas


def _append_publication_panel(
    image_bgr: np.ndarray,
    title: str,
    subtitle: str,
    summary_lines: list[str],
    legend_items: list[tuple[tuple[int, int, int], str]],
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    panel_width = max(420, int(width * 0.22))
    page = np.full((height, width + panel_width, 3), PAGE_BG, dtype=np.uint8)
    page[:, :width] = image_bgr

    panel = page[:, width:]
    cv2.rectangle(panel, (0, 0), (panel_width - 1, height - 1), PANEL_BORDER, 1)
    cv2.rectangle(panel, (0, 0), (panel_width - 1, height - 1), PANEL_BG, -1)
    cv2.rectangle(panel, (0, 0), (panel_width - 1, 88), (242, 245, 248), -1)

    y = 34
    y = _draw_text_block(panel, [title], 20, y, font_scale=0.78, thickness=2, line_gap=30)
    y = _draw_text_block(panel, [subtitle], 20, y, font_scale=0.48, color=TEXT_MUTED, line_gap=24)

    cv2.line(panel, (20, 102), (panel_width - 20, 102), PANEL_BORDER, 1)
    y = 132
    y = _draw_text_block(panel, ["Summary"], 20, y, font_scale=0.62, thickness=2, line_gap=26)
    y = _draw_text_block(panel, summary_lines, 20, y + 6, font_scale=0.5, color=TEXT_DARK, line_gap=24)

    y += 10
    cv2.line(panel, (20, y), (panel_width - 20, y), PANEL_BORDER, 1)
    y += 30
    y = _draw_text_block(panel, ["Legend"], 20, y, font_scale=0.62, thickness=2, line_gap=26)
    _draw_legend_items(panel, legend_items, 20, y + 8)
    return page


def _quadtree_level(tile_id: str) -> int:
    return max(0, tile_id.count("_") - 1)


def _format_ratio(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _write_overlay_variant(
    vis: np.ndarray,
    output_path: str | Path,
    variant: str,
    title: str,
    subtitle: str,
    summary_lines: list[str],
    legend_items: list[tuple[tuple[int, int, int], str]],
) -> Path:
    if variant == "paper":
        page = _append_paper_inset(vis, title=title, subtitle=subtitle, legend_items=legend_items)
    elif variant == "audit":
        page = _append_publication_panel(
            vis,
            title=title,
            subtitle=subtitle,
            summary_lines=summary_lines,
            legend_items=legend_items,
        )
    else:
        raise ValueError(f"Unsupported overlay variant: {variant}")

    cv2.imwrite(str(output_path), page)
    return Path(output_path)


def draw_sahi_overlay(
    dom_path: str | Path,
    patches: list[dict],
    output_path: str | Path,
    max_side: int = 8000,
    line_thickness: int = 2,
    alpha: float = 0.34,
    stats: dict | None = None,
    config: dict | None = None,
    variant: str = "audit",
) -> Path:
    img, scale = load_dom_array(dom_path, max_side)
    overlay = img.copy()

    kept_count = 0
    skipped_count = 0
    for patch in patches:
        x = int(patch["pixel_origin"][0] * scale)
        y = int(patch["pixel_origin"][1] * scale)
        size = int(patch["pixel_size"] * scale)
        x0, y0, x1, y1 = _clamp_rect(x, y, x + size, y + size, img.shape[1], img.shape[0])
        is_kept = patch.get("status") == "kept"
        color = KEPT_FILL if is_kept else SKIPPED_FILL
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
        kept_count += int(is_kept)
        skipped_count += int(not is_kept)

    vis = cv2.addWeighted(img, 1.0 - alpha, overlay, alpha, 0)

    for patch in patches:
        x = int(patch["pixel_origin"][0] * scale)
        y = int(patch["pixel_origin"][1] * scale)
        size = int(patch["pixel_size"] * scale)
        x0, y0, x1, y1 = _clamp_rect(x, y, x + size, y + size, img.shape[1], img.shape[0])
        if patch.get("status") == "kept":
            cv2.rectangle(vis, (x0, y0), (x1, y1), KEPT_OUTLINE, line_thickness)
        else:
            cv2.rectangle(vis, (x0, y0), (x1, y1), SKIPPED_OUTLINE, 1)

    patch_size_px = 0
    overlap = 0.0
    min_content_ratio = 0.0
    coverage_ratio = 0.0
    if stats:
        patch_size_px = int(stats.get("patch_size_distribution", {}).get("pixel_size", 0))
        coverage_ratio = float(stats.get("coverage_ratio", 0.0))
    if config:
        patch_cfg = config.get("patching", {})
        overlap = float(patch_cfg.get("overlap", 0.0))
        min_content_ratio = float(patch_cfg.get("min_content_ratio", 0.0))

    summary_lines = [
        f"Total patches: {len(patches):,}",
        f"Kept patches: {kept_count:,}",
        f"Skipped patches: {skipped_count:,}",
        f"Coverage ratio: {_format_ratio(coverage_ratio)}",
        f"Patch size: {patch_size_px}px",
        f"Overlap: {overlap:.2f}",
        f"Min content ratio: {min_content_ratio:.2f}",
    ]
    legend_items = [
        (KEPT_FILL, "Kept tile footprint"),
        (SKIPPED_FILL, "Skipped tile footprint"),
        (KEPT_OUTLINE, "Kept tile boundary"),
    ]

    return _write_overlay_variant(
        vis=vis,
        output_path=output_path,
        variant=variant,
        title="SAHI Slicing Overlay",
        subtitle="Green tiles are retained for detection; orange tiles are skipped.",
        summary_lines=summary_lines,
        legend_items=legend_items,
    )


def draw_quadtree_overlay(
    dom_path: str | Path,
    tiles: list[dict],
    dom_bounds: list[float],
    img_size: tuple[int, int],
    output_path: str | Path,
    max_side: int = 8000,
    line_thickness: int = 2,
    alpha: float = 0.28,
    show_labels: bool = True,
    stats: dict | None = None,
    config: dict | None = None,
    variant: str = "audit",
) -> Path:
    img, scale = load_dom_array(dom_path, max_side)
    scaled_w, scaled_h = img.shape[1], img.shape[0]
    dom_w, dom_h = img_size
    dom_xmin, dom_ymin, dom_xmax, dom_ymax = dom_bounds

    def world_to_scaled(wx: float, wy: float) -> tuple[int, int]:
        px = (wx - dom_xmin) / (dom_xmax - dom_xmin) * dom_w * scale
        py = (dom_ymax - wy) / (dom_ymax - dom_ymin) * dom_h * scale
        return int(px), int(py)

    overlay = img.copy()
    level_counts = [0] * len(QUAD_LEVEL_COLORS)
    kept_tiles = 0
    skipped_tiles = 0

    for tile in tiles:
        bounds = tile.get("bounds_m", [0, 0, 0, 0])
        x0, y0 = world_to_scaled(bounds[0], bounds[3])
        x1, y1 = world_to_scaled(bounds[2], bounds[1])
        x0, y0, x1, y1 = _clamp_rect(x0, y0, x1, y1, scaled_w, scaled_h)

        if tile.get("skipped", False):
            cv2.rectangle(overlay, (x0, y0), (x1, y1), SKIPPED_FILL, -1)
            skipped_tiles += 1
            continue

        level = min(_quadtree_level(tile.get("tile_id", "")), len(QUAD_LEVEL_COLORS) - 1)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), QUAD_LEVEL_COLORS[level], -1)
        level_counts[level] += 1
        kept_tiles += 1

    vis = cv2.addWeighted(img, 1.0 - alpha, overlay, alpha, 0)

    for tile in tiles:
        bounds = tile.get("bounds_m", [0, 0, 0, 0])
        x0, y0 = world_to_scaled(bounds[0], bounds[3])
        x1, y1 = world_to_scaled(bounds[2], bounds[1])
        x0, y0, x1, y1 = _clamp_rect(x0, y0, x1, y1, scaled_w, scaled_h)

        if tile.get("skipped", False):
            cv2.rectangle(vis, (x0, y0), (x1, y1), SKIPPED_OUTLINE, 1)
            continue

        level = min(_quadtree_level(tile.get("tile_id", "")), len(QUAD_LEVEL_COLORS) - 1)
        cv2.rectangle(vis, (x0, y0), (x1, y1), QUAD_LEVEL_COLORS[level], line_thickness)
        if show_labels and (y1 - y0) > 36 and (x1 - x0) > 64:
            cv2.putText(
                vis,
                f"L{level}",
                (x0 + 4, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    base_size = 0.0
    min_size = 0.0
    overlap_m = 0.0
    coverage_ratio = 0.0
    if stats:
        coverage_ratio = float(stats.get("coverage_ratio", 0.0))
    if config:
        cover_cfg = config.get("cover", {})
        base_size = float(cover_cfg.get("base_tile_size_m", 0.0))
        min_size = float(cover_cfg.get("min_tile_size_m", 0.0))
        overlap_m = float(cover_cfg.get("tile_overlap_m", 0.0))

    summary_lines = [
        f"Total tiles: {len(tiles):,}",
        f"Kept tiles: {kept_tiles:,}",
        f"Skipped tiles: {skipped_tiles:,}",
        f"Coverage ratio: {_format_ratio(coverage_ratio)}",
        f"Base tile size: {base_size:.2f} m",
        f"Minimum tile size: {min_size:.2f} m",
        f"Tile overlap: {overlap_m:.2f} m",
        f"Level counts: {', '.join(f'L{i}={count}' for i, count in enumerate(level_counts) if count > 0)}",
    ]
    legend_items = [
        (QUAD_LEVEL_COLORS[0], "Level 0 / coarse retained tile"),
        (QUAD_LEVEL_COLORS[1], "Level 1 retained tile"),
        (QUAD_LEVEL_COLORS[2], "Level 2 retained tile"),
        (QUAD_LEVEL_COLORS[3], "Level 3+ retained tile"),
        (SKIPPED_FILL, "Skipped tile (empty or low-content)"),
    ]

    return _write_overlay_variant(
        vis=vis,
        output_path=output_path,
        variant=variant,
        title="Quadtree Slicing Overlay",
        subtitle="Tile colors indicate final split level; orange tiles were skipped.",
        summary_lines=summary_lines,
        legend_items=legend_items,
    )


def draw_quadtree_split_overlay(
    dom_path: str | Path,
    tiles: list[dict],
    dom_bounds: list[float],
    img_size: tuple[int, int],
    output_path: str | Path,
    max_side: int = 8000,
    line_thickness: int = 2,
    alpha: float = 0.28,
    stats: dict | None = None,
    config: dict | None = None,
    variant: str = "audit",
) -> Path:
    return draw_quadtree_overlay(
        dom_path=dom_path,
        tiles=tiles,
        dom_bounds=dom_bounds,
        img_size=img_size,
        output_path=output_path,
        max_side=max_side,
        line_thickness=line_thickness,
        alpha=alpha,
        show_labels=(variant == "audit"),
        stats=stats,
        config=config,
        variant=variant,
    )

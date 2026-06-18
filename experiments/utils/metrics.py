"""
切片实验统一指标计算

每个切片方法的输出格式统一为:
    {
        "method": str,                       # 切片方法名称
        "config": dict,                      # 使用参数
        "total_patches": int,                # 总共生成的窗口数
        "kept_patches": int,                 # 有效切片数
        "skipped_patches": int,              # 跳过数
        "coverage_ratio": float,             # 有效覆盖面积 / DOM总面积
        "skipped_area_ratio": float,         # 跳过面积 / DOM总面积
        "elapsed_seconds": float,            # 切片耗时
        "patch_size_distribution": {         # 切片大小分布
            "min_px": int, "max_px": int, "mean_px": float, "std_px": float
        },
        "patches": list[dict],               # 每个切片的详细记录
    }
"""

from __future__ import annotations
import time
from typing import Any


def compute_sahi_stats(
    method_name: str,
    config: dict,
    all_patches: list[dict],
    dom_width: int,
    dom_height: int,
    elapsed: float,
) -> dict[str, Any]:
    """计算 SAHI / 固定滑窗 切片的统计指标"""
    kept = [p for p in all_patches if p.get("status") == "kept"]
    skipped = [p for p in all_patches if p.get("status") != "kept"]

    dom_area = dom_width * dom_height

    # 用 kept patches 的像素包围盒计算实际唯一覆盖面积（考虑重叠）
    if kept:
        min_x = min(p["pixel_origin"][0] for p in kept)
        min_y = min(p["pixel_origin"][1] for p in kept)
        max_x = max(p["pixel_origin"][0] + p["pixel_size"] for p in kept)
        max_y = max(p["pixel_origin"][1] + p["pixel_size"] for p in kept)
        unique_area = (max_x - min_x) * (max_y - min_y)
    else:
        unique_area = 0

    # SAHI patch 大小统计
    patch_sizes = [p.get("pixel_size", 0) for p in all_patches]

    return {
        "method": method_name,
        "config": config,
        "dom_dims": {"width_px": dom_width, "height_px": dom_height},
        "total_patches": len(all_patches),
        "kept_patches": len(kept),
        "skipped_patches": len(skipped),
        "coverage_ratio": round(min(1.0, unique_area / dom_area), 4) if dom_area else 0,
        "skipped_ratio": round(
            sum(p.get("content_ratio", 0) for p in skipped) / max(len(skipped), 1), 4
        ),
        "elapsed_seconds": round(elapsed, 3),
        "patch_size_distribution": {
            "min_px": int(min(patch_sizes)) if patch_sizes else 0,
            "max_px": int(max(patch_sizes)) if patch_sizes else 0,
            "mean_px": round(float(sum(patch_sizes) / len(patch_sizes)), 1) if patch_sizes else 0,
            "pixel_size": int(patch_sizes[0]) if patch_sizes else 0,
            "all_same_size": len(set(int(s) for s in patch_sizes)) == 1,
        },
        "patches": all_patches,
    }


def compute_quadtree_stats(
    method_name: str,
    config: dict,
    tiles: list[dict],
    dom_bounds_world: list[float],
    elapsed: float,
) -> dict[str, Any]:
    """计算四叉树自适应切片的统计指标"""
    kept = [t for t in tiles if not t.get("skipped", False)]
    skipped = [t for t in tiles if t.get("skipped", False)]

    # 世界坐标 转 像素 (近似: 1m ≈ 分辨率 px)
    # DOM bounds: [xmin, ymin, xmax, ymax]
    dx = dom_bounds_world[2] - dom_bounds_world[0]
    dy = dom_bounds_world[3] - dom_bounds_world[1]
    dom_area_m2 = dx * dy

    # tile 面积（用于分布统计，保留原始值）
    tile_areas_m2 = []
    for t in tiles:
        b = t.get("bounds_m", [0, 0, 0, 0])
        area = (b[2] - b[0]) * (b[3] - b[1])
        tile_areas_m2.append(area)

    # 唯一覆盖面积：kept tiles 的包围盒（避免 tile_overlap 导致面积重复计算）
    if kept:
        min_x = min(t["bounds_m"][0] for t in kept)
        min_y = min(t["bounds_m"][1] for t in kept)
        max_x = max(t["bounds_m"][2] for t in kept)
        max_y = max(t["bounds_m"][3] for t in kept)
        unique_area_m2 = (max_x - min_x) * (max_y - min_y)
    else:
        unique_area_m2 = 0

    return {
        "method": method_name,
        "config": config,
        "dom_bounds_world": dom_bounds_world,
        "dom_area_m2": round(dom_area_m2, 2),
        "total_tiles": len(tiles),
        "kept_tiles": len(kept),
        "skipped_tiles": len(skipped),
        "coverage_ratio": round(min(1.0, unique_area_m2 / dom_area_m2), 4) if dom_area_m2 else 0,
        "unique_coverage_m2": round(unique_area_m2, 2),
        "elapsed_seconds": round(elapsed, 3),
        "tile_size_distribution_m2": {
            "min": round(min(tile_areas_m2), 2) if tile_areas_m2 else 0,
            "max": round(max(tile_areas_m2), 2) if tile_areas_m2 else 0,
            "mean": round(sum(tile_areas_m2) / len(tile_areas_m2), 2) if tile_areas_m2 else 0,
        },
        "tiles": tiles,
    }

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
    # SAHI patch 大小固定
    patch_sizes = [p.get("pixel_size", 0) for p in all_patches]
    covered_area = sum(sz * sz for sz in patch_sizes)  # 不考虑重叠的覆盖

    return {
        "method": method_name,
        "config": config,
        "dom_dims": {"width_px": dom_width, "height_px": dom_height},
        "total_patches": len(all_patches),
        "kept_patches": len(kept),
        "skipped_patches": len(skipped),
        "coverage_ratio": round(covered_area / dom_area, 4) if dom_area else 0,
        "skipped_area_ratio": round(
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

    # tile 面积
    tile_areas_m2 = []
    for t in tiles:
        b = t.get("bounds_m", [0, 0, 0, 0])
        area = (b[2] - b[0]) * (b[3] - b[1])
        tile_areas_m2.append(area)

    return {
        "method": method_name,
        "config": config,
        "dom_bounds_world": dom_bounds_world,
        "dom_area_m2": round(dom_area_m2, 2),
        "total_tiles": len(tiles),
        "kept_tiles": len(kept),
        "skipped_tiles": len(skipped),
        "coverage_ratio": round(sum(tile_areas_m2) / dom_area_m2, 4) if dom_area_m2 else 0,
        "elapsed_seconds": round(elapsed, 3),
        "tile_size_distribution_m2": {
            "min": round(min(tile_areas_m2), 2) if tile_areas_m2 else 0,
            "max": round(max(tile_areas_m2), 2) if tile_areas_m2 else 0,
            "mean": round(sum(tile_areas_m2) / len(tile_areas_m2), 2) if tile_areas_m2 else 0,
        },
        "tiles": tiles,
    }

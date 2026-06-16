"""
3D 石块可视化 — 显示单个石块的点云 + 体积信息

两种模式:
  - 单石块: 指定 stone_id，显示该石块的三维形态、BBox、体积
  - 批量预览: 显示所有石块的缩略图列表
"""

from __future__ import annotations
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Any


def visualize_stone(stone: dict, points: np.ndarray | None = None,
                    show_bbox: bool = True, show_axes: bool = True,
                    window_name: str = "Stone Viewer") -> None:
    """在 Open3D 窗口中显示单个石块

    显示内容:
      - 石块点云 (灰色)
      - 3D BBox (绿色)
      - 坐标轴
      - 体积/尺寸信息 (窗口标题)
    """
    pcd = o3d.geometry.PointCloud()
    if points is not None:
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.72, 0.72, 0.72])  # 灰色
    elif stone.get("pointcloud_path") and Path(stone["pointcloud_path"]).exists():
        pcd = o3d.io.read_point_cloud(stone["pointcloud_path"])
    else:
        print(f"  [ERROR] No point cloud for stone {stone.get('stone_id')}")
        return

    geoms = [pcd]

    # 3D BBox
    if show_bbox and stone.get("bbox_3d"):
        b = stone["bbox_3d"]
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[b[0], b[1], b[2]],
            max_bound=[b[3], b[4], b[5]],
        )
        bbox.color = (0.0, 0.8, 0.0)  # 绿色
        geoms.append(bbox)

    # 坐标轴
    if show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        geoms.append(axes)

    # 构建信息文本
    info = []
    info.append(f"Stone: {stone.get('stone_id', 'N/A')}")
    info.append(f"Volume: {stone.get('volume_m3', 0):.4f} m³")
    info.append(f"Diameter: {stone.get('equivalent_diameter_m', 0):.2f} m")
    info.append(f"Points: {stone.get('point_count', len(pcd.points))}")
    if stone.get("surface_area_m2"):
        info.append(f"Surface: {stone['surface_area_m2']:.3f} m²")

    print("\n" + "\n".join(info))
    print("\n[Open3D Window] 鼠标拖拽旋转 | 滚轮缩放 | Ctrl+拖拽平移")

    o3d.visualization.draw_geometries(
        geoms,
        window_name=window_name,
        width=1024, height=768,
        left=50, top=50,
    )


def visualize_multiple_stones(stones: list[dict], stone_dir: str | Path,
                               max_stones: int = 50,
                               show_bbox: bool = True) -> None:
    """批量显示多个石块的点云（不同颜色）"""
    stone_dir = Path(stone_dir)
    colors = plt_colors(len(stones[:max_stones]))

    geoms = []
    for i, s in enumerate(stones[:max_stones]):
        ply_path = stone_dir / f"{s['stone_id']}.ply"
        if not ply_path.exists():
            continue
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if len(pcd.points) == 0:
            continue
        c = colors[i % len(colors)]
        pcd.paint_uniform_color(c)
        geoms.append(pcd)

        if show_bbox and s.get("bbox_3d"):
            b = s["bbox_3d"]
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=[b[0], b[1], b[2]],
                max_bound=[b[3], b[4], b[5]],
            )
            bbox.color = c
            geoms.append(bbox)

    if not geoms:
        print("  No stone point clouds found")
        return

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    geoms.append(axes)

    print(f"\nShowing {len(geoms)} stones (different colors)")
    o3d.visualization.draw_geometries(
        geoms, window_name="Multi-Stone Viewer",
        width=1280, height=800,
    )


def plt_colors(n: int):
    """生成 n 种不同的颜色"""
    import random
    random.seed(42)
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0  # 黄金角分布
        # HSV to RGB
        import colorsys
        colors.append(colorsys.hsv_to_rgb(hue, 0.8, 0.9))
    return colors

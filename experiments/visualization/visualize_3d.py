"""
3D stone visualization — displays stone point clouds from pipeline results

Two modes:
  - Single stone: shows 3D shape, bbox, volume info
  - Batch preview: shows multiple stones in different colors

All views translate to origin so Open3D camera works naturally.
"""

from __future__ import annotations
import numpy as np
import open3d as o3d
from pathlib import Path


def _open_viewer(geoms, title, width=1024, height=768, point_size=1):
    """Open Open3D window with white bg and configurable point size."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=width, height=height)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = point_size        # 小点，绵密感
    opt.background_color = (0.96, 0.96, 0.96)  # 白底
    vis.run()
    vis.destroy_window()


def visualize_stone(stone: dict, points: np.ndarray | None = None,
                    show_bbox: bool = True,
                    window_name: str = "Stone Viewer") -> None:
    """Display a single stone. Loads from PLY or accepts points directly.

    Translates to origin so large CGCS2000 coordinates work correctly.
    """
    pcd = o3d.geometry.PointCloud()

    if points is not None:
        # External points passed in — translate to origin
        centroid = points.mean(axis=0)
        pts_origin = points - centroid
        pcd.points = o3d.utility.Vector3dVector(pts_origin)
        # Height-based grayscale for depth
        z = pts_origin[:, 2]
        zn = (z - z.min()) / (z.max() - z.min() + 1e-8)
        gray = 0.3 + 0.4 * zn
        pcd.paint_uniform_color([0.55, 0.55, 0.55])
    elif stone.get("pointcloud_path") and Path(stone["pointcloud_path"]).exists():
        # Load from PLY
        pcd = o3d.io.read_point_cloud(stone["pointcloud_path"])
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            print(f"  Empty point cloud: {stone.get('stone_id')}")
            return
        centroid = pts.mean(axis=0)
        pcd.translate(-centroid)
        pcd.paint_uniform_color([0.55, 0.55, 0.55])
    else:
        print(f"  No point cloud for stone {stone.get('stone_id')}")
        return

    geoms = [pcd]

    # Green bbox, translated
    if show_bbox and stone.get("bbox_3d"):
        b = np.array(stone["bbox_3d"])
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=b[:3] - centroid, max_bound=b[3:] - centroid)
        bbox.color = (0.0, 0.7, 0.0)
        geoms.append(bbox)

    # Info
    pts = np.asarray(pcd.points)
    print(f"\n  Stone: {stone.get('stone_id', '?')}")
    print(f"  Points: {len(pts)}")
    if len(pts) > 0:
        print(f"  Size: {np.ptp(pts[:,0]):.2f} x {np.ptp(pts[:,1]):.2f} x {np.ptp(pts[:,2]):.2f} m")
    print(f"  Volume: {stone.get('volume_m3', 0):.4f} m3")
    print(f"  Diameter: {stone.get('equivalent_diameter_m', 0):.3f} m")

    _open_viewer(geoms, window_name)


def visualize_multiple_stones(stones: list[dict], stone_dir: str | Path,
                               max_stones: int = 50) -> None:
    """Show all stones at their real relative positions (map view)."""
    stone_dir = Path(stone_dir)
    import colorsys

    # Collect centroids to compute scene center
    centroids = []
    for s in stones[:max_stones]:
        ply_path = stone_dir / f"{s['stone_id']}.ply"
        if not ply_path.exists():
            continue
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points)
        if len(pts) >= 4:
            centroids.append(pts.mean(axis=0))

    if not centroids:
        print("  No stone point clouds found")
        return

    scene_center = np.mean(centroids, axis=0)
    geoms = []

    for i, s in enumerate(stones[:max_stones]):
        ply_path = stone_dir / f"{s['stone_id']}.ply"
        if not ply_path.exists():
            continue
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points)
        if len(pts) < 4:
            continue

        # Real position, centered on scene center
        pcd.translate(-scene_center)

        hue = (i * 0.618033988749895) % 1.0
        pcd.paint_uniform_color(colorsys.hsv_to_rgb(hue, 0.6, 0.8))
        geoms.append(pcd)

    print(f"\n  Showing {len(geoms)} stones at original positions")
    _open_viewer(geoms, "Stone Map View", width=1280, height=800, point_size=2.0)

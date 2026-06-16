"""
石块体积计算方法

支持三种算法:
  1. convex_hull  — 凸包法（最快，略高估）
  2. alpha_shape  — α-形状（推荐，兼顾精度与速度）
  3. grid_2d5     — 2.5D 栅格法（最物理精确，需地面点）

所有方法返回体积(m³)，以及可选的辅助数据。
"""

from __future__ import annotations
import numpy as np
import open3d as o3d
from typing import Any

# 抑制 Open3D 的 alpha shape / qhull 警告（不影响计算结果）
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def compute_volume(points: np.ndarray, method: str = "grid_2d5",
                   alpha: float = 0.5, voxel_size: float = 0.03,
                   **kwargs) -> dict[str, Any]:
    """计算石块点云的体积

    Args:
        points: (N, 3) float32 点云
        method: "convex_hull" | "alpha_shape" | "grid_2d5"
        alpha: α-形状参数，越小越贴合凹面
        voxel_size: 预降采样体素大小

    Returns:
        {"volume_m3": float, "method": str, "surface_area_m2": float,
         "point_count": int, "note": str}
    """
    if len(points) < 4:
        return {"volume_m3": 0.0, "method": method,
                "surface_area_m2": 0.0, "point_count": len(points),
                "note": "点数不足"}

    # 降采样
    if voxel_size > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size)
        pts = np.asarray(pcd.points, dtype=np.float32)
    else:
        pts = points

    if len(pts) < 4:
        pts = points  # 降采样后不够，用原始点

    if method == "convex_hull":
        return _convex_hull_volume(pts)
    elif method == "alpha_shape":
        return _alpha_shape_volume(pts, alpha)
    elif method == "grid_2d5":
        return _grid_2d5_volume(pts, kwargs.get("grid_resolution", 0.1))
    else:
        raise ValueError(f"Unknown volume method: {method}")


def _convex_hull_volume(points: np.ndarray) -> dict[str, Any]:
    """凸包法 — 快速，对凹形石块略微高估"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    try:
        hull, _ = pcd.compute_convex_hull()
        vol = hull.get_volume()
        area = hull.get_surface_area()
        return {
            "volume_m3": float(vol),
            "surface_area_m2": float(area),
            "point_count": len(points),
            "method": "convex_hull",
            "note": "凸包法，对凹面可能高估",
        }
    except Exception as e:
        return {"volume_m3": 0.0, "surface_area_m2": 0.0,
                "point_count": len(points), "method": "convex_hull",
                "note": f"凸包计算失败: {e}"}


def _alpha_shape_volume(points: np.ndarray, alpha: float) -> dict[str, Any]:
    """α-形状 — 通过 α 参数控制曲面紧贴程度

    alpha 会自动根据点云边界框自适应：
      - 若指定的 alpha 导致空网格，会逐步放大到对角线长度的 50%
      - 若始终失败则回退到凸包
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 计算自适应 alpha 范围
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(np.asarray(bbox.get_max_bound()) - np.asarray(bbox.get_min_bound()))
    # alpha 不应超过对角线的一半（否则接近凸包），也不应太小
    alpha_min = max(alpha, diag * 0.05)  # 至少对角线 5%
    alpha_max = diag * 0.5               # 最多对角线 50%

    # 尝试递增 alpha 值，直到重建成功
    alphas_to_try = sorted(set([
        alpha_min,
        alpha_min * 2,
        alpha_min * 4,
        alpha_max * 0.3,
        alpha_max * 0.6,
        alpha_max,
    ]))

    last_error = ""
    for a in alphas_to_try:
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, a)
            if len(mesh.vertices) < 4 or len(mesh.triangles) < 4:
                continue
            if mesh.is_watertight():
                vol = mesh.get_volume()
                area = mesh.get_surface_area()
                if vol > 0:
                    return {
                        "volume_m3": float(vol),
                        "surface_area_m2": float(area),
                        "point_count": len(points),
                        "method": "alpha_shape",
                        "alpha": round(a, 4),
                        "note": f"α-形状水密网格 (α={a:.3f})",
                    }
        except Exception as e:
            last_error = str(e)
            continue

    # 全部失败 → 尝试泊松表面重建（对只有顶面的点云更友好）
    try:
        # 估计法线（泊松重建需要）
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=diag * 0.1, max_nn=30))
        # 定向法线（朝外）
        pcd.orient_normals_consistent_tangent_plane(30)
        poisson_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, width=0, scale=1.1, linear_fit=False)
        if len(poisson_mesh.vertices) > 10 and len(poisson_mesh.triangles) > 10:
            poisson_mesh = poisson_mesh.filter_smooth_simple(number_of_iterations=1)
            if poisson_mesh.is_watertight():
                vol = poisson_mesh.get_volume()
                area = poisson_mesh.get_surface_area()
                if vol > 0:
                    return {
                        "volume_m3": float(vol),
                        "surface_area_m2": float(area),
                        "point_count": len(points),
                        "method": "poisson",
                        "note": "泊松表面重建 (alpha shape 失败后)",
                    }
    except Exception as e:
        last_error = str(e)

    # 全部失败 → 凸包兜底
    fallback = _convex_hull_volume(points)
    fallback["note"] = f"α-形状全部失败，回退凸包 (alphas tried: {alphas_to_try})"
    fallback["method"] = "alpha_shape(fallback)"
    if last_error:
        fallback["note"] += f" | {last_error}"
    return fallback


def _grid_2d5_volume(points: np.ndarray,
                     grid_resolution: float = 0.1) -> dict[str, Any]:
    """2.5D 栅格法 — 将点云投影到 XY 网格，用 Z 值差计算体积

    适用于地面上的石块：顶部有点云覆盖，底部为地面。
    通过 XY 网格上每个 cell 的 Z 范围累加体积。

    grid_resolution: 网格分辨率(m)，越小越精细
    """
    if len(points) < 4:
        return {"volume_m3": 0.0, "surface_area_m2": 0.0,
                "point_count": len(points), "method": "grid_2d5",
                "note": "点数不足"}

    xy = points[:, :2]
    z = points[:, 2]

    # 计算网格范围
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)

    # 如果石块几乎是平的，直接返回 0
    z_range = z.max() - z.min()
    if z_range < grid_resolution:
        return {"volume_m3": 0.0, "surface_area_m2": 0.0,
                "point_count": len(points), "method": "grid_2d5",
                "note": "石块过于扁平"}

    # 构建网格
    nx = max(1, int(np.ceil((xmax - xmin) / grid_resolution)))
    ny = max(1, int(np.ceil((ymax - ymin) / grid_resolution)))

    # 网格化：每个 grid cell 记录点的 Z 值
    # 使用 vectorized 操作
    xi = np.floor((xy[:, 0] - xmin) / grid_resolution).astype(np.int32)
    yi = np.floor((xy[:, 1] - ymin) / grid_resolution).astype(np.int32)
    xi = np.clip(xi, 0, nx - 1)
    yi = np.clip(yi, 0, ny - 1)

    # 每个 cell 的 Z 范围
    cell_z_range = np.zeros((ny, nx), dtype=np.float32)
    cell_count = np.zeros((ny, nx), dtype=np.int32)
    cell_z_min = np.full((ny, nx), np.inf, dtype=np.float32)
    cell_z_max = np.full((ny, nx), -np.inf, dtype=np.float32)

    for i in range(len(xi)):
        cx, cy = xi[i], yi[i]
        cell_z_min[cy, cx] = min(cell_z_min[cy, cx], z[i])
        cell_z_max[cy, cx] = max(cell_z_max[cy, cx], z[i])
        cell_count[cy, cx] += 1

    # 有效 cell（有点覆盖）
    valid = cell_count > 0
    if not valid.any():
        return {"volume_m3": 0.0, "surface_area_m2": 0.0,
                "point_count": len(points), "method": "grid_2d5",
                "note": "无有效网格"}

    # 拟合地面高度：取所有有效 cell 的 Z 最小值附近
    # 假设石块底部是地面，取周边最低点作为地面
    ground_z = np.percentile(z, 5)  # 取底部 5% 分位作为地面

    # 每个 cell 的体积 = cell_area * (z_max - ground_z)
    cell_area = grid_resolution ** 2
    z_heights = cell_z_max[valid] - ground_z
    z_heights = np.maximum(z_heights, 0)
    volume = float(np.sum(z_heights) * cell_area)

    # 表面积 = 有效 cell 面积之和
    surface_area = float(np.sum(valid) * cell_area)

    return {
        "volume_m3": round(volume, 4),
        "surface_area_m2": round(surface_area, 4),
        "point_count": len(points),
        "grid_nx": nx,
        "grid_ny": ny,
        "grid_resolution": grid_resolution,
        "ground_z": round(float(ground_z), 3),
        "method": "grid_2d5",
        "note": "2.5D栅格法，地面取底部5%分位",
    }

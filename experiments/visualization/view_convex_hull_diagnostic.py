"""
Legacy diagnostic for comparing one stone's 2.5D estimate with a convex hull.

This script is not part of the main paper volume comparison. The reported
pipeline compares the 2D equivalent-sphere proxy with GroundDEM-based 2.5D
integration. Keep this tool only for investigating shell-like point clouds and
convex-hull overestimation.

Examples:
  python experiments/visualization/view_convex_hull_diagnostic.py --source quadtree_dom --method correlation_clustering --list-stones 20
  python experiments/visualization/view_convex_hull_diagnostic.py --source quadtree_dom --method correlation_clustering --stone-rank 0
  python experiments/visualization/view_convex_hull_diagnostic.py --source quadtree_dom --method correlation_clustering --stone-id stone_005283
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.scene_reference import CURRENT_SCENE
from experiments.common.pointcloud_index import PointCloudXYGridIndex
from experiments.common.stone_region import crop_stone_point_cloud
from experiments.volume.estimators import (
    _xy_convex_hull,
    estimate_2d5_with_ground,
    estimate_convex_hull,
)
from experiments.volume.ground_estimator import GroundDEM
from experiments.volume.run_volume import _load_fusion
from experiments.visualization.view_stone_mapping import (
    _find_stone,
    _load_local_scene_points,
    _select_stones,
    _sorted_stones,
)

SOURCES = ["sahi", "quadtree_dom"]
METHODS = ["heuristic", "correlation_clustering"]
STONE_MODES = ["accepted", "rejected", "all"]
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "visualization" / "outputs" / "volume_case"


def _load_volume_config() -> dict:
    path = PROJECT_ROOT / "experiments" / "volume" / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[np.sort(indices)]


def _build_convex_hull_mesh(points: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        import open3d as o3d
    except ImportError:
        return None, None

    if len(points) < 4:
        return None, None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    try:
        hull, _ = pcd.compute_convex_hull()
    except Exception:
        return None, None
    if hull is None or len(hull.vertices) < 4 or len(hull.triangles) < 4:
        return None, None
    return np.asarray(hull.vertices), np.asarray(hull.triangles)


def _set_axes_equal(ax, points: np.ndarray) -> None:
    if len(points) == 0:
        return
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins)) / 2.0
    if radius <= 0:
        radius = 0.5
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _style_3d_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("X (m)", labelpad=6)
    ax.set_ylabel("Y (m)", labelpad=6)
    ax.set_zlabel("Z (m)", labelpad=6)
    ax.grid(False)
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))


def _plot_point_cloud(ax, points: np.ndarray, title: str) -> None:
    sampled = points
    color_values = sampled[:, 2] if len(sampled) > 0 else np.asarray([])
    ax.scatter(
        sampled[:, 0],
        sampled[:, 1],
        sampled[:, 2],
        c=color_values,
        cmap="viridis",
        s=2,
        alpha=0.75,
        linewidths=0,
    )
    _style_3d_axis(ax, title)
    _set_axes_equal(ax, sampled)
    ax.view_init(elev=24, azim=-62)


def _plot_convex_hull(ax, points: np.ndarray, hull_vertices: np.ndarray | None, hull_triangles: np.ndarray | None, title: str) -> None:
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="#7c8a99", s=1.5, alpha=0.40, linewidths=0)
    if hull_vertices is not None and hull_triangles is not None:
        max_triangles = min(len(hull_triangles), 800)
        tri_indices = np.linspace(0, len(hull_triangles) - 1, num=max_triangles, dtype=np.int64)
        sampled_triangles = hull_triangles[tri_indices]
        lines = []
        for tri in sampled_triangles:
            a, b, c = hull_vertices[tri]
            lines.append([a, b])
            lines.append([b, c])
            lines.append([c, a])
        wire = Line3DCollection(
            lines,
            colors=(0.78, 0.12, 0.10, 0.55),
            linewidths=0.45,
        )
        ax.add_collection3d(wire)
        bounds = np.vstack([points, hull_vertices])
    else:
        bounds = points
    _style_3d_axis(ax, title)
    _set_axes_equal(ax, bounds)
    ax.view_init(elev=24, azim=-62)


def _plot_2d5(ax, debug: dict | None, title: str) -> None:
    if not debug or len(debug.get("positive_centers_xy", [])) == 0:
        ax.text2D(0.05, 0.5, "No positive 2.5D cells", transform=ax.transAxes, fontsize=12)
        _style_3d_axis(ax, title)
        return

    centers = np.asarray(debug["positive_centers_xy"], dtype=np.float64)
    base_z = np.asarray(debug["positive_ground_z"], dtype=np.float64)
    heights = np.asarray(debug["positive_heights_m"], dtype=np.float64)
    top_z = base_z + heights
    colors = plt.cm.Blues(np.clip(heights / max(float(np.max(heights)), 1e-6), 0.2, 1.0))

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        base_z,
        c="#cfd8dc",
        s=5,
        alpha=0.45,
        linewidths=0,
    )
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        top_z,
        c=colors,
        s=10,
        alpha=0.92,
        linewidths=0,
    )

    line_step = max(1, len(centers) // 220)
    for idx in range(0, len(centers), line_step):
        ax.plot(
            [centers[idx, 0], centers[idx, 0]],
            [centers[idx, 1], centers[idx, 1]],
            [base_z[idx], top_z[idx]],
            color="#4f81bd",
            alpha=0.28,
            linewidth=0.7,
        )

    bounds = np.vstack([
        np.column_stack((centers[:, 0], centers[:, 1], base_z)),
        np.column_stack((centers[:, 0], centers[:, 1], top_z)),
    ])
    _style_3d_axis(ax, title)
    _set_axes_equal(ax, bounds)
    ax.view_init(elev=24, azim=-62)


def _plot_footprint(ax, points: np.ndarray, hull_vertices: np.ndarray | None, debug: dict | None, cell_size: float, title: str) -> None:
    ax.scatter(points[:, 0], points[:, 1], s=2, c="#555555", alpha=0.45, linewidths=0, label="stone points")

    if debug and len(debug.get("positive_centers_xy", [])) > 0:
        centers = np.asarray(debug["positive_centers_xy"], dtype=np.float64)
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            s=max(12.0, cell_size * 180.0),
            c="#2d6cdf",
            alpha=0.55,
            marker="s",
            linewidths=0,
            label="2.5D cells",
        )

    polygon_source = hull_vertices[:, :2] if hull_vertices is not None else points[:, :2]
    hull_xy = _xy_convex_hull(polygon_source)
    if len(hull_xy) >= 3:
        closed = np.vstack([hull_xy, hull_xy[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="#c62828", linewidth=1.6, label="convex hull outline")

    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def _plot_xy_map(ax, points: np.ndarray, title: str) -> None:
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=points[:, 2],
        cmap="viridis",
        s=3,
        alpha=0.60,
        linewidths=0,
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Z (m)")


def _plot_profile_map(
    ax,
    points: np.ndarray,
    hull_vertices: np.ndarray | None,
    dims: tuple[int, int],
    labels: tuple[str, str],
    title: str,
) -> None:
    ax.scatter(
        points[:, dims[0]],
        points[:, dims[1]],
        s=3,
        c="#6f7d8c",
        alpha=0.45,
        linewidths=0,
    )
    hull_source = hull_vertices[:, list(dims)] if hull_vertices is not None else points[:, list(dims)]
    hull_2d = _xy_convex_hull(hull_source)
    if len(hull_2d) >= 3:
        closed = np.vstack([hull_2d, hull_2d[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="#c62828", linewidth=1.8)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.grid(True, alpha=0.18)


def _plot_2d5_cells(ax, debug: dict | None, title: str) -> None:
    if not debug or len(debug.get("positive_centers_xy", [])) == 0:
        ax.text(0.05, 0.5, "No positive 2.5D cells", fontsize=12, transform=ax.transAxes)
        ax.set_axis_off()
        return

    centers = np.asarray(debug["positive_centers_xy"], dtype=np.float64)
    heights = np.asarray(debug["positive_heights_m"], dtype=np.float64)
    scatter = ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c=heights,
        cmap="Blues",
        s=28,
        alpha=0.85,
        marker="s",
        linewidths=0,
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Height above ground (m)")


def _plot_summary_panel(
    ax,
    stone: dict,
    volume_2d5: dict,
    volume_hull: dict,
    point_count: int,
    hull_point_count: int,
    crop_info: dict,
) -> None:
    main_volume = float(volume_2d5.get("volume_m3", 0.0))
    hull_volume = float(volume_hull.get("volume_m3", 0.0))
    ax.set_axis_off()

    ratio = hull_volume / main_volume if main_volume > 0 else 0.0
    details = [
        "Volume comparison",
        "",
        f"stone_id: {stone.get('stone_id')}",
        f"diameter: {float(stone.get('equivalent_diameter_m', 0.0)):.3f} m",
        f"area: {float(stone.get('area_m2', 0.0)):.3f} m2",
        f"source detections: {int(stone.get('source_detection_count', 0))}",
        f"stone points: {point_count:,}",
        f"hull points: {hull_point_count:,}",
        "",
        f"2.5D volume: {main_volume:.4f} m3",
        f"Convex hull volume: {hull_volume:.4f} m3",
        f"hull / 2.5D: {ratio:.3f}",
        f"crop candidates: {int(crop_info.get('bbox_candidate_count', 0))}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(details),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
    )


def _panel_rectangles(width: int, height: int, gap: int = 28, outer: int = 30) -> list[tuple[int, int, int, int]]:
    panel_w = (width - 2 * outer - gap) // 2
    panel_h = (height - 2 * outer - gap) // 2
    rects = []
    for row in range(2):
        for col in range(2):
            x0 = outer + col * (panel_w + gap)
            y0 = outer + row * (panel_h + gap)
            rects.append((x0, y0, x0 + panel_w, y0 + panel_h))
    return rects


def _draw_panel(canvas: np.ndarray, rect: tuple[int, int, int, int], title: str, subtitle: str = "") -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (208, 214, 220), 2)
    cv2.rectangle(canvas, (x0, y0), (x1, y0 + 42), (245, 247, 250), -1)
    cv2.putText(canvas, title, (x0 + 14, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 25, 25), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (x0 + 14, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 80), 1, cv2.LINE_AA)
    inner_top = y0 + 74 if subtitle else y0 + 50
    return (x0 + 20, inner_top, x1 - 20, y1 - 18)


def _project_to_box(coords: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) == 0:
        return np.empty((0, 2), dtype=np.int32)

    x0, y0, x1, y1 = box
    xs = coords[:, 0]
    ys = coords[:, 1]
    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    scale = min(box_w / span_x, box_h / span_y) * 0.92
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    px_center = (x0 + x1) / 2.0
    py_center = (y0 + y1) / 2.0

    px = (xs - cx) * scale + px_center
    py = py_center - (ys - cy) * scale
    return np.column_stack((np.round(px), np.round(py))).astype(np.int32)


def _scalar_colors(values: np.ndarray, colormap: int) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    min_v = float(np.min(vals))
    max_v = float(np.max(vals))
    if max_v - min_v <= 1e-9:
        normalized = np.full(vals.shape, 128, dtype=np.uint8)
    else:
        normalized = np.clip(np.round((vals - min_v) / (max_v - min_v) * 255.0), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized.reshape(-1, 1), colormap).reshape(-1, 3)


def _draw_scatter(canvas: np.ndarray, pixels: np.ndarray, colors: np.ndarray, radius: int = 2) -> None:
    for (px, py), color in zip(pixels, colors, strict=True):
        cv2.circle(canvas, (int(px), int(py)), radius, tuple(int(v) for v in color.tolist()), -1, cv2.LINE_AA)


def _draw_polyline(canvas: np.ndarray, pixels: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    if len(pixels) < 2:
        return
    cv2.polylines(canvas, [pixels.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _render_overview_png(
    *,
    stone: dict,
    sampled_points: np.ndarray,
    hull_vertices: np.ndarray | None,
    volume_2d5: dict,
    volume_hull: dict,
    crop_info: dict,
    point_count: int,
    hull_point_count: int,
    z_range_m: float,
    output_path: Path,
) -> None:
    width, height = 1800, 1380
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    rects = _panel_rectangles(width, height)

    cv2.putText(
        canvas,
        f"Volume case comparison: {stone.get('stone_id')}",
        (36, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (18, 18, 18),
        2,
        cv2.LINE_AA,
    )

    box1 = _draw_panel(
        canvas,
        rects[0],
        "Stone point map",
        f"XY projection colored by Z | points={point_count:,} | z-range={z_range_m:.3f} m",
    )
    px1 = _project_to_box(sampled_points[:, [0, 1]], box1)
    c1 = _scalar_colors(sampled_points[:, 2], cv2.COLORMAP_VIRIDIS)
    _draw_scatter(canvas, px1, c1, radius=2)

    box2 = _draw_panel(
        canvas,
        rects[1],
        "Convex hull profile",
        f"X-Z projection | hull volume={volume_hull.get('volume_m3', 0.0):.4f} m3 | hull points={hull_point_count:,}",
    )
    px2 = _project_to_box(sampled_points[:, [0, 2]], box2)
    c2 = np.tile(np.asarray([[120, 130, 140]], dtype=np.uint8), (len(px2), 1))
    _draw_scatter(canvas, px2, c2, radius=1)
    if hull_vertices is not None and len(hull_vertices) > 0:
        hull_xz = _xy_convex_hull(hull_vertices[:, [0, 2]])
        hull_px = _project_to_box(hull_xz, box2)
        _draw_polyline(canvas, hull_px, (32, 32, 220), thickness=2)

    box3 = _draw_panel(
        canvas,
        rects[2],
        "2.5D positive cells",
        f"XY cell centers colored by height above ground | 2.5D volume={volume_2d5.get('volume_m3', 0.0):.4f} m3",
    )
    debug = volume_2d5.get("debug")
    if debug and len(debug.get("positive_centers_xy", [])) > 0:
        centers = np.asarray(debug["positive_centers_xy"], dtype=np.float64)
        heights = np.asarray(debug["positive_heights_m"], dtype=np.float64)
        px3 = _project_to_box(centers[:, [0, 1]], box3)
        c3 = _scalar_colors(heights, cv2.COLORMAP_OCEAN)
        for (px, py), color in zip(px3, c3, strict=True):
            cv2.rectangle(canvas, (int(px) - 3, int(py) - 3), (int(px) + 3, int(py) + 3), tuple(int(v) for v in color.tolist()), -1)
    else:
        cv2.putText(canvas, "No positive 2.5D cells", (box3[0] + 10, box3[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1, cv2.LINE_AA)

    box4 = _draw_panel(
        canvas,
        rects[3],
        "Case summary",
        "Single-stone comparison for paper inspection",
    )
    ratio = float(volume_hull.get("volume_m3", 0.0)) / max(float(volume_2d5.get("volume_m3", 1e-6)), 1e-6)
    lines = [
        f"stone_id: {stone.get('stone_id')}",
        f"equivalent diameter: {float(stone.get('equivalent_diameter_m', 0.0)):.3f} m",
        f"projected area: {float(stone.get('area_m2', 0.0)):.3f} m2",
        f"source detections: {int(stone.get('source_detection_count', 0))}",
        "",
        f"stone points: {point_count:,}",
        f"hull visualization points: {hull_point_count:,}",
        f"crop candidates: {int(crop_info.get('bbox_candidate_count', 0))}",
        "",
        f"2.5D volume: {float(volume_2d5.get('volume_m3', 0.0)):.4f} m3",
        f"Convex hull volume: {float(volume_hull.get('volume_m3', 0.0)):.4f} m3",
        f"hull / 2.5D ratio: {ratio:.4f}",
        f"2.5D status: {volume_2d5.get('status')}",
        f"hull status: {volume_hull.get('status')}",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (box4[0] + 6, box4[1] + 28 + idx * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def _summary_payload(
    stone: dict,
    rank: int,
    volume_2d5: dict,
    volume_hull: dict,
    point_count: int,
    hull_point_count: int,
    z_range_m: float,
    output_path: Path,
) -> dict:
    main_volume = float(volume_2d5.get("volume_m3", 0.0))
    hull_volume = float(volume_hull.get("volume_m3", 0.0))
    ratio = hull_volume / main_volume if main_volume > 0 else 0.0
    volume_2d5_serializable = {k: v for k, v in volume_2d5.items() if k != "debug"}
    return {
        "stone_id": stone.get("stone_id"),
        "stone_rank": int(rank),
        "bbox_world": stone.get("bbox_world"),
        "centroid_world": stone.get("centroid_world"),
        "equivalent_diameter_m": float(stone.get("equivalent_diameter_m", 0.0)),
        "area_m2": float(stone.get("area_m2", 0.0)),
        "source_detection_count": int(stone.get("source_detection_count", 0)),
        "point_count": int(point_count),
        "convex_hull_visual_point_count": int(hull_point_count),
        "z_range_m": round(float(z_range_m), 4),
        "volume_2d5": volume_2d5_serializable,
        "volume_convex_hull": volume_hull,
        "convex_hull_to_2d5": round(float(ratio), 4),
        "figure": str(output_path),
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy diagnostic: compare one stone's convex-hull and 2.5D volume.")
    parser.add_argument("--source", choices=SOURCES, default="quadtree_dom")
    parser.add_argument("--method", choices=METHODS, default="correlation_clustering")
    parser.add_argument("--mode", choices=STONE_MODES, default="accepted")
    parser.add_argument("--stone-id", default=None)
    parser.add_argument("--stone-rank", type=int, default=None)
    parser.add_argument("--list-stones", type=int, nargs="?", const=20, default=None)
    parser.add_argument("--sample-points", type=int, default=8000, help="Maximum points shown in matplotlib panels")
    parser.add_argument("--hull-max-points", type=int, default=8000, help="Maximum points used for convex-hull visualization and fast hull volume")
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--dem-pad-m", type=float, default=2.0, help="Extra XY padding for local DEM support")
    parser.add_argument("--figure-dpi", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fusion, stones, detections = _load_fusion(args.source, args.method)
    selected = _sorted_stones(_select_stones(fusion, args.mode))
    if not selected:
        raise SystemExit(f"No stones found for mode={args.mode}")

    if args.list_stones is not None:
        limit = min(args.list_stones, len(selected))
        print(f"mode={args.mode}  listing top {limit} stones")
        for rank, stone in enumerate(selected[:limit]):
            print(
                f"[{rank:03d}] {stone.get('stone_id')}  "
                f"diameter={float(stone.get('equivalent_diameter_m', 0.0)):.4f} m  "
                f"area={float(stone.get('area_m2', 0.0)):.4f} m2  "
                f"dets={int(stone.get('source_detection_count', 0))}"
            )
        return

    stone, rank = _find_stone(selected, args.stone_id, args.stone_rank)
    config = _load_volume_config()
    crop_cfg = config.get("crop", {})
    dem_cfg = config.get("ground_dem", {})
    grid_cfg = config.get("grid", {})

    gt = CURRENT_SCENE.load_gt()
    dem_pad_m = max(float(args.dem_pad_m), float(crop_cfg.get("bbox_pad_m", 0.5)))
    point_bbox = CURRENT_SCENE.xy_transform.world_bbox_to_point_bbox(
        stone.get("bbox_world", [0, 0, 0, 0]),
        pad_m=dem_pad_m,
    )
    print("Loading local scene point cloud...")
    local_points, local_info = _load_local_scene_points(point_bbox, chunk_size=args.chunk_size)
    if len(local_points) == 0:
        raise SystemExit("No local scene points were found around the selected stone.")

    print("Building local DEM...")
    ground_dem = GroundDEM(
        local_points,
        resolution=float(dem_cfg.get("resolution_m", 0.5)),
        percentile=int(dem_cfg.get("percentile", 5)),
        subsample_step=int(dem_cfg.get("subsample_step", 100)),
        min_points_per_cell=int(dem_cfg.get("min_points_per_cell", 3)),
    )

    print("Cropping stone point cloud...")
    local_index = PointCloudXYGridIndex.build(
        local_points,
        cell_size=float(crop_cfg.get("index_cell_size_m", 1.0)),
    )
    stone_points, crop_info = crop_stone_point_cloud(
        local_points,
        stone,
        detections,
        gt,
        CURRENT_SCENE.xy_transform,
        bbox_pad_m=float(crop_cfg.get("bbox_pad_m", 0.5)),
        pc_index=local_index,
    )
    if len(stone_points) == 0:
        raise SystemExit("The selected stone produced an empty point-cloud crop.")
    print(f"  stone crop points: {len(stone_points):,}")

    print("Estimating volumes...")
    hull_points = _sample_points(stone_points, max_points=args.hull_max_points, seed=args.seed + 1000 + rank)
    volume_2d5 = estimate_2d5_with_ground(
        stone_points,
        ground_dem,
        grid_resolution=float(grid_cfg.get("resolution_m", 0.05)),
        return_debug=True,
    )
    print("  2.5D estimation done")
    volume_hull = estimate_convex_hull(hull_points)
    volume_hull["visualized_from_point_count"] = int(len(hull_points))
    hull_vertices, hull_triangles = _build_convex_hull_mesh(hull_points)
    print("  convex hull estimation done")

    sampled_points = _sample_points(stone_points, max_points=args.sample_points, seed=args.seed + rank)
    z_range_m = float(np.ptp(stone_points[:, 2])) if len(stone_points) > 0 else 0.0

    output_dir = OUTPUT_ROOT / args.source / args.method / stone.get("stone_id", f"rank_{rank:04d}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "volume_compare.png"
    summary_path = output_dir / "volume_compare_summary.json"
    print("  rendering overview image")
    _render_overview_png(
        stone=stone,
        sampled_points=sampled_points,
        hull_vertices=hull_vertices,
        volume_2d5=volume_2d5,
        volume_hull=volume_hull,
        crop_info=crop_info,
        point_count=len(stone_points),
        hull_point_count=len(hull_points),
        z_range_m=z_range_m,
        output_path=figure_path,
    )
    print("  figure export done")

    summary = _summary_payload(
        stone=stone,
        rank=rank,
        volume_2d5=volume_2d5,
        volume_hull=volume_hull,
        point_count=len(stone_points),
        hull_point_count=len(hull_points),
        z_range_m=z_range_m,
        output_path=figure_path,
    )
    _write_json(summary_path, summary)

    print("")
    print("=" * 72)
    print(f"Stone volume comparison: {stone.get('stone_id')}  (rank={rank})")
    print("=" * 72)
    print(f"  point count      : {len(stone_points):,}")
    print(f"  hull vis points  : {len(hull_points):,}")
    print(f"  local DEM points : {len(local_points):,}")
    print(f"  local bbox       : {local_info.get('point_bbox')}")
    print(f"  z_range_m        : {z_range_m:.4f}")
    print(f"  2.5D volume_m3   : {volume_2d5.get('volume_m3', 0.0):.4f}  [{volume_2d5.get('status')}]")
    print(f"  hull volume_m3   : {volume_hull.get('volume_m3', 0.0):.4f}  [{volume_hull.get('status')}]")
    if volume_2d5.get("volume_m3", 0.0) > 0:
        ratio = float(volume_hull.get("volume_m3", 0.0)) / float(volume_2d5["volume_m3"])
        print(f"  hull / 2.5D      : {ratio:.4f}")
    print(f"  figure           : {figure_path}")
    print(f"  summary json     : {summary_path}")


if __name__ == "__main__":
    main()

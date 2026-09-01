from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import laspy
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import colors
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common.scene_reference import CURRENT_SCENE
from experiments.common.stone_region import crop_stone_point_cloud
from experiments.volume.ground_estimator import GroundDEM


FIGURE_ID = "FIG-3-1"
RUN_ID = "main_dom2_pc2_qt20_d050_cc_v2"
STONE_ID = "stone_000360"
SOURCE = "quadtree_dom"
METHOD = "correlation_clustering"

SLICING_DIR = ROOT / "experiments" / "slicing" / "outputs" / SOURCE
DETECTION_DIR = ROOT / "experiments" / "detection" / "outputs" / SOURCE
FUSION_DIR = ROOT / "experiments" / "fusion" / "outputs" / SOURCE / METHOD
VOLUME_DIR = ROOT / "experiments" / "volume" / "outputs" / SOURCE / METHOD
OUTPUT_DIR = ROOT / "experiments" / "visualization" / "outputs" / FIGURE_ID

PATHS = {
    "slicing_summary": SLICING_DIR / "slicing_summary.json",
    "tile_stats": SLICING_DIR / "tile_stats.json",
    "tile_overview": SLICING_DIR / "tile_overlay_paper.png",
    "detections": DETECTION_DIR / "detections.json",
    "detection_stats": DETECTION_DIR / "detection_stats.json",
    "accepted_stones": FUSION_DIR / "accepted_stones.json",
    "fusion_summary": FUSION_DIR / "fusion_summary.json",
    "stone_volumes": VOLUME_DIR / "stone_volumes.json",
    "volume_stats": VOLUME_DIR / "volume_stats.json",
    "volume_config": ROOT / "experiments" / "volume" / "config.json",
}

COLORS = {
    "ink": "#27313A",
    "muted": "#6B747C",
    "light": "#D9DEE2",
    "blue": "#2F6FA3",
    "blue_light": "#9EC5E5",
    "amber": "#D88A23",
    "amber_light": "#F1C27D",
    "teal": "#2A9D8F",
    "teal_light": "#A8DADC",
    "ground": "#B9BFC4",
    "pass": "#3B8C6E",
    "white": "#FFFFFF",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, with_sha256: bool) -> dict:
    stat = path.stat()
    result = {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
    }
    if with_sha256:
        result["sha256"] = sha256_file(path)
    return result


def git_state() -> dict:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "dirty": bool(run("status", "--porcelain")),
    }


def world_to_pixel(gt: tuple[float, float, float, float, float, float], x: float, y: float) -> tuple[float, float]:
    origin_x, res_x, _, origin_y, _, res_y = gt
    return (float(x - origin_x) / res_x, float(y - origin_y) / res_y)


def world_bbox_to_pixel_box(
    bbox: list[float] | tuple[float, float, float, float],
    gt: tuple[float, float, float, float, float, float],
    *,
    pad_m: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    px0, py0 = world_to_pixel(gt, x0 - pad_m, y0 - pad_m)
    px1, py1 = world_to_pixel(gt, x1 + pad_m, y1 + pad_m)
    return (
        int(math.floor(min(px0, px1))),
        int(math.floor(min(py0, py1))),
        int(math.ceil(max(px0, px1))),
        int(math.ceil(max(py0, py1))),
    )


def pixel_box_extent(
    pixel_box: tuple[int, int, int, int],
    gt: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = pixel_box
    origin_x, res_x, _, origin_y, _, res_y = gt
    wx0 = origin_x + x0 * res_x
    wx1 = origin_x + x1 * res_x
    wy0 = origin_y + y0 * res_y
    wy1 = origin_y + y1 * res_y
    return (min(wx0, wx1), max(wx0, wx1), min(wy0, wy1), max(wy0, wy1))


def read_dom_pixel_box(pixel_box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = pixel_box
    with rasterio.open(CURRENT_SCENE.dom_path) as dataset:
        x0 = max(0, min(dataset.width - 1, x0))
        y0 = max(0, min(dataset.height - 1, y0))
        x1 = max(x0 + 1, min(dataset.width, x1))
        y1 = max(y0 + 1, min(dataset.height, y1))
        rgb = dataset.read(
            [1, 2, 3],
            window=Window(x0, y0, x1 - x0, y1 - y0),
        )
    return np.moveaxis(rgb, 0, -1)


def decode_rle(detection: dict) -> np.ndarray:
    rle = detection["rle_mask"]
    height, width = [int(v) for v in rle["size"]]
    counts = np.asarray(rle["counts"], dtype=np.int64)
    starts_with = rle.get("starts_with")
    if starts_with is None:
        expected = float(detection["area_m2"]) / 0.0001
        odd_area = int(counts[1::2].sum())
        even_area = int(counts[0::2].sum())
        starts_with = 0 if abs(odd_area - expected) <= abs(even_area - expected) else 1
    flat = np.zeros(height * width, dtype=np.uint8)
    position = 0
    for index, count in enumerate(counts):
        stop = position + int(count)
        if (int(starts_with) + index) % 2 == 1:
            flat[position:stop] = 1
        position = stop
    return flat.reshape(height, width)


def mask_in_crop(detection: dict, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
    output = np.zeros((crop_y1 - crop_y0, crop_x1 - crop_x0), dtype=np.uint8)
    mask = decode_rle(detection)
    det_x0, det_y0 = [int(v) for v in detection["pixel_origin"]]
    det_y1 = det_y0 + mask.shape[0]
    det_x1 = det_x0 + mask.shape[1]
    ix0 = max(crop_x0, det_x0)
    iy0 = max(crop_y0, det_y0)
    ix1 = min(crop_x1, det_x1)
    iy1 = min(crop_y1, det_y1)
    if ix0 < ix1 and iy0 < iy1:
        output[iy0 - crop_y0 : iy1 - crop_y0, ix0 - crop_x0 : ix1 - crop_x0] = mask[
            iy0 - det_y0 : iy1 - det_y0,
            ix0 - det_x0 : ix1 - det_x0,
        ]
    return output


def show_dom(ax, image: np.ndarray, extent: tuple[float, float, float, float]) -> None:
    ax.imshow(image, extent=extent, origin="upper")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def overlay_mask(ax, mask: np.ndarray, extent, color: str, *, alpha: float = 0.42, linewidth: float = 1.0) -> None:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgb = colors.to_rgb(color)
    rgba[..., :3] = rgb
    rgba[..., 3] = mask.astype(np.float32) * alpha
    ax.imshow(rgba, extent=extent, origin="upper")
    if np.any(mask):
        ax.contour(
            mask,
            levels=[0.5],
            colors=[color],
            linewidths=linewidth,
            extent=extent,
            origin="upper",
        )


def add_scale_bar(ax, extent, length_m: float = 1.0, *, color: str = "white") -> None:
    x0, x1, y0, y1 = extent
    start_x = x0 + 0.08 * (x1 - x0)
    start_y = y0 + 0.08 * (y1 - y0)
    ax.plot([start_x, start_x + length_m], [start_y, start_y], color=color, lw=2.2, solid_capstyle="butt")
    ax.text(
        start_x + length_m / 2,
        start_y + 0.025 * (y1 - y0),
        f"{length_m:g} m",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color=color,
        weight="bold",
    )


def panel_heading(ax, letter: str, title: str) -> None:
    text_method = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_method(0.0, 1.02, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, weight="bold", color=COLORS["ink"])
    text_method(0.08, 1.02, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, weight="bold", color=COLORS["ink"])


def load_scene_points_and_dem_sample(
    local_bbox: tuple[float, float, float, float],
    *,
    subsample_step: int,
    chunk_size: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    local_parts: list[np.ndarray] = []
    dem_parts: list[np.ndarray] = []
    file_stats: list[dict] = []
    global_offset = 0
    x0, y0, x1, y1 = local_bbox

    for path in CURRENT_SCENE.pointcloud_paths:
        local_count = 0
        sampled_count = 0
        file_count = 0
        with laspy.open(path) as reader:
            total_points = int(reader.header.point_count)
            local_offset = 0
            for chunk in reader.chunk_iterator(chunk_size):
                count = len(chunk)
                xs = np.asarray(chunk.x)
                ys = np.asarray(chunk.y)
                zs = np.asarray(chunk.z)
                local_keep = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
                if np.any(local_keep):
                    local_parts.append(np.column_stack((xs[local_keep], ys[local_keep], zs[local_keep])))
                    local_count += int(np.count_nonzero(local_keep))
                positions = np.arange(global_offset + local_offset, global_offset + local_offset + count)
                dem_keep = positions % subsample_step == 0
                if np.any(dem_keep):
                    dem_parts.append(np.column_stack((xs[dem_keep], ys[dem_keep], zs[dem_keep])))
                    sampled_count += int(np.count_nonzero(dem_keep))
                local_offset += count
                file_count += count
            global_offset += local_offset
        file_stats.append(
            {
                "path": str(path.relative_to(ROOT)),
                "total_points": total_points,
                "streamed_points": file_count,
                "local_points": local_count,
                "ground_dem_sample_points": sampled_count,
            }
        )

    if not local_parts or not dem_parts:
        raise RuntimeError("Point-cloud extraction did not produce local or GroundDEM points.")
    return np.vstack(local_parts), np.vstack(dem_parts), file_stats


def find_records() -> dict:
    detections = load_json(PATHS["detections"])
    stones = load_json(PATHS["accepted_stones"])
    volume_records = load_json(PATHS["stone_volumes"])["stones"]
    tile_payload = load_json(PATHS["tile_stats"])
    stone = next(item for item in stones if item["stone_id"] == STONE_ID)
    volume = next(item for item in volume_records if item["stone_id"] == STONE_ID)
    selected_detections = [detections[index] for index in stone["detection_indices"]]
    tile_by_id = {item["tile_id"]: item for item in tile_payload["tiles"]}
    selected_tiles = [tile_by_id[patch_id] for patch_id in stone["source_patch_ids"]]

    if stone["source_detection_count"] < 2 or stone["source_patches_span"] < 2:
        raise RuntimeError("Selected stone is not a cross-tile duplicate case.")
    if not stone["validation_3d"]["passed"]:
        raise RuntimeError("Selected stone did not pass 3D validation.")
    if not volume["qc"]["passed"] or volume["methods"]["2d5"]["status"] != "ok":
        raise RuntimeError("Selected stone did not pass volume QC.")
    if {item["source_patch_id"] for item in selected_detections} != set(stone["source_patch_ids"]):
        raise RuntimeError("Detection-to-tile provenance mismatch.")
    return {
        "detections": detections,
        "stone": stone,
        "volume": volume,
        "selected_detections": selected_detections,
        "selected_tiles": selected_tiles,
    }


def recompute_2d5_grid(points: np.ndarray, ground_dem: GroundDEM, grid_resolution: float) -> dict:
    """Rebuild the exact positive-height cells needed by panel f."""
    print("  grid debug: converting point rows", flush=True)
    rows_xyz = np.asarray(points, dtype=np.float64).tolist()
    print("  grid debug: computing bounds", flush=True)
    xmin = min(row[0] for row in rows_xyz)
    ymin = min(row[1] for row in rows_xyz)
    xmax = max(row[0] for row in rows_xyz)
    ymax = max(row[1] for row in rows_xyz)
    nx = max(1, int(np.ceil((xmax - xmin) / grid_resolution)))
    ny = max(1, int(np.ceil((ymax - ymin) / grid_resolution)))
    print(f"  grid debug: aggregating {len(rows_xyz):,} points into {nx} x {ny}", flush=True)
    cell_tops: dict[tuple[int, int], float] = {}
    for point_x, point_y, point_z in rows_xyz:
        col = min(nx - 1, max(0, int(math.floor((float(point_x) - xmin) / grid_resolution))))
        row = min(ny - 1, max(0, int(math.floor((float(point_y) - ymin) / grid_resolution))))
        key = (row, col)
        current = cell_tops.get(key)
        if current is None or float(point_z) > current:
            cell_tops[key] = float(point_z)
    sorted_cells = sorted(cell_tops)
    print(f"  grid debug: building {len(sorted_cells):,} cell centers", flush=True)
    centers_x = np.asarray([xmin + (col + 0.5) * grid_resolution for row, col in sorted_cells], dtype=np.float64)
    centers_y = np.asarray([ymin + (row + 0.5) * grid_resolution for row, col in sorted_cells], dtype=np.float64)
    rock_top = np.asarray([cell_tops[cell] for cell in sorted_cells], dtype=np.float64)
    print("  grid debug: querying GroundDEM", flush=True)
    ground_z = np.asarray(ground_dem.get_ground_z(centers_x, centers_y), dtype=np.float64)
    print("  grid debug: filtering positive heights", flush=True)
    finite_ground = 0
    positive_centers_list: list[tuple[float, float]] = []
    positive_heights_list: list[float] = []
    for center_x, center_y, top_z, base_z in zip(centers_x.tolist(), centers_y.tolist(), rock_top.tolist(), ground_z.tolist(), strict=True):
        if not math.isfinite(base_z):
            continue
        finite_ground += 1
        height = top_z - base_z
        if math.isfinite(height) and height > 0:
            positive_centers_list.append((center_x, center_y))
            positive_heights_list.append(height)
    positive_centers = np.asarray(positive_centers_list, dtype=np.float64)
    positive_heights = np.asarray(positive_heights_list, dtype=np.float64)
    print(f"  grid debug: positive cells {len(positive_heights):,}", flush=True)
    return {
        "status": "ok" if len(positive_heights) else "invalid",
        "reason": "ok" if len(positive_heights) else "no_positive_heights",
        "volume_m3": round(float(math.fsum(float(value) for value in positive_heights) * grid_resolution**2), 4),
        "occupied_cells": int(len(cell_tops)),
        "ground_supported_cells": int(finite_ground),
        "valid_cells": int(len(positive_heights)),
        "debug": {
            "grid_resolution_m": float(grid_resolution),
            "positive_centers_xy": positive_centers,
            "positive_heights_m": positive_heights,
        },
    }


def add_flow_arrow(fig, start_ax, end_ax, direction: str) -> None:
    start = start_ax.get_subplotspec().get_position(fig)
    end = end_ax.get_subplotspec().get_position(fig)
    if direction == "right":
        p0 = (start.x1 + 0.004, (start.y0 + start.y1) / 2)
        p1 = (end.x0 - 0.004, (end.y0 + end.y1) / 2)
    elif direction == "left":
        flow_y = start.y0 + 0.86 * (start.y1 - start.y0)
        p0 = (start.x0 - 0.004, flow_y)
        p1 = (end.x1 + 0.004, flow_y)
    else:
        flow_x = start.x1 - 0.012
        p0 = (flow_x, start.y0 - 0.004)
        p1 = (flow_x, end.y1 + 0.004)
    fig.add_artist(
        FancyArrowPatch(
            p0,
            p1,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.05,
            color=COLORS["muted"],
            zorder=50,
        )
    )


def draw_panel_a(fig, ax, stone, local_points, gt) -> dict:
    panel_heading(ax, "a", "Co-registered input data")
    ax.set_axis_off()
    crop_box = world_bbox_to_pixel_box(stone["bbox_world"], gt, pad_m=2.2)
    extent = pixel_box_extent(crop_box, gt)
    dom = read_dom_pixel_box(crop_box)
    left = ax.inset_axes([0.00, 0.10, 0.48, 0.82])
    right = ax.inset_axes([0.52, 0.10, 0.48, 0.82])
    show_dom(left, dom, extent)
    left.set_title("DOM", pad=2, fontsize=6.5)
    add_scale_bar(left, extent, 1.0)
    x, y = local_points[:, 0], local_points[:, 1]
    right.hexbin(x, y, gridsize=80, bins="log", mincnt=1, cmap="Greys", linewidths=0)
    right.add_patch(
        Rectangle(
            (stone["bbox_world"][0], stone["bbox_world"][1]),
            stone["bbox_world"][2] - stone["bbox_world"][0],
            stone["bbox_world"][3] - stone["bbox_world"][1],
            fill=False,
            edgecolor=COLORS["blue"],
            linewidth=1.1,
        )
    )
    right.set_xlim(extent[0], extent[1])
    right.set_ylim(extent[2], extent[3])
    right.set_aspect("equal")
    right.set_xticks([])
    right.set_yticks([])
    right.set_title("Point cloud (top view)", pad=2, fontsize=6.5)
    for spine in right.spines.values():
        spine.set_visible(False)
    ax.text(0.50, 0.01, "Unified world coordinates (EPSG:4536)", ha="center", va="bottom", fontsize=5.7, color=COLORS["muted"], transform=ax.transAxes)
    return {"dom_crop_pixel_box": list(crop_box), "local_point_count": int(len(local_points))}


def draw_panel_b(fig, ax, stone, selected_tiles, slicing_summary, gt) -> dict:
    panel_heading(ax, "b", "Adaptive quadtree tiling")
    ax.set_axis_off()
    crop_box = world_bbox_to_pixel_box(stone["bbox_world"], gt, pad_m=3.6)
    extent = pixel_box_extent(crop_box, gt)
    dom = read_dom_pixel_box(crop_box)
    main = ax.inset_axes([0.00, 0.08, 1.00, 0.86])
    show_dom(main, dom, extent)
    overlap = float(slicing_summary["config"]["tile_overlap_m"])
    half = overlap / 2.0
    tile_colors = [COLORS["amber"], COLORS["blue"]]
    expanded = []
    for tile, color in zip(selected_tiles, tile_colors, strict=True):
        x0, y0, x1, y1 = [float(v) for v in tile["bounds_m"]]
        expanded_box = (x0 - half, y0 - half, x1 + half, y1 + half)
        expanded.append(expanded_box)
        main.add_patch(
            Rectangle(
                (expanded_box[0], expanded_box[1]),
                expanded_box[2] - expanded_box[0],
                expanded_box[3] - expanded_box[1],
                fill=False,
                edgecolor=color,
                linewidth=1.35,
            )
        )
    ox0 = max(expanded[0][0], expanded[1][0])
    oy0 = max(expanded[0][1], expanded[1][1])
    ox1 = min(expanded[0][2], expanded[1][2])
    oy1 = min(expanded[0][3], expanded[1][3])
    if ox0 < ox1 and oy0 < oy1:
        main.add_patch(
            Rectangle((ox0, oy0), ox1 - ox0, oy1 - oy0, facecolor=COLORS["teal"], edgecolor="none", alpha=0.28)
        )
        main.annotate(
            "0.5 m overlap",
            xy=((ox0 + ox1) / 2, (oy0 + oy1) / 2),
            xytext=(extent[0] + 0.08 * (extent[1] - extent[0]), extent[2] + 0.13 * (extent[3] - extent[2])),
            arrowprops={"arrowstyle": "->", "color": COLORS["ink"], "lw": 0.7},
            fontsize=5.4,
            color=COLORS["ink"],
        )
    main.text(0.02, 0.98, "10 m child tiles", transform=main.transAxes, ha="left", va="top", fontsize=5.6, color=COLORS["ink"], bbox={"fc": "white", "ec": "none", "alpha": 0.78, "pad": 1.4})
    overview = plt.imread(PATHS["tile_overview"])
    inset = main.inset_axes([0.70, 0.67, 0.28, 0.30])
    inset.imshow(overview)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color(COLORS["white"])
        spine.set_linewidth(0.8)
    return {"dom_crop_pixel_box": list(crop_box), "source_tiles": [item["tile_id"] for item in selected_tiles], "overlap_m": overlap}


def draw_panel_c(fig, ax, detection, gt) -> dict:
    panel_heading(ax, "c", "Instance mask and 2D measurement")
    ax.set_axis_off()
    crop_box = world_bbox_to_pixel_box(detection["bbox_world"], gt, pad_m=0.75)
    extent = pixel_box_extent(crop_box, gt)
    dom = read_dom_pixel_box(crop_box)
    mask = mask_in_crop(detection, crop_box)
    image_ax = ax.inset_axes([0.00, 0.06, 1.00, 0.88])
    show_dom(image_ax, dom, extent)
    overlay_mask(image_ax, mask, extent, COLORS["teal"], alpha=0.43, linewidth=1.15)
    center_x, center_y = [float(v) for v in detection["centroid_world"]]
    radius = float(detection["equivalent_diameter_m"]) / 2.0
    image_ax.add_patch(Circle((center_x, center_y), radius, fill=False, linestyle=(0, (3, 2)), linewidth=0.9, edgecolor=COLORS["blue"]))
    image_ax.plot([center_x - radius, center_x + radius], [center_y, center_y], color=COLORS["blue"], lw=0.9)
    image_ax.text(
        0.03,
        0.04,
        rf"$A$ = {detection['area_m2']:.4f} m$^2$" + "\n" + rf"$d_{{eq}}$ = {detection['equivalent_diameter_m']:.4f} m",
        transform=image_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
        color=COLORS["ink"],
        bbox={"fc": "white", "ec": "none", "alpha": 0.82, "pad": 2.0},
    )
    add_scale_bar(image_ax, extent, 0.5)
    return {"detection_index": 371, "source_patch_id": detection["source_patch_id"], "area_m2": detection["area_m2"], "equivalent_diameter_m": detection["equivalent_diameter_m"]}


def draw_panel_d(fig, ax, stone, selected_detections, gt) -> dict:
    panel_heading(ax, "d", "Cross-tile duplicate fusion")
    ax.set_axis_off()
    positions = [[0.00, 0.20, 0.29, 0.70], [0.35, 0.20, 0.29, 0.70], [0.71, 0.14, 0.29, 0.79]]
    source_colors = [COLORS["amber"], COLORS["blue"]]
    source_axes = []
    for index, (detection, position, color) in enumerate(zip(selected_detections, positions[:2], source_colors, strict=True)):
        child = ax.inset_axes(position)
        crop_box = world_bbox_to_pixel_box(stone["bbox_world"], gt, pad_m=0.55)
        extent = pixel_box_extent(crop_box, gt)
        show_dom(child, read_dom_pixel_box(crop_box), extent)
        overlay_mask(child, mask_in_crop(detection, crop_box), extent, color, alpha=0.48, linewidth=1.0)
        child.set_title(f"{detection['source_patch_id']}\nscore {detection['score']:.3f}", fontsize=5.2, pad=1.5, color=color)
        source_axes.append(child)
    fused_ax = ax.inset_axes(positions[2])
    fused_crop = world_bbox_to_pixel_box(stone["bbox_world"], gt, pad_m=0.55)
    fused_extent = pixel_box_extent(fused_crop, gt)
    fused_dom = read_dom_pixel_box(fused_crop)
    masks = [mask_in_crop(item, fused_crop) for item in selected_detections]
    union = np.maximum.reduce(masks)
    show_dom(fused_ax, fused_dom, fused_extent)
    overlay_mask(fused_ax, union, fused_extent, COLORS["teal"], alpha=0.43, linewidth=1.25)
    for mask, color in zip(masks, source_colors, strict=True):
        if np.any(mask):
            fused_ax.contour(mask, levels=[0.5], colors=[color], linewidths=0.65, extent=fused_extent, origin="upper")
    fused_ax.set_title(f"1 fused stone\n{STONE_ID}", fontsize=5.5, pad=1.5, color=COLORS["teal"], weight="bold")
    for x0, x1 in [(0.29, 0.35), (0.64, 0.71)]:
        ax.add_patch(
            FancyArrowPatch((x0, 0.53), (x1, 0.53), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=COLORS["muted"])
        )
    ax.text(0.50, 0.05, "2 source detections  ->  1 world-coordinate object", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.7, color=COLORS["muted"])
    return {"detection_indices": stone["detection_indices"], "source_patch_ids": stone["source_patch_ids"], "fusion_method": stone["merge_method"]}


def draw_panel_e(fig, ax, stone, stone_points, ground_dem) -> dict:
    panel_heading(ax, "e", "3D screening with GroundDEM")
    center_x, center_y = [float(v) for v in stone["centroid_world"]]
    bbox = stone["bbox_world"]
    x_grid = np.linspace(float(bbox[0]) - 0.45, float(bbox[2]) + 0.45, 180)
    y_grid = np.full_like(x_grid, center_y)
    ground_profile = np.asarray(ground_dem.get_ground_z(x_grid, y_grid), dtype=np.float64)
    reference_z = float(np.median(ground_profile[np.isfinite(ground_profile)]))
    point_ground = np.asarray(ground_dem.get_ground_z(stone_points[:, 0], stone_points[:, 1]))
    relative_height = stone_points[:, 2] - point_ground
    vmax = max(1e-6, float(np.percentile(relative_height[np.isfinite(relative_height)], 98)))
    ax.fill_between(
        x_grid - center_x,
        ground_profile - reference_z - 0.16,
        ground_profile - reference_z,
        color=COLORS["ground"],
        alpha=0.65,
        linewidth=0,
    )
    ax.plot(x_grid - center_x, ground_profile - reference_z, color=COLORS["muted"], lw=1.0, label="GroundDEM profile")
    scatter = ax.scatter(
        stone_points[:, 0] - center_x,
        stone_points[:, 2] - reference_z,
        c=np.clip(relative_height, 0.0, vmax),
        cmap="viridis",
        s=0.24,
        alpha=0.38,
        linewidths=0,
        rasterized=True,
    )
    p90 = float(stone["validation_3d"]["height_above_ground"]["p90_m"])
    center_ground = float(ground_dem.get_ground_z(center_x, center_y))
    ax.annotate(
        "",
        xy=(0, center_ground + p90 - reference_z),
        xytext=(0, center_ground - reference_z),
        arrowprops={"arrowstyle": "<->", "color": COLORS["amber"], "lw": 1.2},
    )
    ax.text(0.03, 0.91, rf"$h_{{P90}}$ = {p90:.3f} m", transform=ax.transAxes, fontsize=5.8, color=COLORS["amber"], weight="bold")
    ax.text(0.03, 0.83, f"3D accepted | n = {len(stone_points):,}", transform=ax.transAxes, fontsize=5.5, color=COLORS["pass"])
    ax.set_xlabel("local x (m)", labelpad=1)
    ax.set_ylabel("elevation relative to local reference (m)")
    ax.set_xlim(float(bbox[0]) - center_x - 0.42, float(bbox[2]) - center_x + 0.42)
    visible_z = stone_points[:, 2] - reference_z
    ax.set_ylim(min(float(np.min(ground_profile - reference_z)) - 0.12, float(np.percentile(visible_z, 1)) - 0.06), float(np.percentile(visible_z, 99.5)) + 0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2, pad=1)
    ax.legend(loc="lower right", fontsize=5.0, handlelength=1.4)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.02)
    colorbar.set_label("relative height (m)", fontsize=5.3)
    colorbar.ax.tick_params(labelsize=5.0, length=1.5)
    return {
        "point_count": int(len(stone_points)),
        "p90_relative_height_m": p90,
        "elevated_ratio": stone["validation_3d"]["height_above_ground"]["elevated_ratio"],
        "validation_passed": True,
        "ground_dem_profile_samples": int(len(x_grid)),
        "all_stone_points_rendered": True,
    }


def draw_record_icon(ax, y: float, kind: str, color: str) -> None:
    if kind == "position":
        ax.add_patch(Circle((0.11, y), 0.027, transform=ax.transAxes, fill=False, ec=color, lw=1.0))
        ax.plot([0.075, 0.145], [y, y], transform=ax.transAxes, color=color, lw=0.8)
        ax.plot([0.11, 0.11], [y - 0.035, y + 0.035], transform=ax.transAxes, color=color, lw=0.8)
    elif kind == "area":
        ax.add_patch(Rectangle((0.08, y - 0.025), 0.06, 0.05, transform=ax.transAxes, fill=False, ec=color, lw=1.0))
    elif kind == "diameter":
        ax.add_patch(FancyArrowPatch((0.075, y), (0.145, y), transform=ax.transAxes, arrowstyle="<->", mutation_scale=7, lw=1.0, color=color))
    elif kind == "status":
        ax.add_patch(Circle((0.11, y), 0.028, transform=ax.transAxes, fill=False, ec=color, lw=1.0))
        ax.plot([0.094, 0.106, 0.132], [y, y - 0.012, y + 0.014], transform=ax.transAxes, color=color, lw=1.1)
    else:
        heights = [0.03, 0.05, 0.075]
        for idx, height in enumerate(heights):
            ax.add_patch(Rectangle((0.075 + idx * 0.025, y - 0.035), 0.018, height, transform=ax.transAxes, fc=color, ec="none", alpha=0.85))


def draw_isometric_columns(ax, centers: np.ndarray, heights: np.ndarray, resolution: float) -> None:
    x_index = np.rint((centers[:, 0] - centers[:, 0].min()) / resolution).astype(int)
    y_index = np.rint((centers[:, 1] - centers[:, 1].min()) / resolution).astype(int)
    order = np.argsort(x_index + y_index)
    vmax = max(1e-6, float(np.percentile(heights, 98)))
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    cmap = mpl.colormaps["viridis"]
    height_scale = 2.8
    for position in order:
        col = int(x_index[position])
        row = int(y_index[position])
        height = float(heights[position])
        center_x = (col - row) * 0.50
        base_y = (col + row) * 0.25
        top_y = base_y + height * height_scale
        top_color = cmap(norm(min(height, vmax)))
        left_color = tuple(max(0.0, component * 0.72) for component in top_color[:3]) + (0.95,)
        right_color = tuple(max(0.0, component * 0.56) for component in top_color[:3]) + (0.95,)
        left_face = np.asarray(
            [
                [center_x - 0.50, top_y],
                [center_x, top_y - 0.25],
                [center_x, base_y - 0.25],
                [center_x - 0.50, base_y],
            ]
        )
        right_face = np.asarray(
            [
                [center_x, top_y - 0.25],
                [center_x + 0.50, top_y],
                [center_x + 0.50, base_y],
                [center_x, base_y - 0.25],
            ]
        )
        top_face = np.asarray(
            [
                [center_x - 0.50, top_y],
                [center_x, top_y + 0.25],
                [center_x + 0.50, top_y],
                [center_x, top_y - 0.25],
            ]
        )
        ax.fill(left_face[:, 0], left_face[:, 1], color=left_color, linewidth=0)
        ax.fill(right_face[:, 0], right_face[:, 1], color=right_color, linewidth=0)
        ax.fill(top_face[:, 0], top_face[:, 1], color=top_color, linewidth=0)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()


def draw_panel_f(fig, ax, volume, volume_debug) -> dict:
    panel_heading(ax, "f", "2.5D integration and stone record")
    ax.set_axis_off()
    bar_ax = ax.inset_axes([0.00, 0.08, 0.62, 0.82])
    centers = np.asarray(volume_debug["positive_centers_xy"])
    heights = np.asarray(volume_debug["positive_heights_m"])
    resolution = float(volume_debug["grid_resolution_m"])
    draw_isometric_columns(bar_ax, centers, heights, resolution)
    bar_ax.text(0.02, 0.03, r"$V_{2.5D}=\sum h_i\Delta^2$", transform=bar_ax.transAxes, fontsize=6.0, color=COLORS["ink"])

    record_ax = ax.inset_axes([0.64, 0.06, 0.36, 0.86])
    record_ax.set_axis_off()
    record_ax.text(0.04, 0.98, "Stone record", transform=record_ax.transAxes, ha="left", va="top", fontsize=6.2, weight="bold", color=COLORS["ink"])
    rows = [
        (0.82, "position", COLORS["blue"], f"X {volume['geometry']['centroid_world'][0]:.3f}\nY {volume['geometry']['centroid_world'][1]:.3f}"),
        (0.64, "area", COLORS["teal"], rf"$A$ {volume['fusion_prior']['area_m2']:.4f} m$^2$"),
        (0.49, "diameter", COLORS["amber"], rf"$d_{{eq}}$ {volume['fusion_prior']['equivalent_diameter_m']:.4f} m"),
        (0.34, "status", COLORS["pass"], "3D / QC PASS"),
        (0.17, "volume", COLORS["blue"], rf"$V_{{2.5D}}$ {volume['methods']['2d5']['volume_m3']:.4f} m$^3$" + "\n" + rf"$V_{{2D}}$ {volume['methods']['2d_proxy']['volume_m3']:.4f} m$^3$"),
    ]
    for y_pos, kind, color, label in rows:
        draw_record_icon(record_ax, y_pos, kind, color)
        record_ax.text(0.20, y_pos, label, transform=record_ax.transAxes, ha="left", va="center", fontsize=5.3, color=COLORS["ink"], linespacing=1.15)
    return {
        "grid_resolution_m": resolution,
        "positive_cells_rendered": int(len(heights)),
        "volume_2d5_m3": volume["methods"]["2d5"]["volume_m3"],
        "volume_2d_proxy_m3": volume["methods"]["2d_proxy"]["volume_m3"],
        "qc_passed": volume["qc"]["passed"],
    }


def band_heading(ax, title: str, subtitle: str, *, title_fontsize: float = 7.0) -> None:
    ax.set_axis_off()
    lines = title.split("\n")
    line_y = 1.015
    for line in lines:
        ax.text(0.0, line_y, line, transform=ax.transAxes, ha="left", va="bottom", fontsize=title_fontsize, weight="bold", color=COLORS["ink"])
        line_y -= 0.021
    ax.text(0.0, 0.970, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.0, color=COLORS["muted"])
    ax.plot([0.0, 1.0], [0.952, 0.952], transform=ax.transAxes, color=COLORS["light"], lw=0.65, clip_on=False)


def style_image_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_input_band(ax, stone, local_points, selected_tiles, slicing_summary, gt) -> dict:
    """Show real co-registered local inputs and the two full overlapping tiles."""
    band_heading(ax, "DOM Preparation & Adaptive Tiling", "Co-registered DOM + original local point cloud")
    crop_box = world_bbox_to_pixel_box(stone["bbox_world"], gt, pad_m=2.2)
    extent = pixel_box_extent(crop_box, gt)
    dom = read_dom_pixel_box(crop_box)
    dom_ax = ax.inset_axes([0.00, 0.55, 0.47, 0.34])
    cloud_ax = ax.inset_axes([0.53, 0.55, 0.47, 0.34])
    show_dom(dom_ax, dom, extent)
    dom_ax.set_title("DOM local view", fontsize=5.8, pad=1.7, color=COLORS["ink"])

    visible = (
        (local_points[:, 0] >= extent[0])
        & (local_points[:, 0] <= extent[1])
        & (local_points[:, 1] >= extent[2])
        & (local_points[:, 1] <= extent[3])
    )
    cloud_ax.scatter(
        local_points[visible, 0],
        local_points[visible, 1],
        c=local_points[visible, 2],
        cmap="Greys",
        s=0.035,
        alpha=0.50,
        linewidths=0,
        rasterized=True,
    )
    cloud_ax.set_xlim(extent[0], extent[1])
    cloud_ax.set_ylim(extent[2], extent[3])
    cloud_ax.set_aspect("equal")
    cloud_ax.set_facecolor(COLORS["white"])
    style_image_axis(cloud_ax)
    cloud_ax.set_title("Original point cloud", fontsize=5.8, pad=1.7, color=COLORS["ink"])

    tile_ax = ax.inset_axes([0.00, 0.04, 1.00, 0.40])
    tile_info = draw_tile_overlay(tile_ax, stone, selected_tiles, slicing_summary, gt)
    tile_ax.set_title("Adaptive overlapping tiles", fontsize=5.6, pad=1.7, color=COLORS["ink"])
    ax.add_patch(FancyArrowPatch((0.50, 0.535), (0.50, 0.462), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=7.5, lw=0.8, color=COLORS["muted"]))
    return {
        "dom_crop_pixel_box": list(crop_box),
        "displayed_original_point_count": int(np.count_nonzero(visible)),
        "tile_overlay": tile_info,
    }


def draw_tile_overlay(ax, stone, selected_tiles, slicing_summary, gt) -> dict:
    tile_crop = world_bbox_to_pixel_box(stone["bbox_world"], gt, pad_m=3.5)
    extent = pixel_box_extent(tile_crop, gt)
    show_dom(ax, read_dom_pixel_box(tile_crop), extent)
    overlap = float(slicing_summary["config"]["tile_overlap_m"])
    half = overlap / 2.0
    tile_colors = [COLORS["amber"], COLORS["blue"]]
    tile_centers_y: list[float] = []
    for tile, color in zip(selected_tiles, tile_colors, strict=True):
        x0, y0, x1, y1 = [float(value) for value in tile["bounds_m"]]
        x0, y0, x1, y1 = x0 - half, y0 - half, x1 + half, y1 + half
        tile_centers_y.append((y0 + y1) / 2.0)
        tile_ax_box = Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, edgecolor=color, alpha=0.18, linewidth=1.05)
        ax.add_patch(tile_ax_box)
    top_index = int(np.argmax(tile_centers_y))
    for index, (label, color) in enumerate([("Tile A", COLORS["amber"]), ("Tile B", COLORS["blue"])]):
        ax.text(
            0.025,
            0.90 if index == top_index else 0.12,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.0,
            weight="bold",
            color=color,
            bbox={"fc": "white", "ec": "none", "alpha": 0.88, "pad": 1.1},
        )
    return {"source_tiles": [item["tile_id"] for item in selected_tiles], "overlap_m": overlap, "dom_crop_pixel_box": list(tile_crop)}


def draw_fusion_band(ax, stone, selected_detections, gt) -> dict:
    band_heading(ax, "Tile-wise Segmentation & Fusion", "Per-tile inference -> duplicate resolution")
    crop_box = world_bbox_to_pixel_box(stone["bbox_world"], gt, pad_m=0.72)
    extent = pixel_box_extent(crop_box, gt)
    dom = read_dom_pixel_box(crop_box)

    yolo_ax = ax.inset_axes([0.27, 0.850, 0.46, 0.092])
    yolo_ax.set_axis_off()
    yolo_ax.set_facecolor(COLORS["white"])
    yolo_ax.add_patch(
        Rectangle((0.0, 0.0), 1.0, 1.0, transform=yolo_ax.transAxes, fill=False, ec=COLORS["ink"], lw=0.9)
    )
    yolo_ax.text(0.5, 0.64, "YOLO11m-seg", transform=yolo_ax.transAxes, ha="center", va="center", fontsize=5.6, weight="bold", color=COLORS["ink"])
    yolo_ax.text(0.5, 0.28, "tile-wise instance segmentation", transform=yolo_ax.transAxes, ha="center", va="center", fontsize=4.5, color=COLORS["muted"])

    positions = [[0.00, 0.46, 0.43, 0.34], [0.57, 0.46, 0.43, 0.34]]
    source_colors = [COLORS["amber"], COLORS["blue"]]
    masks = []
    for label, detection, position, color in zip(["Tile A candidate", "Tile B candidate"], selected_detections, positions, source_colors, strict=True):
        child = ax.inset_axes(position)
        show_dom(child, dom, extent)
        mask = mask_in_crop(detection, crop_box)
        masks.append(mask)
        overlay_mask(child, mask, extent, color, alpha=0.46, linewidth=1.0)
        child.set_title(label, fontsize=5.7, pad=1.7, color=color, weight="bold")

    # Fan-out from YOLO to the two candidates (distinct start points, no overlap).
    ax.add_patch(FancyArrowPatch((0.40, 0.848), (0.215, 0.802), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=6.5, lw=0.75, color=COLORS["muted"]))
    ax.add_patch(FancyArrowPatch((0.60, 0.848), (0.785, 0.802), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=6.5, lw=0.75, color=COLORS["muted"]))
    # Candidates converge into the fusion step.
    ax.add_patch(FancyArrowPatch((0.23, 0.458), (0.44, 0.402), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=7.2, lw=0.8, color=COLORS["amber"]))
    ax.add_patch(FancyArrowPatch((0.77, 0.458), (0.56, 0.402), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=7.2, lw=0.8, color=COLORS["blue"]))
    ax.text(0.50, 0.388, "correlation clustering", transform=ax.transAxes, ha="center", va="center", fontsize=5.0, weight="bold", color=COLORS["ink"], bbox={"fc": "white", "ec": "none", "alpha": 0.92, "pad": 0.6})
    ax.text(0.50, 0.350, "world-coordinate mask IoU + centroid distance", transform=ax.transAxes, ha="center", va="center", fontsize=4.5, color=COLORS["muted"], bbox={"fc": "white", "ec": "none", "alpha": 0.92, "pad": 0.6})
    ax.add_patch(FancyArrowPatch((0.50, 0.330), (0.50, 0.300), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=7.2, lw=0.8, color=COLORS["muted"]))

    fused_ax = ax.inset_axes([0.17, 0.045, 0.66, 0.25])
    show_dom(fused_ax, dom, extent)
    union = np.maximum.reduce(masks)
    overlay_mask(fused_ax, union, extent, COLORS["teal"], alpha=0.45, linewidth=1.25)
    fused_ax.text(
        0.50,
        0.95,
        "Fused rock footprint",
        transform=fused_ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.6,
        color=COLORS["teal"],
        weight="bold",
        bbox={"fc": "white", "ec": "none", "alpha": 0.82, "pad": 0.7},
    )
    return {"detection_indices": stone["detection_indices"], "source_patch_ids": stone["source_patch_ids"], "fusion_method": stone["merge_method"], "fused_mask_rendered": True}


def clean_3d_axis(ax) -> None:
    ax.set_axis_off()
    ax.set_facecolor(COLORS["white"])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(COLORS["white"])
        axis.pane.set_edgecolor(COLORS["white"])
    ax.grid(False)
    ax.view_init(elev=24, azim=-62)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass


def set_3d_limits(ax, points: np.ndarray, *, bottom: np.ndarray | None = None) -> None:
    x0, y0, z0 = np.min(points, axis=0)
    x1, y1, z1 = np.max(points, axis=0)
    if bottom is not None and len(bottom):
        z0 = min(float(z0), float(np.min(bottom)))
    pad_xy = 0.08 * max(float(x1 - x0), float(y1 - y0))
    pad_z = 0.05 * float(z1 - z0)
    ax.set_xlim(x0 - pad_xy, x1 + pad_xy)
    ax.set_ylim(y0 - pad_xy, y1 + pad_xy)
    ax.set_zlim(z0 - pad_z, z1 + pad_z)
    ax.set_box_aspect((float(x1 - x0), float(y1 - y0), max(float(z1 - z0), 1e-6) * 0.80))


def draw_3d_band(fig, ax, stone_points, volume_debug, ground_dem) -> dict:
    band_heading(ax, "Point-Cloud Mapping & 2.5D Measurement", "Fused footprint + local point cloud -> 2.5D measurement", title_fontsize=6.4)
    mapping_ax = ax.inset_axes([0.05, 0.60, 0.90, 0.26])
    mapping_ax.scatter(
        stone_points[:, 0],
        stone_points[:, 1],
        c=stone_points[:, 2],
        cmap="Greys",
        s=0.14,
        alpha=0.56,
        linewidths=0,
        rasterized=True,
    )
    mapping_ax.set_aspect("equal")
    x_min, x_max = float(np.min(stone_points[:, 0])), float(np.max(stone_points[:, 0]))
    y_min, y_max = float(np.min(stone_points[:, 1])), float(np.max(stone_points[:, 1]))
    mapping_ax.set_xlim(x_min, x_max)
    mapping_ax.set_ylim(y_min, y_max)
    style_image_axis(mapping_ax)
    mapping_ax.set_title("Local point-cloud mapping (top view)", fontsize=5.8, pad=1.8, color=COLORS["ink"], weight="bold")

    integration_ax = ax.inset_axes([0.05, 0.12, 0.90, 0.34])
    integration_ax.scatter(stone_points[:, 0], stone_points[:, 1], s=0.09, c="#6B7F89", alpha=0.23, linewidths=0, rasterized=True)
    centers = np.asarray(volume_debug["positive_centers_xy"], dtype=np.float64)
    heights = np.asarray(volume_debug["positive_heights_m"], dtype=np.float64)
    base = np.asarray(ground_dem.get_ground_z(centers[:, 0], centers[:, 1]), dtype=np.float64)
    valid = np.isfinite(base) & np.isfinite(heights)
    integration_ax.scatter(
        centers[valid, 0],
        centers[valid, 1],
        s=3.0,
        marker="s",
        c=COLORS["pass"],
        alpha=0.42,
        linewidths=0,
        rasterized=True,
    )
    integration_ax.set_aspect("equal")
    integration_ax.set_xlim(x_min, x_max)
    integration_ax.set_ylim(y_min, y_max)
    style_image_axis(integration_ax)
    integration_ax.set_title("2.5D integration cells (top view)", fontsize=5.8, pad=1.8, color=COLORS["ink"], weight="bold")
    ax.add_patch(FancyArrowPatch((0.50, 0.575), (0.50, 0.505), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=7.5, lw=0.8, color=COLORS["muted"]))
    return {"point_count": int(len(stone_points)), "view": "top", "positive_integration_cells_rendered": int(np.count_nonzero(valid)), "ground_dem_displayed": False}


def write_caption(output_dir: Path, stone: dict, volume: dict) -> Path:
    caption = f"""# Fig. 3-1 | Traceable DOM-to-2.5D rock measurement workflow

**English caption.** DOM preparation and adaptive tiling: co-registered DOM and the original photogrammetric point cloud share one local world reference; the quadtree subdivides the 20 m base tiles into 10 m children according to local edge density, and the two overlapping tiles of the example stone are shown as semi-transparent footprints. Tile-wise segmentation and fusion: every tile is processed by YOLO11m-seg instance segmentation; the two tile-level candidate masks of the same boundary-crossing rock are resolved by correlation clustering in world coordinates (weighted mask-IoU and centroid-distance similarity) and fused into one complete footprint. Point-cloud mapping and 2.5D measurement: the fused footprint is mapped to the corresponding local point cloud, and all valid positive-height cells relative to the GroundDEM are integrated into the per-rock 2.5D measurement. All panels originate from the same complete run and representative object, which passed the recorded 3D validation and volume quality-control checks; GroundDEM is used as the computational reference but is intentionally not rendered as a surface in this figure. Source records and panel provenance are provided in `metadata.json`.

**Chinese reference.** DOM 准备与自适应切片：同一局部区域的 DOM 与原始摄影测量点云共享统一空间参考；四叉树依据局部边缘密度将 20 m 基片细分为 10 m 子片，示例岩块所在的两个重叠切片以半透明轮廓显示。逐切片分割与融合：每个切片经 YOLO11m-seg 实例分割；位于切片边界附近的同一岩块形成的两个切片级候选掩膜，在世界坐标下通过相关性聚类（mask IoU 与质心距离的加权相似度）关联，并融合为完整岩块轮廓。点云映射与 2.5D 测量：融合轮廓映射到对应的局部点云，全部相对 GroundDEM 的有效正高程单元积分为逐岩块的 2.5D 测量。本图所有影像均来自同一完整运行与同一代表性岩块；该对象已通过记录的三维验证和体积质量控制。GroundDEM 作为计算参考使用，但本图不显示其表面，以避免显示伪尖峰。
"""
    path = output_dir / "caption.md"
    path.write_text(caption, encoding="utf-8")
    return path


def write_source_data(output_dir: Path, records: dict, slicing_summary: dict) -> Path:
    detection_stats = load_json(PATHS["detection_stats"])
    fusion_summary = load_json(PATHS["fusion_summary"])
    volume_stats = load_json(PATHS["volume_stats"])
    payload = {
        "figure_id": FIGURE_ID,
        "run_id": RUN_ID,
        "stone_id": STONE_ID,
        "scene": CURRENT_SCENE.to_dict(),
        "configs": {
            "slicing": slicing_summary["config"],
            "detection": detection_stats["config"],
            "fusion_and_3d_validation": fusion_summary["config"],
            "volume": volume_stats["config"],
        },
        "records": {
            "selected_tiles": records["selected_tiles"],
            "selected_detections": records["selected_detections"],
            "fused_stone": records["stone"],
            "volume": records["volume"],
        },
    }
    path = output_dir / "source_data.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_metadata(
    output_dir: Path,
    records: dict,
    panel_metadata: dict,
    pointcloud_stats: list[dict],
    ground_dem: GroundDEM,
    volume_recomputed: dict,
) -> Path:
    script_path = Path(__file__).resolve()
    source_jsons = [
        PATHS["slicing_summary"],
        PATHS["tile_stats"],
        PATHS["detections"],
        PATHS["detection_stats"],
        PATHS["accepted_stones"],
        PATHS["fusion_summary"],
        PATHS["stone_volumes"],
        PATHS["volume_stats"],
        PATHS["volume_config"],
    ]
    payload = {
        "figure_id": FIGURE_ID,
        "run_id": RUN_ID,
        "stone_id": STONE_ID,
        "status": "generated_from_complete_real_run",
        "generated_at": datetime.now().astimezone().isoformat(),
        "scene": CURRENT_SCENE.to_dict(),
        "pipeline": {"source": SOURCE, "fusion_method": METHOD},
        "selection_requirements": {
            "cross_tile_detection_count": records["stone"]["source_detection_count"],
            "source_patches_span": records["stone"]["source_patches_span"],
            "validation_3d_passed": records["stone"]["validation_3d"]["passed"],
            "volume_qc_passed": records["volume"]["qc"]["passed"],
        },
        "script": {
            "path": str(script_path.relative_to(ROOT)),
            "sha256": sha256_file(script_path),
            "backend": "Python/matplotlib",
        },
        "source_data_snapshot": file_record(output_dir / "source_data.json", with_sha256=True),
        "git": git_state(),
        "source_files": [file_record(path, with_sha256=True) for path in source_jsons]
        + [file_record(PATHS["tile_overview"], with_sha256=True)]
        + [file_record(CURRENT_SCENE.dom_path, with_sha256=False)]
        + [file_record(path, with_sha256=False) for path in CURRENT_SCENE.pointcloud_paths],
        "pointcloud_stream": pointcloud_stats,
        "ground_dem": {
            **ground_dem.to_dict(),
            "construction": "Exact systematic point selection used by the volume run; rebuilt in streaming mode for the figure.",
        },
        "volume_recomputation_check": {
            "reported_volume_m3": records["volume"]["methods"]["2d5"]["volume_m3"],
            "recomputed_volume_m3": volume_recomputed["volume_m3"],
            "reported_valid_cells": records["volume"]["methods"]["2d5"]["valid_cells"],
            "recomputed_valid_cells": volume_recomputed["valid_cells"],
            "matched": bool(
                volume_recomputed["volume_m3"] == records["volume"]["methods"]["2d5"]["volume_m3"]
                and volume_recomputed["valid_cells"] == records["volume"]["methods"]["2d5"]["valid_cells"]
            ),
        },
        "panel_sources": panel_metadata,
        "image_integrity": {
            "dom": "Raw RGB values; cropping only; no local retouching or intensity manipulation.",
            "point_cloud": "All original points inside the displayed input extent are rendered in the top view; all extracted fused-object points are rendered in both magnified top views.",
            "ground_dem": "The unchanged GroundDEM is queried only to obtain the bases of the integration cells. It is intentionally not rendered as a surface or profile.",
            "volume_grid": "All valid positive-height cells used by the recomputed 2.5D integral are rendered as vertical segments in the integration view.",
        },
        "outputs": [
            "fig_3_1.png",
            "fig_3_1.tiff",
            "fig_3_1.pdf",
            "fig_3_1.svg",
            "fig_3_1_overall_workflow.py",
            "source_data.json",
            "caption.md",
            "metadata.json",
        ],
    }
    path = output_dir / "metadata.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in PATHS.values():
        if not path.exists():
            raise FileNotFoundError(path)

    records = find_records()
    stone = records["stone"]
    volume = records["volume"]
    selected_detections = records["selected_detections"]
    selected_tiles = records["selected_tiles"]
    slicing_summary = load_json(PATHS["slicing_summary"])
    volume_config = load_json(PATHS["volume_config"])
    gt = CURRENT_SCENE.load_gt()

    pad_m = 2.5
    local_bbox = CURRENT_SCENE.xy_transform.world_bbox_to_point_bbox(stone["bbox_world"], pad_m=pad_m)
    dem_cfg = volume_config["ground_dem"]
    print("Loading local point cloud and exact GroundDEM sample...")
    local_points, dem_sample, pointcloud_stats = load_scene_points_and_dem_sample(
        local_bbox,
        subsample_step=int(dem_cfg["subsample_step"]),
    )
    print(f"  local points: {len(local_points):,}")
    print(f"  GroundDEM sample: {len(dem_sample):,}")
    ground_dem = GroundDEM(
        dem_sample,
        resolution=float(dem_cfg["resolution_m"]),
        percentile=int(dem_cfg["percentile"]),
        subsample_step=1,
        min_points_per_cell=int(dem_cfg["min_points_per_cell"]),
    )
    stone_points, _ = crop_stone_point_cloud(
        local_points,
        stone,
        records["detections"],
        gt,
        CURRENT_SCENE.xy_transform,
        bbox_pad_m=float(volume_config["crop"]["bbox_pad_m"]),
    )
    print(f"  extracted stone points: {len(stone_points):,}")
    print("Recomputing the panel-f 2.5D cells...")
    volume_recomputed = recompute_2d5_grid(
        stone_points,
        ground_dem,
        grid_resolution=float(volume_config["grid"]["resolution_m"]),
    )
    if volume_recomputed["status"] != "ok":
        raise RuntimeError(f"2.5D recomputation failed: {volume_recomputed['reason']}")
    if volume_recomputed["volume_m3"] != volume["methods"]["2d5"]["volume_m3"]:
        raise RuntimeError(
            f"Volume provenance mismatch: recomputed {volume_recomputed['volume_m3']} vs recorded {volume['methods']['2d5']['volume_m3']}"
        )

    print(f"  recomputed volume: {volume_recomputed['volume_m3']:.4f} m3", flush=True)
    print("Building three-band workflow layout...", flush=True)
    fig = plt.figure(figsize=(7.087, 3.78))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.00, 1.24, 1.00],
        left=0.035,
        right=0.985,
        bottom=0.075,
        top=0.925,
        wspace=0.22,
    )
    input_band = fig.add_subplot(grid[0, 0])
    fusion_band = fig.add_subplot(grid[0, 1])
    mapping_band = fig.add_subplot(grid[0, 2])

    panel_metadata = {}
    print("  input and partition", flush=True)
    panel_metadata["input_and_partition"] = draw_input_band(input_band, stone, local_points, selected_tiles, slicing_summary, gt)
    print("  cross-tile fusion", flush=True)
    panel_metadata["cross_tile_fusion"] = draw_fusion_band(fusion_band, stone, selected_detections, gt)
    print("  3D mapping and integration", flush=True)
    panel_metadata["mapping_and_integration"] = draw_3d_band(fig, mapping_band, stone_points, volume_recomputed["debug"], ground_dem)

    # The only inter-band arrows are short and horizontal, keeping the reading order unambiguous.
    for start_ax, end_ax in [(input_band, fusion_band), (fusion_band, mapping_band)]:
        start = start_ax.get_subplotspec().get_position(fig)
        end = end_ax.get_subplotspec().get_position(fig)
        fig.add_artist(
            FancyArrowPatch(
                (start.x1 + 0.005, 0.50 * (start.y0 + start.y1)),
                (end.x0 - 0.005, 0.50 * (end.y0 + end.y1)),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=8.5,
                linewidth=0.9,
                color=COLORS["muted"],
                zorder=50,
            )
        )

    output_stem = OUTPUT_DIR / "fig_3_1"
    print("Exporting figure files...")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600)
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)

    for raster_path, save_kwargs in (
        (output_stem.with_suffix(".png"), {"dpi": (600, 600)}),
        (output_stem.with_suffix(".tiff"), {"compression": "tiff_lzw", "dpi": (600, 600)}),
    ):
        with Image.open(raster_path) as raster:
            raster.convert("RGB").save(raster_path, **save_kwargs)

    shutil.copy2(Path(__file__).resolve(), OUTPUT_DIR / Path(__file__).name)

    write_caption(OUTPUT_DIR, stone, volume)
    write_source_data(OUTPUT_DIR, records, slicing_summary)
    write_metadata(OUTPUT_DIR, records, panel_metadata, pointcloud_stats, ground_dem, volume_recomputed)
    print(f"Figure package: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

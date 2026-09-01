from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import laspy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = 500_000_000


ROOT = Path(__file__).resolve().parents[2]
FIGURE_ID = "FIG-2-1"
OUTPUT_DIR = ROOT / "experiments" / "visualization" / "outputs" / FIGURE_ID

PANEL_SIZE_M = 25.0
DOM_PREVIEW_MAX_PIXELS = 1400
POINT_PREVIEW_READ_POINTS = 500_000
POINT_DISPLAY_LIMIT = 120_000
POINT_PREVIEW_IMAGE_SIZE = 1400
FIGURE_DPI = 300

COLORS = {
    "ink": "#25313B",
    "muted": "#65727E",
    "rule": "#D7DEE5",
    "site_a": "#2F6FA3",
    "site_b": "#2A8C7E",
    "arrow": "#4D5964",
    "panel_bg": "#F7F9FA",
}

ELEVATION_STOPS = np.asarray(
    [
        [0xDF, 0xEE, 0xF7],
        [0x91, 0xC6, 0xD9],
        [0x68, 0xA9, 0x8E],
        [0xD5, 0xB5, 0x6F],
        [0xAD, 0x7B, 0x42],
    ],
    dtype=np.float64,
) / 255.0


@dataclass(frozen=True, slots=True)
class SceneSpec:
    key: str
    label: str
    role: str
    dom_path: Path
    tfw_path: Path
    pointcloud_paths: tuple[Path, ...]
    x_shift_m: float = 0.0
    y_shift_m: float = 0.0
    coordinate_note: str = "absolute world coordinates"
    crop_fraction_x: float = 0.5
    crop_fraction_y: float = 0.5


SCENES = (
    SceneSpec(
        key="mine_site_a",
        label="Mine Site A",
        role="Used for parameter setting and workflow freezing",
        dom_path=ROOT / "data" / "dom2" / "DOM.tif",
        tfw_path=ROOT / "data" / "dom2" / "DOM.tfw",
        pointcloud_paths=(
            ROOT / "data" / "pointcloud2" / "Data" / "BlockB.laz",
            ROOT / "data" / "pointcloud2" / "Data" / "BlockY.laz",
        ),
        coordinate_note="absolute world coordinates",
        crop_fraction_x=0.30,
        crop_fraction_y=0.65,
    ),
    SceneSpec(
        key="mine_site_b",
        label="Mine Site B",
        role="Used for independent execution with frozen settings",
        dom_path=ROOT / "data" / "dom3" / "DOM.tif",
        tfw_path=ROOT / "data" / "dom3" / "DOM.tfw",
        pointcloud_paths=(
            ROOT / "data" / "pointcloud3" / "Data" / "BlockB.laz",
            ROOT / "data" / "pointcloud3" / "Data" / "BlockY.laz",
        ),
        x_shift_m=623499.1064384683,
        y_shift_m=4678587.312019404,
        coordinate_note="point cloud translated to the DOM reference",
        crop_fraction_x=0.35,
        crop_fraction_y=0.78,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, include_hash: bool = False) -> dict:
    stat = path.stat()
    record = {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
    }
    if include_hash:
        record["sha256"] = sha256_file(path)
    return record


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


def require_inputs(scene: SceneSpec) -> None:
    for path in (scene.dom_path, scene.tfw_path, *scene.pointcloud_paths):
        if not path.exists():
            raise FileNotFoundError(path)


def elevation_colors(norm: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(norm, dtype=np.float64), 0.0, 1.0)
    scaled = values * (len(ELEVATION_STOPS) - 1)
    left = np.floor(scaled).astype(np.int32)
    right = np.clip(left + 1, 0, len(ELEVATION_STOPS) - 1)
    t = (scaled - left)[:, None]
    rgb = (1.0 - t) * ELEVATION_STOPS[left] + t * ELEVATION_STOPS[right]
    alpha = np.ones((len(values), 1), dtype=np.float64)
    return np.hstack((rgb, alpha))


def parse_tfw(path: Path) -> tuple[float, float, float, float, float, float]:
    values = [float(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != 6:
        raise ValueError(f"Invalid TFW file: {path}")
    a, d, b, e, c, f = values
    return a, d, b, e, c, f


def dom_metadata(scene: SceneSpec) -> tuple[dict, tuple[float, float, float, float], tuple[float, float, float, float, float, float]]:
    a, d, b, e, c, f = parse_tfw(scene.tfw_path)
    with Image.open(scene.dom_path) as image:
        width, height = image.size
    if abs(b) > 1e-9 or abs(d) > 1e-9:
        raise RuntimeError("Rotated DOM world files are not supported by this figure helper.")
    x0 = c
    x1 = c + a * width
    y0 = f + e * height
    y1 = f
    bounds = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    metadata = {
        "width_px": int(width),
        "height_px": int(height),
        "crs": "EPSG:4536",
        "bounds": [float(v) for v in bounds],
        "resolution_m": [float(abs(a)), float(abs(e))],
    }
    return metadata, bounds, (a, d, b, e, c, f)


def pointcloud_world_bounds(scene: SceneSpec) -> tuple[float, float, float, float, list[dict]]:
    records: list[dict] = []
    x0 = math.inf
    y0 = math.inf
    x1 = -math.inf
    y1 = -math.inf
    for path in scene.pointcloud_paths:
        with laspy.open(path) as reader:
            header = reader.header
            wx0 = float(header.x_min + scene.x_shift_m)
            wy0 = float(header.y_min + scene.y_shift_m)
            wx1 = float(header.x_max + scene.x_shift_m)
            wy1 = float(header.y_max + scene.y_shift_m)
            x0 = min(x0, wx0)
            y0 = min(y0, wy0)
            x1 = max(x1, wx1)
            y1 = max(y1, wy1)
            records.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "point_count": int(header.point_count),
                    "world_bounds": [wx0, wy0, wx1, wy1],
                    "z_bounds_raw": [float(header.z_min), float(header.z_max)],
                }
            )
    return x0, y0, x1, y1, records


def choose_crop_bounds(scene: SceneSpec) -> tuple[float, float, float, float, dict]:
    dom_record, dom_bounds, _ = dom_metadata(scene)

    px0, py0, px1, py1, pc_records = pointcloud_world_bounds(scene)
    overlap_left = max(float(dom_bounds[0]), px0)
    overlap_bottom = max(float(dom_bounds[1]), py0)
    overlap_right = min(float(dom_bounds[2]), px1)
    overlap_top = min(float(dom_bounds[3]), py1)
    if overlap_right <= overlap_left or overlap_top <= overlap_bottom:
        raise RuntimeError(f"{scene.label} DOM and point-cloud extents do not overlap.")

    width = min(PANEL_SIZE_M, overlap_right - overlap_left)
    height = min(PANEL_SIZE_M, overlap_top - overlap_bottom)
    size = min(width, height)
    center_x = overlap_left + scene.crop_fraction_x * (overlap_right - overlap_left)
    center_y = overlap_bottom + scene.crop_fraction_y * (overlap_top - overlap_bottom)
    left = min(max(center_x - 0.5 * size, overlap_left), overlap_right - size)
    bottom = min(max(center_y - 0.5 * size, overlap_bottom), overlap_top - size)
    right = left + size
    top = bottom + size
    metadata = {
        "dom": dom_record,
        "pointclouds": pc_records,
        "overlap_bounds": [overlap_left, overlap_bottom, overlap_right, overlap_top],
        "crop_bounds": [left, bottom, right, top],
        "crop_size_m": float(size),
    }
    return left, bottom, right, top, metadata


def read_dom_crop(scene: SceneSpec, bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    left, bottom, right, top = bounds
    _, _, gt = dom_metadata(scene)
    a, _, _, e, c, f = gt
    with Image.open(scene.dom_path) as image:
        width, height = image.size
        col0 = max(0, int(math.floor((left - c) / a)))
        col1 = min(width, int(math.ceil((right - c) / a)))
        row0 = max(0, int(math.floor((top - f) / e)))
        row1 = min(height, int(math.ceil((bottom - f) / e)))
        if col1 <= col0 or row1 <= row0:
            raise RuntimeError("DOM crop window is empty after clamping.")
        crop = image.crop((col0, row0, col1, row1)).convert("RGB")
        scale = min(DOM_PREVIEW_MAX_PIXELS / max(crop.size), 1.0)
        if scale < 1.0:
            crop = crop.resize((max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))), Image.Resampling.LANCZOS)
        image_array = np.asarray(crop, dtype=np.uint8).copy()
        nodata = np.all(image_array <= 2, axis=2)
        image_array[nodata] = 247
    crop_left = c + col0 * a
    crop_right = c + col1 * a
    crop_top = f + row0 * e
    crop_bottom = f + row1 * e
    return image_array, (float(crop_left), float(crop_right), float(crop_bottom), float(crop_top))


def build_pointcloud_sample(scene: SceneSpec, bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, dict]:
    left, bottom, right, top = bounds
    sampled: list[np.ndarray] = []
    streamed_points = 0
    kept_points = 0

    per_file_limit = max(1, POINT_DISPLAY_LIMIT // len(scene.pointcloud_paths))
    for file_index, path in enumerate(scene.pointcloud_paths):
        print(f"  reading preview points: {path.name}", flush=True)
        with laspy.open(path) as reader:
            read_count = min(POINT_PREVIEW_READ_POINTS, int(reader.header.point_count))
            points = reader.read_points(read_count)
        streamed_points += read_count
        xs = np.asarray(points.x, dtype=np.float64) + scene.x_shift_m
        ys = np.asarray(points.y, dtype=np.float64) + scene.y_shift_m
        keep = (xs >= left) & (xs <= right) & (ys >= bottom) & (ys <= top)
        if not np.any(keep):
            continue
        xs = xs[keep]
        ys = ys[keep]
        zs = np.asarray(points.z, dtype=np.float64)[keep]
        kept_points += len(zs)
        selected = np.column_stack((xs, ys, zs))
        if len(selected) > per_file_limit:
            rng = np.random.default_rng(2401 + file_index)
            indices = np.sort(rng.choice(len(selected), size=per_file_limit, replace=False))
            selected = selected[indices]
        sampled.append(selected)

    if not sampled:
        raise RuntimeError(f"No point-cloud points found in {scene.label} crop.")

    points = np.vstack(sampled)
    z = points[:, 2]
    q02, q98 = np.percentile(z, [2, 98])
    norm = np.clip((z - q02) / max(q98 - q02, 1e-9), 0.0, 1.0)
    colors = elevation_colors(norm)
    metadata = {
        "streamed_points": int(streamed_points),
        "points_in_crop": int(kept_points),
        "preview_points_read_per_file": int(POINT_PREVIEW_READ_POINTS),
        "display_limit_total": int(POINT_DISPLAY_LIMIT),
        "display_points": int(len(points)),
        "relative_elevation_quantile_reference": {"q02_raw_z": float(q02), "q98_raw_z": float(q98)},
    }
    print(f"  display points: {len(points):,}", flush=True)
    return np.column_stack((points[:, 0], points[:, 1], z, colors[:, 0], colors[:, 1], colors[:, 2], colors[:, 3])), metadata


def rasterize_pointcloud_preview(sample: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    left, bottom, right, top = bounds
    image = np.full((POINT_PREVIEW_IMAGE_SIZE, POINT_PREVIEW_IMAGE_SIZE, 3), 247, dtype=np.uint8)
    cols = np.floor((sample[:, 0] - left) / (right - left) * (POINT_PREVIEW_IMAGE_SIZE - 1)).astype(np.int32)
    rows = np.floor((top - sample[:, 1]) / (top - bottom) * (POINT_PREVIEW_IMAGE_SIZE - 1)).astype(np.int32)
    cols = np.clip(cols, 0, POINT_PREVIEW_IMAGE_SIZE - 1)
    rows = np.clip(rows, 0, POINT_PREVIEW_IMAGE_SIZE - 1)
    rgb = np.clip(sample[:, 3:6] * 255.0, 0, 255).astype(np.uint8)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        rr = np.clip(rows + dy, 0, POINT_PREVIEW_IMAGE_SIZE - 1)
        cc = np.clip(cols + dx, 0, POINT_PREVIEW_IMAGE_SIZE - 1)
        image[rr, cc] = rgb
    return image


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def paste_image_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    title: str,
    color: str,
    *,
    note: str | None = None,
) -> None:
    x0, y0, x1, y1 = box
    panel = Image.fromarray(image).resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    canvas.paste(panel, (x0, y0))
    draw.rectangle(box, outline=hex_to_rgb(COLORS["rule"]), width=2)
    label_font = load_font(23, bold=True)
    note_font = load_font(18)
    label_text = f"({label})" if not title else f"({label}) {title}"
    tw, th = text_size(draw, label_text, label_font)
    draw.rounded_rectangle((x0 + 12, y0 + 12, x0 + 26 + tw, y0 + 24 + th), radius=4, fill=(255, 255, 255))
    draw.text((x0 + 19, y0 + 17), label_text, fill=hex_to_rgb(COLORS["ink"]), font=label_font)
    if note:
        nw, nh = text_size(draw, note, note_font)
        draw.rounded_rectangle((x1 - nw - 28, y1 - nh - 36, x1 - 12, y1 - 12), radius=4, fill=(255, 255, 255))
        draw.text((x1 - nw - 20, y1 - nh - 28), note, fill=hex_to_rgb(COLORS["muted"]), font=note_font)


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str) -> None:
    rgb = hex_to_rgb(color)
    draw.line((start, end), fill=rgb, width=4)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 20
    spread = 0.45
    p1 = (end[0] - length * math.cos(angle - spread), end[1] - length * math.sin(angle - spread))
    p2 = (end[0] - length * math.cos(angle + spread), end[1] - length * math.sin(angle + spread))
    draw.polygon([end, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))], fill=rgb)


def draw_role_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    site: SceneSpec,
    color: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(255, 255, 255))
    title_font = load_font(24, bold=True)
    site_font = load_font(27, bold=True)
    body_font = load_font(22)
    note_font = load_font(18)
    draw.text((x0 + 18, y0 + 16), f"({label}) Workflow use", fill=hex_to_rgb(COLORS["ink"]), font=title_font)
    draw.text((x0 + 22, y0 + 118), site.label, fill=hex_to_rgb(color), font=site_font)
    y = y0 + 305
    for line in wrap_text(draw, site.role, body_font, x1 - x0 - 48):
        draw.text((x0 + 22, y), line, fill=hex_to_rgb(COLORS["ink"]), font=body_font)
        y += 30
    y = y1 - 92
    for line in wrap_text(draw, site.coordinate_note, note_font, x1 - x0 - 48):
        draw.text((x0 + 22, y), line, fill=hex_to_rgb(COLORS["muted"]), font=note_font)
        y += 24


def draw_figure(assets: dict[str, dict]) -> Image.Image:
    canvas = Image.new("RGB", (1215, 1210), "white")
    draw = ImageDraw.Draw(canvas)

    image_w = 540
    image_h = 540
    gap = 34
    x_dom = 50
    x_pc = x_dom + image_w + gap
    y_top = 45
    y_bottom = y_top + image_h + 80

    color_by_scene = {"mine_site_a": COLORS["site_a"], "mine_site_b": COLORS["site_b"]}
    rows = [
        ("mine_site_a", y_top, "a", "b"),
        ("mine_site_b", y_bottom, "c", "d"),
    ]

    for key, y, dom_label, pc_label in rows:
        color = color_by_scene[key]
        asset = assets[key]
        paste_image_panel(
            canvas,
            draw,
            asset["dom_image"],
            (x_dom, y, x_dom + image_w, y + image_h),
            dom_label,
            "",
            color,
        )
        paste_image_panel(
            canvas,
            draw,
            asset["pointcloud_image"],
            (x_pc, y, x_pc + image_w, y + image_h),
            pc_label,
            "",
            color,
        )
    return canvas


def write_caption() -> Path:
    caption = (
        "图 2-1 两个研究区的 DOM 与摄影测量点云输入对照。"
        "（a）矿区 A 的 DOM 局部影像；（b）矿区 A 同一坐标窗口内的点云俯视高程渲染；"
        "（c）矿区 B 的 DOM 局部影像；（d）矿区 B 同一坐标窗口内的点云俯视高程渲染。"
        "矿区 B 点云根据已记录的场景平移量转换至 DOM 坐标参考。"
    )
    path = OUTPUT_DIR / "caption.md"
    path.write_text(caption, encoding="utf-8")
    return path


def write_metadata(asset_metadata: dict[str, dict], outputs: list[str]) -> Path:
    script_path = Path(__file__).resolve()
    payload = {
        "figure_id": FIGURE_ID,
        "status": "generated_from_real_dom_and_pointcloud_sources",
        "generated_at": datetime.now().astimezone().isoformat(),
        "backend": "Python/PIL/numpy",
        "script": {"path": str(script_path.relative_to(ROOT)), "sha256": sha256_file(script_path)},
        "git": git_state(),
        "layout": {
            "description": "Two rows by two columns; figure-internal text is limited to panel labels only.",
            "panel_size_m": PANEL_SIZE_M,
            "pointcloud_preview_points_read_per_file": POINT_PREVIEW_READ_POINTS,
            "pointcloud_display_limit_total": POINT_DISPLAY_LIMIT,
            "pointcloud_preview_image_size": POINT_PREVIEW_IMAGE_SIZE,
            "output_dpi": FIGURE_DPI,
        },
        "sources": {
            scene.key: {
                "label": scene.label,
                "role": scene.role,
                "dom_path": str(scene.dom_path.relative_to(ROOT)),
                "tfw_path": str(scene.tfw_path.relative_to(ROOT)),
                "pointcloud_paths": [str(path.relative_to(ROOT)) for path in scene.pointcloud_paths],
                "xy_shift_m": [scene.x_shift_m, scene.y_shift_m],
                "coordinate_note": scene.coordinate_note,
                "crop_fraction_xy": [scene.crop_fraction_x, scene.crop_fraction_y],
            }
            for scene in SCENES
        },
        "input_files": [
            file_record(path, include_hash=False)
            for scene in SCENES
            for path in (scene.dom_path, scene.tfw_path, *scene.pointcloud_paths)
        ],
        "panel_sources": asset_metadata,
        "image_integrity": {
            "dom": "Raster windows were read from original GeoTIFF files and downsampled only for display; exact near-zero no-data background pixels were rendered as white.",
            "point_cloud": "A bounded preview sample was read from each LAZ file; true points inside the displayed coordinate window were rasterized into a top-view elevation-colored image panel.",
            "no_fabrication": "No synthetic image content, manual retouching, or mixed-scene substitution was used.",
        },
        "outputs": outputs,
    }
    path = OUTPUT_DIR / "metadata.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assets: dict[str, dict] = {}
    metadata: dict[str, dict] = {}
    for scene in SCENES:
        print(f"Preparing {scene.label}...", flush=True)
        require_inputs(scene)
        left, bottom, right, top, crop_metadata = choose_crop_bounds(scene)
        bounds = (left, bottom, right, top)
        dom_image, extent = read_dom_crop(scene, bounds)
        pointcloud_sample, pc_metadata = build_pointcloud_sample(scene, bounds)
        pointcloud_image = rasterize_pointcloud_preview(pointcloud_sample, bounds)
        assets[scene.key] = {
            "dom_image": dom_image,
            "pointcloud_image": pointcloud_image,
            "extent": extent,
            "bounds": (bounds[0], bounds[2], bounds[1], bounds[3]),
        }
        metadata[scene.key] = {**crop_metadata, "pointcloud_rendering": pc_metadata}

    print("Drawing figure...", flush=True)
    figure = draw_figure(assets)
    output_png = OUTPUT_DIR / "fig_2_1_study_area_inputs.png"
    print("Saving PNG...", flush=True)
    figure.save(output_png, dpi=(FIGURE_DPI, FIGURE_DPI))

    print("Writing caption and metadata...", flush=True)
    write_caption()
    write_metadata(metadata, [output_png.name, "caption.md", "metadata.json"])
    print(f"Figure written to: {output_png}")


if __name__ == "__main__":
    main()

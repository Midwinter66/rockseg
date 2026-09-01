"""Validate a non-destructive XY registration candidate for mine site B.

This script deliberately does not import or modify CURRENT_SCENE. It reads only
dom3 and pointcloud3, derives an extent-based translation, and writes audit
artifacts needed before site B can become an independent test scene.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOM_PATH = PROJECT_ROOT / "data" / "dom3" / "DOM.tif"
POINTCLOUD_PATHS = (
    PROJECT_ROOT / "data" / "pointcloud3" / "Data" / "BlockB.laz",
    PROJECT_ROOT / "data" / "pointcloud3" / "Data" / "BlockY.laz",
)
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "validation" / "outputs" / "scene_b_alignment"


@dataclass(frozen=True)
class Bounds:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) / 2.0

    def to_dict(self) -> dict[str, float]:
        return {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
        }


def union_bounds(items: list[Bounds]) -> Bounds:
    return Bounds(
        xmin=min(item.xmin for item in items),
        ymin=min(item.ymin for item in items),
        xmax=max(item.xmax for item in items),
        ymax=max(item.ymax for item in items),
    )


def load_source_metadata() -> tuple[dict, Bounds, list[dict], Bounds]:
    import laspy
    import rasterio

    with rasterio.open(DOM_PATH) as dataset:
        dom = {
            "path": str(DOM_PATH),
            "width_px": int(dataset.width),
            "height_px": int(dataset.height),
            "resolution_m": [float(dataset.res[0]), float(dataset.res[1])],
            "crs": str(dataset.crs),
        }
        dom_bounds = Bounds(*[float(value) for value in dataset.bounds])

    pointclouds: list[dict] = []
    local_bounds: list[Bounds] = []
    for path in POINTCLOUD_PATHS:
        with laspy.open(path) as reader:
            header = reader.header
            bounds = Bounds(
                xmin=float(header.mins[0]),
                ymin=float(header.mins[1]),
                xmax=float(header.maxs[0]),
                ymax=float(header.maxs[1]),
            )
            local_bounds.append(bounds)
            pointclouds.append(
                {
                    "path": str(path),
                    "point_count": int(header.point_count),
                    "local_bounds": bounds.to_dict(),
                }
            )
    return dom, dom_bounds, pointclouds, union_bounds(local_bounds)


def derive_translation(dom_bounds: Bounds, local_bounds: Bounds) -> tuple[float, float, dict[str, float]]:
    """Use matching union-bounds centers; residuals quantify the extent fit."""
    x_shift = dom_bounds.center_x - local_bounds.center_x
    y_shift = dom_bounds.center_y - local_bounds.center_y
    translated = Bounds(
        local_bounds.xmin + x_shift,
        local_bounds.ymin + y_shift,
        local_bounds.xmax + x_shift,
        local_bounds.ymax + y_shift,
    )
    residuals = {
        "xmin_m": translated.xmin - dom_bounds.xmin,
        "xmax_m": translated.xmax - dom_bounds.xmax,
        "ymin_m": translated.ymin - dom_bounds.ymin,
        "ymax_m": translated.ymax - dom_bounds.ymax,
    }
    return x_shift, y_shift, residuals


def sample_point_clouds(
    *, max_points: int, chunk_size: int, seed: int, x_shift: float, y_shift: float
) -> tuple[np.ndarray, list[dict]]:
    import laspy

    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    sampling_records: list[dict] = []
    point_counts = []
    for path in POINTCLOUD_PATHS:
        with laspy.open(path) as reader:
            point_counts.append(int(reader.header.point_count))
    total_points = sum(point_counts)

    for index, path in enumerate(POINTCLOUD_PATHS):
        target = max(1, round(max_points * point_counts[index] / total_points))
        probability = min(1.0, target / point_counts[index])
        selected: list[np.ndarray] = []
        with laspy.open(path) as reader:
            for points in reader.chunk_iterator(chunk_size):
                keep = rng.random(len(points)) < probability
                if not np.any(keep):
                    continue
                selected.append(
                    np.column_stack(
                        (
                            np.asarray(points.x)[keep] + x_shift,
                            np.asarray(points.y)[keep] + y_shift,
                        )
                    ).astype(np.float64, copy=False)
                )
        sampled = np.concatenate(selected, axis=0) if selected else np.empty((0, 2), dtype=np.float64)
        samples.append(sampled)
        sampling_records.append(
            {
                "path": str(path),
                "source_point_count": point_counts[index],
                "sample_probability": probability,
                "sampled_point_count": int(len(sampled)),
            }
        )
    return np.concatenate(samples, axis=0), sampling_records


def load_dom_preview(*, max_size: int) -> np.ndarray:
    from PIL import Image

    # Rasterio's out-of-size read is unusually slow for this striped GeoTIFF.
    # Pillow provides a fast, read-only display preview; spatial bounds remain
    # sourced from Rasterio in load_source_metadata().
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(DOM_PATH) as source:
        source = source.convert("RGB")
        source.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        image = np.asarray(source, dtype=np.float32) / 255.0
    return np.clip(image, 0.0, 1.0)


def crop_preview(preview: np.ndarray, dom_bounds: Bounds, bounds: Bounds) -> np.ndarray:
    height, width = preview.shape[:2]
    x0 = int(np.floor((bounds.xmin - dom_bounds.xmin) / (dom_bounds.xmax - dom_bounds.xmin) * width))
    x1 = int(np.ceil((bounds.xmax - dom_bounds.xmin) / (dom_bounds.xmax - dom_bounds.xmin) * width))
    y0 = int(np.floor((dom_bounds.ymax - bounds.ymax) / (dom_bounds.ymax - dom_bounds.ymin) * height))
    y1 = int(np.ceil((dom_bounds.ymax - bounds.ymin) / (dom_bounds.ymax - dom_bounds.ymin) * height))
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    return preview[y0:y1, x0:x1]


def select_validation_windows(
    preview: np.ndarray, dom_bounds: Bounds, points: np.ndarray, *, window_size_m: float, count: int
) -> list[tuple[Bounds, float, int]]:
    """Choose distributed windows that contain both DOM content and point samples."""
    # Black areas are outside the valid DOM footprint. The threshold is only
    # used to avoid empty review panels; it is not a registration criterion.
    valid_dom = np.mean(preview, axis=2) > (8.0 / 255.0)
    x_span = dom_bounds.xmax - dom_bounds.xmin
    y_span = dom_bounds.ymax - dom_bounds.ymin
    candidates: list[tuple[Bounds, float, int, float, float]] = []

    # A dense grid permits each intended spatial position to move inward when
    # an image edge is outside the actual surveyed footprint.
    for y_fraction in np.linspace(0.09, 0.91, 13):
        for x_fraction in np.linspace(0.09, 0.91, 13):
            center_x = dom_bounds.xmin + x_span * x_fraction
            center_y = dom_bounds.ymin + y_span * y_fraction
            window = Bounds(
                center_x - window_size_m / 2.0,
                center_y - window_size_m / 2.0,
                center_x + window_size_m / 2.0,
                center_y + window_size_m / 2.0,
            )
            dom_crop = crop_preview(valid_dom[..., np.newaxis], dom_bounds, window)
            coverage = float(dom_crop.mean()) if dom_crop.size else 0.0
            point_count = int(
                np.count_nonzero(
                    (points[:, 0] >= window.xmin)
                    & (points[:, 0] <= window.xmax)
                    & (points[:, 1] >= window.ymin)
                    & (points[:, 1] <= window.ymax)
                )
            )
            candidates.append((window, coverage, point_count, x_fraction, y_fraction))

    eligible = [candidate for candidate in candidates if candidate[1] >= 0.95 and candidate[2] >= 80]
    if len(eligible) < count:
        eligible = [candidate for candidate in candidates if candidate[1] >= 0.85 and candidate[2] > 0]
    if len(eligible) < count:
        raise RuntimeError("Unable to select enough non-empty local validation windows.")

    # Retain a 2 x 3 geographical spread while preferring the closest valid
    # candidate to each target. A candidate cannot be reused in another panel.
    targets = [(x, y) for y in (0.18, 0.50, 0.82) for x in (0.25, 0.75)]
    selected: list[tuple[Bounds, float, int]] = []
    remaining = eligible.copy()
    for target_x, target_y in targets[:count]:
        best_index = min(
            range(len(remaining)),
            key=lambda index: (
                (remaining[index][3] - target_x) ** 2 + (remaining[index][4] - target_y) ** 2
                + 5.0 * (1.0 - remaining[index][1])
                - 0.000002 * min(remaining[index][2], 2_000)
            ),
        )
        window, coverage, point_count, _, _ = remaining.pop(best_index)
        selected.append((window, coverage, point_count))
    return selected


def render_artifacts(dom_bounds: Bounds, points: np.ndarray, output_dir: Path) -> tuple[list[dict], list[int]]:
    from PIL import Image, ImageDraw

    def paint_points(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        x_pixels = np.floor(
            (points[:, 0] - dom_bounds.xmin) / (dom_bounds.xmax - dom_bounds.xmin) * (width - 1)
        ).astype(np.int64)
        y_pixels = np.floor(
            (dom_bounds.ymax - points[:, 1]) / (dom_bounds.ymax - dom_bounds.ymin) * (height - 1)
        ).astype(np.int64)
        valid = (x_pixels >= 0) & (x_pixels < width) & (y_pixels >= 0) & (y_pixels < height)
        rendered = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        ys, xs = y_pixels[valid], x_pixels[valid]
        original = rendered[ys, xs].astype(np.float32)
        rendered[ys, xs] = (original * 0.38 + np.asarray([230, 57, 70], dtype=np.float32) * 0.62).astype(np.uint8)
        return rendered

    artifacts: list[dict] = []
    preview = load_dom_preview(max_size=4800)
    overlay = paint_points(preview)
    overview = output_dir / "site_b_alignment_overview.png"
    Image.fromarray(overlay, mode="RGB").save(overview)
    artifacts.append({"path": str(overview), "description": "Full-scene DOM and transformed point-cloud overlay"})

    window_size_m = 18.0
    selected_windows = select_validation_windows(
        preview, dom_bounds, points, window_size_m=window_size_m, count=6
    )
    panel_size = 600
    padding = 28
    label_height = 28
    mosaic = Image.new("RGB", (panel_size * 2 + padding * 3, (panel_size + label_height) * 3 + padding * 4), "white")
    draw = ImageDraw.Draw(mosaic)
    for index, (window, coverage, point_count) in enumerate(selected_windows):
        row, column = divmod(index, 2)
        crop = crop_preview(overlay, dom_bounds, window)
        panel = Image.fromarray(crop, mode="RGB").resize((panel_size, panel_size), Image.Resampling.LANCZOS)
        x_offset = padding + column * (panel_size + padding)
        y_offset = padding + row * (panel_size + label_height + padding)
        mosaic.paste(panel, (x_offset, y_offset + label_height))
        draw.text(
            (x_offset, y_offset),
            f"Window {index + 1}: DOM {coverage:.0%}; samples {point_count}",
            fill="black",
        )
    windows = output_dir / "site_b_alignment_windows.png"
    mosaic.save(windows)
    artifacts.append({"path": str(windows), "description": "Six distributed local DOM and point-cloud overlays"})
    return artifacts, [int(preview.shape[1]), int(preview.shape[0])]


def write_feature_template(output_dir: Path) -> Path:
    path = output_dir / "manual_feature_residuals_template.csv"
    headers = [
        "feature_id",
        "feature_description",
        "dom_x_world_m",
        "dom_y_world_m",
        "point_x_local_m",
        "point_y_local_m",
        "point_x_world_m",
        "point_y_world_m",
        "residual_m",
        "reviewer_note",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for index in range(1, 11):
            writer.writerow([f"F{index:02d}", "", "", "", "", "", "", "", "", ""])
    return path


def write_summary(
    *, output_dir: Path, dom: dict, dom_bounds: Bounds, pointclouds: list[dict], local_bounds: Bounds,
    x_shift: float, y_shift: float, extent_residuals: dict[str, float], samples: list[dict], artifacts: list[dict], preview_size: list[int],
    feature_template: Path,
) -> Path:
    max_extent_residual = max(abs(value) for value in extent_residuals.values())
    resolution = max(dom["resolution_m"])
    summary = {
        "validation_type": "site_b_extent_translation_and_visual_overlay",
        "status": "provisional_pending_manual_feature_check",
        "inputs": {"dom": dom, "point_clouds": pointclouds},
        "dom_bounds_world_m": dom_bounds.to_dict(),
        "point_cloud_union_bounds_local_m": local_bounds.to_dict(),
        "candidate_xy_transform": {
            "mode": "translated_local",
            "x_shift_m": x_shift,
            "y_shift_m": y_shift,
            "equations": {
                "point_to_world": "x_world = x_local + x_shift; y_world = y_local + y_shift",
                "world_to_point": "x_local = x_world - x_shift; y_local = y_world - y_shift",
            },
        },
        "extent_residuals_m": extent_residuals,
        "max_extent_residual_m": max_extent_residual,
        "extent_check": {
            "criterion": "maximum translated point-cloud extent residual <= one DOM pixel",
            "dom_pixel_m": resolution,
            "passed": bool(max_extent_residual <= resolution + 1e-9),
        },
        "automated_check_limit": "Extent agreement does not prove feature-level alignment. Manual inspection of the generated windows and feature residual recording are required before activating site B.",
        "manual_feature_check": {
            "required_feature_count": 6,
            "recommended_distribution": "image corners, center, and the point-cloud block join",
            "record_template": str(feature_template),
            "acceptance_to_define_before_independent_run": "Record median, RMSE, and maximum planar residual; confirm no systematic directional drift across the image.",
        },
        "sampling": samples,
        "dom_preview_size_px": preview_size,
        "artifacts": artifacts,
        "scope_guard": "This validation reads only dom3 and pointcloud3 and does not alter CURRENT_SCENE or any site A output.",
    }
    path = output_dir / "alignment_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a non-destructive site B XY registration candidate.")
    parser.add_argument("--max-points", type=int, default=180_000, help="Target number of point-cloud samples for overlay plots.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="LAZ reader chunk size.")
    parser.add_argument("--seed", type=int, default=20260817, help="Deterministic point-sampling seed.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.max_points <= 0 or args.chunk_size <= 0:
        raise SystemExit("--max-points and --chunk-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dom, dom_bounds, pointclouds, local_bounds = load_source_metadata()
    x_shift, y_shift, residuals = derive_translation(dom_bounds, local_bounds)
    points, samples = sample_point_clouds(
        max_points=args.max_points,
        chunk_size=args.chunk_size,
        seed=args.seed,
        x_shift=x_shift,
        y_shift=y_shift,
    )
    artifacts, preview_size = render_artifacts(dom_bounds, points, args.output_dir)
    feature_template = write_feature_template(args.output_dir)
    summary_path = write_summary(
        output_dir=args.output_dir,
        dom=dom,
        dom_bounds=dom_bounds,
        pointclouds=pointclouds,
        local_bounds=local_bounds,
        x_shift=x_shift,
        y_shift=y_shift,
        extent_residuals=residuals,
        samples=samples,
        artifacts=artifacts,
        preview_size=preview_size,
        feature_template=feature_template,
    )
    print(f"Candidate translation: x_shift={x_shift:.6f} m, y_shift={y_shift:.6f} m")
    print(f"Sampled points: {len(points):,}")
    print(f"Summary: {summary_path}")
    print("Status: provisional; review the six local overlays and complete feature residuals before activating site B.")


if __name__ == "__main__":
    main()

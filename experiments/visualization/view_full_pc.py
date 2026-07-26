from __future__ import annotations

import argparse
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POINT_CLOUDS = (
    PROJECT_ROOT / "data" / "pointcloud2" / "Data" / "BlockB.laz",
    PROJECT_ROOT / "data" / "pointcloud2" / "Data" / "BlockY.laz",
)
DISPLAY_COLORS = (
    [0.22, 0.22, 0.22],
    [0.22, 0.22, 0.22],
)


def load_random_sample(
    path: Path,
    max_points: int,
    chunk_size: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Load a random point sample without holding the complete LAZ in memory."""
    rng = np.random.default_rng(seed)
    sampled_chunks: list[np.ndarray] = []

    with laspy.open(path) as reader:
        total_points = int(reader.header.point_count)
        sample_probability = 1.0
        if 0 < max_points < total_points:
            sample_probability = max_points / total_points

        for points in reader.chunk_iterator(chunk_size):
            if sample_probability < 1.0:
                keep = rng.random(len(points)) < sample_probability
                if not np.any(keep):
                    continue
            else:
                keep = slice(None)

            xyz = np.column_stack(
                (
                    np.asarray(points.x)[keep],
                    np.asarray(points.y)[keep],
                    np.asarray(points.z)[keep],
                )
            )
            sampled_chunks.append(xyz)

    if not sampled_chunks:
        raise RuntimeError(f"No points were loaded from: {path}")

    sampled = np.concatenate(sampled_chunks, axis=0)
    if 0 < max_points < len(sampled):
        selected = rng.choice(len(sampled), size=max_points, replace=False)
        sampled = sampled[selected]

    return sampled, total_points


def show_point_clouds(
    point_sets: list[tuple[str, np.ndarray]],
    point_size: float,
) -> None:
    # Rendering projected coordinates near Y=4.7 million as float32 can create
    # visible bands. Recenter for display while preserving the exact geometry.
    display_count = sum(len(points) for _, points in point_sets)
    display_origin = sum(
        (points.sum(axis=0) for _, points in point_sets),
        start=np.zeros(3, dtype=np.float64),
    ) / display_count

    print(f"Display origin (world): {display_origin}")

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name="Full point-cloud scene", width=1400, height=900)

    local_bounds_min = np.full(3, np.inf)
    local_bounds_max = np.full(3, -np.inf)
    for index, (name, points_world) in enumerate(point_sets):
        points_local = points_world - display_origin
        local_bounds_min = np.minimum(local_bounds_min, points_local.min(axis=0))
        local_bounds_max = np.maximum(local_bounds_max, points_local.max(axis=0))

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points_local)
        cloud.paint_uniform_color(DISPLAY_COLORS[index % len(DISPLAY_COLORS)])
        visualizer.add_geometry(cloud)
        print(f"  {name}: {len(points_world):,} display points")

    print(f"Local minimum:          {local_bounds_min}")
    print(f"Local maximum:          {local_bounds_max}")

    render_option = visualizer.get_render_option()
    render_option.background_color = np.asarray([1.0, 1.0, 1.0])
    render_option.point_size = float(point_size)

    visualizer.run()
    visualizer.destroy_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize multiple LAZ regions with Open3D.")
    parser.add_argument(
        "--files",
        type=Path,
        nargs="+",
        default=list(DEFAULT_POINT_CLOUDS),
        help="LAZ/LAS files to display (default: pointcloud2 BlockB and BlockY).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=6_000_000,
        help="Maximum random display points across all files; use 0 for every point.",
    )
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point-size", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [path.expanduser().resolve() for path in args.files]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Point-cloud file does not exist:\n{missing_text}")

    point_counts: list[int] = []
    for path in paths:
        with laspy.open(path) as reader:
            point_counts.append(int(reader.header.point_count))

    total_scene_points = sum(point_counts)
    if args.max_points <= 0 or args.max_points >= total_scene_points:
        display_budgets = [0] * len(paths)
    else:
        display_budgets = [
            max(1, round(args.max_points * count / total_scene_points))
            for count in point_counts
        ]

    point_sets: list[tuple[str, np.ndarray]] = []
    for index, (path, total, budget) in enumerate(
        zip(paths, point_counts, display_budgets, strict=True)
    ):
        print(f"Loading: {path}")
        points, _ = load_random_sample(
            path=path,
            max_points=budget,
            chunk_size=args.chunk_size,
            seed=args.seed + index,
        )
        print(f"Loaded for display: {len(points):,} / {total:,} points")
        point_sets.append((path.name, points))

    print(f"Scene points: {total_scene_points:,}")
    print(f"Displayed:    {sum(len(points) for _, points in point_sets):,}")
    show_point_clouds(point_sets, point_size=args.point_size)


if __name__ == "__main__":
    main()

"""
Point-cloud viewer for fused stones.

This viewer uses the same full-scene rendering style as view_full_pc.py:
- pointcloud files are sampled independently and shown together
- the base scene keeps the original full-scene appearance
- stone masks are overlaid by recoloring only the matched points

Layouts:
  - stones: full-scene view with each selected stone mask colored separately
  - scene: full-scene view with accepted/rejected stones colored by status
  - single: local map-context view for one stone with mask coloring and a bbox

Examples:
  python experiments/visualization/run_visualize_pc.py --source quadtree_dom --method correlation_clustering --layout stones --mode accepted
  python experiments/visualization/run_visualize_pc.py --source quadtree_dom --method correlation_clustering --layout scene --mode all
  python experiments/visualization/run_visualize_pc.py --source quadtree_dom --method correlation_clustering --stone-rank 0 --mode accepted
  python experiments/visualization/run_visualize_pc.py --source quadtree_dom --method correlation_clustering --stone-rank 0 --single-context raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.pointcloud_index import PointCloudXYGridIndex
from experiments.common.scene_reference import CURRENT_SCENE
from experiments.common.stone_region import crop_stone_point_cloud

FUSION_ROOT = PROJECT_ROOT / "experiments" / "fusion" / "outputs"
DETECTION_ROOT = PROJECT_ROOT / "experiments" / "detection" / "outputs"

SOURCES = ["sahi", "quadtree_dom"]
METHODS = ["heuristic", "correlation_clustering"]
STONE_MODES = ["accepted", "rejected", "all"]
VIEW_LAYOUTS = ["stones", "scene"]
VISUAL_CROP_MODES = ["bbox", "mask"]
SINGLE_CONTEXT_MODES = ["sampled", "raw"]

FILE_BASE_COLORS = np.asarray(
    [
        [0.86, 0.28, 0.18],
        [0.16, 0.48, 0.86],
        [0.18, 0.72, 0.54],
        [0.90, 0.62, 0.16],
    ],
    dtype=np.float32,
)
STONE_MASK_COLORS = np.asarray(
    [
        [0.95, 0.25, 0.25],
        [0.20, 0.82, 0.38],
        [0.25, 0.52, 0.95],
        [0.98, 0.82, 0.18],
        [0.78, 0.32, 0.92],
        [0.14, 0.86, 0.86],
        [0.92, 0.38, 0.62],
        [0.86, 0.58, 0.20],
    ],
    dtype=np.float32,
)
ACCEPTED_COLOR = np.asarray([0.98, 0.82, 0.18], dtype=np.float32)
REJECTED_COLOR = np.asarray([0.92, 0.12, 0.12], dtype=np.float32)
LOCAL_CONTEXT_COLOR = np.asarray([0.62, 0.62, 0.66], dtype=np.float32)
VIEWER_BACKGROUND = np.asarray([0.03, 0.03, 0.03], dtype=np.float64)

_CACHED_FULL_SCENE_POINTS: np.ndarray | None = None
_CACHED_FULL_SCENE_SOURCE_IDS: np.ndarray | None = None
_CACHED_POINT_INDEX: dict[tuple[int, float], PointCloudXYGridIndex] = {}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_data(source: str, method: str) -> tuple[dict, list[dict]]:
    fusion_path = FUSION_ROOT / source / method / "fusion_stats.json"
    detections_path = DETECTION_ROOT / source / "detections.json"
    if not fusion_path.exists():
        raise FileNotFoundError(f"Missing fusion result: {fusion_path}")
    if not detections_path.exists():
        raise FileNotFoundError(f"Missing detections: {detections_path}")

    fusion = _load_json(fusion_path)
    detections = json.loads(detections_path.read_text(encoding="utf-8"))
    return fusion, detections


def _select_stones(fusion: dict, mode: str) -> list[dict]:
    accepted = list(fusion.get("stones", []))
    rejected = list(fusion.get("rejected_stones_detail", []))
    if mode == "accepted":
        return accepted
    if mode == "rejected":
        return rejected
    return accepted + rejected


def _sorted_stones(stones: list[dict]) -> list[dict]:
    def key_fn(stone: dict) -> tuple[float, float]:
        bbox = stone.get("bbox_world", [0.0, 0.0, 0.0, 0.0])
        bbox_area = max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))
        score = float(stone.get("score_mean", stone.get("score_max", 0.0)))
        return bbox_area, score

    return sorted(stones, key=key_fn, reverse=True)


def _load_random_sample(path: Path, max_points: int, chunk_size: int, seed: int) -> tuple[np.ndarray, int]:
    import laspy

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
            ).astype(np.float64, copy=False)
            sampled_chunks.append(xyz)

    if not sampled_chunks:
        raise RuntimeError(f"No points were loaded from: {path}")

    sampled = np.concatenate(sampled_chunks, axis=0)
    if 0 < max_points < len(sampled):
        selected = rng.choice(len(sampled), size=max_points, replace=False)
        sampled = sampled[selected]

    return sampled, total_points


def _load_scene_sample(max_points: int, chunk_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    paths = list(CURRENT_SCENE.pointcloud_paths)
    point_counts: list[int] = []

    import laspy

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Point-cloud file does not exist: {path}")
        with laspy.open(path) as reader:
            point_counts.append(int(reader.header.point_count))

    total_scene_points = sum(point_counts)
    if max_points <= 0 or max_points >= total_scene_points:
        budgets = [0] * len(paths)
    else:
        budgets = [
            max(1, round(max_points * count / total_scene_points))
            for count in point_counts
        ]

    point_sets: list[np.ndarray] = []
    source_ids: list[np.ndarray] = []
    for index, (path, total_points, budget) in enumerate(zip(paths, point_counts, budgets, strict=True)):
        print(f"Loading scene sample: {path}")
        points, _ = _load_random_sample(
            path=path,
            max_points=budget,
            chunk_size=chunk_size,
            seed=seed + index,
        )
        print(f"  loaded for display: {len(points):,} / {total_points:,} points")
        point_sets.append(points)
        source_ids.append(np.full(len(points), index, dtype=np.int16))

    scene_points = np.concatenate(point_sets, axis=0) if point_sets else np.empty((0, 3), dtype=np.float64)
    scene_source_ids = np.concatenate(source_ids, axis=0) if source_ids else np.empty(0, dtype=np.int16)
    print(f"Scene points: {total_scene_points:,}")
    print(f"Displayed:    {len(scene_points):,}")
    return scene_points, scene_source_ids, paths


def _load_full_scene() -> tuple[np.ndarray, np.ndarray]:
    global _CACHED_FULL_SCENE_POINTS, _CACHED_FULL_SCENE_SOURCE_IDS
    if _CACHED_FULL_SCENE_POINTS is not None and _CACHED_FULL_SCENE_SOURCE_IDS is not None:
        return _CACHED_FULL_SCENE_POINTS, _CACHED_FULL_SCENE_SOURCE_IDS

    import laspy

    parts: list[np.ndarray] = []
    source_ids: list[np.ndarray] = []
    for source_idx, path in enumerate(CURRENT_SCENE.pointcloud_paths):
        if not path.exists():
            raise FileNotFoundError(f"Missing point cloud file: {path}")
        las = laspy.read(str(path))
        points = np.column_stack([las.x, las.y, las.z]).astype(np.float64, copy=False)
        parts.append(points)
        source_ids.append(np.full(len(points), source_idx, dtype=np.int16))
        print(f"  loaded {path.name}: {len(points):,} points")

    _CACHED_FULL_SCENE_POINTS = np.vstack(parts)
    _CACHED_FULL_SCENE_SOURCE_IDS = np.concatenate(source_ids)
    print(f"  full scene points: {len(_CACHED_FULL_SCENE_POINTS):,}")
    return _CACHED_FULL_SCENE_POINTS, _CACHED_FULL_SCENE_SOURCE_IDS


def _build_index(points: np.ndarray, cell_size: float) -> PointCloudXYGridIndex:
    key = (id(points), float(cell_size))
    cached = _CACHED_POINT_INDEX.get(key)
    if cached is not None:
        return cached

    print(f"  building XY grid index @ {cell_size:.2f} m")
    index = PointCloudXYGridIndex.build(points, cell_size=cell_size)
    _CACHED_POINT_INDEX[key] = index
    return index


def _compute_display_origin(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(points, dtype=np.float64).mean(axis=0)


def _open_viewer(geometries: list, title: str, point_size: float = 1.5) -> None:
    import open3d as o3d

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name=title, width=1400, height=900)
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    render_option = visualizer.get_render_option()
    render_option.background_color = VIEWER_BACKGROUND
    render_option.point_size = float(point_size)
    visualizer.run()
    visualizer.destroy_window()


def _base_scene_colors(source_ids: np.ndarray) -> np.ndarray:
    if len(source_ids) == 0:
        return np.empty((0, 3), dtype=np.float32)
    colors = np.zeros((len(source_ids), 3), dtype=np.float32)
    for source_idx in np.unique(source_ids):
        colors[source_ids == source_idx] = FILE_BASE_COLORS[int(source_idx) % len(FILE_BASE_COLORS)]
    return colors


def _match_masked_points(candidate_points: np.ndarray, masked_points: np.ndarray) -> np.ndarray:
    matched = np.zeros(len(candidate_points), dtype=bool)
    if len(candidate_points) == 0 or len(masked_points) == 0:
        return matched

    lookup: dict[tuple[float, float, float], list[int]] = {}
    for idx, row in enumerate(candidate_points):
        key = tuple(np.round(row, 6))
        lookup.setdefault(key, []).append(idx)

    for row in masked_points:
        key = tuple(np.round(row, 6))
        indices = lookup.get(key)
        if not indices:
            continue
        matched[indices.pop()] = True
    return matched


def _query_candidate_indices(
    points: np.ndarray,
    stone: dict,
    index: PointCloudXYGridIndex,
    bbox_pad_m: float,
) -> np.ndarray:
    bbox = stone.get("bbox_world", [0.0, 0.0, 0.0, 0.0])
    x0, y0, x1, y1 = CURRENT_SCENE.xy_transform.world_bbox_to_point_bbox(
        bbox,
        pad_m=bbox_pad_m,
    )
    return index.query_bbox_indices(x0, y0, x1, y1)


def _extract_mask_match(
    points: np.ndarray,
    stone: dict,
    detections: list[dict],
    gt: tuple[float, float, float, float, float, float],
    index: PointCloudXYGridIndex,
    *,
    bbox_pad_m: float = 0.35,
    use_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    candidate_indices = _query_candidate_indices(points, stone, index, bbox_pad_m=bbox_pad_m)
    if len(candidate_indices) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.zeros(0, dtype=bool),
            {
                "query_mode": "bbox_context",
                "bbox_candidate_count": 0,
                "mask_point_count": 0,
                "bbox_pad_m": float(bbox_pad_m),
            },
        )

    candidate_points = points[candidate_indices].copy()
    if use_mask:
        masked_points, crop_info = crop_stone_point_cloud(
            points,
            stone,
            detections,
            gt,
            CURRENT_SCENE.xy_transform,
            bbox_pad_m=bbox_pad_m,
            pc_index=index,
        )
        matched_mask = _match_masked_points(candidate_points, masked_points)
        info = dict(crop_info)
        info["visual_context_point_count"] = int(len(candidate_points))
        info["visual_mask_point_count"] = int(np.count_nonzero(matched_mask))
        info["query_mode"] = "bbox_context_with_mask_coloring"
        return candidate_indices.astype(np.int64, copy=False), matched_mask, info

    matched_mask = np.ones(len(candidate_indices), dtype=bool)
    return (
        candidate_indices.astype(np.int64, copy=False),
        matched_mask,
        {
            "query_mode": "bbox_context_only",
            "bbox_candidate_count": int(len(candidate_indices)),
            "mask_point_count": int(len(candidate_indices)),
            "bbox_pad_m": float(bbox_pad_m),
        },
    )


def _build_scene_geometry(points: np.ndarray, colors: np.ndarray, display_origin: np.ndarray):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64) - display_origin)
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return cloud


def _sample_local_points(
    candidate_indices: np.ndarray,
    matched_mask: np.ndarray,
    *,
    max_points: int,
    seed: int,
) -> np.ndarray:
    if max_points <= 0 or len(candidate_indices) <= max_points:
        return np.arange(len(candidate_indices), dtype=np.int64)

    mask_positions = np.flatnonzero(matched_mask)
    if len(mask_positions) >= max_points:
        return np.sort(
            np.random.default_rng(seed).choice(mask_positions, size=max_points, replace=False)
        ).astype(np.int64, copy=False)

    keep = set(mask_positions.tolist())
    remaining = max_points - len(mask_positions)
    other_positions = np.flatnonzero(~matched_mask)
    if remaining > 0 and len(other_positions) > 0:
        sampled = np.random.default_rng(seed).choice(
            other_positions,
            size=min(remaining, len(other_positions)),
            replace=False,
        )
        keep.update(int(v) for v in sampled.tolist())
    return np.asarray(sorted(keep), dtype=np.int64)


def _paint_scene_by_stones(
    scene_points: np.ndarray,
    scene_source_ids: np.ndarray,
    stones: list[dict],
    detections: list[dict],
    gt: tuple[float, float, float, float, float, float],
    *,
    crop_mode: str,
    index_cell_size: float,
    stone_limit: int,
    status_coloring: bool,
) -> tuple[np.ndarray, dict]:
    scene_index = _build_index(scene_points, index_cell_size)
    colors = _base_scene_colors(scene_source_ids)

    colored_stones = 0
    colored_points = 0
    selected_stones = stones[:stone_limit] if stone_limit > 0 else stones

    for stone_idx, stone in enumerate(selected_stones):
        candidate_indices, matched_mask, _ = _extract_mask_match(
            scene_points,
            stone,
            detections,
            gt,
            scene_index,
            bbox_pad_m=0.35,
            use_mask=(crop_mode == "mask"),
        )
        if len(candidate_indices) == 0 or not np.any(matched_mask):
            continue

        matched_indices = candidate_indices[matched_mask]
        if status_coloring:
            is_rejected = bool(stone.get("validation_3d", {}).get("passed") is False)
            color = REJECTED_COLOR if is_rejected else ACCEPTED_COLOR
        else:
            color = STONE_MASK_COLORS[stone_idx % len(STONE_MASK_COLORS)]

        colors[matched_indices] = color
        colored_points += int(len(matched_indices))
        colored_stones += 1

    return colors, {
        "colored_stones": int(colored_stones),
        "colored_points": int(colored_points),
        "scene_point_count": int(len(scene_points)),
    }


def _view_scene(
    *,
    scene_points: np.ndarray,
    scene_source_ids: np.ndarray,
    stones: list[dict],
    detections: list[dict],
    gt: tuple[float, float, float, float, float, float],
    layout: str,
    crop_mode: str,
    index_cell_size: float,
    max_stones: int,
) -> None:
    status_coloring = layout == "scene"
    colors, stats = _paint_scene_by_stones(
        scene_points=scene_points,
        scene_source_ids=scene_source_ids,
        stones=stones,
        detections=detections,
        gt=gt,
        crop_mode=crop_mode,
        index_cell_size=index_cell_size,
        stone_limit=max_stones,
        status_coloring=status_coloring,
    )

    display_origin = _compute_display_origin(scene_points)
    geometry = _build_scene_geometry(scene_points, colors, display_origin)
    print(f"  layout: {layout}")
    print(f"  crop mode: {crop_mode}")
    print(f"  colored stones: {stats['colored_stones']}")
    print(f"  colored points: {stats['colored_points']:,}")
    print(f"  display origin: {display_origin}")
    _open_viewer([geometry], f"Stone Visualization - {layout}", point_size=1.5)


def _view_single_stone(
    points: np.ndarray,
    source_ids: np.ndarray,
    stone: dict,
    detections: list[dict],
    gt: tuple[float, float, float, float, float, float],
    *,
    crop_mode: str,
    index_cell_size: float,
    stone_max_points: int,
    seed: int,
    context_mode: str,
) -> None:
    import open3d as o3d

    index = _build_index(points, index_cell_size)
    candidate_indices, matched_mask, crop_info = _extract_mask_match(
        points,
        stone,
        detections,
        gt,
        index,
        bbox_pad_m=0.35,
        use_mask=(crop_mode == "mask"),
    )
    if len(candidate_indices) == 0 or not np.any(matched_mask):
        print("  no local context points were extracted for this stone")
        return

    selected_pos = _sample_local_points(
        candidate_indices,
        matched_mask,
        max_points=stone_max_points,
        seed=seed,
    )
    candidate_indices = candidate_indices[selected_pos]
    matched_mask = matched_mask[selected_pos]

    candidate_points = points[candidate_indices]
    candidate_source_ids = source_ids[candidate_indices]
    masked_points = candidate_points[matched_mask]

    display_origin = _compute_display_origin(candidate_points)
    colors = _base_scene_colors(candidate_source_ids)
    is_rejected = bool(stone.get("validation_3d", {}).get("passed") is False)
    colors[matched_mask] = REJECTED_COLOR if is_rejected else ACCEPTED_COLOR

    cloud = _build_scene_geometry(candidate_points, colors, display_origin)

    mask_cloud = o3d.geometry.PointCloud()
    mask_cloud.points = o3d.utility.Vector3dVector(masked_points.astype(np.float64) - display_origin)
    bbox = mask_cloud.get_axis_aligned_bounding_box()
    bbox.color = (0.0, 0.8, 0.0)

    validation = stone.get("validation_3d", {})
    print(f"  stone_id: {stone.get('stone_id')}")
    print(f"  context points: {len(candidate_points):,}")
    print(f"  mask points: {len(masked_points):,}")
    print(f"  visual mode: {context_mode} context + mask coloring + bbox")
    print(f"  crop query: {crop_info.get('query_mode')}")
    print(
        "  mask size_m: "
        f"{np.ptp(masked_points[:, 0]):.3f} x "
        f"{np.ptp(masked_points[:, 1]):.3f} x "
        f"{np.ptp(masked_points[:, 2]):.3f}"
    )
    print(f"  z_range_m: {validation.get('z_range_m', 'n/a')}")
    print(f"  validation: {validation.get('passed', 'n/a')}")
    if validation.get("reasons"):
        print(f"  reasons: {', '.join(validation['reasons'])}")

    _open_viewer([cloud, bbox], f"Stone {stone.get('stone_id')}", point_size=1.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="View fused stones on the point cloud")
    parser.add_argument("--source", choices=SOURCES, default="quadtree_dom")
    parser.add_argument("--method", choices=METHODS, default="correlation_clustering")
    parser.add_argument("--mode", choices=STONE_MODES, default="accepted")
    parser.add_argument("--stone-rank", type=int, default=None, help="Rank after sorting by bbox area")
    parser.add_argument("--list-stones", type=int, nargs="?", const=20, default=None)
    parser.add_argument("--layout", choices=VIEW_LAYOUTS, default="stones")
    parser.add_argument("--visual-crop", choices=VISUAL_CROP_MODES, default="mask")
    parser.add_argument("--max-batch-stones", type=int, default=80)
    parser.add_argument("--stone-max-points", type=int, default=120_000)
    parser.add_argument("--max-overview-points", type=int, default=6_000_000)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--index-cell-size", type=float, default=1.0)
    parser.add_argument("--single-context", choices=SINGLE_CONTEXT_MODES, default="sampled")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fusion, detections = _load_data(args.source, args.method)
    gt = CURRENT_SCENE.load_gt()
    stones = _sorted_stones(_select_stones(fusion, args.mode))
    if not stones:
        print(f"  no stones found for mode={args.mode}")
        return

    if args.list_stones is not None:
        limit = min(args.list_stones, len(stones))
        print(f"  mode={args.mode}  listing top {limit} stones")
        for rank, stone in enumerate(stones[:limit]):
            bbox = stone.get("bbox_world", [0.0, 0.0, 0.0, 0.0])
            bbox_area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
            validation = stone.get("validation_3d", {})
            print(
                f"  [{rank:03d}] {stone.get('stone_id')} "
                f"bbox_area={bbox_area:.3f} "
                f"score={stone.get('score_mean', 0):.3f} "
                f"passed={validation.get('passed', 'n/a')}"
            )
        return

    if args.stone_rank is not None:
        if args.stone_rank < 0 or args.stone_rank >= len(stones):
            raise SystemExit(f"stone-rank out of range: 0..{len(stones) - 1}")
        if args.single_context == "raw":
            points, source_ids = _load_full_scene()
        else:
            points, source_ids, _ = _load_scene_sample(
                max_points=args.max_overview_points,
                chunk_size=args.chunk_size,
                seed=args.seed,
            )
        _view_single_stone(
            points=points,
            source_ids=source_ids,
            stone=stones[args.stone_rank],
            detections=detections,
            gt=gt,
            crop_mode=args.visual_crop,
            index_cell_size=args.index_cell_size,
            stone_max_points=args.stone_max_points,
            seed=args.seed + args.stone_rank,
            context_mode=args.single_context,
        )
        return

    scene_points, scene_source_ids, _ = _load_scene_sample(
        max_points=args.max_overview_points,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    _view_scene(
        scene_points=scene_points,
        scene_source_ids=scene_source_ids,
        stones=stones,
        detections=detections,
        gt=gt,
        layout=args.layout,
        crop_mode=args.visual_crop,
        index_cell_size=args.index_cell_size,
        max_stones=args.max_batch_stones,
    )


if __name__ == "__main__":
    main()

"""Refactored fusion evaluation module.

This script is designed for experiment reporting (paper-ready):
1) automatic quality estimation via point-cloud Z-range heuristic;
2) interactive manual review on DOM crops + empty-tile checks;
3) comprehensive report generation (JSON/Markdown/CSV + charts).

Usage:
  python experiments/evaluation/run_evaluation.py --auto
  python experiments/evaluation/run_evaluation.py --label
  python experiments/evaluation/run_evaluation.py --report
  python experiments/evaluation/run_evaluation.py --all

You can evaluate multiple configurations at once:
  python experiments/evaluation/run_evaluation.py --source all --method all --all
  python experiments/evaluation/run_evaluation.py --source all --method all --compare
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any
import sys

try:
    import cv2
except Exception:
    cv2 = None
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.pointcloud_index import PointCloudXYGridIndex
from experiments.common.scene_reference import CURRENT_SCENE
from experiments.common.stone_region import crop_stone_point_cloud

SELF_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SELF_DIR / "outputs"

# Data paths
DOM_PATH = CURRENT_SCENE.dom_path
TFW_PATH = CURRENT_SCENE.tfw_path
LAZ_PATHS = list(CURRENT_SCENE.pointcloud_paths)
FUSION_ROOT = PROJECT_ROOT / "experiments" / "fusion" / "outputs"
DETECTION_ROOT = PROJECT_ROOT / "experiments" / "detection" / "outputs"

SOURCES = ["sahi", "quadtree_dom"]
METHODS = ["heuristic", "correlation_clustering"]
DEFAULT_SOURCE = SOURCES[0]
DEFAULT_METHOD = METHODS[1]

STONE_PREFIX = "stone:"
EMPTY_KEY = "empty_tiles"
META_KEY = "_meta"


def _require_cv2():
    if cv2 is None:
        raise ModuleNotFoundError("Missing dependency: opencv-python (cv2). Install requirements or activate correct environment.")

def _require_import(module_name: str, pip_hint: str | None = None) -> None:
    try:
        __import__(module_name)
    except Exception as exc:
        hint = f" (install: pip install {pip_hint})" if pip_hint else ""
        raise ModuleNotFoundError(f"Missing dependency: {module_name}.{hint}") from exc


def _pair_is_legacy_default(source: str, method: str) -> bool:
    return source == DEFAULT_SOURCE and method == DEFAULT_METHOD


# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# Utilities
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T

def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num) / float(den) if den > 0 else default


def _wilson_ci(success: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson interval for precision estimate (one proportion)."""
    if n <= 0:
        return 0.0, 0.0
    phat = success / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError:
            break
    # Fallback for legacy/edge files
    raw = path.read_bytes()
    if not raw:
        return default
    try:
        return json.loads(raw.decode("utf-8-sig"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_tfw(path: Path) -> tuple:
    lines = [float(l.strip()) for l in path.read_text("utf-8").splitlines() if l.strip()]
    return (lines[4], lines[0], lines[2], lines[5], lines[1], lines[3])


def _pixel_to_world(gt: tuple, px: float, py: float) -> tuple[float, float]:
    return float(gt[0] + px * gt[1] + py * gt[2]), float(gt[3] + px * gt[4] + py * gt[5])


def _world_to_pixel(gt: tuple, wx: float, wy: float) -> tuple[int, int]:
    """Approximate world to DOM pixel conversion."""
    px = int(round((wx - gt[0]) / gt[1]))
    py = int(round((wy - gt[3]) / gt[5]))
    return px, py


def _rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    mask = np.zeros(h * w, dtype=np.uint8)
    counts = [int(v) for v in rle.get("counts", [])]
    starts_with = rle.get("starts_with")
    if starts_with is None:
        starts_with = 0 if sum(counts[1::2]) <= sum(counts[0::2]) else 1
    pos = 0
    for i, count in enumerate(counts):
        if (int(starts_with) + i) % 2 == 1:
            mask[pos:pos + count] = 255
        pos += count
    return mask.reshape(h, w)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _bbox_area_from_world(bbox_world: list[float]) -> float:
    if len(bbox_world) < 4:
        return 0.0
    w = max(0.0, float(bbox_world[2]) - float(bbox_world[0]))
    h = max(0.0, float(bbox_world[3]) - float(bbox_world[1]))
    return w * h


def _ensure_tuple(v: Any, n: int, default: float = 0.0):
    if v is None:
        return tuple([default] * n)
    out = list(v)
    if len(out) < n:
        out += [default] * (n - len(out))
    return tuple(out[:n])


# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# Paths & config
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T

def _load_config() -> dict:
    return _read_json(SELF_DIR / "config.json", {})


def _pair_output_dir(source: str, method: str) -> Path:
    return OUTPUTS_DIR / source / method


def _pair_outputs(pair_dir: Path) -> dict[str, Path]:
    return {
        "auto_json": pair_dir / "auto_eval.json",
        "manual_json": pair_dir / "manual_labels.json",
        "report_json": pair_dir / "evaluation_report.json",
        "report_md": pair_dir / "evaluation_report.md",
        "report_csv": pair_dir / "evaluation_report.csv",
        "stone_csv": pair_dir / "stone_level_results.csv",
        "z_hist": pair_dir / "plots" / "z_range_distribution.png",
        "sweep_csv": pair_dir / "plots" / "manual_threshold_sweep.csv",
        "sweep_png": pair_dir / "plots" / "manual_threshold_sweep.png",
    }


def _legacy_outputs() -> dict[str, Path]:
    return {
        "manual_labels": OUTPUTS_DIR / "manual_labels.json",
        "evaluation_report": OUTPUTS_DIR / "evaluation_report.json",
        "comparison": OUTPUTS_DIR / "comparison_report.json",
    }


def _resolve_pairs(source: str, method: str) -> list[tuple[str, str]]:
    srcs = SOURCES if source == "all" else [source]
    mths = METHODS if method == "all" else [method]
    return [(s, m) for s in srcs for m in mths]


# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# Load experiment results
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T

def _load_fusion(source: str, method: str) -> tuple[list[dict], list[dict], dict]:
    fp = FUSION_ROOT / source / method / "fusion_stats.json"
    dp = DETECTION_ROOT / source / "detections.json"
    fusion = _read_json(fp, {})
    detections = _read_json(dp, [])
    stones = fusion.get("stones", []) if isinstance(fusion, dict) else []
    return stones, detections, fusion if isinstance(fusion, dict) else {}


# Cached point cloud
_PC_CACHE: np.ndarray | None = None
_PC_INDEX_CACHE: dict[float, PointCloudXYGridIndex] = {}

def _load_point_cloud(min_pts: int = 10) -> np.ndarray:
    global _PC_CACHE
    if _PC_CACHE is not None:
        return _PC_CACHE

    _require_import('laspy', 'laspy')
    import laspy

    all_pts = []
    for path in LAZ_PATHS:
        if not path.exists():
            continue
        las = laspy.read(str(path))
        pts = np.column_stack([las.x, las.y, las.z]).astype(np.float32)
        all_pts.append(pts)
    if not all_pts:
        _PC_CACHE = np.zeros((0, 3), dtype=np.float32)
        return _PC_CACHE

    pc = np.vstack(all_pts)
    if pc.shape[0] < min_pts:
        _PC_CACHE = pc
    else:
        _PC_CACHE = pc.astype(np.float32)
    return _PC_CACHE


def _load_point_cloud_index(pc: np.ndarray, cell_size: float) -> PointCloudXYGridIndex:
    key = float(cell_size)
    cached = _PC_INDEX_CACHE.get(key)
    if cached is not None and cached.points is pc:
        return cached

    index = PointCloudXYGridIndex.build(pc, cell_size=cell_size)
    _PC_INDEX_CACHE[key] = index
    return index


def _point_in_poly_mask(points_xy: np.ndarray, polys: list[np.ndarray]) -> np.ndarray:
    _require_cv2()
    """Points shape (N,2), polys shape list of contour arrays with shape (1,n,2)."""
    if points_xy.size == 0 or not polys:
        return np.zeros(points_xy.shape[0], dtype=bool)
    keep = np.zeros(points_xy.shape[0], dtype=bool)
    for poly in polys:
        keep |= np.array([
            cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
            for x, y in points_xy
        ])
    return keep


def _build_detection_polys(stone: dict, detections: list[dict], gt: tuple) -> tuple[list[np.ndarray], int]:
    _require_cv2()
    polys: list[np.ndarray] = []
    skipped = 0
    for idx in stone.get("detection_indices", []):
        if idx >= len(detections):
            skipped += 1
            continue
        det = detections[idx]
        rle = det.get("rle_mask")
        if not rle:
            skipped += 1
            continue
        mask = _rle_decode(rle)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            skipped += 1
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            skipped += 1
            continue
        eps = max(1.0, 0.01 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, eps, True)
        ox, oy = det.get("pixel_origin", [0, 0])
        poly = []
        for px, py in approx.reshape(-1, 2):
            wx, wy = _pixel_to_world(gt, float(px + ox), float(py + oy))
            poly.append(list(CURRENT_SCENE.xy_transform.world_to_point_xy(wx, wy)))
        if len(poly) >= 3:
            polys.append(np.array([poly], dtype=np.float32))
    return polys, skipped


# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# �� �Զ�����
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T

def _auto_eval(
    stones: list[dict],
    detections: list[dict],
    gt: tuple,
    z_thresh: float,
    cfg: dict,
    source: str,
    method: str,
) -> dict:
    """Automatic pseudo-evaluation based on cloud Z-range statistics per stone."""
    pc = _load_point_cloud(min_pts=int(cfg.get("z_range", {}).get("min_points_per_stone", 10)))
    if pc.size == 0:
        return {
            "error": "No point cloud data",
            "stone_labels": [],
            "pair": {"source": source, "method": method},
        }

    print(f"[Auto] Loaded point cloud: {len(pc)} points")

    results: list[dict] = []
    tp = fp = skipped = 0
    skip_stats = defaultdict(int)
    min_points = int(cfg.get("z_range", {}).get("min_points_per_stone", 10))
    bbox_pad_m = float(cfg.get("z_range", {}).get("bbox_pad_m", 0.5))
    index_cell_size = float(cfg.get("z_range", {}).get("index_cell_size_m", max(1.0, bbox_pad_m * 2.0)))
    pc_index = _load_point_cloud_index(pc, index_cell_size)

    for s in stones:
        sid = s.get("stone_id", "unknown")
        rec = {
            "stone_id": sid,
            "status": "skipped",
            "source_detection_count": int(s.get("source_detection_count", 0)),
            "score_mean": _safe_float(s.get("score_mean", 0.0), 0.0),
            "score_max": _safe_float(s.get("score_max", 0.0), 0.0),
            "bbox_area_m2": _bbox_area_from_world(_ensure_tuple(s.get("bbox_world"), 4)),
            "n_detection_polygons": 0,
            "n_points_bbox": 0,
            "n_points_in_poly": 0,
            "z_min": None,
            "z_max": None,
            "z_range": 0.0,
            "auto_label": "SKIPPED",
            "skip_reason": None,
        }

        if rec["source_detection_count"] <= 0:
            rec["skip_reason"] = "no_detections"
            skip_stats["no_detections"] += 1
            results.append(rec)
            skipped += 1
            continue

        b = _ensure_tuple(s.get("bbox_world"), 4)
        if not b or len(b) < 4:
            rec["skip_reason"] = "invalid_bbox"
            skip_stats["invalid_bbox"] += 1
            results.append(rec)
            skipped += 1
            continue

        matched, crop_info = crop_stone_point_cloud(
            pc,
            s,
            detections,
            gt,
            CURRENT_SCENE.xy_transform,
            bbox_pad_m=bbox_pad_m,
            pc_index=pc_index,
        )
        rec["n_detection_polygons"] = int(crop_info.get("polygon_count", 0))
        rec["n_points_bbox"] = int(crop_info.get("bbox_candidate_count", 0))
        rec["n_points_in_poly"] = int(len(matched))

        if rec["n_detection_polygons"] <= 0:
            rec["skip_reason"] = "no_polygon"
            skip_stats["no_polygon"] += 1
            results.append(rec)
            skipped += 1
            continue

        if rec["n_points_bbox"] < min_points:
            rec["skip_reason"] = "few_points_bbox"
            skip_stats["few_points"] += 1
            results.append(rec)
            skipped += 1
            continue

        if len(matched) < min_points:
            rec["skip_reason"] = "few_points_polygon"
            skip_stats["few_points_in_polygon"] += 1
            results.append(rec)
            skipped += 1
            continue

        z_vals = matched[:, 2]
        z_min = float(np.min(z_vals))
        z_max = float(np.max(z_vals))
        z_range = float(np.ptp(z_vals))
        rec.update({
            "status": "evaluated",
            "z_min": round(z_min, 4),
            "z_max": round(z_max, 4),
            "z_range": round(z_range, 4),
        })

        is_tp = z_range >= z_thresh
        rec["auto_label"] = "TP" if is_tp else "FP"
        if is_tp:
            tp += 1
        else:
            fp += 1
        results.append(rec)

    evaluated = tp + fp
    precision = _safe_div(tp, evaluated)
    auto_precision_ci = _wilson_ci(tp, evaluated)

    out = {
        "pair": {
            "source": source,
            "method": method,
        },
        "generated_by": "run_evaluation.py",
        "z_threshold_m": z_thresh,
        "min_points_per_stone": min_points,
        "total_stones": len(stones),
        "total_evaluated": evaluated,
        "tp": tp,
        "fp": fp,
        "skipped": skipped,
        "skip_detail": dict(skip_stats),
        "auto_precision": round(precision, 4),
        "auto_precision_ci": [round(auto_precision_ci[0], 4), round(auto_precision_ci[1], 4)],
        "stone_labels": results,
    }

    paths = _pair_outputs(_pair_output_dir(source, method))
    _write_json(paths["auto_json"], out)
    print(f"[Auto] Done: TP={tp}, FP={fp}, evaluated={evaluated}, precision={precision:.1%}")
    return out


def _crop_dom_around_bbox(
    dom: np.ndarray,
    gt: tuple,
    bbox: list[float],
    margin_m: float = 2.0,
) -> tuple[np.ndarray, int, int, int, int]:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    half_w = (bbox[2] - bbox[0]) / 2 + margin_m
    half_h = (bbox[3] - bbox[1]) / 2 + margin_m

    px0, py0 = _world_to_pixel(gt, cx - half_w, cy + half_h)
    px1, py1 = _world_to_pixel(gt, cx + half_w, cy - half_h)
    px0, px1 = max(0, px0), min(dom.shape[1], px1)
    py0, py1 = max(0, py0), min(dom.shape[0], py1)
    return dom[py0:py1, px0:px1].copy(), px0, py0, px1, py1


def _draw_bbox_on_crop(
    crop: np.ndarray,
    bbox_world: list[float],
    gt: tuple,
    px0: int,
    py0: int,
    color=(0, 255, 0),
    label: str = "",
) -> np.ndarray:
    _require_cv2()
    vis = crop.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    x0p, y0p = _world_to_pixel(gt, bbox_world[0], bbox_world[3])
    x1p, y1p = _world_to_pixel(gt, bbox_world[2], bbox_world[1])
    x0p -= px0
    y0p -= py0
    x1p -= px0
    y1p -= py0
    cv2.rectangle(vis, (x0p, y0p), (x1p, y1p), color, 2)
    if label:
        cv2.putText(
            vis,
            label,
            (max(0, x0p + 2), max(15, y0p - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return vis


def _select_stones_for_label(stones: list[dict], n: int, seed: int, strategy: str) -> list[str]:
    ids = [s["stone_id"] for s in stones if s.get("source_detection_count", 0) > 0]
    if not ids or n <= 0:
        return []

    rng = random.Random(seed)
    if n >= len(ids):
        return ids

    if strategy == "stratified":
        def _bucket(s: dict) -> str:
            c = int(s.get("source_detection_count", 0))
            if c == 1:
                return "1"
            if c == 2:
                return "2"
            if c <= 4:
                return "3-4"
            return ">=5"

        buckets: dict[str, list[str]] = defaultdict(list)
        for s in stones:
            sid = s.get("stone_id")
            if not sid or int(s.get("source_detection_count", 0)) <= 0:
                continue
            buckets[_bucket(s)].append(sid)

        for v in buckets.values():
            rng.shuffle(v)

        nonempty = [b for b in buckets.values() if b]
        if not nonempty:
            return rng.sample(ids, n)

        quota = max(1, n // len(nonempty))
        selected: list[str] = []
        for v in nonempty:
            selected.extend(v[:quota])

        remaining_pool = [sid for sid in ids if sid not in set(selected)]
        rng.shuffle(remaining_pool)
        selected.extend(remaining_pool[: max(0, n - len(selected))])
        return selected[:n]

    remaining = ids.copy()
    rng.shuffle(remaining)
    return remaining[:n]


def _normalize_manual_file_data(raw: dict) -> tuple[dict[str, str], list[dict], dict]:
    """Returns (stone_labels, empty_tiles, meta)."""
    if not isinstance(raw, dict):
        return {}, [], {}

    meta = raw.get(META_KEY, {}) if isinstance(raw.get(META_KEY, {}), dict) else {}
    empty_tiles = raw.get(EMPTY_KEY, [])
    if not isinstance(empty_tiles, list):
        empty_tiles = []

    stone_labels: dict[str, str] = {}
    for k, v in raw.items():
        if k == EMPTY_KEY or k == META_KEY:
            continue
        if k.startswith(STONE_PREFIX):
            stone_labels[k[len(STONE_PREFIX):]] = str(v)
        elif k.startswith("stone_"):
            stone_labels[k] = str(v)

    return stone_labels, empty_tiles, meta


def _to_manual_payload(stone_labels: dict[str, str], empty_tiles: list[dict], meta: dict) -> dict:
    payload = {META_KEY: meta}
    payload.update({f"{STONE_PREFIX}{k}": v for k, v in sorted(stone_labels.items())})
    payload[EMPTY_KEY] = list(empty_tiles)
    return payload


def _persist_manual_labels(path: Path, stone_labels: dict[str, str], empty_tiles: list[dict], source: str, method: str) -> dict:
    payload = _to_manual_payload(
        stone_labels,
        empty_tiles,
        {
            "source": source,
            "method": method,
            "labeled_stones": len(stone_labels),
            "empty_tiles": len(empty_tiles),
        },
    )
    _write_json(path, payload)
    if _pair_is_legacy_default(source, method):
        # Keep old project-level path for backward compatibility only for the default pair.
        _write_json(_legacy_outputs()["manual_labels"], payload)
    return payload


def _dom_labeler(source: str, method: str, cfg: dict) -> dict:
    """Interactive manual labeling on DOM crops, plus empty-tile checks."""
    _require_cv2()
    _require_import('rasterio', 'rasterio')
    print(f"[Manual] source={source}, method={method}")

    paths = _pair_outputs(_pair_output_dir(source, method))
    label_path = paths["manual_json"]

    gt = CURRENT_SCENE.load_gt()
    stones, detections = _load_fusion(source, method)[:2]
    print(f"  fused stones: {len(stones)} | detections: {len(detections)}")

    import rasterio
    with rasterio.open(str(DOM_PATH)) as src:
        dom = src.read()
    if dom.shape[0] == 3:
        dom = np.transpose(dom, (1, 2, 0))
    elif dom.shape[0] == 1:
        dom = dom[0]
    dom = dom.astype(np.uint8)

    sample_cfg = cfg.get("sampling", {})
    random_seed = int(cfg.get("random", {}).get("seed", sample_cfg.get("random_seed", 0) if isinstance(sample_cfg, dict) else 0))
    n_stones = int(sample_cfg.get("n_stones", 60))
    n_empty = int(sample_cfg.get("n_empty_tiles", 30))
    strategy = str(sample_cfg.get("strategy", "random")).lower()
    z_thresh = float(cfg.get("z_range", {}).get("threshold_m", 0.5))

    existing = _read_json(label_path, {})
    if not existing:
        legacy_raw = _read_json(_legacy_outputs()["manual_labels"], {})
        legacy_meta = legacy_raw.get(META_KEY, {}) if isinstance(legacy_raw, dict) else {}
        if not legacy_meta or (legacy_meta.get("source") == source and legacy_meta.get("method") == method):
            existing = legacy_raw

    stone_label_map, empty_done, _ = _normalize_manual_file_data(existing)
    print(f"  existing manual labels: {len(stone_label_map)} stones, {len(empty_done)} empty tiles")

    sampled = _select_stones_for_label(stones, n_stones, random_seed, strategy)
    to_label = [sid for sid in sampled if sid not in stone_label_map]

    if to_label:
        print(f"[Manual] Need to label {len(to_label)} stones (strategy={strategy})")
        print("[Manual] Key: 1=TP (positive), 2=FP (negative), 3=UNSURE, ESC=save&exit")
        cv2.namedWindow("Stone Labeler", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Stone Labeler", 1000, 800)

        for sid in to_label:
            s = next((x for x in stones if x["stone_id"] == sid), None)
            if s is None:
                continue
            if s.get("source_detection_count", 0) <= 0:
                continue
            bbox = s.get("bbox_world", [])
            if not bbox:
                continue

            crop, px0, py0, _, _ = _crop_dom_around_bbox(
                dom,
                gt,
                bbox,
                float(cfg.get("dom_crop_margin_m", 2.0)),
            )
            if crop.size == 0:
                continue

            vis = _draw_bbox_on_crop(
                crop,
                bbox,
                gt,
                px0,
                py0,
                color=(0, 255, 0),
                label=f"{sid} | dets={s.get('source_detection_count', 0)} | score={s.get('score_mean', 0):.3f}",
            )

            info = np.zeros((60, vis.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                info,
                f"{sid}  |  Detection count: {s.get('source_detection_count', 0)}  |  Auto threshold: {z_thresh}m",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            display = np.vstack([vis, info])

            while True:
                cv2.imshow("Stone Labeler", display)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("1"):
                    stone_label_map[sid] = "TP"
                    print(f"  {sid}: TP")
                    break
                if key == ord("2"):
                    stone_label_map[sid] = "FP"
                    print(f"  {sid}: FP")
                    break
                if key == ord("3"):
                    stone_label_map[sid] = "UNSURE"
                    print(f"  {sid}: UNSURE")
                    break
                if key == 27:  # ESC
                    _persist_manual_labels(path=label_path, stone_labels=stone_label_map, empty_tiles=empty_done, source=source, method=method)
                    cv2.destroyAllWindows()
                    return _to_manual_payload(stone_label_map, empty_done, {"source": source, "method": method})

        cv2.destroyAllWindows()

    # Empty tile sampling
    empty_cfg = cfg.get("empty_sampling", cfg.get("empty_tiles", {})) if isinstance(cfg.get("empty_sampling", {}), dict) else {}
    tile_sz = int(empty_cfg.get("tile_size", 1024))
    tile_stride = int(empty_cfg.get("tile_stride", 512))

    if len(empty_done) < n_empty:
        print(f"[Manual] Empty tile check: {len(empty_done)}/{n_empty}")
        print("[Manual] Key: 1=Has stone (possible FN), 2=None, ESC=save&exit")

        h, w = dom.shape[:2]
        candidates: list[tuple[int, int]] = []
        for y in range(0, max(0, h - tile_sz), tile_stride):
            for x in range(0, max(0, w - tile_sz), tile_stride):
                candidates.append((x, y))

        rng = random.Random(random_seed + 7)
        rng.shuffle(candidates)

        cv2.namedWindow("Empty Tile Check", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Empty Tile Check", 1000, 800)

        for x, y in candidates:
            if len(empty_done) >= n_empty:
                break
            tile = dom[y:y + tile_sz, x:x + tile_sz].copy()
            if tile.size == 0:
                continue

            tile_bgr = tile if tile.ndim == 3 else cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            info = np.zeros((40, tile_sz, 3), dtype=np.uint8)
            cv2.putText(
                info,
                f"Empty tile ({x},{y})  1=Has stone (FN)  2=None  ESC=save&exit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            display = np.vstack([tile_bgr, info])

            while True:
                cv2.imshow("Empty Tile Check", display)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("1"):
                    empty_done.append({"tile": [x, y], "has_stone": True})
                    print(f"  Empty tile ({x},{y}) -> has stone")
                    break
                if key == ord("2"):
                    empty_done.append({"tile": [x, y], "has_stone": False})
                    print(f"  Empty tile ({x},{y}) -> no stone")
                    break
                if key == 27:
                    _persist_manual_labels(path=label_path, stone_labels=stone_label_map, empty_tiles=empty_done, source=source, method=method)
                    cv2.destroyAllWindows()
                    return _to_manual_payload(stone_label_map, empty_done, {"source": source, "method": method})

        cv2.destroyAllWindows()

    _persist_manual_labels(
        path=label_path,
        stone_labels=stone_label_map,
        empty_tiles=empty_done,
        source=source,
        method=method,
    )
    print(f"[Manual] Saved labels -> {label_path}")
    return _to_manual_payload(stone_label_map, empty_done, {"source": source, "method": method})


# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T
# ��������ӻ�
# �T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T�T

def _build_label_sweep(stone_labels: list[dict], manual_map: dict[str, str], steps: int = 40) -> list[dict]:
    """Compute precision/recall/F1 for different auto Z thresholds using manual labels."""
    zs: list[float] = []
    by_sid: dict[str, float] = {}
    for rec in stone_labels:
        sid = rec.get("stone_id")
        if sid not in manual_map:
            continue
        if manual_map[sid] == "UNSURE":
            continue
        try:
            z = float(rec.get("z_range", 0.0))
        except Exception:
            continue
        zs.append(z)
        by_sid[sid] = z

    if not zs:
        return []

    z_min, z_max = min(zs), max(zs)
    thresholds = sorted(set((np.linspace(z_min, z_max, num=max(3, steps)).tolist())))

    out = []
    for thr in thresholds:
        tp = fp = fn = 0
        for sid, mlabel in manual_map.items():
            if mlabel == "UNSURE":
                continue
            z = by_sid.get(sid)
            if z is None:
                continue
            pred_tp = z >= thr
            gt_tp = mlabel == "TP"
            if pred_tp and gt_tp:
                tp += 1
            elif pred_tp and not gt_tp:
                fp += 1
            elif (not pred_tp) and gt_tp:
                fn += 1

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        out.append({
            "threshold": round(float(thr), 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "labeled_items": tp + fp + fn,
        })

    if out:
        best = max(out, key=lambda r: (r["f1"], r["precision"]))
        for r in out:
            r["is_best_f1"] = r is best
    return out


def _make_visualizations(auto: dict, sweep: list[dict], pair_dir: Path, no_plots: bool = False) -> dict[str, str]:
    out = {
        "z_hist_png": str((pair_dir / "plots" / "z_range_distribution.png")),
        "sweep_png": str((pair_dir / "plots" / "manual_threshold_sweep.png")),
        "sweep_csv": str((pair_dir / "plots" / "manual_threshold_sweep.csv")),
    }
    if no_plots:
        return out

    _require_import('matplotlib', 'matplotlib')
    import matplotlib.pyplot as plt

    plot_dir = pair_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    evaluated = [r for r in auto.get("stone_labels", []) if r.get("status") == "evaluated"]
    if evaluated:
        zs = [float(r.get("z_range", 0.0) or 0.0) for r in evaluated]
        labels = [str(r.get("auto_label", "").upper()) for r in evaluated]
        plt.figure(figsize=(9, 4.8))
        plt.hist([z for z, l in zip(zs, labels) if l == "TP"], bins=30, alpha=0.75, label="TP")
        plt.hist([z for z, l in zip(zs, labels) if l == "FP"], bins=30, alpha=0.75, label="FP")
        plt.axvline(x=float(auto.get("z_threshold_m", 0.0)), color="k", linestyle="--", label="threshold")
        plt.title("Z-range distribution by auto label")
        plt.xlabel("z_range (m)")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out["z_hist_png"])
        plt.close()

    if sweep:
        with open(out["sweep_csv"], "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "tp",
                    "fp",
                    "fn",
                    "labeled_items",
                    "is_best_f1",
                ],
            )
            writer.writeheader()
            for row in sweep:
                writer.writerow(row)

        plt.figure(figsize=(9, 4.8))
        x = [r["threshold"] for r in sweep]
        plt.plot(x, [r["precision"] for r in sweep], label="precision")
        plt.plot(x, [r["recall"] for r in sweep], label="recall")
        plt.plot(x, [r["f1"] for r in sweep], label="f1")
        plt.xlabel("Z threshold (m)")
        plt.ylabel("Score")
        plt.title("Manual-labeled stones threshold sweep")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out["sweep_png"])
        plt.close()

    return out


def _manual_cross_stats(auto: dict, manual_map: dict[str, str]) -> tuple[dict[str, int], dict[str, float]]:
    counts = {
        "auto_tp_manual_tp": 0,
        "auto_tp_manual_fp": 0,
        "auto_fp_manual_tp": 0,
        "auto_fp_manual_fp": 0,
        "manual_only_tp": 0,
        "manual_only_fp": 0,
        "paired": 0,
    }

    auto_map = {r.get("stone_id"): r.get("auto_label") for r in auto.get("stone_labels", []) if r.get("status") == "evaluated"}
    for sid, lab in manual_map.items():
        if lab == "UNSURE":
            continue
        al = auto_map.get(sid)
        if al is None:
            continue
        counts["paired"] += 1
        if lab == "TP":
            counts["manual_only_tp"] += 1
        else:
            counts["manual_only_fp"] += 1

        if al == "TP" and lab == "TP":
            counts["auto_tp_manual_tp"] += 1
        elif al == "TP" and lab == "FP":
            counts["auto_tp_manual_fp"] += 1
        elif al == "FP" and lab == "TP":
            counts["auto_fp_manual_tp"] += 1
        elif al == "FP" and lab == "FP":
            counts["auto_fp_manual_fp"] += 1

    tp_on_manual = counts["auto_tp_manual_tp"]
    fp_on_manual = counts["auto_tp_manual_fp"]
    fn_on_manual = counts["auto_fp_manual_tp"]
    precision = _safe_div(tp_on_manual, tp_on_manual + fp_on_manual)
    recall = _safe_div(tp_on_manual, tp_on_manual + fn_on_manual)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return counts, {
        "paired_stones": counts["paired"],
        "auto_manual_confusion": counts,
        "auto_precision_on_labeled": round(precision, 4),
        "auto_recall_on_labeled": round(recall, 4),
        "auto_f1_on_labeled": round(f1, 4),
    }


def _to_stone_level_csv_rows(auto: dict, manual_map: dict[str, str]) -> list[dict]:
    out = []
    for rec in auto.get("stone_labels", []):
        sid = rec.get("stone_id", "")
        out.append(
            {
                "stone_id": sid,
                "status": rec.get("status", ""),
                "skip_reason": rec.get("skip_reason", ""),
                "source_detection_count": rec.get("source_detection_count", 0),
                "score_mean": rec.get("score_mean", 0.0),
                "score_max": rec.get("score_max", 0.0),
                "bbox_area_m2": rec.get("bbox_area_m2", 0.0),
                "n_detection_polygons": rec.get("n_detection_polygons", 0),
                "n_points_bbox": rec.get("n_points_bbox", 0),
                "n_points_in_poly": rec.get("n_points_in_poly", 0),
                "z_min": rec.get("z_min", ""),
                "z_max": rec.get("z_max", ""),
                "z_range": rec.get("z_range", 0.0),
                "auto_label": rec.get("auto_label", "SKIPPED"),
                "manual_label": manual_map.get(sid, "UNLABELED"),
                "is_auto_evaluated": 1 if rec.get("status") == "evaluated" else 0,
            }
        )
    return out


def _write_stone_level_csv(auto: dict, manual_map: dict[str, str], out_path: Path) -> str:
    rows = _to_stone_level_csv_rows(auto, manual_map)
    fieldnames = list(rows[0].keys()) if rows else [
        "stone_id",
        "status",
        "skip_reason",
        "source_detection_count",
        "score_mean",
        "score_max",
        "bbox_area_m2",
        "n_detection_polygons",
        "n_points_bbox",
        "n_points_in_poly",
        "z_min",
        "z_max",
        "z_range",
        "auto_label",
        "manual_label",
        "is_auto_evaluated",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return str(out_path)


def _report_to_markdown(report: dict, out_path: Path) -> None:
    lines = [
        "# Fusion Evaluation Report",
        "",
        f"- source: `{report['pair']['source']}`",
        f"- method: `{report['pair']['method']}`",
        f"- generated_at: {report['generated_at']}",
        "",
    ]

    lines.append("## Automatic Evaluation")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total stones | {report['auto']['total_stones']} |")
    lines.append(f"| Evaluated stones | {report['auto']['total_evaluated']} |")
    lines.append(f"| Auto TP | {report['auto']['auto_tp']} |")
    lines.append(f"| Auto FP | {report['auto']['auto_fp']} |")
    lines.append(f"| Skipped (not evaluable) | {report['auto']['skipped']} |")
    lines.append(f"| Auto precision | {report['auto']['auto_precision']:.4f} |")
    lines.append(
        f"| Auto precision 95% CI | {report['auto'].get('auto_precision_ci', [0, 0])[0]:.4f} ~ "
        f"{report['auto'].get('auto_precision_ci', [0, 0])[1]:.4f} |"
    )
    lines.append(f"| Z threshold | {report['auto']['z_threshold_m']} m |")
    lines.append(f"| Min points threshold | {report['auto']['min_points_per_stone']} |")

    lines.append("")
    if report['auto'].get("skip_detail"):
        lines.append("### Auto skip breakdown")
        lines.append("| Reason | Count |")
        lines.append("|---|---:|")
        for k in [
            "no_detections",
            "invalid_bbox",
            "invalid_polygons",
            "no_polygon",
            "few_points",
            "few_points_in_polygon",
        ]:
            v = report['auto']['skip_detail'].get(k, 0)
            if v:
                lines.append(f"| {k} | {v} |")

    lines.append("")
    lines.append("## Manual Annotation")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Labeled stones | {report['manual']['stones_labeled']} |")
    lines.append(f"| TP | {report['manual']['manual_tp']} |")
    lines.append(f"| FP | {report['manual']['manual_fp']} |")
    lines.append(f"| UNSURE | {report['manual']['manual_unsure']} |")
    lines.append(f"| Manual precision | {report['manual']['manual_precision']:.4f} |")
    lines.append(
        f"| Manual precision 95% CI | {report['manual'].get('manual_precision_ci', [0, 0])[0]:.4f} "
        f"~ {report['manual'].get('manual_precision_ci', [0, 0])[1]:.4f} |"
    )
    lines.append(f"| Empty tiles checked | {report['manual']['empty_tiles_checked']} |")
    lines.append(f"| Empty tiles with stones | {report['manual']['empty_tiles_with_stones']} |")
    lines.append(
        f"| Empty tile hit-rate 95% CI | {report['manual'].get('fn_rate_ci', [0, 0])[0]:.4f} "
        f"~ {report['manual'].get('fn_rate_ci', [0, 0])[1]:.4f} |"
    )
    lines.append(
        f"| Estimated FN (CI) | {report['manual']['fn_estimated']} "
        f"({report['manual']['fn_estimated_ci'][0]:.4f} ~ {report['manual']['fn_estimated_ci'][1]:.4f}) |"
    )
    lines.append(f"| Estimated recall | {report['manual']['manual_recall']:.4f} |")
    lines.append(
        f"| Recall 95% CI | {report['manual']['manual_recall_ci'][0]:.4f} "
        f"~ {report['manual']['manual_recall_ci'][1]:.4f} |"
    )
    lines.append(f"| F1 | {report['manual']['manual_f1']:.4f} |")

    lines.append("")
    lines.append("## Threshold sensitivity")
    best = report['auto'].get("best_threshold_by_manual_f1") or {}
    lines.append(f"- Best threshold by F1: {best.get('threshold', 'N/A')} (F1={best.get('f1', None) if best else 'N/A'})")
    if report['auto'].get("threshold_sweep"):
        lines.append(f"- Threshold-sweep points: {len(report['auto']['threshold_sweep'])}")

    lines.append("")
    lines.append("## Auto vs Manual alignment")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    c = report["cross"]["auto_manual_confusion"]
    lines.append(f"| Paired stones | {report['cross']['paired_stones']} |")
    lines.append(f"| auto TP / manual TP | {c['auto_tp_manual_tp']} |")
    lines.append(f"| auto TP / manual FP | {c['auto_tp_manual_fp']} |")
    lines.append(f"| auto FP / manual TP | {c['auto_fp_manual_tp']} |")
    lines.append(f"| auto FP / manual FP | {c['auto_fp_manual_fp']} |")
    lines.append(f"| Precision on paired | {report['cross']['auto_precision_on_labeled']:.4f} |")
    lines.append(f"| Recall on paired | {report['cross']['auto_recall_on_labeled']:.4f} |")
    lines.append(f"| F1 on paired | {report['cross']['auto_f1_on_labeled']:.4f} |")

    lines.append("")
    lines.append("## Paper metrics")
    lines.append(
        f"- Precision={report['summary']['precision']:.4f}, Recall={report['summary']['recall']:.4f}, "
        f"F1={report['summary']['f1_score']:.4f}"
    )
    lines.append(f"- Artifact: stone-level results => `{report['artifacts'].get('stone_csv', '')}`")

    if report.get("artifacts"):
        lines.append("")
        lines.append("## Artifacts")
        for k, v in report["artifacts"].items():
            lines.append(f"- {k}: `{v}`")

    out_path.write_text("\n".join(lines), encoding="utf-8")

def _generate_report(source: str, method: str, cfg: dict, no_plots: bool = False) -> dict:
    paths = _pair_outputs(_pair_output_dir(source, method))
    auto = _read_json(paths["auto_json"], {})
    if not auto:
        print(f"[Report] auto result missing for {source}/{method}, please run --auto first")
        return {}

    manual_raw = _read_json(paths["manual_json"], {})
    if not manual_raw and _pair_is_legacy_default(source, method):
        manual_raw = _read_json(_legacy_outputs()["manual_labels"], {})
    stone_label_map, empty_tiles, _ = _normalize_manual_file_data(manual_raw)

    tp_labeled = sum(1 for v in stone_label_map.values() if v == "TP")
    fp_labeled = sum(1 for v in stone_label_map.values() if v == "FP")
    unsure = sum(1 for v in stone_label_map.values() if v == "UNSURE")
    total_stones_labeled = tp_labeled + fp_labeled + unsure

    manual_precision = _safe_div(tp_labeled, tp_labeled + fp_labeled)
    manual_precision_ci = _wilson_ci(tp_labeled, tp_labeled + fp_labeled)

    fn_found = sum(1 for t in empty_tiles if bool(t.get("has_stone", False)))
    fn_factor = float(cfg.get("recall_estimation", {}).get("stones_per_empty_fn_tile", 2))
    fn_estimated = fn_found * fn_factor
    empty_n = len(empty_tiles)
    # Recall is estimated via empty-tile sampling. We provide interval by propagating
    # uncertainty of fn-rate measured by binomial Wilson interval.
    fn_rate_ci = _wilson_ci(fn_found, empty_n)
    fn_estimated_ci = [fn_rate_ci[0] * fn_factor * empty_n, fn_rate_ci[1] * fn_factor * empty_n]
    estimated_total = tp_labeled + fn_estimated
    manual_recall = _safe_div(tp_labeled, estimated_total)
    recall_ci_low = _safe_div(tp_labeled, tp_labeled + fn_estimated_ci[1])
    recall_ci_high = _safe_div(tp_labeled, tp_labeled + fn_estimated_ci[0])
    manual_recall_ci = [round(recall_ci_low, 4), round(recall_ci_high, 4)]
    manual_f1 = _safe_div(2 * manual_precision * manual_recall, manual_precision + manual_recall)

    _, cross = _manual_cross_stats(auto, stone_label_map)
    label_sweep = _build_label_sweep(auto.get("stone_labels", []), stone_label_map)
    best_sweep = next((r for r in label_sweep if r.get("is_best_f1")), None)

    pair_dir = _pair_output_dir(source, method)
    artifacts = _make_visualizations(auto, label_sweep, pair_dir, no_plots=no_plots)
    artifacts["stone_csv"] = _write_stone_level_csv(auto, stone_label_map, paths["stone_csv"])

    import datetime
    # min/max z-range only on evaluated stones
    eval_z = [float(r.get("z_range", 0.0)) for r in auto.get("stone_labels", []) if r.get("status") == "evaluated"]

    report = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pair": {
            "source": source,
            "method": method,
        },
        "auto": {
            "total_stones": auto.get("total_stones", 0),
            "total_evaluated": auto.get("total_evaluated", 0),
            "skipped": auto.get("skipped", 0),
            "skip_detail": auto.get("skip_detail", {}),
            "auto_tp": auto.get("tp", 0),
            "auto_fp": auto.get("fp", 0),
            "auto_precision": auto.get("auto_precision", 0),
            "auto_precision_ci": auto.get("auto_precision_ci", [0.0, 0.0]),
            "z_threshold_m": auto.get("z_threshold_m", 0.0),
            "min_points_per_stone": auto.get("min_points_per_stone", 0),
            "z_range_stats": {
                "count": len(eval_z),
                "z_range_min": round(min(eval_z), 4) if eval_z else 0.0,
                "z_range_max": round(max(eval_z), 4) if eval_z else 0.0,
            },
            "threshold_sweep": label_sweep,
            "best_threshold_by_manual_f1": best_sweep,
        },
        "manual": {
            "stones_labeled": total_stones_labeled,
            "manual_tp": tp_labeled,
            "manual_fp": fp_labeled,
            "manual_unsure": unsure,
            "manual_precision": round(manual_precision, 4),
            "manual_precision_ci": [round(manual_precision_ci[0], 4), round(manual_precision_ci[1], 4)],
            "empty_tiles_checked": len(empty_tiles),
            "empty_tiles_with_stones": fn_found,
            "fn_rate_ci": [round(fn_rate_ci[0], 4), round(fn_rate_ci[1], 4)],
            "fn_estimated": round(fn_estimated, 4),
            "fn_estimated_ci": [round(fn_estimated_ci[0], 4), round(fn_estimated_ci[1], 4)],
            "estimated_total_stones": round(estimated_total, 4),
            "manual_recall": round(manual_recall, 4),
            "manual_recall_ci": manual_recall_ci,
            "manual_f1": round(manual_f1, 4),
            "note": "Manual precision-first; recall from sampled empty tiles.",
        },

        "cross": cross,
        "summary": {
            "precision": round(manual_precision, 4),
            "recall": round(manual_recall, 4),
            "f1_score": round(manual_f1, 4),
        },
        "artifacts": artifacts,
    }

    _write_json(paths["report_json"], report)
    if _pair_is_legacy_default(source, method):
        _write_json(_legacy_outputs()["evaluation_report"], report)

    with open(paths["report_csv"], "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "value"])
        writer.writerow(["source", source])
        writer.writerow(["method", method])
        writer.writerow(["auto_total_stones", report["auto"]["total_stones"]])
        writer.writerow(["auto_evaluated", report["auto"]["total_evaluated"]])
        writer.writerow(["auto_skipped", report["auto"]["skipped"]])
        writer.writerow(["auto_precision", report["auto"]["auto_precision"]])
        writer.writerow(["auto_precision_ci_low", report["auto"].get("auto_precision_ci", [0, 0])[0]])
        writer.writerow(["auto_precision_ci_high", report["auto"].get("auto_precision_ci", [0, 0])[1]])
        writer.writerow(["manual_precision", report["manual"]["manual_precision"]])
        writer.writerow(["manual_precision_ci_low", report["manual"].get("manual_precision_ci", [0, 0])[0]])
        writer.writerow(["manual_precision_ci_high", report["manual"].get("manual_precision_ci", [0, 0])[1]])
        writer.writerow(["manual_recall", report["manual"]["manual_recall"]])
        writer.writerow(["manual_recall_ci_low", report["manual"].get("manual_recall_ci", [0, 0])[0]])
        writer.writerow(["manual_recall_ci_high", report["manual"].get("manual_recall_ci", [0, 0])[1]])
        writer.writerow(["manual_f1", report["manual"]["manual_f1"]])
        writer.writerow(["manual_tp", tp_labeled])
        writer.writerow(["manual_fp", fp_labeled])
        writer.writerow(["manual_unsure", unsure])
        writer.writerow(["empty_tiles_checked", len(empty_tiles)])
        writer.writerow(["empty_tiles_with_stones", fn_found])
        writer.writerow(["fn_estimated", fn_estimated])
        writer.writerow(["fn_estimated_ci_low", report["manual"]["fn_estimated_ci"][0]])
        writer.writerow(["fn_estimated_ci_high", report["manual"]["fn_estimated_ci"][1]])
        writer.writerow(["cross_auto_f1", cross["auto_f1_on_labeled"]])
        for k, v in (auto.get("skip_detail", {}) or {}).items():
            writer.writerow([f"skip_{k}", v])

    _report_to_markdown(report, paths["report_md"])
    print(f"[Report] pair={source}/{method}: P={manual_precision:.3f}, R={manual_recall:.3f}, F1={manual_f1:.3f}")
    return report


def _to_comparison_row(report: dict) -> dict:
    return {
        "source": report.get("pair", {}).get("source", ""),
        "method": report.get("pair", {}).get("method", ""),
        "auto_precision": report.get("auto", {}).get("auto_precision", 0),
        "auto_precision_ci_low": report.get("auto", {}).get("auto_precision_ci", [0, 0])[0],
        "auto_precision_ci_high": report.get("auto", {}).get("auto_precision_ci", [0, 0])[1],
        "manual_precision": report.get("manual", {}).get("manual_precision", 0),
        "manual_precision_ci_low": report.get("manual", {}).get("manual_precision_ci", [0, 0])[0],
        "manual_precision_ci_high": report.get("manual", {}).get("manual_precision_ci", [0, 0])[1],
        "manual_recall": report.get("manual", {}).get("manual_recall", 0),
        "manual_recall_ci_low": report.get("manual", {}).get("manual_recall_ci", [0, 0])[0],
        "manual_recall_ci_high": report.get("manual", {}).get("manual_recall_ci", [0, 0])[1],
        "f1": report.get("summary", {}).get("f1_score", 0),
        "auto_total_evaluated": report.get("auto", {}).get("total_evaluated", 0),
        "manual_stones_labeled": report.get("manual", {}).get("stones_labeled", 0),
        "empty_tiles_checked": report.get("manual", {}).get("empty_tiles_checked", 0),
        "best_threshold": report.get("auto", {}).get("best_threshold_by_manual_f1", {}).get("threshold"),
        "best_f1": report.get("auto", {}).get("best_threshold_by_manual_f1", {}).get("f1"),
        "cross_auto_f1": report.get("cross", {}).get("auto_f1_on_labeled", 0),
    }


def _generate_comparison(reports: list[dict], out_dir: Path, no_plots: bool = False) -> dict:
    if not reports:
        return {}

    rows = [_to_comparison_row(r) for r in reports if r]
    rows = [r for r in rows if r.get("source") or r.get("method")]
    if not rows:
        return {}

    rows.sort(key=lambda x: x.get("f1", 0), reverse=True)

    out_json = out_dir / "comparison_report.json"
    out_csv = out_dir / "comparison_report.csv"
    out_md = out_dir / "comparison_report.md"

    _write_json(out_json, {"rows": rows})

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    lines = [
        "# ???????+???",
        "",
        "| rank | source | method | auto_P(95CI) | manual_P(95CI) | manual_R(95CI) | manual_F1 | auto_eval_count | manual_labels | best_thres |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['source']} | {r['method']} "
            f"| {r['auto_precision']:.4f} [{r['auto_precision_ci_low']:.4f},{r['auto_precision_ci_high']:.4f}] "
            f"| {r['manual_precision']:.4f} [{r['manual_precision_ci_low']:.4f},{r['manual_precision_ci_high']:.4f}] "
            f"| {r['manual_recall']:.4f} [{r['manual_recall_ci_low']:.4f},{r['manual_recall_ci_high']:.4f}] "
            f"| {r['f1']:.4f} | {r['auto_total_evaluated']} | {r['manual_stones_labeled']} | {r['best_threshold']} |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    if not no_plots:
        _require_import('matplotlib', 'matplotlib')
        import matplotlib.pyplot as plt
        x = [f"{r['source']}\n{r['method']}" for r in rows]
        p = np.arange(len(x))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(p, [r["manual_precision"] for r in rows], marker="o", label="manual precision")
        ax.plot(p, [r["manual_recall"] for r in rows], marker="o", label="manual recall")
        ax.plot(p, [r["f1"] for r in rows], marker="o", label="manual F1")
        ax.set_xticks(p)
        ax.set_xticklabels(x, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("Method comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "comparison_plot.png")
        plt.close(fig)

    print(f"[Compare] comparison report -> {out_csv}")
    return {"rows": rows, "files": {"json": str(out_json), "csv": str(out_csv), "md": str(out_md)}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Refactored fusion evaluation")
    parser.add_argument("--source", choices=SOURCES + ["all"], default=SOURCES[0], help="input source: sahi / quadtree_dom / all")
    parser.add_argument("--method", choices=METHODS + ["all"], default=METHODS[1], help="fusion method")
    parser.add_argument("--auto", action="store_true", help="run automatic evaluation")
    parser.add_argument("--label", action="store_true", help="run interactive manual labeling")
    parser.add_argument("--report", action="store_true", help="generate report")
    parser.add_argument("--all", action="store_true", help="run auto + label + report")
    parser.add_argument("--compare", action="store_true", help="generate multi-pair comparison after selected reports")
    parser.add_argument("--no-plots", action="store_true", help="skip plot generation")
    parser.add_argument("--seed", type=int, default=None, help="override random seed")
    args = parser.parse_args()

    if not any([args.auto, args.label, args.report, args.all, args.compare]):
        parser.print_help()
        return

    cfg = _load_config()
    if args.seed is not None:
        cfg.setdefault("random", {})["seed"] = args.seed

    gt = CURRENT_SCENE.load_gt()
    pairs = _resolve_pairs(args.source, args.method)

    generated_reports: list[dict] = []
    do_auto = args.auto or args.all
    do_label = args.label or args.all
    do_report = args.report or args.all

    for source, method in pairs:
        stones, detections, _ = _load_fusion(source, method)
        _pair_output_dir(source, method).mkdir(parents=True, exist_ok=True)

        if do_auto:
            z_thresh = float(cfg.get("z_range", {}).get("threshold_m", 0.5))
            _auto_eval(stones, detections, gt, z_thresh, cfg, source, method)
        if do_label:
            _dom_labeler(source, method, cfg)
        if do_report:
            report = _generate_report(source, method, cfg, no_plots=args.no_plots)
            if report:
                generated_reports.append(report)

    if (len(pairs) > 1 and (args.report or args.all)) or args.compare:
        if not generated_reports and not args.report:
            for source, method in pairs:
                rp = _pair_outputs(_pair_output_dir(source, method))["report_json"]
                prev = _read_json(rp, {})
                if prev:
                    generated_reports.append(prev)
        comp = _generate_comparison(generated_reports, OUTPUTS_DIR, no_plots=args.no_plots)
        if comp:
            print(f"[Compare] pairs: {len(comp.get('rows', []))}")


if __name__ == "__main__":
    main()

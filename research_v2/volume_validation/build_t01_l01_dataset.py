"""Build the T01 + L01 V2 shape-aware dataset.

This script only constructs reusable dataset artifacts. It reuses the
current V2 feature implementation from ``enhance_shape_aware.py`` and
does not train a model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from enhance_shape_aware import (
    compute_mesh_volume,
    extract_descriptors,
    load_obj_simple,
    simulate_2_5d_surface,
)
from shape_features_v2 import FEATURE_NAMES, feature_schema, extract_features


DATASETS = ("T01", "L01")
GRID_RES_MM = 0.5
SCALE_FACTOR = 1.0
SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
SOURCE_ROOT = Path("data/experience_rock")
OUTPUT_DIR = Path("research_v2/volume_validation/datasets/t01_l01_v2")
CACHE_DIR = OUTPUT_DIR / "cache"
SUCCESS_STATUSES = {"valid", "success"}


def sample_paths(dataset_ids: tuple[str, ...] = DATASETS) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for dataset_id in dataset_ids:
        for obj_path in sorted((SOURCE_ROOT / dataset_id).glob("*.obj")):
            paths.append((dataset_id, obj_path))
    return paths


def cache_path(dataset_id: str, original_obj_id: str) -> Path:
    return CACHE_DIR / dataset_id / f"{original_obj_id}.json"


def is_finite_array(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(values)))


def quality_flags(row: dict, features: np.ndarray) -> list[str]:
    flags = []
    y_ratio = row["y_ratio"]
    if y_ratio <= 0 or y_ratio > 5:
        flags.append(f"y_ratio_outlier:{y_ratio:.6g}")
    for name, value in zip(FEATURE_NAMES, features):
        if abs(float(value)) > 100:
            flags.append(f"feature_outlier:{name}={float(value):.6g}")
    return flags


def write_cache(record: dict) -> None:
    path = cache_path(record["dataset_id"], record["original_obj_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def read_cache(dataset_id: str, original_obj_id: str) -> dict | None:
    path = cache_path(dataset_id, original_obj_id)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def process_one(dataset_id: str, obj_path: Path) -> dict:
    sample_id = obj_path.stem
    base = {
        "sample_id": sample_id,
        "dataset_id": dataset_id,
        "original_obj_id": sample_id,
        "obj_path": str(obj_path.as_posix()),
        "status": "error",
        "error": "",
        "grid_res_mm": GRID_RES_MM,
        "scale_factor": SCALE_FACTOR,
    }
    t0 = time.perf_counter()
    try:
        vertices, faces = load_obj_simple(obj_path)
        load_s = time.perf_counter() - t0
        if len(vertices) == 0 or len(faces) == 0:
            return {**base, "error": "empty_mesh", "load_time_s": round(load_s, 6)}

        t_volume = time.perf_counter()
        v_true_unscaled = compute_mesh_volume(vertices, faces)
        v_true = v_true_unscaled * (SCALE_FACTOR ** 3)
        volume_s = time.perf_counter() - t_volume
        if not math.isfinite(v_true) or v_true <= 0:
            return {
                **base,
                "error": "invalid_mesh_volume",
                "n_vertices": int(len(vertices)),
                "n_faces": int(len(faces)),
                "V_true_unscaled": float(v_true_unscaled),
                "load_time_s": round(load_s, 6),
                "volume_time_s": round(volume_s, 6),
                "total_time_s": round(time.perf_counter() - t0, 6),
            }

        t_surface = time.perf_counter()
        surface = simulate_2_5d_surface(vertices * SCALE_FACTOR, faces, GRID_RES_MM)
        surface_s = time.perf_counter() - t_surface
        if surface is None or surface["n_valid_cells"] == 0:
            return {
                **base,
                "error": "empty_2_5d_surface",
                "n_vertices": int(len(vertices)),
                "n_faces": int(len(faces)),
                "V_true": float(v_true),
                "V_true_unscaled": float(v_true_unscaled),
                "load_time_s": round(load_s, 6),
                "volume_time_s": round(volume_s, 6),
                "surface_time_s": round(surface_s, 6),
                "total_time_s": round(time.perf_counter() - t0, 6),
            }

        t_features = time.perf_counter()
        desc = extract_descriptors(surface)
        if desc is None:
            return {**base, "error": "descriptor_extraction_failed"}

        v_2_5d = float(desc["V_2_5d"])
        if not math.isfinite(v_2_5d) or v_2_5d <= 0:
            return {**base, "error": "invalid_v_2_5d"}

        features = extract_features(desc)
        if len(features) != len(FEATURE_NAMES):
            return {**base, "error": "feature_count_mismatch"}
        if not is_finite_array(features):
            return {**base, "error": "nan_or_inf_feature"}

        y_ratio = float(v_true / (v_2_5d + 1e-9))
        if not math.isfinite(y_ratio):
            return {**base, "error": "invalid_y_ratio"}

        row = {
            **base,
            "status": "success",
            "V_true": float(v_true),
            "V_true_unscaled": float(v_true_unscaled),
            "V_2_5D": v_2_5d,
            "y_ratio": y_ratio,
            "n_vertices": int(len(vertices)),
            "n_faces": int(len(faces)),
            "n_valid_cells": int(desc["n_valid_cells"]),
            "load_time_s": round(load_s, 6),
            "volume_time_s": round(volume_s, 6),
            "surface_time_s": round(surface_s, 6),
        }
        for name, value in zip(FEATURE_NAMES, features):
            row[name] = float(value)
        row["feature_time_s"] = round(time.perf_counter() - t_features, 6)
        row["total_time_s"] = round(time.perf_counter() - t0, 6)
        row["quality_flags"] = ";".join(quality_flags(row, features))
        return row
    except Exception as exc:
        return {
            **base,
            "error": f"exception:{type(exc).__name__}:{exc}",
            "total_time_s": round(time.perf_counter() - t0, 6),
        }


def build_cache(dataset_ids: tuple[str, ...]) -> None:
    paths = sample_paths(dataset_ids)
    total = len(paths)
    completed = 0
    for index, (dataset_id, obj_path) in enumerate(paths, start=1):
        original_obj_id = obj_path.stem
        cached = read_cache(dataset_id, original_obj_id)
        if cached is not None:
            if (
                float(cached.get("grid_res_mm", -1.0)) != GRID_RES_MM
                or float(cached.get("scale_factor", -1.0)) != SCALE_FACTOR
            ):
                raise RuntimeError(
                    f"Incompatible cache for {dataset_id}/{original_obj_id}: "
                    "grid resolution or scale factor differs from this build."
                )
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"{dataset_id}: {completed} / {total}", flush=True)
            continue
        record = process_one(dataset_id, obj_path)
        write_cache(record)
        completed += 1
        if completed % 10 == 0 or completed == total:
            print(f"{dataset_id}: {completed} / {total}", flush=True)


def load_cached_records() -> tuple[list[dict], list[dict]]:
    records = []
    for dataset_id in DATASETS:
        for cache_file in sorted((CACHE_DIR / dataset_id).glob("*.json")):
            with open(cache_file, encoding="utf-8") as f:
                records.append(json.load(f))
    valid = [r for r in records if r.get("status") in SUCCESS_STATUSES]
    invalid = [r for r in records if r.get("status") not in SUCCESS_STATUSES]
    return valid, invalid


def split_samples(samples: list[dict]) -> dict:
    # Each cache record represents one original OBJ, so a shuffled object-level
    # partition is group-aware and cannot put an OBJ in more than one split.
    object_keys = [f"{row['dataset_id']}::{row['original_obj_id']}" for row in samples]
    if len(set(object_keys)) != len(object_keys):
        raise RuntimeError("Duplicate dataset_id/original_obj_id pairs cannot be split safely.")
    order = np.random.default_rng(SEED).permutation(len(samples))
    n_train = int(round(len(samples) * TRAIN_RATIO))
    n_validation = int(round(len(samples) * VALIDATION_RATIO))
    return {
        "train_idx": order[:n_train].tolist(),
        "validation_idx": order[n_train:n_train + n_validation].tolist(),
        "test_idx": order[n_train + n_validation:].tolist(),
    }


def check_leakage(samples: list[dict], splits: dict) -> dict:
    split_names = list(splits)
    sample_sets = {name: set(splits[name]) for name in split_names}
    overlaps = {}
    for i, left in enumerate(split_names):
        for right in split_names[i + 1:]:
            overlaps[f"{left}__{right}"] = sorted(sample_sets[left] & sample_sets[right])

    obj_to_splits = defaultdict(set)
    for split_name, indices in splits.items():
        for idx in indices:
            obj_to_splits[samples[idx]["original_obj_id"]].add(split_name)
    leaked_objects = {obj: sorted(names) for obj, names in obj_to_splits.items() if len(names) > 1}
    duplicates = {
        f"{dataset_id}::{obj}": count
        for (dataset_id, obj), count in Counter(
            (row["dataset_id"], row["original_obj_id"]) for row in samples
        ).items()
        if count > 1
    }
    return {
        "sample_index_overlaps": overlaps,
        "leaked_original_obj_ids": leaked_objects,
        "duplicate_original_obj_ids": duplicates,
        "has_leakage": any(overlaps.values()) or bool(leaked_objects),
    }


def save_final(elapsed_s: float) -> None:
    samples, invalid = load_cached_records()
    expected = len(sample_paths())
    if len(samples) + len(invalid) != expected:
        raise RuntimeError(
            f"Cache incomplete: cached={len(samples) + len(invalid)}, expected={expected}. "
            "Resume build before finalizing."
        )

    splits = split_samples(samples)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "dataset_id", "original_obj_id", "obj_path",
        *FEATURE_NAMES,
        "V_true", "V_2_5D", "y_ratio",
        "n_vertices", "n_faces", "n_valid_cells", "quality_flags",
    ]
    with open(OUTPUT_DIR / "samples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in samples:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    np.savez(
        OUTPUT_DIR / "dataset_arrays.npz",
        X=np.array([[row[name] for name in FEATURE_NAMES] for row in samples], dtype=np.float64),
        y=np.array([row["y_ratio"] for row in samples], dtype=np.float64),
        V_true=np.array([row["V_true"] for row in samples], dtype=np.float64),
        V_2_5D=np.array([row["V_2_5D"] for row in samples], dtype=np.float64),
        sample_id=np.array([row["sample_id"] for row in samples]),
        dataset_id=np.array([row["dataset_id"] for row in samples]),
        original_obj_id=np.array([row["original_obj_id"] for row in samples]),
    )

    with open(OUTPUT_DIR / "splits.json", "w", encoding="utf-8") as f:
        json.dump({
            "train": [samples[i]["sample_id"] for i in splits["train_idx"]],
            "validation": [samples[i]["sample_id"] for i in splits["validation_idx"]],
            "test": [samples[i]["sample_id"] for i in splits["test_idx"]],
            "members": {
                name: [
                    {
                        "sample_id": samples[i]["sample_id"],
                        "dataset_id": samples[i]["dataset_id"],
                        "original_obj_id": samples[i]["original_obj_id"],
                    }
                    for i in splits[f"{name}_idx"]
                ]
                for name in ("train", "validation", "test")
            },
            "random_seed": SEED,
            "split_ratio": {
                "train": TRAIN_RATIO,
                "validation": VALIDATION_RATIO,
                "test": TEST_RATIO,
            },
            "indices": splits,
        }, f, indent=2)

    with open(OUTPUT_DIR / "invalid_samples.json", "w", encoding="utf-8") as f:
        json.dump(invalid, f, indent=2)

    counts = {
        dataset_id: {
            "total": sum(1 for _, path in sample_paths() if path.parent.name == dataset_id),
            "success": sum(1 for row in samples if row["dataset_id"] == dataset_id),
            "error": sum(1 for row in invalid if row["dataset_id"] == dataset_id),
        }
        for dataset_id in DATASETS
    }
    metadata = {
        "dataset_name": f"t01_l01_v2_shape_aware_{GRID_RES_MM:g}mm",
        "constructed_at": datetime.now().isoformat(timespec="seconds"),
        "construction_time_s": round(elapsed_s, 3),
        "random_seed": SEED,
        "grid_res_mm": GRID_RES_MM,
        "resolution_mm": GRID_RES_MM,
        "scale_factor": SCALE_FACTOR,
        "scale_rule": "uniform XYZ scaling before 2.5D rasterization",
        "split_ratio": {"train": TRAIN_RATIO, "validation": VALIDATION_RATIO, "test": TEST_RATIO},
        "source_directories": {d: f"data/experience_rock/{d}" for d in DATASETS},
        "feature_names": FEATURE_NAMES,
        "feature_order": FEATURE_NAMES,
        "feature_formulas": feature_schema()["feature_formulas"],
        "target_definition": "y_ratio = V_true / V_2_5D",
        "volume_definition": {
            "V_true": "OBJ triangular mesh volume via the divergence theorem, multiplied by scale_factor^3",
            "V_2_5D": "top-down 2.5D height integration at grid_res_mm",
        },
        "counts": counts,
        "n_total": len(samples) + len(invalid),
        "n_success": len(samples),
        "n_error": len(invalid),
        "split_counts": {
            "train": len(splits["train_idx"]),
            "validation": len(splits["validation_idx"]),
            "test": len(splits["test_idx"]),
        },
        "leakage_check": check_leakage(samples, splits),
        "quality_flagged_samples": [
            {"sample_id": r["sample_id"], "dataset_id": r["dataset_id"], "quality_flags": r["quality_flags"]}
            for r in samples if r.get("quality_flags")
        ],
    }
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Final dataset saved to {OUTPUT_DIR.as_posix()}")


def benchmark() -> None:
    for dataset_id in DATASETS:
        obj_path = next((SOURCE_ROOT / dataset_id).glob("*.obj"))
        record = process_one(dataset_id, obj_path)
        print(json.dumps({
            "dataset_id": dataset_id,
            "sample_id": obj_path.stem,
            "status": record["status"],
            "error": record.get("error", ""),
            "n_faces": record.get("n_faces"),
            "surface_time_s": record.get("surface_time_s"),
            "total_time_s": record.get("total_time_s"),
        }, indent=2))


def main() -> None:
    global GRID_RES_MM, SCALE_FACTOR, OUTPUT_DIR, CACHE_DIR
    parser = argparse.ArgumentParser(description="Build resumable T01 + L01 V2 dataset")
    parser.add_argument("--benchmark", action="store_true", help="Run one T01 and one L01 sample only")
    parser.add_argument("--finalize-only", action="store_true", help="Merge an already complete cache")
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        help="Build one source dataset only; use L01 to preserve the completed T01 cache.",
    )
    parser.add_argument(
        "--grid-res-mm",
        type=float,
        default=GRID_RES_MM,
        help="2.5D raster resolution in millimetres.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Independent dataset directory containing cache and finalized artifacts.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=SCALE_FACTOR,
        help="Uniform XYZ scale applied before 2.5D rasterization.",
    )
    args = parser.parse_args()
    if args.grid_res_mm <= 0:
        raise ValueError("--grid-res-mm must be positive")
    if args.scale_factor <= 0:
        raise ValueError("--scale-factor must be positive")
    GRID_RES_MM = float(args.grid_res_mm)
    SCALE_FACTOR = float(args.scale_factor)
    OUTPUT_DIR = Path(args.output_dir)
    CACHE_DIR = OUTPUT_DIR / "cache"

    start = time.time()
    if args.benchmark:
        benchmark()
        return
    if not args.finalize_only:
        build_cache((args.dataset,) if args.dataset else DATASETS)
    save_final(time.time() - start)


if __name__ == "__main__":
    main()

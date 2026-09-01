"""Read-only training/production Shape-Aware V2 feature consistency check."""
from __future__ import annotations

import importlib.util
import json
import random
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CHECK_PATH = ROOT / "research_v2/volume_validation/feature_consistency_check.json"
REPORT_PATH = ROOT / "research_v2/volume_validation/feature_consistency_report.md"
CACHE_ROOT = ROOT / "research_v2/volume_validation/datasets/t01_l01_v2/cache"
MODEL_PATH = ROOT / "research_v2/volume_validation/output_v2_t01_l01/shape_aware_model_v2_t01_l01.txt"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_v2/volume_validation"))

from enhance_shape_aware import extract_descriptors, load_obj_simple, simulate_2_5d_surface
from shape_features_v2 import FEATURE_NAMES, feature_schema, extract_features, validate_model_feature_names


def load_production_adapter():
    """Load rockseg.volume without importing unrelated pipeline dependencies."""
    package = types.ModuleType("rockseg")
    package.__path__ = [str(ROOT / "rockseg")]
    sys.modules.setdefault("rockseg", package)
    spec = importlib.util.spec_from_file_location("rockseg.volume", ROOT / "rockseg/volume.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load production volume adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rockseg.volume"] = module
    spec.loader.exec_module(module)
    return module


def max_differences(reference: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    absolute = np.abs(actual - reference)
    relative = absolute / np.maximum(np.abs(reference), 1e-12)
    return float(np.max(absolute)), float(np.max(relative))


def model_feature_names() -> list[str]:
    for line in MODEL_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("feature_names="):
            return line.partition("=")[2].split()
    raise RuntimeError("Shape-Aware V2 model has no feature_names declaration")


def write_report(result: dict) -> None:
    lines = [
        "# Shape-Aware V2 Feature Consistency",
        "",
        f"Status: **{result['status']}**",
        "",
        "## Canonical Schema",
        "",
        "Feature order: " + ", ".join(FEATURE_NAMES),
        "",
        "- Training implementation: `enhance_shape_aware.extract_descriptors` and `extract_features`.",
        "- Production implementation: `rockseg.volume.compute_shape_descriptors` and `predict_shape_aware`.",
        "- Both delegate 12-feature construction to `shape_features_v2.py`.",
        "- `H_skew_norm` is the unnormalised `H_skew` value.",
        "- Units: geometry uses one consistent length unit per surface; all exported features are dimensionless.",
        "",
        "## Production Mapping",
        "",
        "| Training feature | Production source | Formula | Unit |",
        "| --- | --- | --- | --- |",
    ]
    sources = {
        "C": "valid height-map footprint", "AR": "valid height-map footprint",
        "solidity": "valid footprint convex hull", "compactness": "valid footprint perimeter",
        "eq_diam_ratio": "valid footprint area and extent", "H_mean_norm": "ground-referenced cell heights",
        "H_std_norm": "ground-referenced cell heights", "H_p25_norm": "ground-referenced cell heights",
        "H_p75_norm": "ground-referenced cell heights", "H_skew_norm": "ground-referenced cell heights",
        "fill_ratio": "2.5D volume, L, W, H", "ellipsoid_ratio": "2.5D volume, L, W, H",
    }
    for name in FEATURE_NAMES:
        lines.append(f"| {name} | {sources[name]} | {result['schema']['feature_formulas'][name]} | dimensionless |")
    lines.extend([
        "",
        "## Five Cached Samples",
        "",
        "| Dataset | Sample | Cache vs canonical max abs | Canonical vs production max abs | Canonical vs production max relative |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for sample in result["samples"]:
        lines.append(
            f"| {sample['dataset_id']} | {sample['sample_id']} | {sample['cache_vs_canonical']['max_abs_diff']:.3e} | "
            f"{sample['canonical_vs_production']['max_abs_diff']:.3e} | "
            f"{sample['canonical_vs_production']['max_rel_diff']:.3e} |"
        )
    lines.extend([
        "",
        "## Result",
        "",
        f"Maximum absolute difference: {result['summary']['max_absolute_difference']:.3e}",
        f"Maximum relative difference: {result['summary']['max_relative_difference']:.3e}",
        f"Tolerance: absolute difference <= {result['tolerance']['max_abs_diff']:.1e}",
        "",
        "PASS requires identical feature order, finite values, positive cached volumes, and the stated tolerance.",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cache_files = sorted(CACHE_ROOT.glob("*/*.json"))
    records = []
    for path in cache_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") in {"success", "valid"}:
            records.append((path, record))
    selected = random.Random(42).sample(records, 5)
    production = load_production_adapter()
    validate_model_feature_names(model_feature_names())

    samples = []
    max_abs = 0.0
    max_rel = 0.0
    passed = True
    for _, cached in selected:
        obj_path = ROOT / cached["obj_path"]
        vertices, faces = load_obj_simple(obj_path)
        surface = simulate_2_5d_surface(vertices, faces, grid_resolution=0.5)
        if surface is None:
            raise RuntimeError(f"Empty surface for cached sample {cached['sample_id']}")
        training_desc = extract_descriptors(surface)
        production_desc = production.compute_shape_descriptors(
            surface["height_map"], surface["footprint_mask"], surface["cell_size"]
        )
        cached_features = np.asarray([cached[name] for name in FEATURE_NAMES], dtype=np.float64)
        training_features = extract_features(training_desc)
        production_features = extract_features(production_desc)
        cache_abs, cache_rel = max_differences(cached_features, training_features)
        prod_abs, prod_rel = max_differences(training_features, production_features)
        finite = bool(np.all(np.isfinite(training_features)) and np.all(np.isfinite(production_features)))
        positive_volume = bool(cached["V_true"] > 0 and cached["V_2_5D"] > 0)
        passed = passed and finite and positive_volume and cache_abs <= 1e-10 and prod_abs <= 1e-10
        max_abs = max(max_abs, cache_abs, prod_abs)
        max_rel = max(max_rel, cache_rel, prod_rel)
        samples.append({
            "sample_id": cached["sample_id"],
            "dataset_id": cached["dataset_id"],
            "original_obj_id": cached["original_obj_id"],
            "V_true": cached["V_true"],
            "V_2_5D": cached["V_2_5D"],
            "all_features_finite": finite,
            "volumes_positive": positive_volume,
            "cache_vs_canonical": {"max_abs_diff": cache_abs, "max_rel_diff": cache_rel},
            "canonical_vs_production": {"max_abs_diff": prod_abs, "max_rel_diff": prod_rel},
        })

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASS" if passed else "FAIL",
        "selected_with_seed": 42,
        "sample_count": len(samples),
        "schema": feature_schema(),
        "model_feature_order": model_feature_names(),
        "production_grid_rule": "The adapter accepts its supplied grid resolution; planned mine-site resolution is 0.01 m.",
        "tolerance": {"max_abs_diff": 1e-10},
        "samples": samples,
        "summary": {
            "max_absolute_difference": max_abs,
            "max_relative_difference": max_rel,
            "feature_order_matches": model_feature_names() == FEATURE_NAMES,
        },
    }
    CHECK_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result)
    if not passed:
        raise SystemExit("Feature consistency check failed")


if __name__ == "__main__":
    main()

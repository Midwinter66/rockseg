"""Resumable Shape-Aware V2 inference for frozen DOM2 sample-manifest batches.

This runner intentionally processes an explicit manifest batch only. It never
changes sampling, models, feature definitions, or production modules.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

from rockseg.validation_3d import GroundDEM, PointCloudGridIndex, load_point_cloud
from rockseg.volume import compute_shape_descriptors, extract_height_map, load_shape_aware_model
from research_v2.volume_validation.shape_features_v2 import (
    FEATURE_NAMES,
    FeatureSchemaError,
    extract_features,
    validate_model_feature_names,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "research_v2/volume_validation/real_mine_sampling/real_mine_volume_sample_manifest.csv"
ACCEPTED_PATH = ROOT / "output/dom2_cascade_v2_3d_fixed/accepted_instances.json"
MODEL_PATH = ROOT / "research_v2/volume_validation/output_v2_scaled_10mm/shape_aware_model_v2_scaled_10mm.txt"
DOM_PATH = ROOT / "data/dom2/DOM.tif"
LAZ_PATHS = [ROOT / "data/pointcloud2/Data/BlockB.laz", ROOT / "data/pointcloud2/Data/BlockY.laz"]
MASKS_PATH = ROOT / "output/dom2_cascade_v2/rock_masks.npz"
RESULT_ROOT = ROOT / "research_v2/volume_validation/real_mine_full"
GRID_RESOLUTION_M = 0.01
SCALE_FACTOR = 82.737840
BATCH_SIZE = 500


RESULT_FIELDS = [
    "sample_id", "rock_id", "stratum", "equivalent_diameter_m", "scale_level",
    "association_status", "point_count", "footprint_m2", "height_m", "occupied_cells",
    "V_2_5D_m3", *FEATURE_NAMES, "y_pred", "V_pred_m3", "status", "failure_reason",
    "processing_time_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen DOM2 V2 volume batches")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch-index", type=int, help="One-based batch index")
    group.add_argument("--batch-range", nargs=2, type=int, metavar=("START", "END"), help="Inclusive batch range")
    args = parser.parse_args()
    if args.batch_range and (args.batch_range[0] < 1 or args.batch_range[1] < args.batch_range[0]):
        parser.error("--batch-range must be an increasing range starting at 1")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * p
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def numeric_summary(records: list[dict], field: str) -> dict:
    values = [float(record[field]) for record in records if record.get("status") == "PASS" and finite(record.get(field, math.nan))]
    if not values:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "p90": None, "max": None, "mean": None}
    return {
        "n": len(values), "min": min(values), "p25": quantile(values, 0.25),
        "median": quantile(values, 0.50), "p75": quantile(values, 0.75), "p90": quantile(values, 0.90),
        "max": max(values), "mean": sum(values) / len(values),
    }


def blank_result(row: dict, accepted: dict, started: float) -> dict:
    result = {field: None for field in RESULT_FIELDS}
    result.update({
        "sample_id": row["sample_id"], "rock_id": row["rock_id"], "stratum": row["stratum"],
        "equivalent_diameter_m": float(row["equivalent_diameter_m"]), "scale_level": row["scale_level"],
        "association_status": "accepted_existing", "point_count": accepted["validation_3d"]["point_count"],
        "status": "FAIL", "processing_time_s": time.perf_counter() - started,
    })
    return result


def preflight(batch_index: int) -> tuple[list[dict], dict, dict, Path, Path, Path]:
    if batch_index < 1:
        raise ValueError("batch-index must be >= 1")
    for path in [MANIFEST_PATH, ACCEPTED_PATH, MODEL_PATH, DOM_PATH, MASKS_PATH, *LAZ_PATHS]:
        if not path.exists():
            raise FileNotFoundError(f"Frozen input is missing: {path}")

    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    accepted_list = json.loads(ACCEPTED_PATH.read_text(encoding="utf-8"))
    accepted = {record["instance_id"]: record for record in accepted_list}
    if len(manifest) != 4000 or len({record["sample_id"] for record in manifest}) != 4000:
        raise RuntimeError("Frozen manifest is not exactly 4,000 unique samples")
    if len({record["rock_id"] for record in manifest}) != 4000:
        raise RuntimeError("Frozen manifest has duplicate rock IDs")
    if not {record["rock_id"] for record in manifest} <= set(accepted):
        raise RuntimeError("Manifest contains rocks outside accepted_instances.json")
    if [record["sample_id"] for record in manifest] != [f"dom2_volume_sample_{index:04d}" for index in range(1, 4001)]:
        raise RuntimeError("Manifest sample IDs are not the frozen sequential manifest")

    start = (batch_index - 1) * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(manifest))
    if start >= len(manifest):
        raise ValueError(f"batch-index {batch_index} has no manifest rows")
    batch = manifest[start:end]

    validate_model_feature_names(FEATURE_NAMES)
    model = load_shape_aware_model(MODEL_PATH)
    if not hasattr(model, "num_feature") or int(model.num_feature()) != len(FEATURE_NAMES):
        raise FeatureSchemaError("Frozen model dimension is not 12")
    if list(model.feature_name()) != FEATURE_NAMES:
        raise FeatureSchemaError("Frozen model feature order differs from canonical schema")

    checkpoint_dir = RESULT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"batch_{batch_index:03d}"
    checkpoint_path = checkpoint_dir / f"{stem}_checkpoint.jsonl"
    results_path = checkpoint_dir / f"{stem}_results.csv"
    qc_path = checkpoint_dir / f"{stem}_qc.json"
    preflight_path = checkpoint_dir / f"{stem}_preflight.json"
    if results_path.exists() and not checkpoint_path.exists():
        raise RuntimeError(f"Historical results file exists without checkpoint: {results_path}")

    payload = {
        "status": "PREFLIGHT_PASS", "created_utc": utc_now(), "batch_index": batch_index,
        "batch_range": {"manifest_positions_one_based": [start + 1, end], "sample_ids": [batch[0]["sample_id"], batch[-1]["sample_id"]]},
        "frozen_inputs": {
            "manifest": str(MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
            "accepted_instances": str(ACCEPTED_PATH.relative_to(ROOT)).replace("\\", "/"),
            "model": str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "dom": str(DOM_PATH.relative_to(ROOT)).replace("\\", "/"),
            "pointclouds": [str(path.relative_to(ROOT)).replace("\\", "/") for path in LAZ_PATHS],
            "masks": str(MASKS_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
        "checks": {
            "manifest_count": len(manifest), "manifest_sample_ids_unique": True, "manifest_rock_ids_unique": True,
            "all_manifest_rocks_accepted": True, "accepted_population": len(accepted),
            "canonical_feature_count": len(FEATURE_NAMES), "canonical_feature_order": FEATURE_NAMES,
            "model_feature_count": int(model.num_feature()), "model_feature_order_matches": True,
            "grid_resolution_m": GRID_RESOLUTION_M, "scale_factor_provenance": SCALE_FACTOR,
            "output_collision": False,
        },
    }
    preflight_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return batch, accepted, model, checkpoint_path, results_path, qc_path


def load_latest_checkpoint(checkpoint_path: Path) -> dict[str, dict]:
    latest = {}
    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    latest[record["rock_id"]] = record
    return latest


def write_results_csv(records: list[dict], results_path: Path) -> None:
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows([{field: record.get(field) for field in RESULT_FIELDS} for record in records])


def build_qc(batch_index: int, records: list[dict], elapsed_s: float) -> dict:
    passed = [record for record in records if record["status"] == "PASS"]
    failures = [record for record in records if record["status"] != "PASS"]
    finite_features = sum(
        all(finite(record.get(name, math.nan)) for name in FEATURE_NAMES)
        for record in passed
    )
    strata = {}
    for name in sorted({record["stratum"] for record in records}):
        group = [record for record in records if record["stratum"] == name]
        ok = [record for record in group if record["status"] == "PASS"]
        strata[name] = {
            "total": len(group), "success": len(ok), "fail": len(group) - len(ok),
            "success_rate": len(ok) / len(group) if group else 0.0,
            "diameter_m": numeric_summary(ok, "equivalent_diameter_m"),
            "V_2_5D_m3": numeric_summary(ok, "V_2_5D_m3"),
            "y_pred": numeric_summary(ok, "y_pred"), "V_pred_m3": numeric_summary(ok, "V_pred_m3"),
        }
    start_index = (batch_index - 1) * BATCH_SIZE + 1
    return {
        "status": "BATCH_COMPLETE", "batch_id": f"batch_{batch_index:03d}", "batch_index": batch_index,
        "start_index": start_index, "end_index": start_index + len(records) - 1, "batch_total": len(records),
        "success": len(passed), "fail": len(failures),
        "success_rate": len(passed) / len(records) if records else 0.0,
        "elapsed_seconds": elapsed_s, "feature_count": len(FEATURE_NAMES),
        "feature_order": FEATURE_NAMES, "feature_finite_success_count": finite_features,
        "feature_finite_rate_success": finite_features / len(passed) if passed else 0.0,
        "feature_finite_rate_all_records": finite_features / len(records) if records else 0.0,
        "volumes_positive_for_success": all(float(record["V_2_5D_m3"]) > 0 and float(record["V_pred_m3"]) > 0 for record in passed),
        "formula_max_abs_difference": max((abs(float(record["V_pred_m3"]) - float(record["V_2_5D_m3"]) * float(record["y_pred"])) for record in passed), default=None),
        "V_2_5D_m3": numeric_summary(passed, "V_2_5D_m3"), "y_pred": numeric_summary(passed, "y_pred"),
        "V_pred_m3": numeric_summary(passed, "V_pred_m3"),
        "failure_reasons": dict(Counter(record.get("failure_reason") for record in failures)),
        "strata": strata,
        "model_provenance": {"path": str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/"), "grid_resolution_m": GRID_RESOLUTION_M, "scale_factor": SCALE_FACTOR},
    }


def process_batch(
    batch_index: int,
    batch: list[dict],
    accepted: dict,
    model,
    checkpoint_path: Path,
    results_path: Path,
    qc_path: Path,
    point_index: PointCloudGridIndex,
    ground_dem: GroundDEM,
    transform,
    masks,
) -> dict:
    started = time.perf_counter()
    latest = load_latest_checkpoint(checkpoint_path)
    successes = {rock_id for rock_id, record in latest.items() if record.get("status") == "PASS"}

    structural_failures = Counter()
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        for offset, row in enumerate(batch, start=1):
            rock_id = row["rock_id"]
            if rock_id in successes:
                continue
            started_item = time.perf_counter()
            record = None
            accepted_record = accepted[rock_id]
            try:
                key = f"{rock_id}_mask"
                if key not in masks.files:
                    raise KeyError("missing_mask")
                mask = np.asarray(masks[key], dtype=bool)
                if mask.ndim != 2 or not mask.any():
                    raise ValueError("invalid_mask")
                x0, y0, _, _ = accepted_record["bbox"]
                height_map, footprint, _, _ = extract_height_map(
                    mask, x0, y0, point_index, ground_dem, transform, GRID_RESOLUTION_M
                )
                # Existing adapter returns the original mask with an empty map when
                # no query points exist. This is an individual empty-surface case,
                # not a model/canonical-schema mismatch.
                if height_map.shape != footprint.shape:
                    raise ValueError("empty_2_5d_surface")
                descriptors = compute_shape_descriptors(height_map, footprint, GRID_RESOLUTION_M)
                if int(descriptors["n_valid_cells"]) <= 0 or float(descriptors["V_2_5d"]) <= 0:
                    raise ValueError("empty_2_5d_surface")
                features = extract_features(descriptors)
                if features.shape != (12,) or not np.all(np.isfinite(features)):
                    raise FeatureSchemaError("nonfinite_feature")
                y_pred = float(model.predict(features.reshape(1, -1))[0])
                if not finite(y_pred) or y_pred <= 0 or y_pred < 0.01 or y_pred > 10.0:
                    raise ValueError("extreme_or_nonfinite_y_pred")
                volume_2_5d = float(descriptors["V_2_5d"])
                volume_pred = volume_2_5d * y_pred
                if not finite(volume_pred) or volume_pred <= 0:
                    raise ValueError("nonpositive_or_nonfinite_V_pred")
                record = blank_result(row, accepted_record, started_item)
                record.update({
                    "footprint_m2": float(descriptors["A"]), "height_m": float(descriptors["H"]),
                    "occupied_cells": int(descriptors["n_valid_cells"]), "V_2_5D_m3": volume_2_5d,
                    "y_pred": y_pred, "V_pred_m3": volume_pred, "status": "PASS", "failure_reason": None,
                    "processing_time_s": time.perf_counter() - started_item,
                })
                record.update({name: float(value) for name, value in zip(FEATURE_NAMES, features)})
            except FeatureSchemaError as exc:
                record = blank_result(row, accepted_record, started_item)
                record["failure_reason"] = f"invalid_feature:{exc}"
            except KeyError as exc:
                record = blank_result(row, accepted_record, started_item)
                record["failure_reason"] = str(exc).strip("'")
                structural_failures["missing_mask"] += 1
            except ValueError as exc:
                record = blank_result(row, accepted_record, started_item)
                record["failure_reason"] = str(exc)
            except Exception as exc:  # Record individual geometry errors and continue.
                record = blank_result(row, accepted_record, started_item)
                record["failure_reason"] = f"coordinate_or_processing_error:{type(exc).__name__}:{exc}"
                structural_failures["coordinate_or_processing_error"] += 1

            checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
            checkpoint.flush()
            latest[rock_id] = record
            if structural_failures["missing_mask"] >= 1 or structural_failures["coordinate_or_processing_error"] >= 25:
                raise RuntimeError(f"System-level failure guard triggered: {dict(structural_failures)}")
            if offset % 25 == 0 or offset == len(batch):
                print(f"batch={batch_index:03d} processed={offset}/{len(batch)} checkpoint={checkpoint_path.name}", flush=True)

    records = [latest[row["rock_id"]] for row in batch]
    write_results_csv(records, results_path)
    qc = build_qc(batch_index, records, time.perf_counter() - started)
    qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": qc["status"], "batch": batch_index, "success": qc["success"], "fail": qc["fail"], "qc": str(qc_path)}, ensure_ascii=False))
    return qc


def read_results_csv(path: Path) -> list[dict]:
    numeric_fields = {
        "equivalent_diameter_m", "point_count", "footprint_m2", "height_m", "occupied_cells",
        "V_2_5D_m3", *FEATURE_NAMES, "y_pred", "V_pred_m3", "processing_time_s",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in numeric_fields:
            if row.get(field) not in (None, ""):
                row[field] = float(row[field])
            else:
                row[field] = None
    return rows


def refresh_existing_batch_qc(batch_index: int) -> None:
    checkpoint_dir = RESULT_ROOT / "checkpoints"
    results_path = checkpoint_dir / f"batch_{batch_index:03d}_results.csv"
    qc_path = checkpoint_dir / f"batch_{batch_index:03d}_qc.json"
    if not results_path.exists():
        return
    previous_elapsed = 0.0
    if qc_path.exists():
        previous_elapsed = float(json.loads(qc_path.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))
    qc_path.write_text(json.dumps(build_qc(batch_index, read_results_csv(results_path), previous_elapsed), ensure_ascii=False, indent=2), encoding="utf-8")


def finalise_4000() -> None:
    checkpoint_dir = RESULT_ROOT / "checkpoints"
    all_records = []
    batch_qcs = []
    for batch_index in range(1, 9):
        results_path = checkpoint_dir / f"batch_{batch_index:03d}_results.csv"
        qc_path = checkpoint_dir / f"batch_{batch_index:03d}_qc.json"
        if not results_path.exists() or not qc_path.exists():
            raise RuntimeError(f"Cannot finalise: Batch {batch_index:03d} checkpoint is incomplete")
        rows = read_results_csv(results_path)
        if len(rows) != BATCH_SIZE:
            raise RuntimeError(f"Cannot finalise: Batch {batch_index:03d} does not contain 500 rows")
        all_records.extend(rows)
        batch_qcs.append(json.loads(qc_path.read_text(encoding="utf-8")))

    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    accepted_ids = {record["instance_id"] for record in json.loads(ACCEPTED_PATH.read_text(encoding="utf-8"))}
    expected_sample_ids = [row["sample_id"] for row in manifest]
    all_records.sort(key=lambda record: record["sample_id"])
    actual_sample_ids = [record["sample_id"] for record in all_records]
    rock_ids = [record["rock_id"] for record in all_records]
    passed = [record for record in all_records if record["status"] == "PASS"]
    failures = [record for record in all_records if record["status"] != "PASS"]
    nonfinite_features = sum(
        1 for record in passed for name in FEATURE_NAMES if not finite(record.get(name))
    )
    nonfinite_predictions = sum(
        1 for record in passed if not finite(record.get("y_pred")) or not finite(record.get("V_pred_m3"))
    )
    formula_max = max(
        (abs(float(record["V_pred_m3"]) - float(record["V_2_5D_m3"]) * float(record["y_pred"])) for record in passed),
        default=None,
    )
    strata = {}
    for name in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        group = [record for record in all_records if record["stratum"] == name]
        ok = [record for record in group if record["status"] == "PASS"]
        bad = [record for record in group if record["status"] != "PASS"]
        strata[name] = {
            "sample_count": len(group), "success_count": len(ok), "failure_count": len(bad),
            "success_rate": len(ok) / len(group) if group else 0.0,
            "diameter_m": numeric_summary(ok, "equivalent_diameter_m"),
            "V_2_5D_m3": numeric_summary(ok, "V_2_5D_m3"),
            "y_pred": numeric_summary(ok, "y_pred"), "V_pred_m3": numeric_summary(ok, "V_pred_m3"),
            "failure_reasons": dict(Counter(record.get("failure_reason") for record in bad)),
        }
    final_qc = {
        "manifest_count": len(manifest), "result_count": len(all_records),
        "sample_ids_match_manifest": actual_sample_ids == expected_sample_ids,
        "sample_ids_unique": len(set(actual_sample_ids)) == 4000,
        "rock_ids_unique": len(set(rock_ids)) == 4000,
        "missing_manifest_samples": sorted(set(expected_sample_ids) - set(actual_sample_ids)),
        "extra_result_samples": sorted(set(actual_sample_ids) - set(expected_sample_ids)),
        "all_rocks_accepted": set(rock_ids) <= accepted_ids,
        "feature_dimension": len(FEATURE_NAMES), "feature_order": FEATURE_NAMES,
        "nonfinite_feature_values_success": nonfinite_features,
        "nonfinite_prediction_or_volume_values_success": nonfinite_predictions,
        "nonpositive_V_2_5D_success": sum(float(record["V_2_5D_m3"]) <= 0 for record in passed),
        "nonpositive_V_pred_success": sum(float(record["V_pred_m3"]) <= 0 for record in passed),
        "formula_max_abs_difference": formula_max,
    }
    passed_qc = (
        final_qc["manifest_count"] == final_qc["result_count"] == 4000
        and final_qc["sample_ids_match_manifest"] and final_qc["sample_ids_unique"]
        and final_qc["rock_ids_unique"] and final_qc["all_rocks_accepted"]
        and final_qc["nonfinite_feature_values_success"] == 0
        and final_qc["nonfinite_prediction_or_volume_values_success"] == 0
        and final_qc["nonpositive_V_2_5D_success"] == 0
        and final_qc["nonpositive_V_pred_success"] == 0
        and final_qc["formula_max_abs_difference"] == 0.0
    )
    summary = {
        "status": "FINAL_QC_PASS" if passed_qc else "FINAL_QC_FAIL",
        "run_scope": "Stratified representative 4,000-rock sample from 69,911 accepted DOM2 instances; not a 69,911-rock full-mine volume run.",
        "model_provenance": {
            "model": str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/"), "grid_resolution_m": GRID_RESOLUTION_M,
            "scale_factor": SCALE_FACTOR, "feature_order": FEATURE_NAMES,
            "external_scaled_10mm_test": {"MAPE_percent": 5.823076769898146, "R2": 0.9838121446742769},
        },
        "population": {"accepted": 69911, "sample_manifest": 4000},
        "overall": {
            "sample_count": len(all_records), "success_count": len(passed), "failure_count": len(failures),
            "success_rate": len(passed) / len(all_records), "V_2_5D_m3": numeric_summary(passed, "V_2_5D_m3"),
            "y_pred": numeric_summary(passed, "y_pred"), "V_pred_m3": numeric_summary(passed, "V_pred_m3"),
            "failure_reasons": dict(Counter(record.get("failure_reason") for record in failures)),
            "processing_time_seconds_sum_per_rock": sum(float(record["processing_time_s"] or 0.0) for record in all_records),
            "batch_wall_time_seconds": sum(float(qc.get("elapsed_seconds", 0.0)) for qc in batch_qcs),
        },
        "strata": strata, "batch_qc": batch_qcs, "final_qc": final_qc,
        "scientific_boundary": "Real-mine absolute volume accuracy was not independently validated because per-rock DOM2 ground-truth volumes are unavailable.",
    }
    write_results_csv(all_records, RESULT_ROOT / "real_mine_volume_4000_results.csv")
    (RESULT_ROOT / "real_mine_volume_4000_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# DOM2 Shape-Aware V2: 4,000-Rock Representative Volume Estimation",
        "", "## Scope", "",
        "Shape-Aware V2 was applied to a stratified representative sample of 4,000 accepted rock instances from the DOM2 mine area.",
        "This is not a full 69,911-rock volume calculation.", "",
        "## Overall", "", "| Item | Value |", "| --- | ---: |",
        f"| Accepted population | 69,911 |", f"| Frozen sample manifest | 4,000 |",
        f"| Success / failure | {len(passed)} / {len(failures)} |", f"| Success rate | {len(passed) / len(all_records):.2%} |",
        f"| Feature dimension | {len(FEATURE_NAMES)} |", f"| Feature finite values among successes | {nonfinite_features} non-finite |",
        f"| Formula max absolute difference | {formula_max} |", "",
        "## Size-Stratified Results", "",
        "| Stratum | Sample | Success | Failure | Success rate | Diameter median (m) | V_2.5D median (m3) | y_pred median | V_pred median (m3) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, value in strata.items():
        lines.append(
            f"| {name} | {value['sample_count']} | {value['success_count']} | {value['failure_count']} | {value['success_rate']:.2%} | "
            f"{value['diameter_m']['median']} | {value['V_2_5D_m3']['median']} | {value['y_pred']['median']} | {value['V_pred_m3']['median']} |"
        )
    lines.extend([
        "", "## Distribution Details", "",
        "| Overall successful records | Min | P25 | Median | P75 | P90 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label, values in [("V_2.5D (m3)", summary["overall"]["V_2_5D_m3"]), ("y_pred", summary["overall"]["y_pred"]), ("V_pred (m3)", summary["overall"]["V_pred_m3"])]:
        lines.append(f"| {label} | {values['min']} | {values['p25']} | {values['median']} | {values['p75']} | {values['p90']} | {values['max']} |")
    for name, value in strata.items():
        lines.extend([
            f"", f"### {name}", "",
            "| Metric | Min | P25 | Median | P75 | P90 | Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for label, values in [("Diameter (m)", value["diameter_m"]), ("V_2.5D (m3)", value["V_2_5D_m3"]), ("y_pred", value["y_pred"]), ("V_pred (m3)", value["V_pred_m3"])]:
            lines.append(f"| {label} | {values['min']} | {values['p25']} | {values['median']} | {values['p75']} | {values['p90']} | {values['max']} |")
        lines.append(f"- Failure reasons: {value['failure_reasons']}")
    lines.extend([
        "", "## Feature and Model Provenance", "",
        "- Grid resolution: 0.01 m (10 mm).", "- Scale-factor provenance: 82.737840.",
        "- Canonical feature order: `C, AR, solidity, compactness, eq_diam_ratio, H_mean_norm, H_std_norm, H_p25_norm, H_p75_norm, H_skew_norm, fill_ratio, ellipsoid_ratio`.",
        "- `H_skew_norm` is the unnormalised `H_skew` value.",
        "- Frozen scaled-10mm external test performance: MAPE 5.82%, R2 0.9838.",
        "", "## Scientific Boundary", "",
        "Real-mine absolute volume accuracy was not independently validated due to the absence of per-rock ground-truth volumes. The reported DOM2 values are model-applied estimates, not an accuracy claim for the mine area.",
        "", "## Final QC", "", f"**{summary['status']}**", "",
    ])
    for key, value in final_qc.items():
        lines.append(f"- {key}: {value}")
    (RESULT_ROOT / "real_mine_volume_4000_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    indices = [args.batch_index] if args.batch_index else list(range(args.batch_range[0], args.batch_range[1] + 1))
    contexts = [(index, *preflight(index)) for index in indices]
    # Batch 001 is retained; refresh its QC schema from the completed CSV only.
    refresh_existing_batch_qc(1)

    # Loading the two LAZ files, GroundDEM, and index is intentionally once per process.
    point_cloud = load_point_cloud(LAZ_PATHS)
    ground_dem = GroundDEM(point_cloud)
    point_index = PointCloudGridIndex(point_cloud, cell_size=1.0)
    with rasterio.open(DOM_PATH) as src:
        transform = src.transform
    masks = np.load(MASKS_PATH, allow_pickle=False)

    for index, batch, accepted, model, checkpoint_path, results_path, qc_path in contexts:
        process_batch(index, batch, accepted, model, checkpoint_path, results_path, qc_path, point_index, ground_dem, transform, masks)
    if indices == list(range(2, 9)):
        finalise_4000()


if __name__ == "__main__":
    main()

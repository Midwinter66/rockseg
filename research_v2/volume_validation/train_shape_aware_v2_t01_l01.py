"""Train the new Shape-Aware V2 model from the finalized T01 + L01 dataset.

The script only reads the pre-built Dataset B artifacts. It does not build
meshes, regenerate splits, or load any historical model.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

from metrics import compute_metrics
from shape_features_v2 import FEATURE_NAMES, feature_schema


DATASET_DIR = Path("research_v2/volume_validation/datasets/t01_l01_v2")
OUTPUT_DIR = Path("research_v2/volume_validation/output_v2_t01_l01")
EXPECTED_FEATURES = FEATURE_NAMES
SEED = 42
PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 12,
    "learning_rate": 0.02,
    "n_estimators": 500,
    "verbose": -1,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "min_child_samples": 8,
    "reg_alpha": 0.2,
    "reg_lambda": 0.5,
    "min_gain_to_split": 0.0001,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "data_random_seed": SEED,
}


def load_dataset(dataset_dir: Path) -> tuple[dict, dict, np.lib.npyio.NpzFile]:
    required = {
        "samples.csv",
        "dataset_arrays.npz",
        "metadata.json",
        "splits.json",
    }
    missing = [name for name in required if not (dataset_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Dataset B artifacts missing: {missing}")

    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    splits = json.loads((dataset_dir / "splits.json").read_text(encoding="utf-8"))
    arrays = np.load(dataset_dir / "dataset_arrays.npz", allow_pickle=False)

    if metadata.get("feature_order") != EXPECTED_FEATURES:
        raise RuntimeError("Dataset feature order does not match the V2 schema.")
    if metadata.get("target_definition") != "y_ratio = V_true / V_2_5D":
        raise RuntimeError("Dataset target definition does not match the V2 ratio target.")
    if arrays["X"].ndim != 2 or arrays["X"].shape[1] != len(EXPECTED_FEATURES):
        raise RuntimeError("Dataset feature matrix must have exactly 12 columns.")
    return metadata, splits, arrays


def indices_from_saved_splits(splits: dict, sample_ids: np.ndarray) -> dict[str, np.ndarray]:
    required = ("train", "validation", "test")
    if any(name not in splits for name in required):
        raise RuntimeError("Saved splits.json must contain train, validation, and test IDs.")

    id_to_index = {str(sample_id): idx for idx, sample_id in enumerate(sample_ids)}
    if len(id_to_index) != len(sample_ids):
        raise RuntimeError("Dataset sample_id values are not unique.")

    result = {}
    for name in required:
        ids = [str(sample_id) for sample_id in splits[name]]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Duplicate sample IDs in saved {name} split.")
        unknown = sorted(set(ids) - set(id_to_index))
        if unknown:
            raise RuntimeError(f"Saved {name} split contains unknown sample IDs: {unknown[:5]}")
        result[name] = np.asarray([id_to_index[sample_id] for sample_id in ids], dtype=np.int64)

    split_sets = {name: set(indices.tolist()) for name, indices in result.items()}
    if split_sets["train"] & split_sets["validation"] or split_sets["train"] & split_sets["test"] or split_sets["validation"] & split_sets["test"]:
        raise RuntimeError("Saved Train/Validation/Test splits overlap.")
    if len(set().union(*split_sets.values())) != len(sample_ids):
        raise RuntimeError("Saved splits do not cover every dataset sample exactly once.")
    return result


def ratio_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mape = float(np.mean(np.abs(residual) / (np.abs(y_true) + 1e-12)) * 100)
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0,
        "n_samples": int(len(y_true)),
    }


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return compute_metrics(y_true, y_pred).to_dict()


def write_predictions(
    path: Path,
    arrays: np.lib.npyio.NpzFile,
    test_indices: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    fieldnames = [
        "sample_id", "dataset_id", "original_obj_id", "y_ratio_true", "y_ratio_pred",
        "V_true", "V_2_5D", "V_pred", "raw_2_5D_abs_error", "corrected_abs_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, pred in zip(test_indices, y_pred):
            v_true = float(arrays["V_true"][index])
            v_2_5d = float(arrays["V_2_5D"][index])
            v_pred = v_2_5d * float(pred)
            writer.writerow({
                "sample_id": str(arrays["sample_id"][index]),
                "dataset_id": str(arrays["dataset_id"][index]),
                "original_obj_id": str(arrays["original_obj_id"][index]),
                "y_ratio_true": float(arrays["y"][index]),
                "y_ratio_pred": float(pred),
                "V_true": v_true,
                "V_2_5D": v_2_5d,
                "V_pred": v_pred,
                "raw_2_5D_abs_error": abs(v_2_5d - v_true),
                "corrected_abs_error": abs(v_pred - v_true),
            })


def quality_check(metadata: dict, arrays: np.lib.npyio.NpzFile, split_indices: dict[str, np.ndarray]) -> dict:
    X = np.asarray(arrays["X"], dtype=np.float64)
    y = np.asarray(arrays["y"], dtype=np.float64)
    v_true = np.asarray(arrays["V_true"], dtype=np.float64)
    v_2_5d = np.asarray(arrays["V_2_5D"], dtype=np.float64)
    split_overlaps = {
        "train_validation": int(len(set(split_indices["train"]) & set(split_indices["validation"]))),
        "train_test": int(len(set(split_indices["train"]) & set(split_indices["test"]))),
        "validation_test": int(len(set(split_indices["validation"]) & set(split_indices["test"]))),
    }
    result = {
        "feature_count": int(X.shape[1]),
        "feature_order_matches": metadata.get("feature_order") == EXPECTED_FEATURES,
        "nan_count": int(np.isnan(X).sum() + np.isnan(y).sum()),
        "inf_count": int(np.isinf(X).sum() + np.isinf(y).sum()),
        "non_positive_V_true": int(np.count_nonzero(v_true <= 0)),
        "non_positive_V_2_5D": int(np.count_nonzero(v_2_5d <= 0)),
        "non_positive_y_ratio": int(np.count_nonzero(y <= 0)),
        "split_overlaps": split_overlaps,
    }
    result["passed"] = (
        result["feature_count"] == len(EXPECTED_FEATURES)
        and result["feature_order_matches"]
        and result["nan_count"] == 0
        and result["inf_count"] == 0
        and result["non_positive_V_true"] == 0
        and result["non_positive_V_2_5D"] == 0
        and result["non_positive_y_ratio"] == 0
        and not any(split_overlaps.values())
    )
    return result


def id_quality_check(arrays: np.lib.npyio.NpzFile, split_indices: dict[str, np.ndarray]) -> dict:
    dataset_ids = np.asarray(arrays["dataset_id"]).astype(str)
    original_ids = np.asarray(arrays["original_obj_id"]).astype(str)
    keys = [f"{dataset}::{obj}" for dataset, obj in zip(dataset_ids, original_ids)]
    unique_keys = len(set(keys)) == len(keys)
    test_keys = [keys[index] for index in split_indices["test"]]
    return {
        "all_dataset_original_obj_keys_unique": unique_keys,
        "test_sample_count_is_69": len(test_keys) == 69,
        "test_dataset_original_obj_keys_unique": len(set(test_keys)) == len(test_keys),
    }


def final_status(raw: dict, corrected: dict) -> tuple[str, str]:
    mae_improvement = 1.0 - corrected["mae"] / raw["mae"] if raw["mae"] > 0 else 0.0
    mape_improvement = 1.0 - corrected["mape"] / raw["mape"] if raw["mape"] > 0 else 0.0
    if mae_improvement >= 0.10 and mape_improvement >= 0.10:
        return "PASS", "Corrected test MAE and MAPE are each at least 10% lower than raw 2.5D."
    if corrected["mae"] < raw["mae"] and corrected["mape"] < raw["mape"]:
        return "WARNING", "Correction improves the fixed test set, but not by the predefined 10% margin."
    return "FAIL", "Correction does not improve both fixed-test MAE and MAPE."


def write_report(path: Path, results: dict, quality: dict, metadata: dict) -> None:
    raw = results["test_volume_metrics"]["raw_2_5D"]
    corrected = results["test_volume_metrics"]["shape_aware_v2_corrected"]
    constant = results["test_volume_metrics"]["constant_correction"]
    lines = [
        "# Shape-Aware V2 10 mm Training Report",
        "",
        f"Final status: **{results['final_status']}**",
        "",
        "## Dataset",
        "",
        f"- Dataset: `{metadata['dataset_name']}`",
        f"- Resolution: {metadata['grid_res_mm']} mm",
        f"- Samples: {metadata['n_success']} valid / {metadata['n_error']} error / {metadata['n_total']} total",
        f"- Scale factor: {metadata.get('scale_factor')}; grid: {metadata['grid_res_mm']} mm ({metadata['grid_res_mm'] / 1000.0} m)",
        "",
        "## Feature Schema",
        "",
        "Feature order: " + ", ".join(EXPECTED_FEATURES),
        "",
        "`H_skew_norm` is the unnormalised `H_skew` value.",
        "",
        "## Split",
        "",
        f"Train / Validation / Test: {results['split_counts']['train']} / {results['split_counts']['validation']} / {results['split_counts']['test']}",
        f"Seed: {SEED}; group: dataset_id + original_obj_id; overlap: {quality['split_overlaps']}",
        "",
        "## Data Quality",
        "",
        f"PASS: {quality['passed']}; NaN: {quality['nan_count']}; Inf: {quality['inf_count']}; non-positive volumes: {quality['non_positive_V_true'] + quality['non_positive_V_2_5D']}",
        "",
        "## LightGBM Parameters",
        "",
        "```json",
        json.dumps(PARAMS, indent=2),
        "```",
        "",
        "## Fixed Test Results",
        "",
        "| Method | MAE | RMSE | MAPE | R2 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Raw 2.5D | {raw['mae']:.6g} | {raw['rmse']:.6g} | {raw['mape']:.4f}% | {raw['r2']:.4f} |",
        f"| Constant correction | {constant['mae']:.6g} | {constant['rmse']:.6g} | {constant['mape']:.4f}% | {constant['r2']:.4f} |",
        f"| Shape-Aware V2 | {corrected['mae']:.6g} | {corrected['rmse']:.6g} | {corrected['mape']:.4f}% | {corrected['r2']:.4f} |",
        "",
        "## Feature Importance",
        "",
        "| Feature | Split | Gain |",
        "| --- | ---: | ---: |",
        *[
            f"| {name} | {results['feature_importance']['split'][name]} | {results['feature_importance']['gain'][name]:.6g} |"
            for name in EXPECTED_FEATURES
        ],
        "",
        "## Error Distribution",
        "",
        f"- Test predicted ratio: {results['test_prediction_summary']}",
        f"- Test corrected volume relative error: {results['test_error_summary']}",
        "",
        "## Decision",
        "",
        results["final_reason"],
        "",
        "Next step: proceed to a small real-mine single-rock interface test only when status is PASS.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global DATASET_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Train Shape-Aware V2 from a finalized dataset")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--run-suffix", default="v2_t01_l01")
    args = parser.parse_args()
    DATASET_DIR = Path(args.dataset_dir)
    OUTPUT_DIR = Path(args.output_dir)

    metadata, saved_splits, arrays = load_dataset(DATASET_DIR)
    split_indices = indices_from_saved_splits(saved_splits, arrays["sample_id"])
    quality = quality_check(metadata, arrays, split_indices)
    if not quality["passed"]:
        raise RuntimeError(f"Dataset quality check failed: {quality}")

    X = np.asarray(arrays["X"], dtype=np.float64)
    y = np.asarray(arrays["y"], dtype=np.float64)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise RuntimeError("Dataset contains NaN or Inf values.")

    train_idx = split_indices["train"]
    validation_idx = split_indices["validation"]
    test_idx = split_indices["test"]
    train_set = lgb.Dataset(X[train_idx], label=y[train_idx], feature_name=EXPECTED_FEATURES)
    validation_set = lgb.Dataset(X[validation_idx], label=y[validation_idx], reference=train_set)
    model = lgb.train(
        PARAMS,
        train_set,
        num_boost_round=500,
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    best_iteration = model.best_iteration or 500
    validation_pred = model.predict(X[validation_idx], num_iteration=best_iteration)
    train_pred = model.predict(X[train_idx], num_iteration=best_iteration)
    test_pred = model.predict(X[test_idx], num_iteration=best_iteration)
    v_true_test = np.asarray(arrays["V_true"][test_idx], dtype=np.float64)
    v_2_5d_test = np.asarray(arrays["V_2_5D"][test_idx], dtype=np.float64)
    v_pred_test = v_2_5d_test * test_pred
    constant_ratio = float(np.mean(y[train_idx]))
    v_constant_test = v_2_5d_test * constant_ratio

    prediction_checks = {
        "test_prediction_count": int(len(test_pred)),
        "test_prediction_finite": bool(np.all(np.isfinite(test_pred))),
        "test_prediction_nonfinite_count": int(np.count_nonzero(~np.isfinite(test_pred))),
        "test_prediction_nonpositive_count": int(np.count_nonzero(test_pred <= 0)),
        "test_volume_prediction_finite": bool(np.all(np.isfinite(v_pred_test))),
        "test_volume_prediction_positive": bool(np.all(v_pred_test > 0)),
        "formula_max_abs_difference": float(np.max(np.abs(v_pred_test - v_2_5d_test * test_pred))),
    }
    id_checks = id_quality_check(arrays, split_indices)
    if not (
        prediction_checks["test_prediction_finite"]
        and prediction_checks["test_volume_prediction_finite"]
        and prediction_checks["test_volume_prediction_positive"]
        and prediction_checks["formula_max_abs_difference"] <= 1e-12
        and all(id_checks.values())
    ):
        raise RuntimeError(f"Training output quality checks failed: {prediction_checks}, {id_checks}")

    split_counts = {name: int(len(indices)) for name, indices in split_indices.items()}
    split_feature_importance = {
        "split": {name: int(value) for name, value in zip(EXPECTED_FEATURES, model.feature_importance(importance_type="split"))},
        "gain": {name: float(value) for name, value in zip(EXPECTED_FEATURES, model.feature_importance(importance_type="gain"))},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = args.run_suffix
    model_path = OUTPUT_DIR / f"shape_aware_model_{suffix}.txt"
    meta_path = OUTPUT_DIR / f"model_meta_{suffix}.json"
    predictions_path = OUTPUT_DIR / f"test_predictions_{suffix}.csv"
    results_path = OUTPUT_DIR / f"training_results_{suffix}.json"
    model.save_model(str(model_path), num_iteration=best_iteration)
    write_predictions(predictions_path, arrays, test_idx, test_pred)

    results = {
        "model_name": f"shape_aware_model_{suffix}",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": metadata["dataset_name"],
        "dataset_directory": DATASET_DIR.as_posix(),
        "target_definition": metadata["target_definition"],
        "best_iteration": int(best_iteration),
        "split_counts": split_counts,
        "training_ratio_metrics": ratio_metrics(y[train_idx], train_pred),
        "validation_ratio_metrics": ratio_metrics(y[validation_idx], validation_pred),
        "test_ratio_metrics": ratio_metrics(y[test_idx], test_pred),
        "test_volume_metrics": {
            "raw_2_5D": metric_dict(v_true_test, v_2_5d_test),
            "constant_correction": metric_dict(v_true_test, v_constant_test),
            "shape_aware_v2_corrected": metric_dict(v_true_test, v_pred_test),
        },
        "constant_correction_ratio_train_mean": constant_ratio,
        "feature_importance": split_feature_importance,
        "data_quality": quality,
        "id_checks": id_checks,
        "prediction_checks": prediction_checks,
        "test_prediction_summary": {
            "min": float(np.min(test_pred)), "median": float(np.median(test_pred)),
            "max": float(np.max(test_pred)), "mean": float(np.mean(test_pred)),
        },
        "test_error_summary": {
            "relative_error_min": float(np.min(np.abs(v_pred_test - v_true_test) / v_true_test)),
            "relative_error_median": float(np.median(np.abs(v_pred_test - v_true_test) / v_true_test)),
            "relative_error_p90": float(np.percentile(np.abs(v_pred_test - v_true_test) / v_true_test, 90)),
            "relative_error_max": float(np.max(np.abs(v_pred_test - v_true_test) / v_true_test)),
        },
    }
    status, reason = final_status(
        results["test_volume_metrics"]["raw_2_5D"],
        results["test_volume_metrics"]["shape_aware_v2_corrected"],
    )
    results["final_status"] = status
    results["final_reason"] = reason
    results["data_quality"] = quality
    model_meta = {
        "model_name": results["model_name"],
        "created_at": results["evaluated_at"],
        "model_file": model_path.name,
        "dataset_name": metadata["dataset_name"],
        "dataset_directory": DATASET_DIR.as_posix(),
        "dataset_constructed_at": metadata["constructed_at"],
        "feature_order": EXPECTED_FEATURES,
        "n_features": len(EXPECTED_FEATURES),
        "target_definition": metadata["target_definition"],
        "grid_res_mm": metadata["grid_res_mm"],
        "grid_resolution_m": float(metadata["grid_res_mm"]) / 1000.0,
        "scale_factor": metadata.get("scale_factor"),
        "dataset_id_counts": metadata.get("counts"),
        "dataset_ids": sorted(set(str(value) for value in arrays["dataset_id"])),
        "random_seed": SEED,
        "training_parameters": PARAMS,
        "best_iteration": int(best_iteration),
        "split_counts": split_counts,
        "split_source": str(DATASET_DIR / "splits.json"),
        "model_path": str(model_path),
        "historical_model_used": False,
    }
    meta_path.write_text(json.dumps(model_meta, indent=2), encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(OUTPUT_DIR / "training_report_v2_scaled_10mm.md", results, quality, metadata)

    print(json.dumps({
        "model": str(model_path),
        "best_iteration": int(best_iteration),
        "split_counts": split_counts,
        "test_volume_metrics": results["test_volume_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()

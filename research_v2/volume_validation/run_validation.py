"""Main validation pipeline — Experiment E5.

Runs the complete external volume validation:

    3D mesh (ground truth)
      → 2.5-D surface simulation
      → shape descriptors
      → volume estimation (box / ellipsoid / 2.5-D / shape-aware)
      → comparison with reference volume

Usage
-----
From the project root::

    python -m research_v2.volume_validation.run_validation \\
        --data-dir data/capek_868 \\
        --output-dir research_v2/volume_validation/output

Or as a script::

    python research_v2/volume_validation/run_validation.py \\
        --data-dir data/capek_868

The data directory should contain::

    data/capek_868/
    ├── L01/  (386 OBJ files, hypervelocity impact, L3-6 chondrite)
    ├── L02/  (403 OBJ files, explosive charge,   L3-6 chondrite)
    ├── T01/  ( 79 OBJ files, explosive charge,   tephriphonolite)
    └── shapeList.txt  (optional metadata)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# pandas is optional — fall back to csv module if not available
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False
    import csv

# Allow running as a script or as a module
if __package__ is None or __package__ == "":
    _project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_project_root))
    from research_v2.volume_validation.config import Config
    from research_v2.volume_validation.mesh_utils import find_obj_files, get_mesh_info
    from research_v2.volume_validation.simulate_2_5d import simulate_2_5d_surface
    from research_v2.volume_validation.shape_descriptors import extract_descriptors, ShapeDescriptors
    from research_v2.volume_validation.volume_estimators import (
        ShapeAwareModel,
        LinearCorrectionModel,
        estimate_box_volume,
        estimate_ellipsoid_volume,
        estimate_2_5d_volume,
    )
    from research_v2.volume_validation.metrics import compute_metrics, format_metrics_table
    from research_v2.volume_validation.data_split import group_aware_split
    from research_v2.volume_validation.visualize import (
        plot_scatter,
        plot_error_boxplot,
        plot_ablation_bars,
        plot_per_group_errors,
        plot_feature_importance,
    )
else:
    from .config import Config
    from .mesh_utils import find_obj_files, get_mesh_info
    from .simulate_2_5d import simulate_2_5d_surface
    from .shape_descriptors import extract_descriptors, ShapeDescriptors
    from .volume_estimators import (
        ShapeAwareModel,
        LinearCorrectionModel,
        estimate_box_volume,
        estimate_ellipsoid_volume,
        estimate_2_5d_volume,
    )
    from .metrics import compute_metrics, format_metrics_table
    from .data_split import group_aware_split
    from .visualize import (
        plot_scatter,
        plot_error_boxplot,
        plot_ablation_bars,
        plot_per_group_errors,
        plot_feature_importance,
    )

logger = logging.getLogger("volume_validation")


# ──────────────────────────────────────────────────────────────────────
# Processing
# ──────────────────────────────────────────────────────────────────────

def process_single_sample(
    sample_id: str,
    group: str,
    obj_path: Path,
    config: Config,
    rng: np.random.Generator,
) -> dict | None:
    """Process one rock fragment: mesh → 2.5-D → descriptors → volumes.

    Returns a dict with all results, or None if processing failed.
    """
    try:
        # 1. Reference volume (ground truth)
        mesh_info = get_mesh_info(obj_path, sample_id, group)
        if mesh_info.volume_mm3 <= 0:
            logger.warning("[%s] Zero/negative volume, skipping.", sample_id)
            return None

        # 2. Simulate 2.5-D surface
        surface = simulate_2_5d_surface(obj_path, config, rng)
        if surface.n_valid_cells == 0:
            logger.warning("[%s] No valid 2.5-D cells, skipping.", sample_id)
            return None

        # 3. Extract shape descriptors
        desc = extract_descriptors(surface)

        # 4. Volume estimates
        V_true = mesh_info.volume_mm3
        V_box = estimate_box_volume(desc)
        V_ellipsoid = estimate_ellipsoid_volume(desc)
        V_2_5d = estimate_2_5d_volume(desc)

        return {
            "sample_id": sample_id,
            "group": group,
            "obj_path": str(obj_path),
            "V_true": V_true,
            "V_box": V_box,
            "V_ellipsoid": V_ellipsoid,
            "V_2_5d": V_2_5d,
            "descriptors": desc,
            "mesh_info": mesh_info,
            "n_valid_cells": surface.n_valid_cells,
        }
    except Exception as e:
        logger.error("[%s] Processing failed: %s", sample_id, e)
        return None


def process_all_samples(
    config: Config,
) -> list[dict]:
    """Process all OBJ files in the data directory.

    Supports caching: if a cache file exists and ``config.use_cache`` is True,
    loads from cache instead of reprocessing.
    """
    # Try cache
    if config.use_cache and config.cache_path.exists():
        logger.info("Loading cached data from %s", config.cache_path)
        cached = np.load(config.cache_path, allow_pickle=True)
        results = cached["results"].tolist()
        logger.info("Loaded %d cached samples.", len(results))
        return results

    # Discover files
    data_dir = config.data_dir
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        logger.info("Set --data-dir to the folder containing L01/, L02/, T01/.")
        return []

    file_list = find_obj_files(data_dir, config.groups)
    logger.info("Discovered %d OBJ files across %d groups.",
                len(file_list), len(config.groups))

    if len(file_list) == 0:
        logger.error("No OBJ files found. Check the data directory structure.")
        return []

    # Process
    rng = np.random.default_rng(config.random_seed)
    results = []

    try:
        from tqdm import tqdm
        iterator = tqdm(file_list, desc="Processing", unit="mesh")
    except ImportError:
        iterator = file_list

    for sample_id, group, obj_path in iterator:
        result = process_single_sample(sample_id, group, obj_path, config, rng)
        if result is not None:
            results.append(result)

    logger.info("Successfully processed %d / %d samples.",
                len(results), len(file_list))

    # Cache
    if config.use_cache and len(results) > 0:
        np.savez(config.cache_path, results=results)
        logger.info("Cached processed data to %s", config.cache_path)

    return results


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

METHOD_NAMES = [
    "Bounding Box",
    "Ellipsoid",
    "2.5D Integration",
    "Linear Correction",
    "Shape-Aware",
]


def evaluate_volume_methods(
    results: list[dict],
    config: Config,
) -> dict:
    """Split data, train shape-aware model, evaluate all methods on test set.

    Returns
    -------
    dict with:
        - test_predictions: {method: (n_test,) array}
        - test_metrics: {method: MetricSet}
        - test_true: (n_test,) array
        - test_groups: (n_test,) array
        - train_info: dict from model training
        - feature_importance: dict
        - split_info: dict
    """
    n = len(results)
    if n < 10:
        logger.error("Too few samples (%d) for a meaningful split.", n)
        return {}

    # Extract arrays
    groups = np.array([r["group"] for r in results])
    V_true = np.array([r["V_true"] for r in results])
    V_box = np.array([r["V_box"] for r in results])
    V_ellipsoid = np.array([r["V_ellipsoid"] for r in results])
    V_2_5d = np.array([r["V_2_5d"] for r in results])
    descriptors = [r["descriptors"] for r in results]

    # ── Split ────────────────────────────────────────────────────────
    logger.info("Splitting data (train %.0f%% / val %.0f%% / test %.0f%%, group-aware)...",
                config.train_ratio * 100,
                config.val_ratio * 100,
                config.test_ratio * 100)

    split = group_aware_split(
        groups,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.random_seed,
    )
    logger.info("Split: train=%d, val=%d, test=%d",
                split.n_train, split.n_val, split.n_test)

    # ── Train models ────────────────────────────────────────────────
    train_desc = [descriptors[i] for i in split.train_idx]
    train_y = V_true[split.train_idx]
    val_desc = [descriptors[i] for i in split.val_idx]
    val_y = V_true[split.val_idx]

    # 4. Linear correction: V = α × V_2.5D (single scalar, most transferable)
    logger.info("Training linear correction model (α × V_2.5D)...")
    linear_model = LinearCorrectionModel()
    linear_info = linear_model.train(train_desc, train_y)

    # 5. Shape-aware ratio model: r = f(shape) → V = r × V_2.5D
    logger.info("Training shape-aware ratio model (LightGBM, transferable features)...")
    model = ShapeAwareModel(config.lgbm_params, mode="transferable")
    train_info = model.train(train_desc, train_y, val_desc, val_y)

    # ── Evaluate on test set ────────────────────────────────────────
    test_idx = split.test_idx
    test_true = V_true[test_idx]
    test_groups = groups[test_idx]
    test_desc = [descriptors[i] for i in test_idx]

    test_box = V_box[test_idx]
    test_ellipsoid = V_ellipsoid[test_idx]
    test_2_5d = V_2_5d[test_idx]
    test_linear = linear_model.predict(test_desc)
    test_shape = model.predict(test_desc)

    test_predictions = {
        "Bounding Box": test_box,
        "Ellipsoid": test_ellipsoid,
        "2.5D Integration": test_2_5d,
        "Linear Correction": test_linear,
        "Shape-Aware": test_shape,
    }

    # ── Correction ratio analysis ───────────────────────────────────
    # r = V_true / V_2.5D  — the key diagnostic
    train_ratios = train_y / (np.array([d.V_2_5d for d in train_desc]) + 1e-9)
    test_ratios = test_true / (test_2_5d + 1e-9)
    ratio_analysis = {
        "train_mean_ratio": float(np.mean(train_ratios)),
        "train_median_ratio": float(np.median(train_ratios)),
        "train_std_ratio": float(np.std(train_ratios)),
        "test_mean_ratio": float(np.mean(test_ratios)),
        "test_median_ratio": float(np.median(test_ratios)),
        "test_std_ratio": float(np.std(test_ratios)),
        "linear_alpha": linear_info["alpha"],
        "interpretation": (
            f"2.5D overestimates by factor {1.0/np.median(train_ratios):.2f}x "
            f"(median ratio r={np.median(train_ratios):.3f}). "
            f"Linear correction α={linear_info['alpha']:.3f} applies a global fix. "
            f"Shape-aware model predicts per-rock r from 5 dimensionless features "
            f"(C, AR, H_mean/H, H_std/H, L/W) that are scale-invariant and "
            f"transferable to blast-rock scenes."
        ),
    }
    logger.info("Correction ratio analysis: %s", ratio_analysis["interpretation"])

    # ── Metrics ──────────────────────────────────────────────────────
    test_metrics = {}
    for name in METHOD_NAMES:
        ms = compute_metrics(test_true, test_predictions[name])
        test_metrics[name] = ms
        logger.info("  %s: MAPE=%.2f%%, SMAPE=%.2f%%, R²=%.4f",
                     name, ms.mape, ms.smape, ms.r2)

    # ── Acceptance check ─────────────────────────────────────────────
    e_shape = test_metrics["Shape-Aware"].mean_rel_error
    e_linear = test_metrics["Linear Correction"].mean_rel_error
    e_box = test_metrics["Bounding Box"].mean_rel_error
    e_ellipsoid = test_metrics["Ellipsoid"].mean_rel_error
    e_2_5d = test_metrics["2.5D Integration"].mean_rel_error

    acceptance = {
        "E_shape < E_box": e_shape < e_box,
        "E_shape < E_ellipsoid": e_shape < e_ellipsoid,
        "E_shape < E_2.5D": e_shape < e_2_5d,
        "E_linear < E_2.5D": e_linear < e_2_5d,
        "passed": e_shape < e_box and e_shape < e_ellipsoid,
        "E_shape": e_shape,
        "E_linear": e_linear,
        "E_box": e_box,
        "E_ellipsoid": e_ellipsoid,
        "E_2.5D": e_2_5d,
    }

    logger.info("Acceptance: %s", "PASSED" if acceptance["passed"] else "FAILED")
    logger.info("  E_box=%.1f%%  E_ellipsoid=%.1f%%  E_2.5D=%.1f%%  "
                "E_linear=%.1f%%  E_shape=%.1f%%",
                e_box, e_ellipsoid, e_2_5d, e_linear, e_shape)

    return {
        "test_predictions": test_predictions,
        "test_metrics": test_metrics,
        "test_true": test_true,
        "test_groups": test_groups,
        "test_idx": test_idx,
        "train_info": train_info,
        "linear_info": linear_info,
        "ratio_analysis": ratio_analysis,
        "feature_importance": model.feature_importance(),
        "acceptance": acceptance,
        "split_info": split.to_dict(),
        "split": split,
        "model": model,
        "linear_model": linear_model,
    }


# ──────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────

def _write_csv(filepath: Path, rows: list[dict]) -> None:
    """Write list of dicts to CSV (uses pandas if available, else csv module)."""
    if _HAS_PANDAS:
        pd.DataFrame(rows).to_csv(filepath, index=False)
    else:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def save_results(
    eval_output: dict,
    all_results: list[dict],
    config: Config,
) -> None:
    """Save all results: CSV, JSON, and figures."""
    results_dir = config.results_path
    figures_dir = config.figures_path

    # ── Per-sample CSV (all samples) ────────────────────────────────
    rows = []
    for r in all_results:
        d = r["descriptors"]
        rows.append({
            "sample_id": r["sample_id"],
            "group": r["group"],
            "V_true": r["V_true"],
            "V_box": r["V_box"],
            "V_ellipsoid": r["V_ellipsoid"],
            "V_2_5d": r["V_2_5d"],
            "L": d.L, "W": d.W, "H": d.H,
            "A": d.A, "P": d.P,
            "C": d.C, "AR": d.AR,
            "H_mean": d.H_mean, "H_max": d.H_max, "H_std": d.H_std,
            "fill_ratio": d.fill_ratio,
            "solidity": d.solidity,
            "n_valid_cells": r["n_valid_cells"],
            "is_watertight": r["mesh_info"].is_watertight,
            "n_vertices": r["mesh_info"].n_vertices,
            "n_faces": r["mesh_info"].n_faces,
        })
    _write_csv(results_dir / "all_samples.csv", rows)
    logger.info("Saved all-sample CSV to %s", results_dir / "all_samples.csv")

    # ── Test-set CSV ────────────────────────────────────────────────
    test_idx = eval_output["test_idx"]
    test_rows = []
    for i, idx in enumerate(test_idx):
        r = all_results[idx]
        test_rows.append({
            "sample_id": r["sample_id"],
            "group": r["group"],
            "V_true": eval_output["test_true"][i],
            "V_box": eval_output["test_predictions"]["Bounding Box"][i],
            "V_ellipsoid": eval_output["test_predictions"]["Ellipsoid"][i],
            "V_2_5d": eval_output["test_predictions"]["2.5D Integration"][i],
            "V_linear": eval_output["test_predictions"]["Linear Correction"][i],
            "V_shape": eval_output["test_predictions"]["Shape-Aware"][i],
            "ratio_r": eval_output["test_true"][i] / (eval_output["test_predictions"]["2.5D Integration"][i] + 1e-9),
        })
    df_test = test_rows  # list of dicts
    _write_csv(results_dir / "test_predictions.csv", df_test)
    logger.info("Saved test predictions to %s", results_dir / "test_predictions.csv")

    # ── Metrics JSON ────────────────────────────────────────────────
    metrics_json = {}
    for name in METHOD_NAMES:
        ms = eval_output["test_metrics"][name]
        metrics_json[name] = ms.to_dict()

    summary = {
        "n_total": len(all_results),
        "n_train": eval_output["split"].n_train,
        "n_val": eval_output["split"].n_val,
        "n_test": eval_output["split"].n_test,
        "metrics": metrics_json,
        "acceptance": eval_output["acceptance"],
        "train_info": eval_output["train_info"],
        "linear_info": eval_output["linear_info"],
        "ratio_analysis": eval_output["ratio_analysis"],
        "feature_importance": eval_output["feature_importance"],
        "config": {
            "grid_resolution_mm": config.grid_resolution_mm,
            "height_noise_std_mm": config.height_noise_std_mm,
            "point_sparsity": config.point_sparsity,
            "random_seed": config.random_seed,
            "lgbm_params": config.lgbm_params,
        },
    }
    with open(results_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved metrics summary to %s", results_dir / "metrics_summary.json")

    # ── Figures ─────────────────────────────────────────────────────
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    test_true = eval_output["test_true"]
    test_preds = eval_output["test_predictions"]

    for name, color in zip(METHOD_NAMES, colors):
        plot_scatter(
            test_true,
            test_preds[name],
            name,
            color=color,
            save_path=figures_dir / f"scatter_{name.lower().replace(' ', '_')}.png",
        )
    logger.info("Saved scatter plots.")

    plot_error_boxplot(
        test_true,
        test_preds,
        save_path=figures_dir / "error_boxplot.png",
    )
    logger.info("Saved error boxplot.")

    plot_ablation_bars(
        METHOD_NAMES,
        [eval_output["test_metrics"][n] for n in METHOD_NAMES],
        save_path=figures_dir / "ablation_bars.png",
    )
    logger.info("Saved ablation bar chart.")

    for name in METHOD_NAMES:
        plot_per_group_errors(
            test_true,
            test_preds[name],
            eval_output["test_groups"],
            name,
            save_path=figures_dir / f"per_group_{name.lower().replace(' ', '_')}.png",
        )
    logger.info("Saved per-group error charts.")

    if eval_output["feature_importance"]:
        plot_feature_importance(
            eval_output["feature_importance"],
            save_path=figures_dir / "feature_importance.png",
        )
        logger.info("Saved feature importance chart.")

    # ── Save models ─────────────────────────────────────────────────
    model_path = results_dir / "shape_aware_model.txt"
    eval_output["model"].save(model_path)
    logger.info("Saved shape-aware model to %s", model_path)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="External volume validation (E5) — "
                    "2.5D-to-volume estimation benchmark on 868 rock fragments."
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing L01/, L02/, T01/ subfolders with OBJ files.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results and figures.",
    )
    parser.add_argument(
        "--grid-resolution", type=float, default=None,
        help="Grid cell size in mm (default: 0.5).",
    )
    parser.add_argument(
        "--noise-std", type=float, default=None,
        help="Height noise std in mm (default: 0.0 = clean).",
    )
    parser.add_argument(
        "--sparsity", type=float, default=None,
        help="Fraction of cells to remove (default: 0.0).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable caching of processed mesh data.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = Config()
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.grid_resolution is not None:
        config.grid_resolution_mm = args.grid_resolution
    if args.noise_std is not None:
        config.height_noise_std_mm = args.noise_std
    if args.sparsity is not None:
        config.point_sparsity = args.sparsity
    if args.seed is not None:
        config.random_seed = args.seed
    if args.no_cache:
        config.use_cache = False

    config.ensure_output_dir()

    logger.info("=" * 70)
    logger.info("RockSeg V2 — Experiment E5: External Volume Validation")
    logger.info("=" * 70)
    logger.info("Data dir:     %s", config.data_dir)
    logger.info("Output dir:   %s", config.output_dir)
    logger.info("Grid res:     %.2f mm", config.grid_resolution_mm)
    logger.info("Noise std:    %.2f mm", config.height_noise_std_mm)
    logger.info("Sparsity:     %.2f", config.point_sparsity)
    logger.info("Random seed:  %d", config.random_seed)
    logger.info("=" * 70)

    # ── Process ─────────────────────────────────────────────────────
    t0 = time.time()
    results = process_all_samples(config)
    t1 = time.time()
    logger.info("Processing completed in %.1f s (%d samples).", t1 - t0, len(results))

    if len(results) == 0:
        logger.error("No samples processed. Exiting.")
        return 1

    # ── Evaluate ────────────────────────────────────────────────────
    t0 = time.time()
    eval_output = evaluate_volume_methods(results, config)
    t1 = time.time()
    logger.info("Evaluation completed in %.1f s.", t1 - t0)

    if not eval_output:
        logger.error("Evaluation failed. Exiting.")
        return 1

    # ── Print summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Volume Ablation Results (held-out test set)")
    print("=" * 70)
    print(format_metrics_table(
        METHOD_NAMES,
        [eval_output["test_metrics"][n] for n in METHOD_NAMES],
    ))
    print("=" * 70)

    acc = eval_output["acceptance"]
    print(f"\nAcceptance criterion: E_shape < E_box AND E_shape < E_ellipsoid")
    print(f"  E_box       = {acc['E_box']:.2f}%")
    print(f"  E_ellipsoid = {acc['E_ellipsoid']:.2f}%")
    print(f"  E_2.5D      = {acc['E_2.5D']:.2f}%")
    print(f"  E_linear    = {acc['E_linear']:.2f}%")
    print(f"  E_shape     = {acc['E_shape']:.2f}%")
    print(f"  Result      = {'PASSED' if acc['passed'] else 'FAILED'}")

    ra = eval_output["ratio_analysis"]
    print(f"\nCorrection Ratio Analysis (r = V_true / V_2.5D):")
    print(f"  Train: mean={ra['train_mean_ratio']:.3f}  median={ra['train_median_ratio']:.3f}  std={ra['train_std_ratio']:.3f}")
    print(f"  Test:  mean={ra['test_mean_ratio']:.3f}  median={ra['test_median_ratio']:.3f}  std={ra['test_std_ratio']:.3f}")
    print(f"  Linear α = {ra['linear_alpha']:.3f}")
    print(f"\n  {ra['interpretation']}")

    # ── Save ────────────────────────────────────────────────────────
    save_results(eval_output, results, config)
    print(f"\nResults saved to: {config.output_dir}")
    print(f"  CSV:     {config.results_path}")
    print(f"  Figures: {config.figures_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

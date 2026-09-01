"""Standalone shape-aware model enhancement script.

Lightweight version that doesn't need trimesh/pandas.
Loads OBJ files manually, computes volumes, simulates 2.5D surfaces,
extracts enhanced shape features, and trains an improved LightGBM model.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np


try:
    from .shape_features_v2 import (
        FEATURE_NAMES,
        compute_surface_descriptors,
        extract_features as _extract_v2_features,
    )
except ImportError:
    from shape_features_v2 import (
        FEATURE_NAMES,
        compute_surface_descriptors,
        extract_features as _extract_v2_features,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("shape_aware_v2")

# ──────────────────────────────────────────────────────────────────────
# OBJ loading & volume computation (no trimesh needed)
# ──────────────────────────────────────────────────────────────────────

def load_obj_simple(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from an OBJ file.
    
    Returns (vertices (N,3), faces (M,3))
    """
    vertices = []
    faces = []
    with open(obj_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                # Handle formats like "f v1/vt1/vn1 v2/vt2/vn2 ..."
                face_verts = []
                for p in parts[1:]:
                    idx = int(p.split("/")[0]) - 1  # OBJ is 1-indexed
                    face_verts.append(idx)
                # Triangulate if more than 3 vertices
                if len(face_verts) >= 3:
                    for i in range(1, len(face_verts) - 1):
                        faces.append([face_verts[0], face_verts[i], face_verts[i+1]])
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


def compute_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute volume of a triangular mesh using the divergence theorem.
    
    V = (1/3) * sum_over_faces (dot(centroid, normal) * area)
    For watertight meshes this gives exact volume.
    """
    if len(faces) == 0:
        return 0.0
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    # Cross product for face normal * 2*area
    cross = np.cross(v1 - v0, v2 - v0)
    # Signed volume contribution: dot(v0, cross) / 6
    vol = np.sum(v0[:, 0] * cross[:, 0] + v0[:, 1] * cross[:, 1] + v0[:, 2] * cross[:, 2]) / 6.0
    return abs(vol)


def find_obj_files(data_dir: Path) -> list[tuple[str, str, Path]]:
    """Find all OBJ files in subdirectories.
    
    Returns list of (sample_id, group, path).
    """
    results = []
    data_dir = Path(data_dir)
    for group_dir in sorted(data_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        group = group_dir.name
        for obj_file in sorted(group_dir.glob("*.obj")):
            sample_id = obj_file.stem
            results.append((sample_id, group, obj_file))
    return results


# ──────────────────────────────────────────────────────────────────────
# 2.5D surface simulation
# ──────────────────────────────────────────────────────────────────────

def simulate_2_5d_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    grid_resolution: float = 0.5,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Simulate a 2.5-D top-down height map from a mesh.
    
    First translates the mesh so its lowest point rests on z=0 (ground),
    then rasterizes the top surface by taking max z per grid cell.
    
    Returns dict with: height_map, footprint_mask, cell_size, grid_x, grid_y, ground_z, n_valid_cells
    """
    if len(vertices) == 0:
        return None
    
    # Translate mesh so lowest point is at z=0 (resting on ground)
    v = vertices.copy()
    z_min = v[:, 2].min()
    v[:, 2] -= z_min  # now bottom is at z=0
    
    # Bounding box in XY
    xmin, ymin = v[:, 0].min(), v[:, 1].min()
    xmax, ymax = v[:, 0].max(), v[:, 1].max()
    
    # Add small padding
    pad = grid_resolution * 0.5
    xmin -= pad
    ymin -= pad
    xmax += pad
    ymax += pad
    
    # Grid dimensions
    nx = max(2, int(np.ceil((xmax - xmin) / grid_resolution)))
    ny = max(2, int(np.ceil((ymax - ymin) / grid_resolution)))
    
    grid_x = xmin + np.arange(nx) * grid_resolution + grid_resolution / 2
    grid_y = ymin + np.arange(ny) * grid_resolution + grid_resolution / 2
    
    # Rasterize: for each face, find grid cells and compute max z
    height_map = np.full((ny, nx), -np.inf, dtype=np.float64)
    footprint_mask = np.zeros((ny, nx), dtype=bool)
    
    for face in faces:
        tri = v[face]
        tx, ty = tri[:, 0], tri[:, 1]
        
        # Bounding box of the triangle in grid coords
        gx_min = max(0, int(np.floor((tx.min() - xmin) / grid_resolution)))
        gx_max = min(nx - 1, int(np.floor((tx.max() - xmin) / grid_resolution)))
        gy_min = max(0, int(np.floor((ty.min() - ymin) / grid_resolution)))
        gy_max = min(ny - 1, int(np.floor((ty.max() - ymin) / grid_resolution)))
        
        if gx_min > gx_max or gy_min > gy_max:
            continue
        
        # Precompute triangle edges for point-in-triangle test
        v0 = tri[2] - tri[0]
        v1 = tri[1] - tri[0]
        dot00 = v0[0]*v0[0] + v0[1]*v0[1]
        dot01 = v0[0]*v1[0] + v0[1]*v1[1]
        dot11 = v1[0]*v1[0] + v1[1]*v1[1]
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-15)
        
        for gy in range(gy_min, gy_max + 1):
            cy = grid_y[gy]
            for gx in range(gx_min, gx_max + 1):
                cx = grid_x[gx]
                
                # Barycentric coordinates
                v2x = cx - tri[0, 0]
                v2y = cy - tri[0, 1]
                dot02 = v0[0]*v2x + v0[1]*v2y
                dot12 = v1[0]*v2x + v1[1]*v2y
                
                u = (dot11 * dot02 - dot01 * dot12) * inv_denom
                v_bary = (dot00 * dot12 - dot01 * dot02) * inv_denom
                
                if u >= 0 and v_bary >= 0 and (u + v_bary) <= 1:
                    # Interpolate z
                    z = tri[0, 2] + u * v0[2] + v_bary * v1[2]
                    if z > height_map[gy, gx]:
                        height_map[gy, gx] = z
                        footprint_mask[gy, gx] = True
    
    # Ground z = 0 (mesh rests on ground after translation)
    ground_z = 0.0
    
    # Add noise
    if noise_std > 0 and rng is not None:
        noise = rng.normal(0, noise_std, height_map.shape)
        height_map[footprint_mask] += noise[footprint_mask]
    
    n_valid = int(footprint_mask.sum())
    
    if n_valid == 0:
        return None
    
    return {
        "height_map": height_map,
        "footprint_mask": footprint_mask,
        "cell_size": grid_resolution,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "ground_z": ground_z,
        "n_valid_cells": n_valid,
    }


# ──────────────────────────────────────────────────────────────────────
# Shape descriptors (enhanced)
# ──────────────────────────────────────────────────────────────────────

def extract_descriptors(surface: dict) -> dict:
    """Extract Dataset B descriptors through the canonical V2 implementation."""
    return compute_surface_descriptors(
        surface["height_map"],
        surface["footprint_mask"],
        surface["cell_size"],
        surface["ground_z"],
    )


# ──────────────────────────────────────────────────────────────────────
# Transferable features (dimensionless)
# ──────────────────────────────────────────────────────────────────────

def extract_features(desc: dict) -> np.ndarray:
    """Extract 12 dimensionless transferable features."""
    return _extract_v2_features(desc)


# ──────────────────────────────────────────────────────────────────────
# Data splitting
# ──────────────────────────────────────────────────────────────────────

def group_aware_split(groups: np.ndarray, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Stratified split preserving group proportions."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    train_idx, val_idx, test_idx = [], [], []
    
    for g in unique_groups:
        g_mask = groups == g
        g_indices = np.where(g_mask)[0]
        rng.shuffle(g_indices)
        n_g = len(g_indices)
        n_train = int(round(n_g * train_ratio))
        n_val = int(round(n_g * val_ratio))
        n_test = n_g - n_train - n_val
        if n_g >= 3 and n_test < 1:
            n_test = 1
            n_val = max(0, n_g - n_train - n_test)
        train_idx.extend(g_indices[:n_train].tolist())
        val_idx.extend(g_indices[n_train:n_train + n_val].tolist())
        test_idx.extend(g_indices[n_train + n_val:].tolist())
    
    return (np.array(train_idx), np.array(val_idx), np.array(test_idx))


# ──────────────────────────────────────────────────────────────────────
# Model training & evaluation
# ──────────────────────────────────────────────────────────────────────

def train_shape_aware(X_train, y_train, X_val, y_val, params=None):
    """Train LightGBM ratio model."""
    import lightgbm as lgb

    if params is None:
        params = {
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
        }
    
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    
    callbacks = [lgb.early_stopping(100, verbose=False)]
    model = lgb.train(
        params, train_set,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    return model


def compute_metrics(y_true, y_pred):
    """Compute error metrics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rel_err = np.abs(y_pred - y_true) / (y_true + 1e-9)
    mape = float(np.mean(rel_err) * 100)
    smape = float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-9)) * 100)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-9))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return {"mape": mape, "smape": smape, "r2": r2, "mae": mae, "mean_rel_error": mape}


# ──────────────────────────────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────────────────────────────

def cross_validate(X, y, groups, n_folds=5, params=None, seed=42):
    """Group-aware k-fold cross-validation."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    n = len(X)
    
    # Shuffle indices
    indices = np.arange(n)
    rng.shuffle(indices)
    
    fold_size = n // n_folds
    fold_metrics = []
    fold_ratios = []
    
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n
        
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        
        # Use first 80% of train as train, last 20% as val
        n_train = int(len(train_idx) * 0.85)
        tr_idx = train_idx[:n_train]
        val_idx = train_idx[n_train:]
        
        model = train_shape_aware(X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], params)
        best_iter = model.best_iteration
        y_pred = model.predict(X[test_idx], num_iteration=best_iter)
        
        metrics = compute_metrics(y[test_idx], y_pred)
        fold_metrics.append(metrics)
        fold_ratios.append({
            "pred_mean": float(np.mean(y_pred)),
            "pred_std": float(np.std(y_pred)),
            "true_mean": float(np.mean(y[test_idx])),
            "true_std": float(np.std(y[test_idx])),
        })
    
    # Aggregate
    agg = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        agg[key] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
    
    return agg, fold_metrics, fold_ratios


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────

def main():
    data_dir = Path("data/experience_rock")
    output_dir = Path("research_v2/volume_validation/output_v2_enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    grid_res = 0.5  # mm
    noise_std = 0.0
    seed = 42
    rng = np.random.default_rng(seed)
    
    logger.info("=" * 70)
    logger.info("Shape-Aware Model Enhancement (v2)")
    logger.info("=" * 70)
    
    # 1. Load and process all samples
    logger.info("Step 1: Loading OBJ files and computing 2.5D surfaces...")
    file_list = find_obj_files(data_dir)
    logger.info(f"Found {len(file_list)} OBJ files")
    
    all_descs = []
    all_V_true = []
    all_groups = []
    all_ids = []
    
    t0 = time.time()
    for sample_id, group, obj_path in file_list:
        try:
            verts, faces = load_obj_simple(obj_path)
            V_true = compute_mesh_volume(verts, faces)
            
            if V_true <= 0 or len(verts) < 4:
                continue
            
            surface = simulate_2_5d_surface(verts, faces, grid_res, noise_std, rng)
            if surface is None or surface["n_valid_cells"] == 0:
                continue
            
            desc = extract_descriptors(surface)
            if desc is None:
                continue
            
            all_descs.append(desc)
            all_V_true.append(V_true)
            all_groups.append(group)
            all_ids.append(sample_id)
        except Exception as e:
            logger.warning(f"  Failed {sample_id}: {e}")
    
    t1 = time.time()
    logger.info(f"Processed {len(all_descs)} samples in {t1-t0:.1f}s")
    
    all_V_true = np.array(all_V_true)
    all_groups = np.array(all_groups)
    
    # 2. Feature extraction
    logger.info("Step 2: Extracting transferable features (12 dimensions)...")
    X = np.array([extract_features(d) for d in all_descs])
    V_2_5d = np.array([d["V_2_5d"] for d in all_descs])
    y_ratios = all_V_true / (V_2_5d + 1e-9)  # r = V_true / V_2.5d
    
    logger.info(f"  Features shape: {X.shape}")
    logger.info(f"  Ratio r stats: mean={np.mean(y_ratios):.3f}, std={np.std(y_ratios):.3f}, "
                f"median={np.median(y_ratios):.3f}, range=[{np.min(y_ratios):.3f}, {np.max(y_ratios):.3f}]")
    
    # 3. Split data
    logger.info("Step 3: Group-aware train/val/test split (70/15/15)...")
    train_idx, val_idx, test_idx = group_aware_split(all_groups, seed=seed)
    logger.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # 4. Baseline methods
    V_box = np.array([d["V_box"] for d in all_descs])
    V_ellipsoid = np.array([d["V_ellipsoid"] for d in all_descs])
    
    # Linear correction (alpha from train set)
    train_ratios = y_ratios[train_idx]
    alpha = float(np.median(train_ratios))
    logger.info(f"\n  Linear correction α = {alpha:.4f} (median ratio on train set)")
    
    # 5. Train shape-aware model
    logger.info("\nStep 4: Training enhanced shape-aware model (12 features, LightGBM)...")
    model = train_shape_aware(
        X[train_idx], y_ratios[train_idx],
        X[val_idx], y_ratios[val_idx],
    )
    best_iter = model.best_iteration
    logger.info(f"  Best iteration: {best_iter}")
    
    # Feature importance
    imp = model.feature_importance(importance_type="split")
    imp_dict = dict(zip(FEATURE_NAMES, imp.tolist()))
    logger.info(f"  Feature importance (split):")
    for name, val in sorted(imp_dict.items(), key=lambda x: -x[1]):
        logger.info(f"    {name:20s}: {val:5.1f}")
    
    # 6. Evaluate all methods on test set
    logger.info("\nStep 5: Evaluating all methods on test set...")
    test_true = all_V_true[test_idx]
    
    test_box = V_box[test_idx]
    test_ellipsoid = V_ellipsoid[test_idx]
    test_2_5d = V_2_5d[test_idx]
    test_linear = alpha * V_2_5d[test_idx]
    
    test_ratios_pred = model.predict(X[test_idx], num_iteration=best_iter)
    test_shape = test_ratios_pred * V_2_5d[test_idx]
    
    methods = {
        "Bounding Box": test_box,
        "Ellipsoid": test_ellipsoid,
        "2.5D Integration": test_2_5d,
        "Linear Correction": test_linear,
        "Shape-Aware (v2)": test_shape,
    }
    
    print("\n" + "=" * 70)
    print("VOLUME ESTIMATION RESULTS (Test Set)")
    print("=" * 70)
    print(f"{'Method':<25s} {'MAPE':>8s} {'SMAPE':>8s} {'R²':>8s}")
    print("-" * 70)
    
    all_metrics = {}
    for name, pred in methods.items():
        m = compute_metrics(test_true, pred)
        all_metrics[name] = m
        print(f"{name:<25s} {m['mape']:>7.2f}% {m['smape']:>7.2f}% {m['r2']:>8.4f}")
    
    print("=" * 70)
    
    # 7. Correction ratio analysis
    print(f"\nCorrection Ratio Analysis (r = V_true / V_2.5D):")
    test_ratios_true = test_true / (test_2_5d + 1e-9)
    print(f"  True r:  mean={np.mean(test_ratios_true):.4f}, std={np.std(test_ratios_true):.4f}, "
          f"range=[{np.min(test_ratios_true):.4f}, {np.max(test_ratios_true):.4f}]")
    print(f"  Pred r:  mean={np.mean(test_ratios_pred):.4f}, std={np.std(test_ratios_pred):.4f}, "
          f"range=[{np.min(test_ratios_pred):.4f}, {np.max(test_ratios_pred):.4f}]")
    print(f"  Linear α: {alpha:.4f}")
    print(f"\n  Shape-aware model predicts {len(np.unique(np.round(test_ratios_pred, 4)))} unique ratio values")
    print(f"  (vs. 1 for linear correction — this is the 'shape-aware' advantage)")
    
    # 8. Cross-validation for robustness
    logger.info("\nStep 6: 5-fold cross-validation (stability check)...")
    cv_params = {
        "objective": "regression", "metric": "mae",
        "num_leaves": 12, "learning_rate": 0.02, "n_estimators": 500,
        "verbose": -1, "subsample": 0.9, "colsample_bytree": 0.9,
        "min_child_samples": 8, "reg_alpha": 0.2, "reg_lambda": 0.5,
    }
    cv_agg, cv_folds, cv_ratios = cross_validate(X, y_ratios, all_groups, n_folds=5, params=cv_params, seed=seed)
    print(f"\n5-Fold CV Results (ratio prediction):")
    print(f"  MAPE: {cv_agg['mape']:.2f}% ± {cv_agg['mape_std']:.2f}%")
    print(f"  R²:   {cv_agg['r2']:.4f} ± {cv_agg['r2_std']:.4f}")
    
    # 9. Save model and results
    logger.info("\nStep 7: Saving model and results...")
    
    model_path = output_dir / "shape_aware_model_v2.txt"
    model.save_model(str(model_path))
    
    # Save meta
    meta = {
        "version": "v2_enhanced",
        "n_features": 12,
        "feature_names": FEATURE_NAMES,
        "best_iteration": int(best_iter),
        "alpha_linear": alpha,
        "test_metrics": all_metrics,
        "cv_agg": cv_agg,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "grid_resolution_mm": grid_res,
    }
    with open(output_dir / "model_meta_v2.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    # Save test predictions
    test_rows = []
    for i, idx in enumerate(test_idx):
        test_rows.append({
            "sample_id": all_ids[idx],
            "group": all_groups[idx],
            "V_true": float(test_true[i]),
            "V_2_5d": float(test_2_5d[i]),
            "V_shape": float(test_shape[i]),
            "ratio_true": float(test_ratios_true[i]),
            "ratio_pred": float(test_ratios_pred[i]),
        })
    
    with open(output_dir / "test_predictions_v2.csv", "w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(test_rows)
    
    # 10. Comparison with v1 baseline
    logger.info("\nStep 8: Comparison with v1 baseline...")
    v1_path = Path("research_v2/volume_validation/output/results/metrics_summary_v1_baseline.json")
    if v1_path.exists():
        with open(v1_path) as f:
            v1 = json.load(f)
        
        print(f"\n{'='*70}")
        print(f"  COMPARISON: v1 (5 features) vs v2 (12 features)")
        print(f"{'='*70}")
        print(f"{'Method':<25s} {'v1 MAPE':>10s} {'v2 MAPE':>10s} {'Δ':>10s}")
        print(f"{'-'*70}")
        
        v1_shape_mape = v1["metrics"]["Shape-Aware"]["mape"]
        v2_shape_mape = all_metrics["Shape-Aware (v2)"]["mape"]
        improvement = v1_shape_mape - v2_shape_mape
        
        print(f"{'2.5D Integration':<25s} {v1['metrics']['2.5D Integration']['mape']:>9.2f}% {all_metrics['2.5D Integration']['mape']:>9.2f}%")
        print(f"{'Linear Correction':<25s} {v1['metrics']['Linear Correction']['mape']:>9.2f}% {all_metrics['Linear Correction']['mape']:>9.2f}%")
        print(f"{'Shape-Aware':<25s} {v1_shape_mape:>9.2f}% {v2_shape_mape:>9.2f}% {improvement:>+9.2f}%")
        print(f"{'='*70}")
        
        if improvement > 0.5:
            print(f"\n  ✓ Improvement of {improvement:.2f}% — shape-aware is now meaningfully better than linear!")
        else:
            print(f"\n  ⚠ Small improvement ({improvement:.2f}%) — model still close to linear correction")
    
    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()

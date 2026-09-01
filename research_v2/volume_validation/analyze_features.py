"""Analyze feature-ratio correlation to understand why shape-aware improvement is small."""
import sys
import numpy as np
from pathlib import Path

# Import our functions
sys.path.insert(0, str(Path(__file__).parent))
from enhance_shape_aware import (
    load_obj_simple, compute_mesh_volume, find_obj_files,
    simulate_2_5d_surface, extract_descriptors, extract_features, FEATURE_NAMES,
)

data_dir = Path("data/experience_rock")
grid_res = 0.5

print("Loading and processing all samples...")
file_list = find_obj_files(data_dir)

all_descs = []
all_V_true = []
all_ratios = []
all_ids = []

for sample_id, group, obj_path in file_list:
    try:
        verts, faces = load_obj_simple(obj_path)
        V_true = compute_mesh_volume(verts, faces)
        if V_true <= 0:
            continue
        surface = simulate_2_5d_surface(verts, faces, grid_res)
        if surface is None:
            continue
        desc = extract_descriptors(surface)
        if desc is None:
            continue
        all_descs.append(desc)
        all_V_true.append(V_true)
        all_ratios.append(V_true / desc["V_2_5d"])
        all_ids.append(sample_id)
    except Exception as e:
        pass

all_V_true = np.array(all_V_true)
all_ratios = np.array(all_ratios)
X = np.array([extract_features(d) for d in all_descs])

print(f"\nTotal samples: {len(all_descs)}")
print(f"Ratio r = V_true/V_2.5d:")
print(f"  mean={np.mean(all_ratios):.4f}, std={np.std(all_ratios):.4f}")
print(f"  median={np.median(all_ratios):.4f}")
print(f"  range=[{np.min(all_ratios):.4f}, {np.max(all_ratios):.4f}]")
print(f"  CV (std/mean) = {np.std(all_ratios)/np.mean(all_ratios)*100:.1f}%")

print(f"\n{'='*70}")
print(f"  Feature - Ratio Correlation Analysis")
print(f"{'='*70}")
print(f"{'Feature':<20s} {'Pearson r':>12s} {'Spearman ρ':>12s} {'Abs corr':>10s}")
print(f"{'-'*70}")

from scipy import stats

correlations = []
for i, name in enumerate(FEATURE_NAMES):
    feat = X[:, i]
    if np.std(feat) < 1e-12:
        continue
    pearson_r, _ = stats.pearsonr(feat, all_ratios)
    spearman_r, _ = stats.spearmanr(feat, all_ratios)
    correlations.append((name, pearson_r, spearman_r, abs(pearson_r)))

correlations.sort(key=lambda x: -x[3])
for name, pr, sr, absr in correlations:
    print(f"{name:<20s} {pr:>+12.4f} {sr:>+12.4f} {absr:>10.4f}")

# Multi-variate: R² of linear regression
print(f"\n{'='*70}")
print(f"  Multivariate Linear Regression (all 12 features)")
print(f"{'='*70}")

from numpy.linalg import lstsq
X_with_bias = np.column_stack([np.ones(len(X)), X])
coeffs, residuals, rank, sv = lstsq(X_with_bias, all_ratios, rcond=None)
y_pred = X_with_bias @ coeffs
ss_res = np.sum((all_ratios - y_pred) ** 2)
ss_tot = np.sum((all_ratios - np.mean(all_ratios)) ** 2)
r2_linear = 1 - ss_res / ss_tot
print(f"  R² (linear model, all 12 features) = {r2_linear:.4f}")
print(f"  This means shape features explain {r2_linear*100:.1f}% of ratio variance")
print(f"  Remaining {100-r2_linear*100:.1f}% is noise / unexplained by shape")

# What's the best possible MAPE improvement?
# If we could predict r perfectly, error would be 0 for ratio prediction
# But volume error = V_pred/V_true - 1 = (r_pred * V_2.5d) / (r_true * V_2.5d) - 1 = r_pred/r_true - 1
# So volume MAPE ≈ mean(|r_pred/r_true - 1|) * 100%
# If r_pred is perfect: MAPE = 0%
# If r_pred = mean(r): MAPE = mean(|mean(r)/r_true - 1|) * 100% = linear correction error
best_mape = 0.0  # perfect prediction
linear_mape = np.mean(np.abs(np.mean(all_ratios)/all_ratios - 1)) * 100
print(f"\n  Best possible MAPE (perfect r prediction): {best_mape:.2f}%")
print(f"  Linear correction MAPE (constant α): {linear_mape:.2f}%")
print(f"  Maximum possible improvement: {linear_mape - best_mape:.2f}%")
print(f"  But with R²={r2_linear:.2f}, realistic improvement: ~{linear_mape * (1 - np.sqrt(r2_linear)):.2f}%")

# Top features only
print(f"\n{'='*70}")
print(f"  Top 5 features only (linear model)")
print(f"{'='*70}")
top5_idx = [FEATURE_NAMES.index(c[0]) for c in correlations[:5]]
X_top5 = X[:, top5_idx]
X_top5_bias = np.column_stack([np.ones(len(X_top5)), X_top5])
coeffs5, _, _, _ = lstsq(X_top5_bias, all_ratios, rcond=None)
y_pred5 = X_top5_bias @ coeffs5
ss_res5 = np.sum((all_ratios - y_pred5) ** 2)
r2_top5 = 1 - ss_res5 / ss_tot
print(f"  Top 5 features: {[c[0] for c in correlations[:5]]}")
print(f"  R² = {r2_top5:.4f} ({r2_top5*100:.1f}% variance explained)")

print(f"\n{'='*70}")
print(f"  Conclusion")
print(f"{'='*70}")
print(f"  The correction ratio r has CV = {np.std(all_ratios)/np.mean(all_ratios)*100:.1f}%")
print(f"  Shape features explain only {r2_linear*100:.1f}% of r's variance")
print(f"  This puts a ceiling on shape-aware improvement")
print(f"  ")
print(f"  Ways to increase shape-aware value:")
print(f"  1. Use more diverse data (wider r distribution)")
print(f"  2. Add richer shape features (e.g. Fourier descriptors)")
print(f"  3. Frame as 'transferable correction' not just accuracy")
print(f"  4. Use data augmentation to increase sample diversity")

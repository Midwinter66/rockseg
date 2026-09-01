# Final Research Status

> Status date: 2026-08-26. This document consolidates frozen outputs only. It does not authorize retraining, resampling, production-code changes, or additional mine-wide processing.

## 1. Research Question

Can physical-scale-aware DOM segmentation, resolution-matched 2.5D surface reconstruction, and a canonical shape-aware correction model provide a reproducible route from real-mine rock instances to volume estimates suitable for subsequent fragmentation analysis?

## 2. Final Pipeline

`UAV DOM -> physical-scale analysis -> multi-scale instance segmentation -> fusion/deduplication -> 2D-3D association -> ground/background separation -> 10 mm ground-referenced 2.5D surface -> canonical 12 features -> Shape-Aware LightGBM -> V_pred = V_2.5D x y_pred -> stratified volume analysis -> PSD/P80`.

## 3. External Mesh Validation

The external validation dataset contains T01 (`79`) and L01 (`386`) OBJ meshes, for `465` objects. The 0.5 mm Shape-Aware V2 result is retained as an external mesh methodological validation; it is not used directly on real-mine point clouds.

## 4. Resolution Matching

DOM resolution is `10 mm/pixel`. Pointcloud2 local XY spacing has P90 `6.0-6.4 mm`; local 3D spacing P90 is approximately `8.5-8.6 mm`. The real-mine 2.5D grid is therefore frozen at `10 mm = 0.01 m`. The original external OBJ meshes were too small for direct 10 mm rasterization and produced predominantly empty surfaces; that 63-sample path is not used for training.

## 5. Scale Adaptation

External OBJ and mine-footprint audits produced a pre-specified uniform scale factor of `82.737840`. The 20-object pilot completed `20/20` valid 10 mm surfaces with finite canonical features and no abnormal y-ratio observations. The adaptation is accepted for this operational methodology, while remaining a geometric-similarity/domain-adaptation assumption rather than independent proof of lithologic equivalence.

## 6. Shape-Aware V2 Model

The final training set is `research_v2/volume_validation/datasets/t01_l01_scaled_10mm/`: all `465/465` objects passed QC with a group-aware `326/70/69` train/validation/test split. The frozen model is:

`research_v2/volume_validation/output_v2_scaled_10mm/shape_aware_model_v2_scaled_10mm.txt`

It uses the ordered canonical schema:

`C, AR, solidity, compactness, eq_diam_ratio, H_mean_norm, H_std_norm, H_p25_norm, H_p75_norm, H_skew_norm, fill_ratio, ellipsoid_ratio`.

`H_skew_norm` remains the unnormalised `H_skew`. Training and production adapters agree exactly on a five-object consistency check.

The target is `y_ratio = V_true / V_2.5D`; prediction is `V_pred = V_2.5D x y_pred`. Best iteration is `356`.

## 7. Real-Mine Pilot

The frozen model, 10 mm grid, and canonical feature schema completed the DOM2 12-rock pilot successfully (`12/12`). Predicted correction factors ranged from `0.6769` to `0.6939`; `V_2.5D` ranged from `0.01018` to `0.30359 m3`; corrected estimates ranged from `0.00692` to `0.20857 m3`. No per-rock DOM2 reference volume was available.

## 8. 4,000-Rock Volume Estimation

The DOM2 inventory contains `76,407` final instances. The existing association accepted `69,911` and rejected `6,496`. A frozen `stratified_quantile_systematic` sample selected `4,000` accepted rocks solely from DOM-footprint equivalent diameter: S1 `400`, S2 `600`, S3 `1,000`, S4 `1,000`, S5 `600`, S6 `400`.

The representative application completed with `3,639` successes and `361` failures (`90.98%` pipeline success rate). Every failure was `empty_2_5d_surface`; success increased with diameter: S1 `84.0%`, S2 `86.3%`, S3 `87.5%`, S4 `92.3%`, S5 `98.0%`, S6 `99.75%`. All successful records had finite 12-feature inputs, positive volumes, exact prediction-formula agreement, and no model-input mismatch.

## 9. Current Quantitative Results

| Result | Frozen value |
| --- | --- |
| External scaled-10mm test sample count | 69 |
| Ratio MAE / RMSE / MAPE / R2 | 0.0374 / 0.0451 / 5.82% / 0.3280 |
| Raw 2.5D external volume MAPE / R2 | 54.24% / 0.1895 |
| Constant correction external volume MAPE / R2 | 6.99% / 0.9705 |
| Shape-Aware external volume MAE / RMSE / MAPE / R2 | 7,167,711 mm3 / 11,890,780 mm3 / 5.82% / 0.9838 |
| DOM2 sampled rocks processed | 4,000 |
| DOM2 successful pipeline records | 3,639 / 4,000 (90.98%) |
| DOM2 correction factor range / median | 0.5541-0.7280 / 0.6797 |
| DOM2 V_2.5D range | 2.99e-09-4.3480 m3 |
| DOM2 V_pred range | 1.99e-09-2.9962 m3 |

## 10. Scientific Interpretation

The completed chain demonstrates a resolution-matched, reproducible application path from DOM2 accepted instances through 2D-3D association, 10 mm 2.5D reconstruction, canonical shape features, and frozen-model volume correction. The external scaled test supports the correction module, while the real-mine run establishes operational applicability on a stratified representative sample. Smaller rocks fail more often because a valid 10 mm surface is less often available; that signal is retained rather than masked.

## 11. Limitations

1. Per-rock real-mine ground-truth volumes are unavailable, so mine absolute volume accuracy is not independently validated.
2. The external-to-mine scale factor relies on a geometric-similarity assumption.
3. The model's 5.82% MAPE belongs only to the scaled external held-out test.
4. Empty 2.5D surfaces create size-dependent non-completion, especially in S1-S3.
5. PSD/P80 has not yet been calculated from the frozen results and requires a predefined statistical protocol.

## 12. What Can Be Claimed

- Shape-Aware V2 improved over raw 2.5D on the held-out scaled external mesh test.
- The frozen 10 mm pipeline completed successfully for 3,639 of 4,000 size-stratified accepted DOM2 rocks.
- All successful real-mine records passed feature, positivity, and prediction-formula QC.
- Empty-surface failures are explicitly reported and are more frequent in smaller size strata.

## 13. What Cannot Be Claimed

- A `5.82%` real-mine volume error.
- A `90.98%` real-mine volume accuracy.
- Validation of all `69,911` accepted DOM2 rocks; the run covers a 4,000-rock representative sample only.
- PSD/P80 accuracy or field fragmentation accuracy before their dedicated analysis and validation.

## 14. Next Step: Paper Consolidation

1. Create manuscript-ready tables from the frozen external and real-mine results.
2. Draft Methods around physical scale, resolution matching, scale adaptation, and canonical correction.
3. Define and run the PSD/P80 analysis on the frozen volume outputs without resampling.
4. Design final figures after claims and tables are fixed.
5. Complete manuscript writing with a focused uncertainty and reviewer-risk check.

## Result Provenance

- Model metrics: `research_v2/volume_validation/output_v2_scaled_10mm/training_results_v2_scaled_10mm.json`
- Model metadata: `research_v2/volume_validation/output_v2_scaled_10mm/model_meta_v2_scaled_10mm.json`
- Feature consistency: `research_v2/volume_validation/feature_consistency_check.json`
- Point spacing: `research_v2/volume_validation/resolution_study/pointcloud_spacing_results.json`
- Sampling: `research_v2/volume_validation/real_mine_sampling/real_mine_sampling_report.md`
- Real-mine summary: `research_v2/volume_validation/real_mine_full/real_mine_volume_4000_summary.json`

# RockSeg Research Overview

> 快速查看版。状态：2026-08-26，所有数值均来自已冻结的结果文件。本文不代表新的实验、训练、重算或全矿区处理。

## 1. One-Line Summary

RockSeg 将物理尺度驱动的多尺度 DOM 实例分割，与已有 2D-3D 关联、10 mm 地面参考 2.5D 表面和 Shape-Aware V2 体积校正连接为一条可复现流程，并已在 DOM2 的 4,000 个分层代表性石块上完成应用。

## 2. Complete Workflow

```text
UAV DOM (10 mm/pixel)
  -> physical-scale-driven multi-scale segmentation
  -> within-scale fusion + cascade deduplication
  -> final DOM2 rock inventory
  -> existing 2D-3D association / accepted-instance filtering
  -> ground/background separation
  -> 10 mm ground-referenced 2.5D top surface
  -> canonical 12 shape features
  -> frozen scaled-10mm Shape-Aware LightGBM
  -> V_pred = V_2.5D x y_pred
  -> size-stratified representative volume results
  -> future PSD / P80 analysis
```

## 3. Data and Evidence Roles

| Data / output | Size | Role | Status |
| --- | ---: | --- | --- |
| DOM2 | 10 mm/pixel | Real-mine segmentation and application domain | FROZEN |
| Pointcloud2 | 146,721,392 points | DOM2 2D-3D association and height information | FROZEN |
| T01 external OBJ | 79 meshes | External mesh methodology evidence | FROZEN |
| L01 external OBJ | 386 meshes | External mesh methodology evidence | FROZEN |
| Scaled 10 mm dataset | 465 / 465 valid | Resolution-matched final model training | FROZEN |
| DOM2 final instances | 76,407 | Multi-scale fusion / deduplication output | FROZEN |
| DOM2 accepted instances | 69,911 | Eligible population for mine application | FROZEN |
| DOM2 representative manifest | 4,000 | Deterministic size-stratified application sample | FROZEN |

## 4. Resolution and Scale Decisions

| Decision | Frozen value | Evidence / reason |
| --- | --- | --- |
| Mine 2.5D grid | `0.01 m` (10 mm) | DOM is 10 mm/pixel; pointcloud2 XY P90 is 6.0-6.4 mm; 3D P90 is about 8.5-8.6 mm. |
| External 0.5 mm V2 | External validation only | It is not resolution-matched to the mine and is not applied to DOM2. |
| Original OBJ at 10 mm | Rejected for training | Only 63 of 465 objects formed valid surfaces; most failed as empty surfaces. |
| External-to-mine scale factor | `82.737840` | Pre-specified from independent footprint statistics; 20-object pilot passed. |
| Final model training grid | 10 mm after scaling | Matches the frozen mine application grid. |

## 5. Canonical Shape-Aware V2 Definition

The feature order is fixed and identical in training and mine inference:

`C, AR, solidity, compactness, eq_diam_ratio, H_mean_norm, H_std_norm, H_p25_norm, H_p75_norm, H_skew_norm, fill_ratio, ellipsoid_ratio`

- Target: `y_ratio = V_true / V_2.5D`.
- Prediction: `V_pred = V_2.5D x y_pred`.
- `H_skew_norm` is the original `H_skew`, not `H_skew / H`.
- Training and production adapters passed the five-object consistency check with maximum absolute difference `0`.

## 6. Experiment Chain

| Step | Question | Frozen result |
| --- | --- | --- |
| External OBJ V2 | Can shape-aware correction improve external 2.5D volume? | 0.5 mm external V2 validation completed; retained as methodology evidence. |
| Resolution benchmark | Can original OBJ transfer directly to 10 mm? | No; strong feature drift at 5-10 mm and predominantly empty surfaces at coarser grids. |
| Point spacing audit | What mine grid is justified? | 10 mm selected. |
| Scale audit | Can external geometry be adapted to mine footprint scale? | Plausible but not independently proven; uniform factor fixed. |
| 20-object pilot | Does scaled geometry form valid 10 mm surfaces? | `20/20` successful. |
| Full dataset QC | Is scaled 10 mm training data valid? | `465/465` successful; split `326/70/69`; no leakage, NaN, or Inf. |
| Final model | Does correction improve held-out scaled external data? | Shape-Aware external Test MAPE `5.82%`, R2 `0.9838`. |
| 12-rock mine pilot | Can the full mine inference chain run? | `12/12` successful. |
| 4,000-rock application | Can the frozen pipeline operate on a representative mine sample? | `3,639/4,000` successful; `FINAL_QC_PASS`. |

## 7. Model Results: External Scaled 10 mm Test

| Method | MAE (mm3) | RMSE (mm3) | MAPE | R2 |
| --- | ---: | ---: | ---: | ---: |
| Raw 2.5D | 59,903,925 | 84,139,845 | 54.24% | 0.1895 |
| Constant correction | 8,723,245 | 16,049,133 | 6.99% | 0.9705 |
| Shape-Aware V2 | 7,167,711 | 11,890,780 | 5.82% | 0.9838 |

Final model: `shape_aware_model_v2_scaled_10mm.txt`; best iteration `356`; external held-out Test count `69`.

## 8. DOM2 Volume Application Results

### Overall

| Item | Value |
| --- | ---: |
| Accepted population | 69,911 |
| Frozen sample | 4,000 |
| Successful inference records | 3,639 |
| Failed records | 361 |
| Pipeline success rate | 90.98% |
| Failure type | `empty_2_5d_surface` only |
| Feature count | 12 |
| Non-finite features / predictions among successes | 0 |
| Formula check | `V_pred = V_2.5D x y_pred`, maximum difference 0 |

### Size-Stratified Completion

| Stratum | Diameter percentile | Sample | Success | Failure | Success rate |
| --- | --- | ---: | ---: | ---: | ---: |
| S1 | P0-P10 | 400 | 336 | 64 | 84.00% |
| S2 | P10-P25 | 600 | 518 | 82 | 86.33% |
| S3 | P25-P50 | 1,000 | 875 | 125 | 87.50% |
| S4 | P50-P75 | 1,000 | 923 | 77 | 92.30% |
| S5 | P75-P90 | 600 | 588 | 12 | 98.00% |
| S6 | P90-P100 | 400 | 399 | 1 | 99.75% |

### Successful-Record Output Ranges

| Quantity | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| `y_pred` | 0.5541 | 0.6797 | 0.7280 |
| `V_2.5D` (m3) | 2.99e-09 | 0.001593 | 4.3480 |
| `V_pred` (m3) | 1.99e-09 | 0.001031 | 2.9962 |

## 9. Interpretation and Boundaries

**Supported:** the external scaled test supports the 10 mm Shape-Aware correction module; the DOM2 run demonstrates a complete, QC-passing inference path for a deterministic representative sample; lower completion in small strata is an observed 10 mm surface-availability limitation.

**Not supported:** a 5.82% real-mine volume error; a 90.98% real-mine volume accuracy; validation of all 69,911 accepted instances; PSD/P80 results before their separate analysis.

## 10. Frozen Result Locations

| Artifact | Location |
| --- | --- |
| Final status | `research_v2/FINAL_RESEARCH_STATUS.md` |
| Model metrics | `research_v2/volume_validation/output_v2_scaled_10mm/training_results_v2_scaled_10mm.json` |
| Model metadata | `research_v2/volume_validation/output_v2_scaled_10mm/model_meta_v2_scaled_10mm.json` |
| Feature consistency | `research_v2/volume_validation/feature_consistency_check.json` |
| Point spacing | `research_v2/volume_validation/resolution_study/pointcloud_spacing_results.json` |
| Sampling report | `research_v2/volume_validation/real_mine_sampling/real_mine_sampling_report.md` |
| 4,000-rock report | `research_v2/volume_validation/real_mine_full/real_mine_volume_4000_report.md` |
| 4,000-rock summary | `research_v2/volume_validation/real_mine_full/real_mine_volume_4000_summary.json` |

## 11. Immediate Use

Use this document for project review, team handoff, and selecting frozen numbers for manuscript tables. The next analysis is PSD/P80 design and result consolidation, without altering the completed model, manifest, or 4,000-rock output.

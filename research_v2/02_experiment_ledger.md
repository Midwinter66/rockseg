# Experiment Ledger

> **FROZEN LEDGER SUMMARY -- 2026-08-26.** Experiments V2-01 to V2-09 are complete. The legacy experiment queue below is retained as **SUPERSEDED / ARCHIVED** provenance.

| Experiment | Status | Frozen outcome |
| --- | --- | --- |
| V2-01 | COMPLETE | 0.5 mm external OBJ methodology validation; retained external-only. |
| V2-02 | COMPLETE | Resolution study; original-scale 10 mm OBJ training rejected. |
| V2-03 | COMPLETE | Point-cloud spacing supports 10 mm mine grid. |
| V2-04 | COMPLETE | Scale audit; factor 82.737840 fixed. |
| V2-05 | COMPLETE | Scaled 10 mm pilot, 20/20 valid. |
| V2-06 | COMPLETE | Scaled 10 mm dataset, 465/465 valid, split 326/70/69. |
| V2-07 | COMPLETE | Frozen model, best iteration 356, external test MAPE 5.82%. |
| V2-08 | COMPLETE | DOM2 12-rock transfer pilot, 12/12 success. |
| V2-09 | COMPLETE | DOM2 4,000-rock stratified application, 3,639 success / 361 empty-surface failures; FINAL_QC_PASS. |

## Historical Experiment Queue (SUPERSEDED; retained for provenance)

This file is the queue and log for V2 experiments. Add one entry before
promoting any result to the manuscript evidence package.

## Status Labels

- `planned`: design exists, not run.
- `running`: currently executing.
- `complete`: outputs and checks are recorded.
- `blocked`: cannot proceed without missing data or a decision.
- `archived`: historical or superseded.

## E0. Baseline Freeze

| Field | Content |
|---|---|
| Status | planned |
| Purpose | Establish one authoritative historical baseline before V2 changes |
| Inputs | Existing configs, result snapshots, run manifests, README/PRD claims |
| Outputs | Baseline freeze note, source-to-number table, accepted historical baseline |
| Gate | One coherent command sequence and one coherent result snapshot |

## E1. Particle-Size And Scale Selection

| Field | Content |
|---|---|
| Status | planned |
| Purpose | Determine physical scales and network input size from actual rock-size distribution |
| Inputs | Manual DOM annotations, candidate input sizes (960, 1024), DOM GSD (0.01 m/px) |
| Outputs | Diameter histogram, pixel-size mapping table, selected scale set, selected input size, overlap candidate |
| Gate | Final scale set, input size, and overlap recorded in decision register (D01, D15, D03) |

Steps:

1. Statistic real rock size distribution (D_physical).
2. Compute pixel size for each rock: D_pixel = D_physical / GSD.
3. Test segmentation performance at different input sizes.
4. Determine the pixel size at which segmentation accuracy stabilizes.
5. Map physical rock size to optimal image scale.
6. Select Fine / Medium / Coarse scales and network input size.

## E2. Multi-Scale Segmentation

| Field | Content |
|---|---|
| Status | planned |
| Purpose | Compare single-, two-, and three-scale segmentation |
| Inputs | Fixed YOLO11m-seg, annotated windows, physical tile manifests |
| Outputs | Mask AP, Recall, Precision, F1, per-size-bin metrics |
| Gate | Selected scale set improves relevant size bins without unacceptable cost |

Comparison: Single-scale vs. Multi-scale, with metrics reported per size bin
(small, medium, large).

## E3. Boundary And Cross-Scale Fusion

| Field | Content |
|---|---|
| Status | planned |
| Purpose | Evaluate duplicate removal and wrong-merge control using multi-feature fusion score |
| Inputs | Candidate masks from E2, fusion-review annotations |
| Outputs | Duplicate rate, boundary recall, over-merge rate, under-merge rate, instance count error, mask AP |
| Gate | Hierarchical fusion selected or rejected with evidence |

Fusion score:

```text
S = w1*IoU + w2*Sc + w3*SA + w4*Sb
```

Thresholds: tau_boundary (within-scale), tau_match (cross-scale).

## E4. 2D-3D Association

| Field | Content |
|---|---|
| Status | planned |
| Purpose | Validate that DOM masks retrieve intended point-cloud objects after coordinate unification and ground removal |
| Inputs | Final masks, LAZ point cloud, manual 2D-3D review set, ground removal method |
| Outputs | Point precision, point recall, 3D IoU, failure taxonomy, ground-removal QC |
| Gate | Association quality and limitations are auditable |

Ground removal candidates: RANSAC plane, local ground surface, DEM,
morphological filtering.

## E5. External Volume Validation

| Field | Content |
|---|---|
| Status | planned |
| Purpose | Validate 2.5D-to-volume estimation independently with ablation |
| Inputs | 868 external 3D rock objects, reference volumes, 2.5D simulation settings |
| Outputs | MAE, RMSE, MAPE/SMAPE, median relative error, R2, per-method ablation table |
| Gate | Volume ablation: bounding box vs. ellipsoid vs. shape-aware, compared on held-out data |

Split: 70% train / 15% validation / 15% test, group-aware by rock
source/batch.

Ablation:

| Method | Volume |
|---|---|
| Bounding box | V_box = LWH |
| Ellipsoid | V_ellipsoid = (pi/6) * LWH |
| Shape-aware (LightGBM) | V_shape |

Acceptance: E_shape < E_box AND E_shape < E_ellipsoid on held-out test data.

## E6. Full-Scene Fragmentation Application

| Field | Content |
|---|---|
| Status | planned |
| Purpose | Run final V2 pipeline on the main scene and compute P80 |
| Inputs | Frozen DOM-side config, validated association/volume settings |
| Outputs | Rock count, volume distribution, equivalent diameter distribution, volume-weighted P80 |
| Gate | All final numbers trace to one frozen run; P80 uses volume weighting |

End-to-end comparison:

| Configuration | Output |
|---|---|
| Single-scale | P80_single |
| Multi-scale | P80_multi |
| Multi-scale + fusion | P80_fused |
| Multi-scale + fusion + 2D-3D | P80_3d |
| Multi-scale + fusion + 2D-3D + shape-aware volume | P80_final |

# Paper Tables

All values below are transcribed from frozen reports, metadata, or direct
inventory metadata. Units are stated where available. No new computation or
experiment was performed for this document.

## Table 1. Study Data and Frozen Analysis Assets

| Item | Value | Provenance |
| --- | ---: | --- |
| DOM2 resolution | 0.01 m/pixel | `rockseg/config.py` |
| Coarse tile coverage / tile count | 10.24 m / 297 | DOM2 discovery report |
| Medium tile coverage / tile count | 5.12 m / 1,188 | DOM2 discovery report |
| Fine tile coverage / tile count | 2.56 m / 4,708 | DOM2 discovery report |
| External T01 meshes | 79 | scaled dataset metadata |
| External L01 meshes | 386 | scaled dataset metadata |
| External mesh total | 465 | scaled dataset metadata |
| Scaled dataset split | 326 train / 70 validation / 69 test | training results |
| Mine accepted population | 69,911 | sampling report |
| Mine representative manifest | 4,000 | sampling report |

## Table 2. DOM2 Multi-Scale Segmentation and Fusion

| Stage | Coarse | Medium | Fine | Total |
| --- | ---: | ---: | ---: | ---: |
| Raw detections | 37,470 | 101,642 | 179,286 | 318,398 |
| Same-scale fused pool | NOT VERIFIED per scale | NOT VERIFIED per scale | NOT VERIFIED per scale | 112,983 |
| Final cascade instances | 5,925 | 10,890 | 59,592 | 76,407 |

The raw and same-scale-fused aggregate counts are recorded in
`real_mine_full/discovery_report.md`. The final per-scale count is the direct
`scale_level` inventory count in `rock_instances.json`. Per-scale counts after
same-scale fusion were not located in frozen metadata and are intentionally not
inferred.

## Table 3. 2D-3D Association and Quality Screening

| Item | Value |
| --- | ---: |
| Final DOM instances | 76,407 |
| Accepted | 69,911 |
| Rejected | 6,496 |
| Accepted rate | 91.50% |
| Minimum point count | 60 |
| Minimum z-range | 0.18 m |
| Minimum P90 height above ground | 0.12 m |
| Elevated-height threshold | 0.08 m |
| Minimum elevated ratio | 0.20 |
| GroundDEM grid / percentile / subsampling | 0.5 m / P5 / every 100th point |

| Rejection reason | Recorded count |
| --- | ---: |
| `too_few_points` | 184 |
| `insufficient_p90_height` | 5,207 |
| `insufficient_elevated_ratio` | 5,847 |
| `insufficient_z_range` | 357 |
| `insufficient_ground_points` | 3 |

Reasons may co-occur for a rejected record; their counts are not expected to
sum to the rejected total.

## Table 4. External OBJ Dataset and Resolution-Matching Evidence

| Item | Value |
| --- | ---: |
| Dataset B | T01 79 + L01 386 = 465 OBJ meshes |
| Original external rasterization | 0.5 mm |
| Original-scale OBJ valid surfaces at 10 mm | 63 / 465 |
| Frozen mine 2.5D grid | 10 mm = 0.01 m |
| Uniform external-mesh scale factor | 82.737840 |
| Scale audit conclusion | `SCALE_ADAPTATION_PLAUSIBLE_BUT_NOT_PROVEN` |
| Scaled 10 mm pilot | 20 / 20 valid surfaces |
| Pilot occupied cells | 1,634 to 51,202 |
| Full scaled 10 mm dataset | 465 / 465 successful |
| Full scaled dataset non-finite features | 0 |

Point-cloud spacing audit: BlockB XY P90 = 6.00 mm and 3D P90 = 8.54 mm;
BlockY XY P90 = 6.40 mm and 3D P90 = 8.60 mm. These sampled spacing statistics
support resolution selection, not per-rock density guarantees.

## Table 5. Canonical Shape-Aware V2 Descriptor

| Order | Feature | Definition |
| ---: | --- | --- |
| 1 | `C` | `min(4*pi*A/P^2, 1)` |
| 2 | `AR` | `L/W` |
| 3 | `solidity` | `min(A/A_convex, 1)` |
| 4 | `compactness` | `P/sqrt(A)` |
| 5 | `eq_diam_ratio` | `sqrt(4*A/pi)/L` |
| 6 | `H_mean_norm` | `H_mean/H` |
| 7 | `H_std_norm` | `H_std/H` |
| 8 | `H_p25_norm` | `H_p25/H` |
| 9 | `H_p75_norm` | `H_p75/H` |
| 10 | `H_skew_norm` | `H_skew` (not divided by `H`) |
| 11 | `fill_ratio` | `V_2.5D/V_box` |
| 12 | `ellipsoid_ratio` | `V_2.5D/V_ellipsoid` |

Here `A`, `P`, `L`, `W`, and `H` denote footprint area, perimeter, length,
width, and height, respectively. The train/inference feature consistency check
reported maximum absolute and relative differences of zero on five checked
samples.

## Table 6. Frozen LightGBM Configuration and External Test

| Parameter / result | Value |
| --- | ---: |
| Target | `y_ratio = V_true / V_2.5D` |
| Prediction | `V_pred = V_2.5D * y_pred` |
| Learning rate | 0.02 |
| Number of leaves | 12 |
| Minimum child samples | 8 |
| Subsample / frequency | 0.9 / 1 |
| Column sample by tree | 0.9 |
| L1 / L2 regularization | 0.2 / 0.5 |
| Minimum gain to split | 0.0001 |
| Maximum rounds / early stopping | 500 / 100 |
| Seed | 42 |
| Best iteration | 356 |
| Test objects | 69 |
| Ratio MAE / RMSE / MAPE / R2 | 0.0374 / 0.0451 / 5.82% / 0.3280 |

## Table 7. External Scaled-10 mm True-Volume Test Comparison

| Method | MAE (mm3) | RMSE (mm3) | MAPE | R2 |
| --- | ---: | ---: | ---: | ---: |
| Raw 2.5D | 59,903,925 | 84,139,845 | 54.24% | 0.1895 |
| Constant correction | 8,723,245 | 16,049,133 | 6.99% | 0.9705 |
| Shape-Aware V2 | 7,167,711 | 11,890,780 | 5.82% | 0.9838 |

This is a held-out scaled external OBJ test. It is not a real-mine accuracy
table.

## Table 8. Real-Mine Deterministic Sample Design

| Stratum | Diameter interval | Population | Frozen sample |
| --- | --- | ---: | ---: |
| S1 | [P0, P10) | 6,950 | 400 |
| S2 | [P10, P25) | 10,523 | 600 |
| S3 | [P25, P50) | 17,465 | 1,000 |
| S4 | [P50, P75) | 17,488 | 1,000 |
| S5 | [P75, P90) | 10,493 | 600 |
| S6 | [P90, P100] | 6,992 | 400 |
| Total | Accepted population | 69,911 | 4,000 |

| Equivalent-diameter quantile | m |
| --- | ---: |
| Min | 0.022567583 |
| P10 | 0.090270333 |
| P25 | 0.113400703 |
| P50 | 0.161952878 |
| P75 | 0.257062507 |
| P90 | 0.430118246 |
| Max | 3.449593713 |

Sampling was `stratified_quantile_systematic`: ascending equivalent diameter,
then `rock_id`, followed by fixed systematic positions. It used no random
generator, volume, target, prediction, or error field.

## Table 9. Frozen 4,000-Rock Mine Application QC

| Metric | Value |
| --- | ---: |
| Manifest / result records | 4,000 / 4,000 |
| Unique rock IDs | 4,000 |
| Successful estimates | 3,639 |
| Failed estimates | 361 |
| Pipeline completion rate | 90.98% |
| Failure reason | `empty_2_5d_surface` only |
| Non-finite features, successful records | 0 |
| Non-finite prediction/volume, successful records | 0 |
| Non-positive `V_2.5D`, successful records | 0 |
| Non-positive `V_pred`, successful records | 0 |
| Maximum formula difference | 0.0 |

| Stratum | Sample | Success | Failure | Completion rate |
| --- | ---: | ---: | ---: | ---: |
| S1 | 400 | 336 | 64 | 84.0% |
| S2 | 600 | 518 | 82 | 86.3% |
| S3 | 1,000 | 875 | 125 | 87.5% |
| S4 | 1,000 | 923 | 77 | 92.3% |
| S5 | 600 | 588 | 12 | 98.0% |
| S6 | 400 | 399 | 1 | 99.75% |

## Table 10. Successful Real-Mine Estimate Distribution

| Quantity | n | Min | P25 | Median | P75 | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `V_2.5D` (m3) | 3,639 | 2.99e-09 | 0.0003706 | 0.0015928 | 0.0064182 | 0.0271358 | 4.3480 |
| `y_pred` | 3,639 | 0.5541 | 0.6231 | 0.6797 | 0.6889 | 0.6974 | 0.7280 |
| `V_pred` (m3) | 3,639 | 1.99e-09 | 0.0002463 | 0.0010305 | 0.0041323 | 0.0182448 | 2.9962 |

The 4,000-rock run is a representative DOM2 application. Per-rock real-mine
ground-truth volumes are unavailable, so this table does not quantify mine
absolute volume accuracy.

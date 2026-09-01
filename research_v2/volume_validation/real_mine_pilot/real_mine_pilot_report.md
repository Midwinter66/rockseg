# Real Mine Shape-Aware V2 Pilot

## Scope

Frozen scaled-10 mm LightGBM model applied to the same 12 existing Mine Site B single-rock associations. No retraining, Dataset B modification, model modification, production-code modification, or image generation was performed.

## Interface and units

- Model: shape_aware_model_v2_scaled_10mm.txt
- Feature count/order: 12, C, AR, solidity, compactness, eq_diam_ratio, H_mean_norm, H_std_norm, H_p25_norm, H_p75_norm, H_skew_norm, fill_ratio, ellipsoid_ratio
- Grid: 0.01 m (10 mm)
- XY/Z coordinates: metres; volumes: m3
- Ground removal: existing GroundDEM at 0.500 m with non-negative height clamp
- Coordinate transform: configured Site B local translation applied to the DOM rasterio transform
- Association: P1 accepted stones with existing 3D screening passed

## Selection

- Available accepted associations: 8538
- Selected: 12
- Rule: sort by existing fused equivalent diameter and take 12 deterministic positions
- Selected IDs: stone_006809, stone_007326, stone_001114, stone_002781, stone_000604, stone_003525, stone_007077, stone_004344, stone_008883, stone_007635, stone_004380, stone_005149

## Results

| Item | Value |
|---|---:|
| Selected / PASS / FAIL | 12 / 12 / 0 |
| y_pred min / median / max | 0.676942 / 0.687087 / 0.693914 |
| V_2.5D min / median / max (m3) | 0.0101817 / 0.0642484 / 0.303588 |
| V_pred min / median / max (m3) | 0.0069209 / 0.044205 / 0.20857 |
| occupied cells min / median / max | 1714 / 2654.5 / 15647 |
| formula max absolute difference | 0 |

## Per-sample status

| sample_id | footprint diameter (m) | points | occupied cells | V_2.5D (m3) | y_pred | V_pred (m3) | status |
|---|---:|---:|---:|---:|---:|---:|---|
| stone_006809 | 0.5046 | 7294 | 1715 | 0.0576203 | 0.689284 | 0.0397168 | PASS |
| stone_007326 | 0.5271 | 3335 | 1714 | 0.0101817 | 0.679742 | 0.0069209 | PASS |
| stone_001114 | 0.5492 | 4454 | 1851 | 0.0327804 | 0.689185 | 0.0225917 | PASS |
| stone_002781 | 0.5761 | 4647 | 2090 | 0.0236326 | 0.687158 | 0.0162394 | PASS |
| stone_000604 | 0.6047 | 5803 | 2453 | 0.0402325 | 0.689063 | 0.0277227 | PASS |
| stone_003525 | 0.6389 | 6772 | 2572 | 0.0145395 | 0.676942 | 0.00984241 | PASS |
| stone_007077 | 0.6795 | 7976 | 2737 | 0.117074 | 0.678257 | 0.0794064 | PASS |
| stone_004344 | 0.7332 | 9636 | 3398 | 0.0708765 | 0.687015 | 0.0486933 | PASS |
| stone_008883 | 0.7937 | 10647 | 4133 | 0.175917 | 0.677127 | 0.119118 | PASS |
| stone_007635 | 0.8933 | 15065 | 4844 | 0.185949 | 0.687355 | 0.127813 | PASS |
| stone_004380 | 1.0574 | 25550 | 7112 | 0.221776 | 0.693914 | 0.153893 | PASS |
| stone_005149 | 1.609 | 40768 | 15647 | 0.303588 | 0.687015 | 0.20857 | PASS |

## Interpretation

The frozen model and canonical 12-feature adapter were executed at 10 mm. Without independent real-mine single-rock ground truth, these are transfer/interface estimates only and do not establish real-mine volume accuracy.

Recommendation: do not run the entire mine yet. Complete pending manual coordinate and association QC, then review the 12 cases before any controlled batch.

# Shape-Aware V2 Feature Consistency

Status: **PASS**

## Canonical Schema

Feature order: C, AR, solidity, compactness, eq_diam_ratio, H_mean_norm, H_std_norm, H_p25_norm, H_p75_norm, H_skew_norm, fill_ratio, ellipsoid_ratio

- Training implementation: `enhance_shape_aware.extract_descriptors` and `extract_features`.
- Production implementation: `rockseg.volume.compute_shape_descriptors` and `predict_shape_aware`.
- Both delegate 12-feature construction to `shape_features_v2.py`.
- `H_skew_norm` is the unnormalised `H_skew` value.
- Units: geometry uses one consistent length unit per surface; all exported features are dimensionless.

## Production Mapping

| Training feature | Production source | Formula | Unit |
| --- | --- | --- | --- |
| C | valid height-map footprint | min(4*pi*A/P^2, 1) | dimensionless |
| AR | valid height-map footprint | L/W | dimensionless |
| solidity | valid footprint convex hull | min(A/A_convex, 1) | dimensionless |
| compactness | valid footprint perimeter | P/sqrt(A) | dimensionless |
| eq_diam_ratio | valid footprint area and extent | sqrt(4*A/pi)/L | dimensionless |
| H_mean_norm | ground-referenced cell heights | H_mean/H | dimensionless |
| H_std_norm | ground-referenced cell heights | H_std/H | dimensionless |
| H_p25_norm | ground-referenced cell heights | H_p25/H | dimensionless |
| H_p75_norm | ground-referenced cell heights | H_p75/H | dimensionless |
| H_skew_norm | ground-referenced cell heights | H_skew | dimensionless |
| fill_ratio | 2.5D volume, L, W, H | V_2_5D/V_box | dimensionless |
| ellipsoid_ratio | 2.5D volume, L, W, H | V_2_5D/V_ellipsoid | dimensionless |

## Five Cached Samples

| Dataset | Sample | Cache vs canonical max abs | Canonical vs production max abs | Canonical vs production max relative |
| --- | --- | ---: | ---: | ---: |
| L01 | L01E083a | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| L01 | L01D024a | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| L01 | L01C024a | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| L01 | L01E136a | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| L01 | L01D108a | 0.000e+00 | 0.000e+00 | 0.000e+00 |

## Result

Maximum absolute difference: 0.000e+00
Maximum relative difference: 0.000e+00
Tolerance: absolute difference <= 1.0e-10

PASS requires identical feature order, finite values, positive cached volumes, and the stated tolerance.

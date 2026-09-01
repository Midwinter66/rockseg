# Shape-Aware V2 10 mm Training Report

Final status: **PASS**

## Dataset

- Dataset: `t01_l01_v2_shape_aware_10mm`
- Resolution: 10.0 mm
- Samples: 465 valid / 0 error / 465 total
- Scale factor: 82.73784; grid: 10.0 mm (0.01 m)

## Feature Schema

Feature order: C, AR, solidity, compactness, eq_diam_ratio, H_mean_norm, H_std_norm, H_p25_norm, H_p75_norm, H_skew_norm, fill_ratio, ellipsoid_ratio

`H_skew_norm` is the unnormalised `H_skew` value.

## Split

Train / Validation / Test: 326 / 70 / 69
Seed: 42; group: dataset_id + original_obj_id; overlap: {'train_validation': 0, 'train_test': 0, 'validation_test': 0}

## Data Quality

PASS: True; NaN: 0; Inf: 0; non-positive volumes: 0

## LightGBM Parameters

```json
{
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
  "seed": 42,
  "feature_fraction_seed": 42,
  "bagging_seed": 42,
  "data_random_seed": 42
}
```

## Fixed Test Results

| Method | MAE | RMSE | MAPE | R2 |
| --- | ---: | ---: | ---: | ---: |
| Raw 2.5D | 5.99039e+07 | 8.41398e+07 | 54.2385% | 0.1895 |
| Constant correction | 8.72325e+06 | 1.60491e+07 | 6.9864% | 0.9705 |
| Shape-Aware V2 | 7.16771e+06 | 1.18908e+07 | 5.8231% | 0.9838 |

## Feature Importance

| Feature | Split | Gain |
| --- | ---: | ---: |
| C | 141 | 0.530638 |
| AR | 213 | 0.589342 |
| solidity | 389 | 1.00792 |
| compactness | 132 | 0.281242 |
| eq_diam_ratio | 407 | 1.1326 |
| H_mean_norm | 254 | 0.892423 |
| H_std_norm | 486 | 8.64757 |
| H_p25_norm | 224 | 0.895862 |
| H_p75_norm | 456 | 0.753969 |
| H_skew_norm | 524 | 1.51991 |
| fill_ratio | 320 | 0.740527 |
| ellipsoid_ratio | 30 | 0.0854979 |

## Error Distribution

- Test predicted ratio: {'min': 0.5711148138803616, 'median': 0.6632253451222518, 'max': 0.7262379983944951, 'mean': 0.6537115774164356}
- Test corrected volume relative error: {'relative_error_min': 0.0002661761074258548, 'relative_error_median': 0.05323090498575092, 'relative_error_p90': 0.10995842658970595, 'relative_error_max': 0.163297357542845}

## Decision

Corrected test MAE and MAPE are each at least 10% lower than raw 2.5D.

Next step: proceed to a small real-mine single-rock interface test only when status is PASS.

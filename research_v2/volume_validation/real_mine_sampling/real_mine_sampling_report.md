# DOM2 Real-Mine Volume Estimation Sample Manifest

## Population

- Accepted: 69,911
- Rejected: 6,496 (not sampled)
- Sampling method: `stratified_quantile_systematic`
- Random seed: not used; selection is deterministic.
- Size variable: DOM-footprint equivalent diameter only. No point cloud, 2.5D, volume, target, prediction, or model-error field was read or used.

## Quantiles

| Statistic | Equivalent diameter (m) |
| --- | ---: |
| Min | 0.022567583 |
| P10 | 0.090270333 |
| P25 | 0.113400703 |
| P50 | 0.161952878 |
| P75 | 0.257062507 |
| P90 | 0.430118246 |
| Max | 3.449593713 |
| Mean | 0.225721909 |

## Stratum Sizes

| Stratum | Interval | Population | Target sample | Actual sample | Min diameter (m) | Median diameter (m) | Max diameter (m) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | [P0, P10) | 6,950 | 400 | 400 | 0.022567583 | 0.078986542 | 0.089562320 |
| S2 | [P10, P25) | 10,523 | 600 | 600 | 0.090270333 | 0.102179079 | 0.112837917 |
| S3 | [P25, P50) | 17,465 | 1,000 | 1,000 | 0.113400703 | 0.133033942 | 0.161559310 |
| S4 | [P50, P75) | 17,488 | 1,000 | 1,000 | 0.161952878 | 0.199630653 | 0.256814736 |
| S5 | [P75, P90) | 10,493 | 600 | 600 | 0.257062507 | 0.320547081 | 0.429970210 |
| S6 | [P90, P100] | 6,992 | 400 | 400 | 0.430118246 | 0.607546002 | 3.449593713 |

## Final Sample

- Final sample count: **4,000**
- Every stratum met its requested allocation; no redistribution was required.
- Within each stratum, records were sorted by `equivalent_diameter_m ASC`, then `rock_id ASC`; systematic positions span the stratum range, including both endpoints.

## Coverage

| Statistic | Accepted population (m) | Final sample (m) |
| --- | ---: | ---: |
| Min | 0.022567583 | 0.022567583 |
| P10 | 0.090270333 | 0.090199532 |
| P25 | 0.113400703 | 0.113260006 |
| P50 | 0.161952878 | 0.161756094 |
| P75 | 0.257062507 | 0.256876678 |
| P90 | 0.430118246 | 0.429985014 |
| Max | 3.449593713 | 3.449593713 |

## QC

| Check | Result |
| --- | --- |
| sample_id_unique | True |
| rock_id_unique | True |
| sample_count | 4000 |
| sample_count_expected | 4000 |
| all_samples_accepted | True |
| no_rejected_rocks | True |
| no_duplicate_rocks | True |
| all_diameters_finite_positive | True |
| stratum_counts | {'S1': 400, 'S2': 600, 'S3': 1000, 'S4': 1000, 'S5': 600, 'S6': 400} |
| stratum_counts_match_actual | True |
| strata_sum_matches_final | True |
| reproducibility | PASS: deterministic diameter+rock_id sort and fixed systematic positions; no random generator used |
| volume_or_model_fields_used | False |

## Status

**SAMPLING_PASS**

This manifest is the frozen sample list for the next real-mine volume-estimation phase. No automatic processing was started after manifest generation.

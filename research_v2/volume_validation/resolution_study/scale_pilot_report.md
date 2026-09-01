# Pilot Validation of Prescribed Scale Adaptation

## Conclusion

**PILOT_PASS**

Prescribed scale factor: `82.737840`. No scale fitting, true-volume selection, model training, production modification, or mine-site calculation was performed.

## Selection

- T01: Sort by original 0.5 mm footprint equivalent diameter, select 10 equally spaced rank indices. Population 79; selected ranks [0, 9, 17, 26, 35, 43, 52, 61, 69, 78].
- L01: Sort by original 0.5 mm footprint equivalent diameter, select 10 equally spaced rank indices. Population 386; selected ranks [0, 43, 86, 128, 171, 214, 257, 299, 342, 385].

## Surface and Feature Gates

- Success / failure: 20 / 0
- Occupied cells: {'n': 20, 'min': 1634.0, 'p25': 2898.5, 'median': 3992.0, 'p75': 6793.5, 'max': 51202.0, 'mean': 8200.55, 'iqr': 3895.0}
- All 12 features finite: True
- Collapsed features: none

## y_ratio Stability

- All samples: {'n': 20, 'min': 0.5489925853618866, 'p25': 0.6366504163566132, 'median': 0.6647648005713689, 'p75': 0.7004553950557929, 'max': 0.7397977299857078, 'mean': 0.6645562830898843, 'iqr': 0.06380497869917967}
- T01: {'n': 10, 'min': 0.5489925853618866, 'p25': 0.632445798615183, 'median': 0.6547952278732012, 'p75': 0.670815692448604, 'max': 0.7279001675624077, 'mean': 0.6524700494263753, 'iqr': 0.03836989383342093}
- L01: {'n': 10, 'min': 0.6098266220978377, 'p25': 0.6453080427339712, 'median': 0.6769307182374147, 'p75': 0.7150189044450679, 'max': 0.7397977299857078, 'mean': 0.6766425167533933, 'iqr': 0.06971086171109664}
- T01/L01 median relative gap: 0.0332; IQR overlap: True

## 12-Feature Stability vs Original 0.5 mm

| Feature | Median abs. change | Median relative change | Max abs. change |
| --- | ---: | ---: | ---: |
| C | 0.0714568 | 0.0724 | 0.181725 |
| AR | 0.0218455 | 0.0140 | 0.0779221 |
| solidity | 0.0286137 | 0.0286 | 0.085979 |
| compactness | 0.213082 | 0.0620 | 0.419348 |
| eq_diam_ratio | 0.0119929 | 0.0161 | 0.0482207 |
| H_mean_norm | 0.0122257 | 0.0173 | 0.0173154 |
| H_std_norm | 0.00195823 | 0.0124 | 0.00912068 |
| H_p25_norm | 0.00985828 | 0.0154 | 0.0247421 |
| H_p75_norm | 0.00971992 | 0.0118 | 0.0261299 |
| H_skew_norm | 0.040482 | 0.1009 | 0.232389 |
| fill_ratio | 0.0301017 | 0.0557 | 0.0630426 |
| ellipsoid_ratio | 0.05749 | 0.0557 | 0.120403 |

## Pilot Samples

| Dataset | ID | Original footprint (mm) | Nominal scaled footprint (mm) | 10 mm cells | y_ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| T01 | T01F076a | 5.890 | 487.4 | 1939 | 0.7017400869230358 |
| T01 | T01F072a | 7.421 | 614.0 | 2976 | 0.630343489744468 |
| T01 | T01F005a | 7.838 | 648.5 | 3325 | 0.6387527252273283 |
| T01 | T01F049a | 8.612 | 712.5 | 4045 | 0.7279001675624077 |
| T01 | T01F058a | 9.322 | 771.3 | 4722 | 0.6262212166849112 |
| T01 | T01F051a | 10.045 | 831.1 | 5378 | 0.6701082656227626 |
| T01 | T01F026a | 11.085 | 917.1 | 6669 | 0.6684299945856651 |
| T01 | T01F032a | 12.283 | 1016.3 | 8175 | 0.671051501390551 |
| T01 | T01F006a | 13.900 | 1150.1 | 10442 | 0.5489925853618866 |
| T01 | T01E144a | 30.856 | 2552.9 | 51202 | 0.6411604611607372 |
| L01 | L01E031a | 5.470 | 452.6 | 1634 | 0.6610996065570726 |
| L01 | L01E067a | 6.257 | 517.7 | 2127 | 0.7000271644333786 |
| L01 | L01E071a | 6.580 | 544.4 | 2335 | 0.7200161511156309 |
| L01 | L01E098a | 7.024 | 581.2 | 2666 | 0.7273390403845869 |
| L01 | L01E002a | 7.506 | 621.0 | 3069 | 0.6442726926593868 |
| L01 | L01D170a | 7.959 | 658.5 | 3454 | 0.6927618299177569 |
| L01 | L01E137a | 8.482 | 701.8 | 3939 | 0.7397977299857078 |
| L01 | L01D039a | 9.236 | 764.2 | 4664 | 0.6228702374248498 |
| L01 | L01D013a | 11.521 | 953.2 | 7167 | 0.6098266220978377 |
| L01 | L01C014a | 25.181 | 2083.4 | 34083 | 0.6484140929577246 |

## Decision

- 0.5 mm V2 retained: yes.
- Current unscaled 63-sample 10 mm dataset used: no.
- Full scaled 10 mm construction allowed: True.
- Next action: Build the full scaled 10 mm dataset with this frozen scale factor only.

## Limitations

- This pilot tests dataset-construction feasibility only, not LightGBM accuracy or real-mine volume accuracy.
- The scale factor is a geometric-similarity assumption because external OBJ physical units remain unverified.
- No model was trained and no mine-site point cloud was processed.

# External OBJ Scale Audit for 10 mm Shape-Aware V2

## Conclusion

**SCALE_ADAPTATION_PLAUSIBLE_BUT_NOT_PROVEN**

The proposed transformation is a geometric-similarity domain-adaptation hypothesis, not a verified OBJ unit conversion. It is technically feasible in the six-object benchmark, but no mine-site volume ground truth has validated it.

## Scale Audit

### T01
- OBJ bbox maximum dimension: n=79, median=12.800, IQR=10.576-15.553, range=7.364-36.005 mm
- 0.5 mm footprint equivalent diameter: n=79, median=9.657, IQR=8.107-12.028, range=5.890-30.856 mm
- Reconstructed 2.5D height H: n=79, median=5.633, IQR=4.672-6.603, range=3.762-20.038 mm

### L01
- OBJ bbox maximum dimension: n=386, median=9.731, IQR=8.433-11.737, range=6.096-34.621 mm
- 0.5 mm footprint equivalent diameter: n=386, median=7.715, IQR=6.682-9.053, range=5.470-25.181 mm
- Reconstructed 2.5D height H: n=386, median=4.832, IQR=4.154-6.047, range=2.591-14.408 mm

### Mine Site B
- Existing equivalent diameter: 500.1-3346.3 mm; median 658.5 mm; IQR 565.3-823.9 mm.

## Proposed Uniform Scale Rule

`s = 658.500 / 7.958873 = 82.737840`

Scale every XYZ coordinate by this one factor before 10 mm rasterization. The numerator comes from independent existing Site B diameter statistics; no true volume or test result is used.

## Feature Interpretation

- Under continuous uniform geometric similarity, all 12 V2 features are dimensionless and invariant.
- At fixed 10 mm rasterization this is approximate: all feature values and the ratio target can change through discretization.
- Normalized height features are stable only for sufficiently resolved geometrically similar surfaces.

## Six-Object Feasibility Benchmark

| Dataset | Percentile | Sample | Scaled eq. diameter (mm) | 10 mm cells | Surface | Features finite |
| --- | ---: | --- | ---: | ---: | --- | --- |
| T01 | 25 | T01F050a | 663.4 | 3440 | ok | True |
| T01 | 50 | T01F046a | 799.0 | 5031 | ok | True |
| T01 | 75 | T01F025a | 999.0 | 7853 | ok | True |
| L01 | 25 | L01E025a | 552.3 | 2418 | ok | True |
| L01 | 50 | L01D097a | 638.3 | 3189 | ok | True |
| L01 | 75 | L01D047a | 749.8 | 4465 | ok | True |

## Decision

- The 0.5 mm V2 model remains the external mesh validation result.
- The current 63-sample unscaled 10 mm dataset must not be used for training.
- Do not yet build a full scaled 10 mm Dataset or train a model.
- The next permissible experiment is a pre-registered limited pilot using this exact uniform scale rule and `original_obj_id` group-aware splits.
- Main risks: unverified external physical units, similarity transfer from fragments to mine rocks, rasterization feature drift, and no real-mine volume ground truth.

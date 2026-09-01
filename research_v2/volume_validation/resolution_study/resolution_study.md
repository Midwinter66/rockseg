# Shape-Aware V2 2.5D Resolution Consistency Study

## Scope

This is a representative-object benchmark, not a new Dataset B build or a
model-training experiment. It reused the training-side mesh rasterizer and
12-feature extraction code on one T01 object (`T01E143a`, 360,368 faces) and
one L01 object (`L01C010a`, 272,256 faces). No production code, Dataset B,
split, model file, OBJ source data, or real mine-site batch was changed.

The tested grid resolutions were 0.5, 1, 2.5, 5, 10, 25 and 50 mm. Detailed
per-object records are in `resolution_results.csv`.

## Benchmark Results

| Grid (mm) | T01 surface | T01 time (s) | T01 raw 2.5D MAPE (%) | L01 surface | L01 time (s) | L01 raw 2.5D MAPE (%) | 12 finite features |
|---:|---|---:|---:|---|---:|---:|---|
| 0.5 | valid | 4.402 | 48.21 | valid | 3.304 | 33.97 | yes |
| 1.0 | valid | 4.215 | 48.86 | valid | 3.176 | 34.53 | yes |
| 2.5 | valid | 4.146 | 45.75 | valid | 3.140 | 33.58 | yes |
| 5.0 | valid | 4.140 | 41.46 | valid | 3.128 | 21.31 | yes |
| 10.0 | valid | 4.116 | 16.62 | valid | 3.120 | 13.76 | yes |
| 25.0 | empty surface | 4.105 | NOT AVAILABLE | empty surface | 3.121 | NOT AVAILABLE | no |
| 50.0 | empty surface | 4.125 | NOT AVAILABLE | empty surface | 3.129 | NOT AVAILABLE | no |

The benchmark timing does not improve meaningfully at coarser grids because
the current mesh rasterizer traverses every triangle. It should not be used as
a production timing estimate for a point-cloud implementation.

## Feature Consistency

All valid surfaces generated 12 finite values in the training order. However,
the feature values are not resolution-stable. Examples:

| Object | Feature | 0.5 mm | 5 mm | 10 mm |
|---|---|---:|---:|---:|
| T01E143a | H_skew_norm | -0.183 | 0.311 | 0.704 |
| T01E143a | AR | 1.692 | 2.000 | 3.000 |
| T01E143a | fill_ratio | 0.447 | 0.626 | 0.774 |
| L01C010a | H_std_norm | 0.212 | 0.263 | 0.001 |
| L01C010a | H_skew_norm | -0.176 | -0.319 | 0.000 |
| L01C010a | fill_ratio | 0.429 | 0.526 | 0.999 |

At 10 mm, `L01C010a` effectively collapses to an almost single-height surface.
At 25 and 50 mm, the grid has no sampled cell centers inside the mesh faces,
so the current training rasterizer produces no 2.5D surface at all. These are
scientific compatibility failures, not missing-value cases to impute.

## Model Performance Availability

The existing 0.5 mm Dataset B model has Test MAPE 5.83% and R2 0.9845.

Test MAPE, RMSE and R2 at 1-50 mm are **NOT AVAILABLE**. They require all
fixed-split objects to be rasterized at each resolution and a model trained and
selected under the same resolution. That experiment was not started because
the representative benchmark already found severe feature drift and invalid
25/50 mm surfaces.

## Answers

1. **Why 0.5 mm?** It is appropriate for the mesh-training experiment because
   it preserves fine mesh geometry and yields valid, non-degenerate features.
   It is not directly suitable for mine-site deployment: it is 20 times finer
   than the known 10 mm DOM pixel size and 100 times finer than the current
   50 mm production grid.
2. **Is 10 mm reasonable?** It is a defensible *candidate* for a controlled
   study because it matches the known DOM pixel size (10 mm/pixel), and both
   representative meshes remained valid. It is not yet validated: feature
   drift is substantial and point-cloud spacing has not been established.
3. **Is 50 mm too coarse?** For the current training mesh rasterization
   algorithm, yes: both representatives returned empty surfaces at 50 mm.
   It therefore cannot generate compatible 12-feature inputs in this test.
4. **Is V2 resolution-robust?** No robustness claim is supported. Feature
   values change materially by 5-10 mm, and 25/50 mm fail outright.
5. **What should mine-site production use?** Do not change production now.
   First measure available point-cloud spacing. If it supports about 10 mm,
   validate 10 mm as the initial candidate; otherwise select a supported grid
   and repeat a resolution-matched validation.
6. **What model strategy is required?** Option A first: train a new
   resolution-matched model after constructing a fixed-split dataset at the
   selected production-supported grid. Option B, multi-resolution training,
   should be evaluated only afterwards as a robustness extension. The current
   0.5 mm model must not be applied directly to 10 mm or 50 mm descriptors.

## Limitation

Only two representative meshes were processed. No full 465-object resolution
sweep, no resolution-specific model training, and no mine-site point-cloud
recomputation were performed. The benchmark is sufficient to reject direct
50 mm transfer under the existing training rasterizer, but not to establish a
final production grid.

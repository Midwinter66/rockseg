# RockSeg

RockSeg is a physical-scale-driven rock segmentation and volume-estimation
pipeline for UAV DOM and OSEB-derived point clouds. It connects multi-scale
DOM instance segmentation, 2D-3D association, ground-referenced 2.5D surface
reconstruction, and a canonical shape-aware LightGBM correction model into a
single reproducible workflow.

## Method Pipeline

```text
UAV DOM (10 mm/pixel)
  -> physical-scale multi-scale instance segmentation
  -> within-scale fusion + cascade deduplication
  -> final rock inventory (76,407 instances)
  -> existing 2D-3D association / accepted-instance filtering (69,911 accepted)
  -> ground/background separation
  -> 10 mm ground-referenced 2.5D top surface
  -> canonical 12 shape features
  -> frozen scaled-10mm Shape-Aware LightGBM
  -> V_pred = V_2.5D x y_pred
  -> size-stratified representative volume results
  -> PSD / P80 analysis (next step)
```

The external mesh methodology (T01: 79 meshes + L01: 386 meshes, 465 objects)
uses a pre-specified uniform scale factor of 82.737840 to adapt external OBJ
geometry to the mine footprint scale. The 0.5 mm V2 result is external
validation only; it is not applied to real-mine point clouds.

## Frozen Results

### External Scaled 10 mm Test (69 held-out objects)

| Method | MAPE | R2 |
| --- | ---: | ---: |
| Raw 2.5D | 54.24% | 0.1895 |
| Constant correction | 6.99% | 0.9705 |
| Shape-Aware V2 | 5.82% | 0.9838 |

### DOM2 4,000-Rock Representative Application

| Item | Result |
| --- | ---: |
| DOM2 final instances | 76,407 |
| DOM2 accepted instances | 69,911 |
| Stratified representative sample | 4,000 |
| Successful pipeline records | 3,639 / 4,000 (90.98%) |
| Failures | 361 (all `empty_2_5d_surface`) |

### Size-Stratified Completion

| Stratum | Sample | Success | Rate |
| --- | ---: | ---: | ---: |
| S1 (P0-P10) | 400 | 336 | 84.00% |
| S2 (P10-P25) | 600 | 518 | 86.33% |
| S3 (P25-P50) | 1,000 | 875 | 87.50% |
| S4 (P50-P75) | 1,000 | 923 | 92.30% |
| S5 (P75-P90) | 600 | 588 | 98.00% |
| S6 (P90-P100) | 400 | 399 | 99.75% |

The external 5.82% MAPE is not real-mine accuracy. The DOM2 90.98% is pipeline
completion rate, not volume accuracy; per-rock mine ground truth is not
available.

## Canonical Shape Features

The 12-dimensional feature schema (fixed in training and inference):

`C, AR, solidity, compactness, eq_diam_ratio, H_mean_norm, H_std_norm,
H_p25_norm, H_p75_norm, H_skew_norm, fill_ratio, ellipsoid_ratio`

- Target: `y_ratio = V_true / V_2.5D`
- Prediction: `V_pred = V_2.5D x y_pred`
- Best iteration: 356

## Project Structure

| Role | Path |
| --- | --- |
| Pipeline package | `rockseg/` |
| Detection entry point | `run_rockseg.py` |
| 3D association entry point | `run_3d_validation_fast.py` |
| Volume estimation entry point | `run_volume_estimation.py` |
| Detection model | `models/best.pt` |
| Volume validation code | `research_v2/volume_validation/` |
| Frozen scaled-10mm model | `research_v2/volume_validation/output_v2_scaled_10mm/` |
| 4,000-rock results | `research_v2/volume_validation/real_mine_full/` |
| Scaled-10mm dataset | `research_v2/volume_validation/datasets/t01_l01_scaled_10mm/` |
| Experiment configs | `experiments/configs/` |

## Source Data (not in repository)

| Data | Path | Size |
| --- | --- | ---: |
| DOM2 orthophoto | `data/dom2/DOM.tif` | 546 MB |
| DOM2 point clouds | `data/pointcloud2/Data/BlockB.laz`, `BlockY.laz` | 1.08 GB |
| DOM3 orthophoto | `data/dom3/DOM.tif` | 473 MB |
| DOM3 point clouds | `data/pointcloud3/Data/BlockB.laz`, `BlockY.laz` | 1.39 GB |
| External meshes | `data/experience_rock/T01/`, `L01/` | — |

Raw data is excluded from Git via `.gitignore`. See `docs/project_context/`
for data preparation notes.

## Documentation

| Content | Path |
| --- | --- |
| Research overview | `research_v2/RESEARCH_OVERVIEW.md` |
| Final research status | `research_v2/FINAL_RESEARCH_STATUS.md` |
| Project context and decisions | `docs/project_context/` |
| Presentations | `docs/presentations/` |
| Reference literature | `docs/references/` |
| Auxiliary scripts | `scripts/` (see `scripts/README.md`) |

## Environment

Install dependencies from `requirements.txt` or `environment.yml`. The local
`Ultralytics/` checkout is not required; the pipeline imports the installed
`ultralytics` package.

## Important Notes

- Do not modify frozen datasets, model files, sample manifests, or result
  files without registering a new experiment.
- Pipeline outputs (`output/`) and archived experiments (`archive/`) are
  excluded from Git; they contain large binary result files.
- Raw data (`data/`) is excluded from Git; transfer separately.

# Paper Method Pipeline

## Scope

This is the implemented and frozen method correspondence for manuscript
writing. It distinguishes operationally verified code paths from scientific
claims that require external validation.

```text
DOM2 (0.01 m/pixel)
  |
  v
Physical-scale multi-scale tiling and instance segmentation
  coarse: 10.24 m coverage; medium: 5.12 m; fine: 2.56 m
  |
  v
Same-scale fusion
  spatial candidate lookup -> bbox IoU prefilter -> nonzero mask IoU
  -> weighted score >= 0.50 -> canonical best mask
  |
  v
Cross-scale cascade deduplication
  bbox IoU >= 0.05 + nonzero mask IoU + diameter ratio >= 0.30
  + centroid distance <= larger radius
  -> primary scale by maximum equivalent diameter
  |
  v
Final DOM2 instance inventory (76,407)
  |
  v
Existing 2D-3D association and quality screening
  point-cloud spatial index / candidate query -> GroundDEM-relative metrics
  -> accepted inventory (69,911)
  |
  v
Frozen diameter-stratified systematic manifest (4,000)
  |
  v
Per-rock DOM mask + associated local point-cloud data
  |
  v
Ground-referenced 10 mm 2.5D surface
  top observed height per occupied grid cell above local GroundDEM
  |
  v
Canonical 12-feature descriptor + V_2.5D
  |
  v
Frozen scaled-10 mm Shape-Aware V2 LightGBM
  y_pred
  |
  v
V_pred = V_2.5D * y_pred
  |
  v
Representative DOM2 mine-volume statistics for successful records
```

## Implementation Correspondence

| Stage | Frozen implementation / output | Implemented behavior |
| --- | --- | --- |
| Configuration | `rockseg/config.py` | Defines GSD 0.01 m/pixel, three physical scales, 20% overlap, confidence threshold 0.25, fusion weights, and fusion thresholds. |
| Pipeline orchestration | `run_rockseg.py`, `rockseg/pipeline.py` | Runs the DOM pipeline using the configured scales and cascade option. |
| Instance representation | `rockseg/models.py` and `output/dom2_cascade_v2/rock_instances.json` | Stores masks, bounding boxes, scale level, and DOM footprint metadata. |
| Same-scale fusion | `rockseg/fusion.py::within_scale_fusion` | Uses candidate spatial grid, bbox IoU >= 0.05, nonzero mask IoU, and weighted score threshold 0.50. |
| Weighted fusion score | `rockseg/config.py::FusionWeights`, `rockseg/fusion.py::compute_fusion_score` | `0.30*IoU + 0.20*centroid + 0.20*area_ratio + 0.15*boundary + 0.15*confidence`. Canonical representative is the maximum `confidence * boundary_completeness`. |
| Cascade deduplication | `rockseg/fusion.py::cascade_deduplication` | Groups compatible cross-scale duplicates; chooses coarse at diameter >= 0.50 m, medium at 0.30-0.50 m, fine below 0.30 m. |
| 3D screen | `rockseg/validation_3d_fast.py`, `output/dom2_cascade_v2_3d_fixed/` | Uses float64 point coordinates, a reusable spatial index, GroundDEM-relative quality metrics, and fixed thresholds. The final accepted inventory comes from this fixed result. |
| Mask-aware validation implementation | `rockseg/validation_3d.py` | Contains a mask-aware validation routine. It is not used here to claim separately validated mask-to-point correspondence accuracy for the final fast screened inventory. |
| Ground reference | `rockseg/validation_3d_fast.py::GroundDEM` | Builds a scene-level 0.5 m grid from finite points subsampled every 100 points; cells with >=3 points use P5 elevation; neighboring valid values fill holes. |
| 2.5D surface | `rockseg/volume.py::extract_height_map` | Queries local point candidates once indexed, computes height above GroundDEM, and records maximum relative height per 10 mm grid cell. |
| Canonical descriptor | `research_v2/volume_validation/shape_features_v2.py` | Defines the train/inference canonical 12-feature schema; consistency report found zero difference on checked samples. |
| Mine batch adapter | `research_v2/volume_validation/run_dom2_volume_batch.py` | Loads frozen model, grid, accepted associations, masks, GroundDEM, and point-cloud index for resumable 4,000-rock inference. |
| Model training | `research_v2/volume_validation/train_shape_aware_v2_t01_l01.py` | Trained once on the frozen scaled external OBJ dataset; no training occurs in mine application. |

## 3D Association and Quality Criteria

The fixed screen requires at least 60 candidate points, z-range >= 0.18 m,
P90 ground-referenced height >= 0.12 m, and elevated ratio >= 0.20 where
"elevated" means height >= 0.08 m. It records `too_few_points`,
`insufficient_ground_points`, `insufficient_z_range`,
`insufficient_p90_height`, and `insufficient_elevated_ratio` without changing
the thresholds per rock.

## 2.5D Surface and Observability Boundary

For each local rock surface, the model uses the maximum observed relative point
height in each 10 mm cell. The DEM provides a local ground reference; it does
not create measurements beneath occlusion. Therefore `V_2.5D` and `V_pred` are
estimates based on the observable, ground-referenced surface. They must not be
interpreted as verified complete geometry for buried, hidden, or overlapping
rock portions.

## Shape-Aware Model Contract

```text
y_ratio = V_true / V_2.5D                 (external mesh training target)
y_pred = LightGBM(X_12)                   (frozen mine inference)
V_pred = V_2.5D * y_pred                  (reported corrected estimate)
```

`X_12` must follow the exact order in `PAPER_TABLES.md`. In particular,
`H_skew_norm = H_skew`; it is not divided by height. The operational mine grid
is fixed at 0.01 m, and model development used external OBJ geometries uniformly
scaled by 82.737840 before 10 mm rasterization.

## External-to-Mine Evidence Chain

```text
Known-volume external OBJ meshes
  -> original 0.5 mm methodology validation
  -> original-scale 10 mm failure identified
  -> DOM GSD and sampled point-spacing audit support 10 mm mine grid
  -> fixed uniform scale adaptation (82.737840)
  -> 20-object feasibility pilot: 20/20 valid
  -> 465-object scaled 10 mm dataset: 465/465 valid
  -> one frozen LightGBM model, held-out external test
  -> 12-rock mine transfer pilot
  -> frozen 4,000-rock representative mine application
```

The scale adaptation is feasible operationally but remains
`SCALE_ADAPTATION_PLAUSIBLE_BUT_NOT_PROVEN`. It does not establish real-mine
absolute accuracy.

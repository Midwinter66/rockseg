# Paper Final Status

## 1. Final Research Question

Can physical-scale-driven multi-scale DOM segmentation, existing 2D-3D point
cloud association, a ground-referenced 10 mm 2.5D surface, and a frozen
shape-aware correction model produce a reproducible representative rock-volume
application for the DOM2 mine area?

## 2. Frozen Research Line

```text
DOM2 UAV/Digital Orthophoto Map
  -> physical-scale multi-scale instance segmentation
  -> same-scale fusion and cross-scale cascade deduplication
  -> final DOM instances
  -> existing 2D-3D association and quality screening
  -> GroundDEM reference
  -> 10 mm observable 2.5D rock surface
  -> canonical 12-feature descriptor
  -> frozen Shape-Aware V2 LightGBM correction
  -> V_pred = V_2.5D * y_pred
  -> representative mine-scale volume statistics
```

All model, dataset, split, grid, scale factor, sampling manifest, and reported
results in this document are frozen. This is a documentation state, not a new
experiment.

## 3. Data Sources and Frozen Assets

| Component | Frozen source | Scope |
| --- | --- | --- |
| DOM2 | `data/dom2/DOM.tif` | 0.01 m/pixel orthophoto |
| Mine point clouds | `data/pointcloud2/Data/BlockB.laz`, `data/pointcloud2/Data/BlockY.laz` | Existing associated mine point-cloud input |
| Final DOM inventory | `output/dom2_cascade_v2/rock_instances.json` | 76,407 final instances |
| 3D association | `output/dom2_cascade_v2_3d_fixed/` | 69,911 accepted; 6,496 rejected |
| External meshes | `data/experience_rock/T01/`, `data/experience_rock/L01/` | 79 + 386 OBJ meshes |
| Scaled 10 mm dataset | `research_v2/volume_validation/datasets/t01_l01_scaled_10mm/` | 465 samples; split 326/70/69 |
| Frozen model | `research_v2/volume_validation/output_v2_scaled_10mm/shape_aware_model_v2_scaled_10mm.txt` | Final 10 mm Shape-Aware V2 model |
| Mine sample manifest | `research_v2/volume_validation/real_mine_sampling/real_mine_volume_sample_manifest.csv` | Deterministic 4,000-rock sample |
| Mine results | `research_v2/volume_validation/real_mine_full/real_mine_volume_4000_*` | Frozen representative application |

## 4. DOM2 Segmentation and Association

DOM2 uses a 0.01 m/pixel ground sampling distance and three physical tile
scales: coarse 10.24 m, medium 5.12 m, and fine 2.56 m. The completed cascade
pipeline produced 76,407 final instances. Existing metadata records 37,470,
101,642, and 179,286 raw detections at coarse, medium, and fine scale,
respectively; 112,983 same-scale-fused detections then entered cascade
deduplication. The final inventory contains 5,925 coarse, 10,890 medium, and
59,592 fine retained instances.

The float64-fixed association accepted 69,911 / 76,407 instances (91.50%) and
rejected 6,496. The accepted set is the only population used by the frozen
mine-volume manifest. The relevant final source is
`output/dom2_cascade_v2_3d_fixed/validation_summary.json`.

## 5. External Mesh Methodology and Resolution Matching

T01 (79 OBJ meshes) and L01 (386 OBJ meshes) form Dataset B (465 meshes).
They provide known mesh volume (`V_true`), a rasterized 2.5D volume
(`V_2.5D`), the canonical descriptor, and the correction target
`y_ratio = V_true / V_2.5D`. They are methodological training and validation
data, not real-mine ground truth.

The 0.5 mm V2 result remains an external-mesh methodological validation only.
Original-scale OBJ meshes were not viable at 10 mm: only 63 / 465 surfaces were
valid. Mine point-cloud spacing evidence and DOM GSD supported a fixed mine
grid of 10 mm (0.01 m). A pre-registered uniform geometry scale factor
`s = 82.737840` was obtained independently from footprint-size statistics:
`658.500 / 7.958873`. The audit concluded
`SCALE_ADAPTATION_PLAUSIBLE_BUT_NOT_PROVEN`.

The scale pilot passed 20 / 20 surfaces, after which the full scaled 10 mm
dataset completed 465 / 465 samples with a group-aware split of 326 / 70 / 69,
no split leakage, and no non-finite features. Scale adaptation is an explicit
domain-adaptation assumption, not a mine-volume accuracy validation.

## 6. Frozen Shape-Aware V2 Model

The final model is LightGBM regression, trained once on the frozen scaled 10 mm
dataset with target `y_ratio = V_true / V_2.5D`. It uses 12 canonical features
in the fixed order stated in `PAPER_TABLES.md`; `H_skew_norm` is the raw
`H_skew`, not `H_skew / H`. The best iteration is 356. Mine inference uses:

```text
V_pred = V_2.5D * y_pred
```

The external scaled-10 mm test has 69 held-out objects. Shape-Aware V2 achieved
MAE 7,167,711 mm3, RMSE 11,890,780 mm3, MAPE 5.82%, and R2 0.9838. These are
external scaled-mesh test values only.

## 7. Real-Mine Application

The real-mine application did not sample by volume, prediction, target, or
error. From 69,911 accepted DOM2 instances, the frozen deterministic
`stratified_quantile_systematic` manifest selected 4,000 records using only DOM
footprint equivalent diameter: S1/S2/S3/S4/S5/S6 =
400/600/1,000/1,000/600/400.

The frozen 10 mm inference run completed 3,639 / 4,000 samples (90.98%). All
361 failures were `empty_2_5d_surface`; no successful sample had a feature
dimension mismatch, NaN/Inf feature, non-positive volume, or volume-formula
failure. Completion rate increased by size stratum from S1 (84.0%) to S6
(99.75%). The successful mine estimates have `y_pred` from 0.5541 to 0.7280,
`V_2.5D` from 2.99e-09 to 4.3480 m3, and `V_pred` from 1.99e-09 to 2.9962 m3.

This is a stratified representative application, not a volume calculation for
all 69,911 accepted rocks.

## 8. What Can Be Claimed

- The completed DOM2 pipeline produced 76,407 final multi-scale instances and
  an existing quality-screened association inventory of 69,911 accepted rocks.
- The frozen external-mesh evidence supports the shape-aware 2.5D correction
  methodology at the scaled 10 mm training resolution.
- The frozen model completed the full prescribed inference chain for 3,639 of
  4,000 deterministically selected accepted DOM2 rocks.
- Surface availability at a 10 mm grid was lower in small-size strata, with all
  recorded failures classified as `empty_2_5d_surface`.
- The result provides reproducible representative mine-scale volume estimates
  conditioned on observable, ground-referenced point-cloud surfaces.

## 9. What Cannot Be Claimed

- Do not state that 5.82% is real-mine volume accuracy. It is the held-out
  scaled external OBJ test MAPE.
- Do not state that 90.98% is volume accuracy. It is the 4,000-rock pipeline
  completion rate.
- Do not state that all 69,911 accepted rocks received volume estimates.
- Do not claim independent segmentation accuracy, DEM accuracy, 2D-3D
  association accuracy, or per-rock mine volume accuracy without reference
  labels or ground-truth measurements.
- Do not claim recovery of hidden lower-rock geometry, buried volume, or
  otherwise unobservable geometry from the 2.5D surface.

## 10. Historical Documents and Current Authority

The current frozen numerical authority is the result metadata cited in Section
3, especially the scaled-10 mm training results, `validation_summary.json`,
the sampling report/manifest, and the 4,000-rock summary. Existing documents
are preserved as follows:

| Document group | Status for paper drafting |
| --- | --- |
| `docs/project_context/EXPERIMENT_LOG.md` entries EXP-V2-01 to EXP-V2-09 | HISTORICAL experiment record; consistent with current frozen chain |
| Historical V1 entries in that log | HISTORICAL / SUPERSEDED for the current V2 paper chain where inconsistent |
| `research_v2/RESEARCH_OVERVIEW.md` and `research_v2/FINAL_RESEARCH_STATUS.md` | HISTORICAL status summaries; this `docs/paper/` package is the current paper-writing fact base |
| Bugged float32 validation/volume runs | SUPERSEDED by the float64-fixed association and frozen 4,000-rock application |
| Original-scale `t01_l01_v2_10mm` training path | SUPERSEDED / not used for training because only 63 / 465 surfaces were valid |

## 11. Paper Sections Ready to Draft

The paper can now draft the study data, physical-scale rationale, segmentation
and fusion method, association/quality screen, 2.5D and descriptor method,
external validation, real-mine representative application, results tables,
limitations, and discussion. The evidence gaps in `PAPER_EVIDENCE_MAP.md`
must remain explicit in the methods, results, and claims.

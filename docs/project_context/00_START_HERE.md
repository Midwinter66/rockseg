# RockSeg -- AI Agent Start Here

> Minimal current context for AI agents. Read this file first. Detailed
> research evidence is maintained in `docs/paper/`.

## 1. Project

Project: RockSeg

Domain: post-blast rock-fragmentation analysis from DOM, instance segmentation,
2D--3D point-cloud association, observable rock-volume estimation, and future
particle-size distribution (PSD) analysis.

Current research question:

How can physical-scale-aware multi-scale DOM segmentation be integrated with
2D--3D geometric reconstruction and shape-aware correction to estimate
observable rock volumes over a wide footprint-size range?

## 2. Frozen research pipeline

```text
Study-area DOM
  -> physical-scale multi-scale tiling
  -> instance segmentation
  -> within-scale duplicate resolution across overlapping tiles
  -> cross-scale size-aware cascade deduplication
  -> final DOM instance inventory
  -> 2D--3D point-cloud association
  -> fixed 3D quality filtering
  -> local GroundDEM reference
  -> ground-referenced 10 mm 2.5D surface
  -> canonical 12-feature descriptor
  -> scaled-10 mm Shape-Aware V2 LightGBM correction
  -> y_pred
  -> V_pred = V_2.5D * y_pred
  -> representative real-mine volume statistics
  -> future PSD / P80 analysis
```

The method targets surface-visible / observable rock geometry. GroundDEM
provides a local height reference; it does not recover buried, occluded, or
otherwise unobserved geometry.

## 3. Frozen configuration

| Item | Frozen value |
| --- | --- |
| DOM GSD | 0.01 m/pixel |
| Physical tile coverages | coarse 10.24 m; medium 5.12 m; fine 2.56 m |
| Tile overlap | 20% |
| Segmentation confidence threshold | 0.25 |
| Within-scale fusion threshold | 0.50 |
| Cross-scale candidate rules | bbox IoU >= 0.05; non-zero mask IoU; diameter ratio >= 0.30; centroid distance within larger radius |
| Primary scale boundaries | fine < 0.30 m; medium 0.30--0.50 m; coarse >= 0.50 m |
| 2.5D grid | 0.01 m (10 mm) |
| External-to-mine scale factor | 82.737840 |
| Shape-Aware input | canonical 12-feature schema, fixed order |
| Correction target | y_ratio = V_true / V_2.5D |
| Final correction | V_pred = V_2.5D * y_pred |

The tenth feature is defined as `H_skew_norm = H_skew`; it is not divided by
height. DOM GSD, point-cloud spacing, and 2.5D grid resolution are distinct
concepts.

## 4. Frozen assets and results

| Asset or result | Current frozen source / value |
| --- | --- |
| DOM | `data/dom2/DOM.tif` |
| Mine point clouds | `data/pointcloud2/Data/BlockB.laz`, `data/pointcloud2/Data/BlockY.laz` |
| Final DOM inventory | `output/dom2_cascade_v2/rock_instances.json`; 76,407 instances |
| 3D association result | `output/dom2_cascade_v2_3d_fixed/`; 69,911 accepted; 6,496 rejected |
| External mesh data | T01 79 + L01 386 = 465 meshes; methodological data only |
| Scaled 10 mm dataset | `research_v2/volume_validation/datasets/t01_l01_scaled_10mm/`; split 326/70/69 |
| Frozen model | `research_v2/volume_validation/output_v2_scaled_10mm/shape_aware_model_v2_scaled_10mm.txt` |
| Mine sample manifest | `research_v2/volume_validation/real_mine_sampling/real_mine_volume_sample_manifest.csv`; 4,000 rocks |
| Mine application results | `research_v2/volume_validation/real_mine_full/real_mine_volume_4000_*` |
| External scaled-10 mm Test | Shape-Aware V2 MAPE 5.82%; R2 0.9838; external mesh test only |
| Real-mine application | 3,639 / 4,000 successful; 361 failures, all `empty_2_5d_surface` |
| Real-mine completion rate | 90.98%; this is not volume accuracy |

The 0.5 mm Shape-Aware model is retained for external OBJ methodological
validation only. It is not the real-mine production model. The scaled-10 mm
model is the only model for the current real-mine application.

## 5. Current status

Status: PAPER WRITING / EXPERIMENTS FROZEN

Completed and frozen:

- physical-scale multi-scale DOM segmentation;
- within-scale duplicate resolution and cross-scale cascade deduplication;
- 2D--3D association and fixed 3D screening;
- GroundDEM and 10 mm observable 2.5D processing;
- canonical 12-feature consistency;
- 0.5 mm external mesh methodological validation;
- point-spacing, resolution, and scale adaptation studies;
- scaled 10 mm external dataset and LightGBM training;
- 12-rock real-mine pilot;
- deterministic 4,000-rock stratified real-mine application;
- final result QC;
- Chinese-first / English methodology draft for Chapter 3, Sections 3.1--3.6.

Current writing task:

- draft Chapter 4 Results in the order 4.1 segmentation/fusion, 4.2 association/filtering, 4.3 external validation, 4.4 resolution/scale adaptation, and 4.5 real-mine application;
- then prepare Discussion, Conclusion, and final translation/polishing.

## 6. Scientific boundaries

Do not claim:

- 5.82% as real-mine volume accuracy; it is the scaled external held-out Test MAPE;
- 90.98% as volume accuracy; it is the real-mine pipeline completion rate;
- volume estimates for all 69,911 accepted instances; only 4,000 were processed;
- independently validated segmentation, fusion, association, or DEM accuracy;
- recovery of buried or occluded rock geometry;
- real-mine absolute volume accuracy without per-rock reference volumes;
- universal transferability to other mines.

Known evidence gaps:

- no independent manual segmentation benchmark;
- no manually adjudicated duplicate-resolution benchmark;
- no independently labelled 2D--3D correspondence benchmark;
- no independent absolute GroundDEM accuracy benchmark;
- no per-rock real-mine ground-truth volume;
- physical validity of the external-to-mine scale adaptation is plausible but not proven;
- PSD/P80 analysis is not yet completed.

## 7. Paper-writing documents

Read these after this file, only when relevant:

```text
docs/paper/README.md
docs/paper/PAPER_FINAL_STATUS.md
docs/paper/PAPER_EVIDENCE_MAP.md
docs/paper/PAPER_TABLES.md
docs/paper/PAPER_METHOD_PIPELINE.md
docs/paper/PAPER_WRITING_BASELINE.md
docs/paper/PAPER_OUTLINE.md
docs/paper/METHODOLOGY_WRITING_STANDARD.md
docs/paper/PAPER_DRAFT.md
docs/paper/PAPER_DRAFT_CN.md
```

Current manuscript authority:

- `PAPER_FINAL_STATUS.md`: final facts and claim boundaries;
- `PAPER_EVIDENCE_MAP.md`: evidence status and gaps;
- `PAPER_TABLES.md`: frozen quantitative values;
- `PAPER_METHOD_PIPELINE.md`: method-to-implementation correspondence;
- `PAPER_DRAFT.md`: English manuscript draft;
- `PAPER_DRAFT_CN.md`: Chinese corresponding draft;
- `PAPER_OUTLINE.md`: chapter architecture and Section 4 linkage.

Historical experiment records remain historical. Do not use an older plan to
override the current frozen paper documents.

## 8. Mandatory restrictions

Without explicit new authorization:

- do not retrain any model;
- do not modify Dataset B or the scaled 10 mm dataset;
- do not modify model parameters, split, scale factor, feature definitions, or grid resolution;
- do not rerun segmentation, 2D--3D association, or volume inference;
- do not run all 69,911 accepted rocks;
- do not modify `rockseg/volume.py` or other production code;
- do not generate new images, PPT, or exploratory experiments;
- do not replace failed samples or transfer external metrics to the mine;
- do not scan the entire repository when the paper documents are sufficient.

For paper drafting, use `EVIDENCE GAP` internally in evidence documents only.
In manuscript prose, express the limitation directly and do not fabricate
missing evidence, citations, values, or figures.

## 9. Agent reading policy

Always read this file first. Then read only the paper document or frozen result
file needed for the current task. Do not automatically read every context file,
source file, or historical experiment.

## 10. Last updated

2026-08-30

Updated for the frozen research baseline and Chapter 3 manuscript rewrite.

# Paper Figure Plan

Status: `PARTIALLY GENERATED - 2026-08-31`

## Generated Figure Log

| Figure   | File                                                   | Format   | Date       | Status                                                              |
| -------- | ------------------------------------------------------ | -------- | ---------- | ------------------------------------------------------------------- |
| Figure 1 | `docs/paper/figures/Figure_1_Overall_Methodological_Framework.svg` | SVG | 2026-08-31 | GENERATED (v3, compact 3×3 serpentine, SCI-grade) |
| Figure 6 | `docs/paper/figures/figure6_12feature_descriptor.html` | HTML/SVG | 2026-08-31 | GENERATED                                                           |
| Figure 7 | `docs/paper/figures/figure7_4000rock_results.html`     | HTML/SVG | 2026-08-31 | GENERATED                                                           |
| Figure 2 | --                                                     | --       | --         | EVIDENCE GAP: no fixed example identified                           |
| Figure 3 | --                                                     | --       | --         | EVIDENCE GAP: no traceable cross-scale group identified             |
| Figure 4 | --                                                     | --       | --         | PARTLY READY: aggregate reasons exist, visual examples not selected |
| Figure 5 | --                                                     | --       | --         | EVIDENCE GAP: no figure-ready DEM/surface arrays confirmed          |

本文件只定义图件目的、所需数据、推荐来源和证据状态。实际图片由 TRAE 完成。TRAE 应优先使用冻结输出，不重新训练、分割、关联或计算体积；若现有输出不包含图件所需中间数据，应标记缺口并停止，而不是重跑实验。

| Figure   | Content                                           | Required data                                                                                                                                               | Recommended source                                                                                                            | Status                                                                                                                                |
| -------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Figure 1 | Complete research workflow                        | Frozen method sequence, key resolutions, external-to-mine model path, final sample scope                                                                    | `PAPER_METHOD_PIPELINE.md`; `PAPER_WRITING_BASELINE.md`; `PAPER_FINAL_STATUS.md`                                              | GENERATED 2026-08-31 v3; `docs/paper/figures/Figure_1_Overall_Methodological_Framework.svg`; compact 3×3 serpentine layout, external model dev as inset; old v1/v2 files deleted                                                              |
| Figure 2 | Within-scale overlapping-tile fusion example      | One rock duplicated in neighboring same-scale tiles; pre-fusion masks/bboxes/confidence/boundary completeness; selected representative                      | Existing per-tile detections and fusion metadata cited by `PAPER_METHOD_PIPELINE.md`                                          | EVIDENCE GAP: a fixed example and figure-ready before/after records are not identified in the paper package                           |
| Figure 3 | Cross-scale cascade example                       | Same physical rock detected at coarse/medium/fine scales; equivalent diameter, centroid, overlap, quality score, primary-scale selection, retained instance | Existing multi-scale detection/final inventory outputs; cascade rules in `PAPER_METHOD_PIPELINE.md`                           | EVIDENCE GAP: a traceable cross-scale group example is not identified in the paper package                                            |
| Figure 4 | 2D-3D association and filtering                   | Accepted and rejected instance examples; associated point candidates; relative-height statistics; rejection reason                                          | Frozen accepted/rejected association outputs and validation summary cited by `PAPER_EVIDENCE_MAP.md`                          | PARTLY READY; aggregate reasons exist, but fixed visual examples must be selected from existing records without rerunning association |
| Figure 5 | GroundDEM, observed rock surface, and 2.5D volume | Local ground plane/DEM, observed point elevations, 10 mm cells, maximum height per cell, integrated volume relationship                                     | GroundDEM and 2.5D method in `PAPER_METHOD_PIPELINE.md`; existing pilot/checkpoint data if cell-level surfaces were persisted | EVIDENCE GAP: figure-ready DEM/surface arrays are not confirmed in the paper package; no recomputation is authorized                  |
| Figure 6 | Canonical 12-feature concept                      | Footprint geometry variables A/P/L/W/convex area; height distribution H/mean/std/P25/P75/skew; box and ellipsoid reference volumes                          | Formula table in `PAPER_TABLES.md`; feature order in `PAPER_WRITING_BASELINE.md`                                              | GENERATED 2026-08-31; `docs/paper/figures/figure6_12feature_descriptor.html`                                                          |
| Figure 7 | Frozen 4,000-rock real-mine results               | Six size strata, sample/success/failure counts, `y_pred`, `V_2.5D`, `V_pred` distributions, failure reason                                                  | Frozen sampling manifest/report and 4,000 summary/results cited by `PAPER_TABLES.md`                                          | GENERATED 2026-08-31; `docs/paper/figures/figure7_4000rock_results.html`                                                              |

## Figure 1. Overall Framework

### Intended message

展示唯一研究主线，而不是将 segmentation、3D association 和 volume correction 画成互不相关的模块。

```text
DOM2
  -> multi-scale segmentation
  -> within-scale fusion
  -> cross-scale cascade deduplication
  -> 2D-3D association and filtering
  -> GroundDEM
  -> 10 mm 2.5D surface
  -> canonical 12 features
  -> frozen Shape-Aware LightGBM
  -> V_pred
  -> 4,000-rock representative application
```

### Required annotations

* DOM GSD: 0.01 m/pixel.

* Multi-scale coverage: 10.24 m, 5.12 m, 2.56 m.

* External model path shown as a side branch: 465 OBJ -> scale 82.737840 -> scaled 10 mm training -> frozen model.

* Real-mine scope: 69,911 accepted population -> fixed 4,000 sample -> 3,639 successful estimates.

* Do not place `5.82%` on the real-mine branch; label it as scaled external Test MAPE only.

## Figure 2. Within-Scale Fusion Example

### Intended message

同一石块因相邻 tile 重叠而在同一尺度被重复检测，融合过程将重复记录组成一组并保留质量最高的代表 mask。

### Required panels

1. Two neighboring overlapping tiles and the same rock location.
2. Duplicate masks/bboxes before fusion.
3. Decision variables: bbox IoU, mask IoU, centroid score, area ratio, boundary completeness, confidence.
4. Weighted score and threshold 0.50.
5. Final representative selected by `confidence * boundary completeness`.

### Evidence boundary

当前论文底稿只有实现规则和总体计数，没有冻结的示例 rock ID。TRAE 必须从已有检测/融合记录中选择可追溯实例；若中间映射未保存，标记 `EVIDENCE GAP`，不得重新运行分割。

## Figure 3. Cross-Scale Cascade Example

### Intended message

展示同一物理石块在 coarse、medium 和 fine 结果中的重复检测，以及最终如何依据空间、尺寸和 primary scale 规则保留一个实例。

### Required panels

1. Coarse/medium/fine observations of one location.
2. bbox and mask overlap.
3. Diameter ratio >= 0.30 and centroid distance <= larger radius.
4. Primary-scale rule: `<0.30 m` fine, `0.30-0.50 m` medium, `>=0.50 m` coarse.
5. Retained representative and discarded duplicates.

### Evidence boundary

最终主路径是 cascade deduplication。不要把通用 cross-scale weighted fusion 画成产生 76,407 最终实例的算法。现有论文包未固定示例组，因此状态为 `EVIDENCE GAP`，只能从既有记录中取例。

## Figure 4. 2D-3D Association and Filtering

### Intended message

展示 DOM instance 如何进入点云候选查询、GroundDEM-relative statistics 和 acceptance/rejection gate。

### Required panels

1. DOM instance/mask and spatial extent.
2. Associated point-cloud candidates.
3. Relative-height profile or summary statistics.
4. One accepted example.
5. Rejected examples covering available reasons where possible: `too_few_points`, `insufficient_z_range`, `insufficient_p90_height`, `insufficient_elevated_ratio`, `insufficient_ground_points`.

### Required annotations

* Minimum 60 points.

* z-range >= 0.18 m.

* P90 relative height >= 0.12 m.

* Elevated threshold 0.08 m and ratio >= 0.20.

* Acceptance rate 91.50% is a screening result, not independently validated association accuracy.

## Figure 5. GroundDEM and 2.5D Surface

### Intended message

说明局部地面参考如何将绝对高程转为相对高度，以及 10 mm grid 如何从可观测上表面形成 `V_2.5D`。

### Required panels

1. GroundDEM cell and P5 ground elevation concept.
2. Observed point cloud above ground.
3. Ground-relative height `h = z_point - z_ground`.
4. Maximum observed height in each 10 mm occupied cell.
5. `V_2.5D = sum(h_cell * cell_area)`.

### Mandatory limitation annotation

`Observable surface only: hidden lower-rock geometry and burial depth are not recovered.`

如果既有输出没有保存 cell-level DEM/height map，TRAE 不得重新计算，应将该图保留为方法概念图或标记 `EVIDENCE GAP`。

## Figure 6. Canonical 12-Feature Descriptor

### Intended message

将 12 features 分成 footprint geometry、normalized height statistics 和 volume-ratio descriptors 三组，但保持模型输入顺序不变。

### Required content

* Footprint: `C`, `AR`, `solidity`, `compactness`, `eq_diam_ratio`.

* Height: `H_mean/H`, `H_std/H`, `H_p25/H`, `H_p75/H`, `H_skew`.

* Volume ratios: `fill_ratio`, `ellipsoid_ratio`.

* Explicit note: `H_skew_norm = H_skew`, not `H_skew/H`.

* Formula source: `PAPER_TABLES.md`.

This figure is conceptual and must not imply that feature importance or SHAP analysis was performed.

## Figure 7. Real-Mine 4,000-Rock Application

### Intended message

展示固定粒径分层样本的覆盖范围、pipeline completion、失败随粒径层变化的趋势，以及成功记录的预测分布。

### Recommended panels

1. Population and sample flow: 69,911 accepted -> 4,000 fixed sample -> 3,639 success + 361 failure.
2. Sample allocation by S1-S6: 400/600/1,000/1,000/600/400.
3. Completion rate by stratum: 84.0%, 86.3%, 87.5%, 92.3%, 98.0%, 99.75%.
4. `y_pred` distribution, range 0.5541-0.7280, median 0.6797.
5. `V_2.5D` and `V_pred` distributions in m3.
6. Failure reason: all 361 are `empty_2_5d_surface`.

### Mandatory caption boundary

The 4,000-rock run is a deterministic size-stratified representative application, not a full 69,911-rock volume census. The 90.98% value is pipeline completion, not volume accuracy. No per-rock real-mine ground-truth volume was available.

## TRAE Execution Rules

* Use only frozen data and outputs cited in `docs/paper/`.

* Do not rerun segmentation, association, 2.5D reconstruction, model training, or mine inference to create a figure.

* Every example figure must preserve rock ID or another traceable source identifier in its working record.

* Distinguish implementation diagrams from quantitative result figures in captions.

* Do not convert `EVIDENCE GAP` into an invented visual example.

* Preserve the scientific boundaries stated in `PAPER_WRITING_BASELINE.md`.


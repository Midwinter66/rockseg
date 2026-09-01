# EXPERIMENT_LOG.md

> **CURRENT FROZEN EXPERIMENT LOG -- 2026-08-30.** The completed experiments below are authoritative. Earlier V1-oriented entries are retained after the historical divider and are **SUPERSEDED** where inconsistent. The project is now in paper-writing preparation; no new experiment is implied by this log.

## EXP-V2-01: 0.5 mm External OBJ Shape-Aware Validation

| Field | Record |
| --- | --- |
| Purpose | Validate the external mesh 2.5D-to-volume correction methodology. |
| Input | T01 (79) + L01 (386) OBJ meshes. |
| Frozen parameters | 0.5 mm grid; canonical 12 features. |
| Method | 2.5D surface, LightGBM correction of `V_true / V_2.5D`. |
| Result | Dataset B completed and the 0.5 mm external-mesh methodological validation was retained. The `5.82%` metric belongs to the separate scaled-10 mm held-out test recorded in EXP-V2-07. |
| Interpretation | Establishes external mesh methodological evidence only. |
| Limitation | Not a production-resolution or real-mine accuracy result. |
| Status | COMPLETE / FROZEN |

## EXP-V2-02: Resolution Benchmark

| Field | Record |
| --- | --- |
| Purpose | Determine whether the external-mesh and mine resolutions can be used directly. |
| Input | Existing external OBJ cache and 2.5D rasterization. |
| Frozen parameters | Tested candidate grids without retraining. |
| Result | 5-10 mm changed feature values; 25/50 mm produced empty surfaces in the current rasterizer. Original-scale OBJ at 10 mm yielded only 63 valid objects. |
| Interpretation | The original OBJ scale cannot directly support a 10 mm training dataset. |
| Limitation | A resolution-robust model was not demonstrated. |
| Status | COMPLETE; original-scale 10 mm training path REJECTED |

## EXP-V2-03: Point-Cloud Spacing Audit

| Field | Record |
| --- | --- |
| Purpose | Match real-mine 2.5D grid size to observed point sampling. |
| Input | Local, KD-tree-based pointcloud2/pointcloud3 windows; no all-pairs calculation. |
| Frozen parameters | DOM GSD `10 mm/pixel`; deterministic sampled windows. |
| Result | Pointcloud2 XY P90 `6.0-6.4 mm`; pointcloud3 XY P90 `6.32 mm`; 3D P90 about `8.5-8.6 mm`. |
| Interpretation | `10 mm = 0.01 m` is the frozen mine grid. |
| Limitation | Statistics are spatial samples, not a per-stone density guarantee. |
| Status | COMPLETE / FROZEN |

## EXP-V2-04: External OBJ Scale Audit

| Field | Record |
| --- | --- |
| Purpose | Assess a scientifically controlled scale mapping from external OBJ to mine rock size. |
| Input | T01/L01 footprint metadata and Site B footprint statistics. |
| Frozen parameters | Uniform factor `82.737840`; no volume, test, or model-error information used to set it. |
| Result | `SCALE_ADAPTATION_PLAUSIBLE_BUT_NOT_PROVEN`; original-scale 10 mm surfaces were inadequate. |
| Interpretation | Geometric-similarity adaptation is operationally admissible but remains a domain-adaptation assumption. |
| Limitation | External OBJ physical scale is not independently verified against mine lithology. |
| Status | COMPLETE / FROZEN |

## EXP-V2-05: Scaled 10 mm Feasibility Pilot

| Field | Record |
| --- | --- |
| Purpose | Verify the pre-registered scale factor before full dataset construction. |
| Input | Ten T01 and ten L01 objects, stratified by footprint. |
| Frozen parameters | Scale `82.737840`; grid `10 mm`; canonical 12 features. |
| Result | `20/20` valid surfaces; occupied cells `1,634-51,202`; all features finite; no abnormal y-ratio values. |
| Interpretation | `PILOT_PASS`; full scaled dataset construction permitted. |
| Limitation | Feasibility only, not a model-accuracy experiment. |
| Status | COMPLETE / FROZEN |

## EXP-V2-06: Full Scaled 10 mm Dataset QC

| Field | Record |
| --- | --- |
| Purpose | Construct a resolution-matched, resumable external dataset. |
| Input | All 465 T01/L01 OBJ meshes. |
| Frozen parameters | Scale `82.737840`; grid `10 mm`; group-aware split seed `42`. |
| Result | T01 `79/79`, L01 `386/386`; split `326/70/69`; zero leakage, NaN, or Inf. |
| Interpretation | Training input passed QC. |
| Limitation | Scaled geometry remains an adaptation assumption. |
| Status | COMPLETE / FROZEN |

## EXP-V2-07: Scaled 10 mm Shape-Aware LightGBM

| Field | Record |
| --- | --- |
| Purpose | Train one final resolution-matched correction model. |
| Input | Frozen 465-object scaled dataset and fixed split. |
| Frozen parameters | 12 features; LightGBM seed `42`; maximum 500 rounds; early stopping; best iteration `356`. |
| Result | Test Shape-Aware MAPE `5.82%`, MAE `7,167,711 mm3`, RMSE `11,890,780 mm3`, R2 `0.9838`; raw 2.5D MAPE `54.24%`; constant correction MAPE `6.99%`. |
| Interpretation | External scaled-test `PASS`; model frozen. |
| Limitation | These values are not real-mine accuracy. |
| Status | COMPLETE / FROZEN |

## EXP-V2-08: DOM2 12-Rock Transfer Pilot

| Field | Record |
| --- | --- |
| Purpose | Verify the frozen model can complete the mine inference chain. |
| Input | Twelve fixed DOM2 accepted rocks and existing 2D-3D associations. |
| Frozen parameters | 10 mm grid; canonical schema; frozen scaled-10mm model. |
| Result | `12/12` success; `y_pred = 0.6769-0.6939`; `V_2.5D = 0.01018-0.30359 m3`; `V_pred = 0.00692-0.20857 m3`; formula exact. |
| Interpretation | Transfer pipeline passed operational QC. |
| Limitation | No per-rock mine reference volumes. Early temporary affine/mask adapter errors were corrected without changing scientific parameters. |
| Status | COMPLETE / FROZEN |

## EXP-V2-09: DOM2 4,000-Rock Stratified Application

| Field | Record |
| --- | --- |
| Purpose | Apply the frozen pipeline to a representative, size-stratified DOM2 sample. |
| Input | 4,000 deterministic accepted instances from the 69,911-instance population. |
| Frozen parameters | `stratified_quantile_systematic`; S1/S2/S3/S4/S5/S6 = `400/600/1000/1000/600/400`; 10 mm; scale `82.737840`; frozen model. |
| Result | `3,639/4,000` success (`90.98%` pipeline completion); `361` failures, all `empty_2_5d_surface`; feature and formula QC passed. Stratum success: S1 `84.0%`, S2 `86.3%`, S3 `87.5%`, S4 `92.3%`, S5 `98.0%`, S6 `99.75%`. |
| Interpretation | `FINAL_QC_PASS`; failures are size-dependent surface-availability observations, not corrected or resampled away. |
| Limitation | No mine-site per-rock ground truth, therefore no mine absolute-accuracy claim. |
| Status | COMPLETE / FROZEN |

## Paper Writing Handoff

The frozen experiment sequence is EXP-V2-01 through EXP-V2-09. It supplies the
evidence chain for the current manuscript. Chapter 3 is drafted in
`docs/paper/PAPER_DRAFT.md` and `docs/paper/PAPER_DRAFT_CN.md` using six main
method sections. The next writing step is Chapter 4 Results, in the same order
as the experiment evidence:

1. segmentation and duplicate-resolution inventory;
2. 2D-3D association and quality filtering;
3. external scaled-mesh Shape-Aware validation;
4. resolution and scale adaptation;
5. representative real-mine application.

These are writing tasks over frozen outputs, not requests to rerun experiments.
The external Test MAPE and real-mine pipeline completion rate must remain
scientifically distinct.

## Historical Record (SUPERSEDED; retained for provenance)

> 已执行实验记录
> 仅记录有实际输出文件支撑的实验

---

## EXP-01: V1 SAHI 分块检测

| 字段 | 值 |
|------|-----|
| Experiment ID | EXP-01 |
| 日期 | UNKNOWN (2026-07前) |
| 目的 | SAHI固定滑动窗口分块 + YOLO11m-seg检测 |
| 数据集 | dom2 (DOM.tif, GSD=0.01m) |
| 模型 | YOLO11m-seg (models/best.pt) |
| 参数 | patch_size=1024, overlap=0.15, conf=0.35 |
| 运行命令 | UNKNOWN |
| 输出目录 | experiments/detection/outputs/ (具体路径 UNKNOWN) |
| 结果 (from current_results.json) | 250 tiles, 128 kept, 30,993 raw candidates |
| 结论 | 作为baseline, 与Quadtree对比 |
| 保留 | 是, 作为baseline |
| 状态 | BASELINE |

---

## EXP-02: V1 Quadtree_DOM 分块检测

| 字段 | 值 |
|------|-----|
| Experiment ID | EXP-02 |
| 日期 | UNKNOWN (2026-07前) |
| 目的 | 边缘密度引导自适应分块 + YOLO11m-seg检测 |
| 数据集 | dom2 |
| 模型 | YOLO11m-seg |
| 参数 | base_tile=20m, min=10m, max=20m, canny_low=30, canny_high=90 |
| 运行命令 | UNKNOWN |
| 输出目录 | experiments/detection/outputs/ (具体路径 UNKNOWN) |
| 结果 (from current_results.json) | 130 tiles, 98 kept, 30,993 raw → 7,823 filtered |
| 结论 | V1主方法, 比SAHI更高效(98 vs 128 tiles) |
| 保留 | 是, V1主方法 |
| 状态 | SUPPORTED (V1) |

---

## EXP-03: V1 融合 + 3D验证 + 体积估算

| 字段 | 值 |
|------|-----|
| Experiment ID | EXP-03 |
| 日期 | UNKNOWN (2026-07前) |
| 目的 | 完整V1流水线: 检测→融合→3D验证→体积 |
| 数据集 | dom2 + pointcloud2 |
| 模型 | YOLO11m-seg + correlation_clustering |
| 参数 | max_distance_m=3.5, distance_sigma=0.7, min_points=60, min_p90_height=0.12 |
| 输出目录 | docs/results/current_results.json |
| 结果 | 6,933 accepted stones, 3D通过率96.5%, 2.5D体积1,451m³ |
| 结论 | V1主场景完整结果 |
| 保留 | 是, 论文V1结果 |
| 状态 | SUPPORTED |

---

## EXP-04: V2 多尺度级联检测 (DOM2)

| 字段 | 值 |
|------|-----|
| Experiment ID | EXP-04 |
| 日期 | 2026-08-25 |
| 目的 | 物理尺度驱动3尺度检测 + 级联去重 |
| 数据集 | dom2 |
| 模型 | YOLO11m-seg |
| 参数 | scales=coarse(10.24m),medium(5.12m),fine(2.56m), cascade=True |
| 运行命令 | `python run_rockseg.py --dom data/dom2/DOM.tif --model models/best.pt --output output/dom2_cascade_v2 --cascade` |
| 输出目录 | output/dom2_cascade_v2/ |
| 结果 | 76,407 实例 |
| 结论 | V2主检测结果 |
| 保留 | 是, 论文V2主方法 |
| 状态 | SUPPORTED |

---

## EXP-05: V2 3D验证 (float32 bug版)

| 字段 | 值 |
|------|-----|
| Experiment ID | EXP-05 |
| 日期 | 2026-08-25 |
| 目的 | 3D点云验证 (bbox快速版) |
| 数据集 | dom2 + pointcloud2 |
| 参数 | min_points=60, min_z_range=0.18, min_p90_height=0.12 |
| 运行命令 | `python run_3d_validation_fast.py ...` |
| 输出目录 | output/dom2_cascade_v2_3d/ |
| 结果 | 73,896/76,407 通过 = 96.7% |
| 问题 | float32精度bug导致结果不准确 |
| 结论 | **废弃, 被 EXP-06 取代** |
| 状态 | REJECTED (bug) |

---

## EXP-06: V2 3D验证 (float64修复版)

| 字段 | 值 |
|------|-----|
| Experiment ID | EXP-06 |
| 日期 | 2026-08-25 |
| 目的 | 3D点云验证 (float64修复后) |
| 数据集 | dom2 + pointcloud2 |
| 参数 | 同EXP-05, float32→float64 |
| 输出目录 | output/dom2_cascade_v2_3d_fixed/ |
| 结果 | 69,911/76,407 通过 = 91.5%, 拒绝6,496 |
| 结论 | 修复后的正确结果 |
| 保留 | 是 |
| 状态 | SUPPORTED |

---

## EXP-07: V2 体积估算 (float32 bug版)

| 字段 | 值 |
|------|-----|
| Experiment ID | EXP-07 |
| 日期 | 2026-08-25 |
| 目的 | 2.5D体积估算 + shape-aware校正 |
| 输出目录 | output/dom2_cascade_v2_volume/ |
| 结果 | 总体积160m³ (严重低估) |
| 问题 | float32精度bug, 高度图覆盖仅25% |
| 结论 | **废弃, 被 EXP-08 取代** |
| 状态 | REJECTED (bug) |

---

## EXP-08: V2 体积估算 (float64修复版)

| 字段 | 值 |
|------|-----|
| Experiment ID | EXP-08 |
| 日期 | 2026-08-25 |
| 目的 | 体积估算 (float64修复后) |
| 输出目录 | output/dom2_cascade_v2_volume_fixed/ |
| 结果 | 65,826有效实例, 总体积999m³ (shape-aware), 1,359m³ (2.5D raw) |
| 校正比 | 0.7348 (mean), α=0.731 (linear) |
| 结论 | 修复后的正确结果, V/(fp×h_max)中位数0.570符合岩石形态 |
| 保留 | 是, 论文体积结果 |
| 状态 | SUPPORTED |

---

## EXP-09: E5 体积验证实验 V1 (5特征)

| 字段 | 值 |
|------|-----|
| Experiment ID | EXP-09 |
| 日期 | UNKNOWN |
| 目的 | 外部OBJ数据验证体积估算方法 (5方法消融) |
| 数据集 | data/experience_rock/T01/ (79个OBJ) |
| 模型 | LightGBM, 5特征, 1棵树6叶 |
| 参数 | grid_res=0.5mm, train=55, val=12, test=12 |
| 输出目录 | research_v2/volume_validation/output/results/ |
| 结果 | Shape-Aware MAPE=7.89%, Linear=8.05%, 2.5D=31.68% |
| 观察 | 模型退化为常数, 校正比0.731~0.736, 与线性几乎无差别 |
| 结论 | 验证了2.5D+校正方法, 但shape-aware优势微弱 |
| 保留 | 是, V1 baseline |
| 状态 | SUPPORTED (V1 baseline) |

---

## EXP-10: E5 体积验证实验 V2 (12特征增强)

| 字段 | 值 |
|------|-----|
| Experiment ID | EXP-10 |
| 日期 | 2026-08-26 |
| 目的 | 增强shape-aware模型: 5→12特征, 优化超参数 |
| 数据集 | data/experience_rock/T01/ (79个OBJ) |
| 模型 | LightGBM, 12特征, best_iter=27 |
| 参数 | num_leaves=12, lr=0.02, early_stopping=100 |
| 输出目录 | research_v2/volume_validation/output_v2_enhanced/ |
| 结果 | Shape-Aware V2 MAPE=7.58%, Linear=7.57% (改进仅0.3%) |
| CV结果 | 5折CV MAPE=6.99% ± 1.35% |
| 观察 | 新增特征: solidity, H_p25, H_p75, H_skew, fill_ratio等。校正比std仅0.0067 |
| 问题 | 校正比r变异系数仅9%, 形状特征能解释的方差有限 |
| 结论 | 改进有限, V2模型未部署到生产代码 |
| 保留 | 是, 但需要重新评估策略 |
| 状态 | TESTING |

---

## 其他运行过的方法

| 方法 | 运行过 | 结果 | 状态 |
|------|--------|------|------|
| FCN | UNKNOWN | UNKNOWN | UNKNOWN |
| DeepLab | UNKNOWN | UNKNOWN | UNKNOWN |
| LRASPP | UNKNOWN | UNKNOWN | UNKNOWN |

注: experiments/ 目录中存在 config 和代码, 但是否有实际运行结果 UNKNOWN。current_results.json 中仅报告了 YOLO11m-seg 的结果。

# Paper Writing Baseline

Status: `FROZEN WRITING BASELINE`

本文件只整理 `docs/paper/` 中已经冻结的事实、方法证据和科学边界。它不是新实验记录，不改变模型、数据集、生产代码或既有结果。

## A. 唯一研究主线

```text
DOM
  -> Multi-scale DOM instance segmentation
  -> Same-tile / within-scale fusion across overlapping tiles
  -> Cross-scale fusion logic and final cascade deduplication
  -> 2D-3D association
  -> 3D filtering
  -> Ground / DEM construction
  -> 2.5D surface reconstruction
  -> Canonical 12-feature Shape-Aware descriptor
  -> LightGBM volume correction
  -> External mesh validation
  -> Resolution / scale adaptation
  -> Real-mine 4,000-rock volume estimation
```

论文应围绕一个连续问题展开：如何通过物理尺度驱动的多尺度分割、分辨率匹配和 Shape-Aware 体积校正，将 UAV DOM 与矿区点云转化为可复现的可观测石块体积估计。

### 术语固定

- `DOM GSD`：影像地面采样距离，冻结为 `0.01 m/pixel`。
- `point spacing`：点云局部点间距统计，不等于影像分辨率。
- `2.5D grid resolution`：表面重建与体积积分的计算栅格，冻结为 `0.01 m`。
- `within-scale fusion across overlapping tiles`：同一物理尺度下，重叠 tile 产生的重复实例融合。论文中不应将其误写为“单个 tile 内部融合”。
- `cross-scale cascade deduplication`：不同物理尺度结果之间的级联去重，而不是跨尺度 mask 几何求并。
- `observable rock volume estimate`：基于可见点云表面和局部地面参考的估计；不代表不可见埋深或被遮挡几何已恢复。

## B. 方法与论文证据

| Paper Section | Method / Claim | Existing Evidence | Source File | Quantitative Result | Status |
| --- | --- | --- | --- | --- | --- |
| Study Area and Data | DOM2 is the image source used by the frozen pipeline | GSD and frozen inventory provenance | `PAPER_FINAL_STATUS.md`; `PAPER_TABLES.md` | 0.01 m/pixel | PASS |
| Multi-scale segmentation | Physical coverage controls coarse/medium/fine detection | Raw and final per-scale counts | `PAPER_TABLES.md`; `PAPER_METHOD_PIPELINE.md` | 10.24/5.12/2.56 m; 318,398 raw; 76,407 final | PASS for inventory; segmentation accuracy is EVIDENCE GAP |
| Same-tile fusion | Same-scale duplicates from overlapping tiles are grouped and represented by the best mask | Implemented spatial, overlap, and weighted-score rules; aggregate fused pool | `PAPER_METHOD_PIPELINE.md`; cited `rockseg/fusion.py`, `rockseg/config.py` | threshold 0.50; 112,983 fused pool | IMPLEMENTATION_VERIFIED |
| Cross-scale fusion | Generic cross-scale fusion logic compares detections from different scales | Implementation is documented, but it is not the final selection path used for 76,407 | `PAPER_METHOD_PIPELINE.md`; cited `rockseg/fusion.py` | generic threshold 0.55 in frozen config | IMPLEMENTATION_VERIFIED; do not equate with final cascade output |
| Cascade strategy | Compatible cross-scale duplicates are grouped, then one primary-scale representative is retained | Implemented bbox/mask/diameter/centroid rules and final inventory | `PAPER_METHOD_PIPELINE.md`; `PAPER_TABLES.md` | 112,983 to 76,407; diameter boundaries 0.30/0.50 m | PASS for implementation and counts; fusion accuracy is EVIDENCE GAP |
| 2D-3D association | DOM instance extent is used to query point-cloud candidates through the existing spatial index | Fixed float64 screening inventory and implementation record | `PAPER_EVIDENCE_MAP.md`; `PAPER_METHOD_PIPELINE.md` | 76,407 records screened | IMPLEMENTATION_VERIFIED; correspondence accuracy is EVIDENCE GAP |
| 3D filtering | Ground-relative point and height statistics determine acceptance | Fixed thresholds and rejection reasons | `PAPER_TABLES.md`; `PAPER_METHOD_PIPELINE.md` | 69,911 accepted; 6,496 rejected; 91.50% | PASS for screening outcome |
| Ground / DEM | Scene-level P5 GroundDEM supplies local ground elevation | Frozen implementation description | `PAPER_METHOD_PIPELINE.md`; `PAPER_EVIDENCE_MAP.md` | 0.5 m grid; every 100th point; >=3 points/cell | IMPLEMENTATION_VERIFIED; DEM accuracy is EVIDENCE GAP |
| 2.5D reconstruction | Maximum observed ground-relative height is integrated on the fixed grid | Successful mine-output and formula QC | `PAPER_METHOD_PIPELINE.md`; `PAPER_TABLES.md` | 10 mm grid; 3,639 positive successful estimates | PASS for operational output |
| Shape-Aware 12 features | Train and mine inference use one canonical ordered descriptor | Formula table and consistency check | `PAPER_TABLES.md`; cited `shape_features_v2.py` and consistency report | 12 features; checked max absolute/relative difference 0 | PASS |
| External dataset | Known-volume T01/L01 OBJ meshes provide `V_true` and correction target | Dataset composition and frozen split | `PAPER_FINAL_STATUS.md`; `PAPER_TABLES.md` | 79 + 386 = 465; split 326/70/69 | PASS |
| 0.5 mm external validation | Establishes external-mesh methodological feasibility only | Status is frozen, but the approved five-document package lacks the exact metric table | `PAPER_FINAL_STATUS.md` | Exact numerical table NOT VERIFIED in approved input | EVIDENCE GAP for numerical citation; methodological status PASS |
| Scale audit | Uniform geometric scale adapts external footprints to mine-size range | Audit, spacing evidence, pilot, and full scaled dataset are summarized | `PAPER_FINAL_STATUS.md`; `PAPER_TABLES.md` | scale 82.737840; 63/465 unscaled valid; pilot 20/20 | PASS for feasibility; physical validity NOT PROVEN |
| Scaled 10 mm model | Frozen LightGBM corrects `V_2.5D` using 12 features | Held-out Test and baseline comparison | `PAPER_TABLES.md`; frozen model results cited there | Shape-Aware MAPE 5.82%, R2 0.9838; best iteration 356 | PASS on scaled external Test only |
| Real-mine 4,000-rock estimation | Frozen model is applied to a deterministic diameter-stratified sample | Manifest and final QC summary | `PAPER_FINAL_STATUS.md`; `PAPER_TABLES.md` | 4,000 total; 3,639 success; 361 fail; 90.98% completion | PASS for application; absolute mine accuracy is EVIDENCE GAP |

## C. 论文固定关键数字

### C1. DOM2 segmentation and fusion

| Item | Frozen value |
| --- | ---: |
| DOM GSD | 0.01 m/pixel |
| Coarse coverage / tiles / raw detections | 10.24 m / 297 / 37,470 |
| Medium coverage / tiles / raw detections | 5.12 m / 1,188 / 101,642 |
| Fine coverage / tiles / raw detections | 2.56 m / 4,708 / 179,286 |
| Total raw detections | 318,398 |
| Within-scale fused pool | 112,983 |
| Final cascade instances | 76,407 |
| Final coarse / medium / fine instances | 5,925 / 10,890 / 59,592 |

Per-scale post-within-scale-fusion counts are `NOT VERIFIED` and must not be inferred from the aggregate 112,983.

### C2. Point cloud, association, and GroundDEM

| Item | Frozen value |
| --- | ---: |
| BlockB XY P90 point spacing | 6.00 mm |
| BlockB 3D P90 point spacing | 8.54 mm |
| BlockY XY P90 point spacing | 6.40 mm |
| BlockY 3D P90 point spacing | 8.60 mm |
| Final instances screened | 76,407 |
| Accepted / rejected | 69,911 / 6,496 |
| Accepted rate | 91.50% |
| Minimum point count | 60 |
| Minimum z-range | 0.18 m |
| Minimum P90 height above ground | 0.12 m |
| Elevated-height threshold / minimum elevated ratio | 0.08 m / 0.20 |
| GroundDEM grid / statistic / subsampling | 0.5 m / P5 / every 100th point |
| 2.5D grid resolution | 0.01 m (10 mm) |

Point spacing、DOM GSD 和 2.5D grid resolution 是三个不同概念。现有 spacing 统计支持 10 mm 作为可操作候选分辨率，但不证明每个石块都具有相同点密度。

### C3. External Dataset B and scale adaptation

| Item | Frozen value |
| --- | ---: |
| T01 / L01 / total | 79 / 386 / 465 OBJ meshes |
| Original external rasterization | 0.5 mm |
| Original-scale valid surfaces at 10 mm | 63 / 465 |
| Uniform scale factor | 82.737840 |
| Scale-audit conclusion | `SCALE_ADAPTATION_PLAUSIBLE_BUT_NOT_PROVEN` |
| Scaled 10 mm pilot | 20 / 20 valid |
| Pilot occupied cells | 1,634 to 51,202 |
| Full scaled 10 mm dataset | 465 / 465 successful |
| Train / validation / test | 326 / 70 / 69 |
| Split leakage / non-finite features | 0 / 0 |

0.5 mm 模型只保留为 external mesh methodological validation。当前五份冻结论文文档没有给出其完整指标表，因此本基线不补写数值；需要引用时标记 `EVIDENCE GAP`，再由作者指定已冻结结果文件核验。

### C4. Canonical 12-feature schema

固定顺序：

1. `C = min(4*pi*A/P^2, 1)`
2. `AR = L/W`
3. `solidity = min(A/A_convex, 1)`
4. `compactness = P/sqrt(A)`
5. `eq_diam_ratio = sqrt(4*A/pi)/L`
6. `H_mean_norm = H_mean/H`
7. `H_std_norm = H_std/H`
8. `H_p25_norm = H_p25/H`
9. `H_p75_norm = H_p75/H`
10. `H_skew_norm = H_skew`，不除以 `H`
11. `fill_ratio = V_2.5D/V_box`
12. `ellipsoid_ratio = V_2.5D/V_ellipsoid`

### C5. Frozen scaled-10 mm model

| Item | Frozen value |
| --- | ---: |
| Model | LightGBM regression |
| Target | `y_ratio = V_true/V_2.5D` |
| Learning rate / num leaves | 0.02 / 12 |
| Minimum child samples | 8 |
| Subsample / frequency | 0.9 / 1 |
| Column sample by tree | 0.9 |
| L1 / L2 regularization | 0.2 / 0.5 |
| Minimum gain to split | 0.0001 |
| Maximum rounds / early stopping | 500 / 100 |
| Random seed | 42 |
| Best iteration | 356 |
| Test objects | 69 |
| Ratio MAE / RMSE | 0.0374 / 0.0451 |
| Ratio MAPE / R2 | 5.82% / 0.3280 |

| Test method | MAE (mm3) | RMSE (mm3) | MAPE | R2 |
| --- | ---: | ---: | ---: | ---: |
| Raw 2.5D | 59,903,925 | 84,139,845 | 54.24% | 0.1895 |
| Constant correction | 8,723,245 | 16,049,133 | 6.99% | 0.9705 |
| Shape-Aware V2 | 7,167,711 | 11,890,780 | 5.82% | 0.9838 |

以上全部是 scaled external OBJ held-out Test 结果，不是 DOM2 真实矿区误差。

### C6. Real-mine 4,000-rock application

| Item | Frozen value |
| --- | ---: |
| Accepted population | 69,911 |
| Sampling method | `stratified_quantile_systematic` |
| Frozen sample | 4,000 |
| S1 / S2 / S3 / S4 / S5 / S6 | 400 / 600 / 1,000 / 1,000 / 600 / 400 |
| Successful / failed inference | 3,639 / 361 |
| Pipeline completion rate | 90.98% |
| Failure reason | `empty_2_5d_surface` only |
| S1 / S2 / S3 success rate | 84.0% / 86.3% / 87.5% |
| S4 / S5 / S6 success rate | 92.3% / 98.0% / 99.75% |

| Successful estimates | Min | Median | Max |
| --- | ---: | ---: | ---: |
| `V_2.5D` (m3) | 2.99e-09 | 0.0015928 | 4.3480 |
| `y_pred` | 0.5541 | 0.6797 | 0.7280 |
| `V_pred` (m3) | 1.99e-09 | 0.0010305 | 2.9962 |

`4,000` 是固定的粒径分层代表性样本；`3,639` 是成功完成推理的样本数。二者都不是整个矿区全部石块数量。

## D. 科学问题检查

### D1. DOM resolution

论文可以表述为：DOM2 processing used a ground sampling distance of 0.01 m per pixel。该值有冻结配置和方法文档支持。不能仅由该值推导 segmentation accuracy 或 point-cloud density。

### D2. Point-cloud resolution

论文应使用 `sampled local point spacing`，不要笼统写成 `point-cloud resolution = 10 mm`。BlockB/BlockY 的 XY P90 为 6.00/6.40 mm，3D P90 为 8.54/8.60 mm；10 mm 是随后选定的 2.5D operational grid。

### D3. DEM / ground surface

GroundDEM 为每个观测点提供局部地面高程参考，使石块高度表示为 `z_point - z_ground`。它支持 ground-referenced 2.5D surface，但没有独立的 DEM vertical-accuracy benchmark。

### D4. Visible-rock limitation

DOM 和空中点云主要观测上表面。当前估计针对 `surface-visible / observable rocks`，不能恢复埋藏体、下层遮挡石块完整几何或不可见埋深。

### D5. Shape-Aware evidence levels

- 0.5 mm：external OBJ methodological validation only。
- Scaled 10 mm：resolution-matched external dataset and frozen LightGBM validation。
- Real mine：frozen model 的 application and pipeline-completion evidence。
- 不得把 external Test MAPE 5.82% 写成 real-mine accuracy。

### D6. Real-mine sample scope

69,911 是 accepted inventory；4,000 是 deterministic size-stratified manifest；3,639 是成功完成 10 mm Shape-Aware inference 的记录。论文中三者必须保持区分。

## E. EVIDENCE GAP Register

| Gap | What is missing | Why it matters | Can existing results support it? |
| --- | --- | --- | --- |
| Segmentation accuracy | Independent manual labels and precision/recall/mAP or equivalent | 76,407 是输出数量，不是检测精度 | No; existing counts only support inventory scale |
| Fusion accuracy | Manually adjudicated duplicate/false-merge benchmark | Count reduction alone不能证明融合正确率 | No; implementation and stage counts can be reported only |
| Per-scale post-within-scale counts | Coarse/medium/fine fused counts before cascade | Needed only for a detailed per-scale fusion table | Not in the five approved documents; aggregate 112,983 is supported |
| 2D-3D association accuracy | Independently labeled image-point correspondence set | Accepted rate is a quality-gate outcome, not correspondence accuracy | No; report workflow and screen outcome only |
| DEM vertical accuracy | Ground checkpoints or reference DEM comparison | Affects absolute ground-relative height uncertainty | No; method is implementation-verified only |
| Occluded/buried rock geometry | Subsurface or multi-view ground truth | Needed for complete physical rock volume | No; explicitly outside observable 2.5D scope |
| Scale-adaptation physical validity | Independent evidence that scaled OBJ geometry represents mine-rock domain | Determines domain-transfer strength | Partly; feasibility is supported, absolute validity is not proven |
| Real-mine absolute volume accuracy | Per-rock reference volume for DOM2 | Required for mine-site MAE/MAPE/R2 | No; must not transfer external metrics |
| 0.5 mm detailed metric table | Exact frozen result source is not identified in the approved five-document input | Needed only if numerical 0.5 mm comparison is included | Potentially, but requires author-approved frozen result-file lookup |
| Literature and novelty positioning | Verified external literature matrix and citations | Required for Introduction and Discussion | No literature package was part of this task; sources pending |

这些缺口只限定论文表述，不自动触发任何新实验。

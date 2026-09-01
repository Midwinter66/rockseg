# TODO.md

> **CURRENT FROZEN WORK STATUS -- 2026-08-30.** This section supersedes the historical task queue below. It does not authorize changes to models, Dataset B, sampling, scale, resolution, or production algorithms.

## COMPLETED / FROZEN

- Canonical 12-feature schema, feature order, and training/production-adapter consistency check.
- 0.5 mm external OBJ Shape-Aware V2 methodological validation.
- Resolution study and point-cloud spatial-sampling analysis.
- Scale audit and frozen uniform scale factor `82.737840`.
- Scaled 10 mm 20-object pilot (`20/20` valid).
- Scaled 10 mm 465-object dataset, QC, and group-aware split (`326/70/69`).
- Frozen scaled-10mm LightGBM V2 (`best_iteration = 356`; external test MAPE `5.82%`).
- DOM2 12-rock real-mine inference pilot (`12/12` success).
- Deterministic 4,000-rock size-stratified manifest and DOM2 representative volume application.
- 4,000-rock final QC (`3,639` success, `361` empty-surface failures, `FINAL_QC_PASS`).
- Chapter 3 methodology draft consolidated into six main sections, with English and Chinese versions in `docs/paper/`.

## NEXT: PAPER WRITING

1. Draft Chapter 4 Results using the frozen tables and result reports.
2. Connect Sections 4.1--4.5 directly to the corresponding Chapter 3 methods.
3. Keep external scaled-mesh metrics separate from real-mine application outcomes.
4. Draft Discussion and Conclusion after the Results chapter is internally consistent.
5. Prepare the Chinese-to-English manuscript translation and final claim audit.
6. Define PSD/P80 analysis only as a subsequent approved analytical stage; no PSD result currently exists.

## SUPERSEDED / NOT PLANNED

- Retraining the 0.5 mm V2 model.
- Training from the invalid original-scale 10 mm 63-sample dataset.
- Re-running all 69,911 accepted rocks without a separately approved scope.
- Unnecessary historical-model comparisons or parameter searches.
- Editing `rockseg/volume.py` merely to replace frozen research outputs.
- Generating images, PPT, or new visual analyses during the current documentation stage.

## Historical Record (SUPERSEDED; retained for provenance)

The task tables below are historical snapshots from before the scaled-10 mm
real-mine application was frozen. They are retained only to preserve project
provenance. They are not an active task queue, and their `待完成` / `CURRENT`
labels must not override the current `NEXT: PAPER WRITING` section above.

> 可执行任务清单

---

## P0 — Critical

| ID | 任务 | 状态 | 说明 |
|----|------|------|------|
| P0-1 | 部署V2 shape-aware模型到volume.py | 待完成 | V2模型(12特征)已训练但未集成到生产代码, volume.py仍用V1(5特征) |
| P0-2 | 检测精度评估 (mAP/Precision/Recall) | 待完成 | 无人工标注ground truth, 无法量化检测准确率。需标注1-2个窗口 |
| P0-3 | 不确定性分析 (GUM级别) | 待完成 | Measurement期刊硬性要求, 当前完全缺失 |

## P1 — Core Research

| ID | 任务 | 状态 | 说明 |
|----|------|------|------|
| P1-1 | 多尺度 vs 单尺度消融实验 | 待完成 | 证明物理尺度多尺度优于固定尺度, 论文核心论据 |
| P1-2 | 边界融合消融 (有无融合对比) | 待完成 | 证明融合算法减少重复检测 |
| P1-3 | 级联去重定量分析 | 待完成 | 统计跨尺度重复检测的合并效果 |
| P1-4 | 用全部868样本重训shape-aware | 被阻塞 | L01/L02数据不在本地, 无法执行 |
| P1-5 | DOM3场景交叉验证 | 待完成 | 数据已有(data/dom3), 未运行, 可验证方法泛化性 |
| P1-6 | P80粒径分布计算 | 待完成 | 代码中未找到实现, 论文需要 |

## P2 — Engineering

| ID | 任务 | 状态 | 说明 |
|----|------|------|------|
| P2-1 | 安装 pandas/trimesh | 被阻塞 | sandbox权限限制(WinError 5), 需在conda环境手动安装 |
| P2-2 | V1/V2代码统一 | 待完成 | 两套代码并存(experiments/ vs rockseg/), 部分功能重叠 |
| P2-3 | 清理临时分析脚本 | 待完成 | 根目录有大量临时脚本(debug_*.py, compare_*.py等) |
| P2-4 | V2模型特征对齐 | 待完成 | shape_descriptors.py已扩展为12特征, 但volume.py仍用5特征, 需同步 |

## P3 — Optional

| ID | 任务 | 状态 | 说明 |
|----|------|------|------|
| P3-1 | 跨域验证 (陨石训练→岩石测试) | 待完成 | 用L01/L02训练, T01测试, 验证可迁移性 |
| P3-2 | 端到端体积误差评估 | 待完成 | 在标注窗口做完整流水线误差评估 |
| P3-3 | 噪声鲁棒性分析 | 待完成 | 模拟点云噪声/稀疏度对体积的影响 |

---

## 已完成

| ID | 任务 | 完成日期 | 说明 |
|----|------|---------|------|
| DONE-1 | V2多尺度检测 (DOM2) | 2026-08-25 | 76,407实例, 3尺度级联 |
| DONE-2 | 3D点云验证 | 2026-08-25 | 91.5%通过率(float64修复后) |
| DONE-3 | 体积估算 | 2026-08-25 | 999m³(float64修复后) |
| DONE-4 | float32精度bug修复 | 2026-08-25 | 4文件8处, float32→float64 |
| DONE-5 | E5体积验证V1 | UNKNOWN | 5方法消融, MAPE=7.89% |
| DONE-6 | E5体积验证V2增强 | 2026-08-26 | 12特征, MAPE=7.58%, CV=6.99% |
| DONE-7 | SimpleTreeModel fallback | UNKNOWN | 纯numpy解析LightGBM模型 |
| DONE-8 | V1完整流水线 | UNKNOWN | SAHI+Quadtree对比, 6,933石块 |

## 已放弃

| ID | 任务 | 放弃原因 |
|----|------|---------|
| DROP-1 | 安装pandas/trimesh via pip | sandbox权限拒绝, 无法在TRAE内安装 |
| DROP-2 | 增加LightGBM特征改进shape-aware | 从5→12特征仅改善0.3%, 边际收益过低 |
| DROP-3 | DOM3处理 (此前) | 用户指示跳过DOM3, 专注3D筛查和体积 |

---

## Historical Snapshot (NOT CURRENT ACTION)

The former action below is retained only as a record of the pre-freeze
engineering state. It is **SUPERSEDED** and must not be executed during the
current paper-writing phase:

> Deploy the V2 shape-aware model to `volume.py` by upgrading
> `predict_shape_aware` from five to twelve features and changing the model
> path to `shape_aware_model_v2.txt`.

The current real-mine result was produced by the frozen scaled-10 mm
application path documented in `docs/paper/`. No production-code deployment
or replacement of the frozen result is authorized by this handoff.

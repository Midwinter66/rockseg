# DECISIONS.md

> 重要研究与工程决策记录

> **CURRENT PAPER-WRITING BASELINE -- 2026-08-30.** The decisions below are
> retained for provenance. The current frozen paper chain is the scaled-10 mm
> Shape-Aware V2 workflow documented in `docs/paper/`; any older status that
> conflicts with this section is HISTORICAL / SUPERSEDED.

## Current final decisions

| Decision | Current status |
| --- | --- |
| Main research line | Physical-scale multi-scale DOM segmentation -> 2D-3D association -> GroundDEM -> 10 mm 2.5D -> canonical 12-feature Shape-Aware correction -> representative mine application |
| External 0.5 mm model | External mesh methodological validation only; not used for the mine application |
| Real-mine model | Scaled 10 mm Shape-Aware V2 LightGBM; scale factor `82.737840`; 10 mm grid; model and parameters unchanged |
| Real-mine scope | Frozen 4,000-rock diameter-stratified sample from 69,911 accepted instances; no 69,911-rock full volume run |
| Paper status | Chapter 3 draft complete in `docs/paper/PAPER_DRAFT.md` and `PAPER_DRAFT_CN.md`; Chapter 4 Results is the next writing task |
| Scientific boundary | External Test metrics are not real-mine accuracy; mine completion rate is not volume accuracy; observable geometry is not buried geometry |

No new experiment, retraining, production-code modification, or result
replacement is authorized by this handoff state.

---

## DEC-01: 选择 YOLO11m-seg 作为检测模型

| 字段 | 值 |
|------|-----|
| Decision ID | DEC-01 |
| Decision | 使用 YOLO11m-seg (ultralytics) 作为实例分割模型 |
| Date | UNKNOWN |
| Alternatives | FCN, DeepLab, LRASPP (experiments/ 中有配置但运行结果 UNKNOWN) |
| Why | YOLO11m-seg 在实例分割任务上性能优秀, ultralytics 生态成熟, 支持 imgsz=1024 大输入 |
| Evidence | models/best.pt (45.2MB), experiments/configs/detection/default.json |
| Status | FINAL |

---

## DEC-02: 采用 Physical-scale-driven Multi-scale 架构

| 字段 | 值 |
|------|-----|
| Decision ID | DEC-02 |
| Decision | V2主方法采用物理尺度驱动多尺度: coarse(10.24m), medium(5.12m), fine(2.56m) |
| Date | UNKNOWN (V2开发期间) |
| Alternatives | SAHI固定窗口(1024px), Quadtree自适应, 单尺度检测 |
| Why | 固定像素窗口无法同时覆盖0.5m小石块和3.5m大石块的最优检测尺寸; 物理尺度匹配确保每个石块在网络输入中保持合理像素比例 |
| Evidence | rockseg/config.py (ScaleConfig 3尺度), rockseg/pipeline.py, README.md |
| Status | FINAL |

---

## DEC-03: 使用 Cascade Deduplication 替代跨尺度融合

| 字段 | 值 |
|------|-----|
| Decision ID | DEC-03 |
| Decision | V2采用级联去重(cascade_deduplication)替代cross_scale_fusion处理跨尺度重复 |
| Date | 2026-08-25 |
| Alternatives | cross_scale_fusion (fusion.py中也实现了, use_cascade=False时使用) |
| Why | 级联去重按粒径选择主尺度(coarse≥0.5m, medium 0.3-0.5m, fine<0.3m), 更符合物理逻辑: 每个粒径的石块应由最适合的尺度检测 |
| Evidence | rockseg/fusion.py (cascade_deduplication函数), run_rockseg.py --cascade 标志 |
| Status | FINAL |

---

## DEC-04: SAHI 作为 Baseline, 不作为主方法

| 字段 | 值 |
|------|-----|
| Decision ID | DEC-04 |
| Decision | SAHI 固定滑动窗口仅作为对比baseline |
| Date | UNKNOWN |
| Alternatives | 可作为主方法, 但不考虑物理尺度 |
| Why | SAHI固定patch_size=1024px, 在GSD=0.01m下覆盖10.24m, 小石块仅占~50px, 检测分辨率不足 |
| Evidence | experiments/configs/slicing/sahi.json, current_results.json (SAHI 250 tiles) |
| Status | FINAL |

---

## DEC-05: Quadtree_DOM 作为 V1 方法, V2不再使用

| 字段 | 值 |
|------|-----|
| Decision ID | DEC-05 |
| Decision | Quadtree边缘密度引导分块仅用于V1, V2改用物理尺度多尺度 |
| Date | UNKNOWN |
| Alternatives | 继续使用Quadtree作为V2分块方法 |
| Why | Quadtree基于图像纹理(边缘密度)分块, 不直接对应物理尺度; V2的物理尺度分块更直接解决"石块像素比例"问题 |
| Evidence | experiments/configs/slicing/quadtree_dom.json (V1), rockseg/tiling.py (V2) |
| Status | FINAL |

---

## DEC-06: 2D-3D Association 设计方向

| 字段 | 值 |
|------|-----|
| Decision ID | DEC-06 |
| Decision | 2D mask(footprint) + 点云(高度) → 2.5D高度图, 非完整3D重建 |
| Date | UNKNOWN |
| Alternatives | 完整3D Mesh重建, 凸包体积 |
| Why | 2.5D高度图计算简单且稳定, 不需要密集点云; 3D Mesh重建在稀疏点云下不可靠; 凸包体积仅作为诊断工具 |
| Evidence | rockseg/volume.py (extract_height_map函数), README.md (明确排除convex hull) |
| Status | FINAL |

---

## DEC-07: Volume Estimation 主方法

| 字段 | 值 |
|------|-----|
| Decision ID | DEC-07 |
| Decision | 当前主方法: 10 mm ground-referenced 2.5D integration + scaled-10 mm Shape-Aware LightGBM correction; constant correction 仅作为外部 Test comparison |
| Date | UNKNOWN |
| Alternatives | Bounding Box, Ellipsoid, Convex Hull |
| Why | 2.5D integration provides the observed ground-referenced geometric base, while the scaled-10 mm Shape-Aware model predicts a correction ratio from the canonical 12-feature descriptor. |
| Evidence | Frozen scaled external Test: Shape-Aware MAPE `5.82%`, volume R2 `0.9838`; raw 2.5D and constant correction are comparison methods. |
| Status | FINAL / FROZEN |

---

## DEC-08: Particle Size 定义为等效直径

| 字段 | 值 |
|------|-----|
| Decision ID | DEC-08 |
| Decision | 粒径 = 等效圆直径 d = sqrt(4A/π), A为footprint面积 |
| Date | UNKNOWN |
| Alternatives | 长轴L, 短轴W, bbox对角线 |
| Why | 等效直径是矿业标准, 与筛分粒度直接可比 |
| Evidence | current_results.json (diameter_m统计), config中 min_stone_diameter_m=0.5 |
| Status | FINAL |

---

## DEC-09: P80 计算思路

| 字段 | 值 |
|------|-----|
| Decision ID | DEC-09 |
| Decision | P80 = 体积累积分布的第80百分位对应的等效直径 |
| Date | UNKNOWN |
| Alternatives | 按数量累积, 按面积累积 |
| Why | 按体积累积更符合矿业P80定义(80%体积通过该筛孔) |
| Evidence | 无代码实现 |
| Status | PROPOSED |

---

## DEC-10: float32 → float64 精度修复

| 字段 | 值 |
|------|-----|
| Decision ID | DEC-10 |
| Decision | 将点云坐标存储从float32改为float64 |
| Date | 2026-08-25 |
| Alternatives | 使用局部坐标(减去基准点)后float32 |
| Why | UTM坐标(4,678,551)在float32下量化为0.5m间隔, 导致高度图覆盖从~100%降至25%, 体积低估6.2倍 |
| Evidence | 修复前: 体积160m³, 修复后: 999m³; debug脚本确认Y坐标量化 |
| Status | FINAL |

---

## DEC-11: shape-aware模型使用 LightGBM 而非神经网络

| 字段 | 值 |
|------|-----|
| Decision ID | DEC-11 |
| Decision | 使用LightGBM梯度提升树预测校正比r |
| Date | UNKNOWN |
| Alternatives | MLP神经网络 (导师提到"神经网络", 但实际用的是LightGBM) |
| Why | 79个样本太少, 深度学习容易过拟合; LightGBM对小数据集更稳健; 模型可保存为文本格式便于部署 |
| Evidence | shape_aware_model.txt (LightGBM文本格式), volume.py SimpleTreeModel fallback |
| Status | FINAL / FROZEN for scaled-10 mm V2 |
| Note | The current paper model is the scaled-10 mm 12-feature LightGBM. Earlier V1 and unscaled paths are historical and are not the real-mine model. |

---

## DEC-12: 跳过 DOM3 矿区处理

| 字段 | 值 |
|------|-----|
| Decision ID | DEC-12 |
| Decision | 跳过DOM3矿区检测, 专注3D筛查和体积估算 |
| Date | 2026-08-25 |
| Alternatives | 运行DOM3完整流水线验证方法一致性 |
| Why | 用户指示, 优先完成DOM2的3D筛查和体积估算 |
| Evidence | 无DOM3输出文件 |
| Status | FINAL (但DOM3可用于后续交叉验证) |

---

## DEC-13: 跳过精度评估

| 字段 | 值 |
|------|-----|
| Decision ID | DEC-13 |
| Decision | 暂时跳过基于人工标注草稿的精度评估 |
| Date | 2026-08-25 |
| Alternatives | 使用947条草稿标注做mAP评估 |
| Why | 草稿标注质量不足以作为ground truth, 缺少实际体积值 |
| Evidence | 用户指示 |
| Status | FINAL (但论文发表需要补做) |

---

## DEC-14: 不加入 Convex Hull 体积到主pipeline

| 字段 | 值 |
|------|-----|
| Decision ID | DEC-14 |
| Decision | 凸包体积仅作为诊断工具, 不在主pipeline中报告 |
| Date | UNKNOWN |
| Alternatives | 作为第六种体积方法参与消融 |
| Why | 凸包在壳状点云行为下不稳定, 增加复杂度但不增加可靠性 |
| Evidence | README.md 明确说明 |
| Status | FINAL |

---

## 暂不加入的方法

| 方法 | 原因 |
|------|------|
| 3D Mesh重建 | 稀疏点云下不可靠, 复杂度高 |
| 傅里叶形状描述子 | 12特征已足够, 增加复杂度收益不明 |
| 多模型集成 | 79样本不足以支撑, 过拟合风险 |

---

## 状态说明

- **PROPOSED**: 讨论过但未确定
- **TESTING**: 已实现正在评估
- **SUPPORTED**: 有实验证据支撑
- **REJECTED**: 已否定
- **FINAL**: 最终确定

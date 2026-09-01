# RESEARCH_DIRECTION.md

> **CURRENT FROZEN DIRECTION -- 2026-08-30.** The method is now an integrated physical-scale-to-volume pipeline and the project is in paper-writing preparation. The former planning record below is retained as **SUPERSEDED / ARCHIVED** provenance and must not be read as the current experiment plan.

## Final Paper Method Logic

1. UAV DOM acquisition.
2. Physical-scale and point-cloud sampling analysis.
3. Multi-scale DOM instance segmentation.
4. Cross-scale fusion and cascade deduplication.
5. Existing 2D-3D association and accepted-instance filtering.
6. Ground/background separation.
7. Ground-referenced 10 mm 2.5D top-surface reconstruction.
8. Canonical 12-feature shape descriptor extraction.
9. Shape-Aware LightGBM correction of `y_ratio = V_true / V_2_5D`.
10. `V_pred = V_2.5D x y_pred`.
11. Stratified representative real-mine volume estimation.
12. Fragmentation / PSD / P80 analysis as the next analytical stage.

## Central Contribution

The scientific line is **physical scale -> resolution matching -> shape-aware volume correction**. Multi-scale segmentation supplies non-duplicated DOM footprints; 2D-3D association supplies their observed height field; the 10 mm 2.5D surface and canonical features enable a resolution-matched correction model.

## Evidence Architecture

- **External mesh methodological validation:** T01+L01 (`465` OBJ) validates the correction module. The retained 0.5 mm V2 result is not deployed to the mine.
- **Resolution matching:** DOM GSD is 10 mm; point-cloud XY P90 is 6.0-6.4 mm. Therefore the mine application grid is frozen at 10 mm.
- **Scale adaptation:** original OBJ scale was not adequate at 10 mm. A pre-specified uniform factor `82.737840` produced a valid 20-object pilot and the 465-object scaled training dataset.
- **Real-mine application:** the frozen scaled-10mm model was applied only to a deterministic, size-stratified sample of 4,000 of 69,911 accepted DOM2 instances. PSD/P80 is not yet an evidence-supported result.

## Required Claim Discipline

- External held-out MAPE `5.82%` and R2 `0.9838` belong to the scaled external mesh test, not DOM2 accuracy.
- DOM2 `3,639/4,000` is a successful pipeline completion count, not a 90.98% volume accuracy claim.
- Empty 2.5D surfaces are retained failures and increase toward smaller size strata; they are an observed resolution/surface-availability limitation.

## Paper Writing Handoff -- CURRENT

Chapter 3 has been consolidated into six main sections so that the manuscript
follows the scientific chain rather than a list of software modules:

```text
3.1 Overall framework
3.2 Physical-scale multi-scale DOM instance segmentation
3.3 Duplicate resolution and cascade deduplication
3.4 2D-3D association, ground reference and 2.5D reconstruction
3.5 Shape-aware descriptor, scale adaptation and volume correction
3.6 Representative real-mine application
```

The current English and Chinese drafts are
`docs/paper/PAPER_DRAFT.md` and `docs/paper/PAPER_DRAFT_CN.md`. Chapter 4
should follow the same evidence order: segmentation/fusion inventory,
2D-3D association and filtering, external Shape-Aware validation, resolution
and scale adaptation, and representative real-mine application.

The current writing baseline is documentation only. Do not reopen the former
plans for retraining, DOM3 processing, full 69,911-rock inference, unapproved
ablation, or uncertainty analysis. Any unsupported paper claim must be marked
as an evidence gap in the paper evidence documents rather than filled by a new
experiment.

## Historical Record (SUPERSEDED; retained for provenance)

> 研究思路与论文主线
> 核心方向: **Physical-scale-driven Multi-scale Rock Segmentation**

---

## 1. 整体流程

```
DOM → 物理尺度分析 → Scale Matching → Multi-scale Detection
  → Instance Fusion → 2D-3D Association → Point Cloud
  → Volume → Particle Size → PSD → P80 → Validation
```

## 2. 核心科学故事

论文要讲清楚两件事（导师确认）：

### 问题1: 宽粒径范围高精度识别
石块粒径跨度极大（0.5m~3.5m+），如何让所有大小的石块都处于高识别精度？

**方法**: 物理尺度驱动多尺度分割 — 不是简单缩放图像，而是让不同物理覆盖面积的图像块输入网络，使目标石块始终保持最优像素尺寸。

**两个衍生问题**:
- **1a 边界裁切**: 分块多→边界多，边界上被裁切的石块如何处理？→ 边界感知融合（20%重叠 + 多特征评分）
- **1b 跨尺度去重**: 多尺度检测同一石块如何不重复计算？→ 级联去重（按粒径选主尺度）

### 问题2: 点云数据如何使用
- **2a 精确截面尺寸**: 点云高度数据辅助计算石块三维形态（2D mask提供水平边界，点云提供垂直高度）
- **2b shape-aware体积校正**: 2.5D轮廓→形状特征→校正比r→体积 V=r×V_2.5D

## 3. 各环节解释

### 3.1 为什么要物理尺度匹配
- 固定像素尺度（如SAHI固定1024px）在不同GSD下对应的物理尺寸不同
- 小石块在大物理窗口中像素占比太小 → 检测困难
- 大石块在小物理窗口中被截断 → 边界问题
- 需要: 每个粒径的石块在网络输入中保持相似的像素比例

### 3.2 固定像素尺度存在的问题
- SAHI: 固定patch_size=1024px, 在GSD=0.01m下覆盖10.24m → 小石块(0.5m)仅占~50px
- 所有石块用同一尺度 → 小石块分辨率不足, 大石块可能超出窗口

### 3.3 为什么需要Multi-scale
- 3个尺度: coarse(10.24m), medium(5.12m), fine(2.56m)
- 每个尺度都输入N=1024像素 → 大石块用coarse, 中等用medium, 小石块用fine
- 保证: 所有石块在输入网络时都有合理的像素占比

### 3.4 SAHI的角色
- **Baseline / 对比方法**, 非主方法
- SAHI是固定滑动窗口分块, 不考虑物理尺度
- 在V1实验中作为对比: SAHI vs Quadtree_DOM
- 配置: `experiments/configs/slicing/sahi.json` (patch_size=1024, overlap=0.15)

### 3.5 Quadtree的角色
- **V1主方法**: 边缘密度引导的自适应分块
- 基于Canny边缘密度决定分块大小
- 配置: `experiments/configs/slicing/quadtree_dom.json` (base_tile=20m, min=10m, max=20m)
- **在V2中已被物理尺度多尺度取代**, 但保留作为V1实验对比

### 3.6 当前主要方法
- **V2: rockseg/ 包** — 物理尺度驱动多尺度检测
  - 3尺度分块 + YOLO11m-seg + 多特征融合 + 级联去重
  - 是论文的**主方法**

### 3.7 哪些是baseline
- **SAHI分块**: 固定滑动窗口 baseline
- **Quadtree_DOM**: V1自适应分块 baseline
- **Bounding Box体积**: V=L×W×H 上界 baseline
- **Ellipsoid体积**: V=(π/6)×L×W×H 几何近似 baseline
- **2.5D Integration**: 无校正的原始积分 baseline
- **Linear Correction**: V=α×V_2.5D 单参数校正 baseline

### 3.8 哪些是ablation
- 体积估算5种方法消融: Box vs Ellipsoid vs 2.5D vs Linear vs Shape-Aware
- 分块方法对比: SAHI vs Quadtree_DOM
- 多尺度 vs 单尺度: **PROPOSED / 未执行**
- 融合 vs 无融合: **PROPOSED / 未执行**
- 级联去重 vs 跨尺度融合: **PROPOSED / 未执行**

### 3.9 2D到3D空间关联
- 2D mask来自DOM图像分割 → 提供水平边界 (footprint)
- 点云提供z方向高度信息 → GroundDEM计算相对高度
- 两者通过世界坐标关联: mask像素 → (x,y)世界坐标 → 查询点云
- 输出: 2.5D高度图 (z=f(x,y) over footprint)

### 3.10 体积计算候选方法
1. **Bounding Box**: V=L×W×H (上界)
2. **Ellipsoid**: V=(π/6)×L×W×H
3. **2.5D Integration**: V=Σ max(z_top-z_ground,0)×Δ² (主方法)
4. **Linear Correction**: V=α×V_2.5D (α=0.731, 来自训练集median ratio)
5. **Shape-Aware**: V=r_pred×V_2.5D (LightGBM预测校正比r)

### 3.11 粒径定义
- **等效直径**: d = sqrt(4A/π), A为footprint面积
- 最小报告粒径: 0.5m (detection阶段过滤)
- 分箱: 0.5-0.75, 0.75-1.0, 1.0-1.5, 1.5+

### 3.12 P80的作用
- P80: 80%石块通过的筛孔尺寸 (粒径分布的第80百分位)
- 是矿业爆破效果评估的关键指标
- **计算方法**: 从体积估算等效直径 → 排序 → P80
- **当前状态**: PROPOSED / 代码中未找到实现

### 3.13 论文应讲的科学故事
1. 宽粒径范围石块测量需要尺度匹配 → 物理尺度多尺度方法
2. 多尺度带来的边界和重复问题 → 融合+级联去重
3. 2D图像+3D点云融合 → 2.5D体积估算
4. shape-aware校正 → 从形状特征预测校正比
5. 外部验证 → 868(79本地)个OBJ样本验证体积方法

### 3.14 不能随意加入的方法
- **不加入纯为增加复杂度的方法**: 凸包体积、3D Mesh重建等
- **Convex hull volume**: 仅作为诊断工具, 不在主pipeline中 (README明确说明)
- **不加入未经验证的方法**: 任何方法需有实验支撑

---

## 4. 论文结构 (导师确认框架)

**两大创新点 × 两个子问题 = 4个技术贡献:**

| 问题 | 子问题 | 技术贡献 | 代码状态 |
|------|--------|---------|---------|
| 1 宽粒径高精度识别 | 1a 边界融合 | 边界感知多特征融合 | ✅ 已实现 |
| 1 宽粒径高精度识别 | 1b 跨尺度去重 | 按粒径选主尺度的级联去重 | ✅ 已实现 |
| 2 点云体积测量 | 2a 2D-3D关联 | mask+点云→2.5D高度图 | ✅ 已实现 |
| 2 点云体积测量 | 2b shape-aware校正 | LightGBM校正比预测 | ⚠️ V1部署, V2未部署 |

---

## 5. PROPOSED / NOT FINAL 的内容

- P80 计算: PROPOSED (代码中未找到实现)
- 多尺度 vs 单尺度消融实验: PROPOSED (未执行)
- 不确定性分析 (GUM级别): PROPOSED (Measurement期刊要求)
- 检测精度评估 (mAP): PROPOSED (无人工标注)
- DOM3 场景验证: PROPOSED (数据已有, 未运行)
- 用全部868样本重训shape-aware: PROPOSED (L01/L02不在本地)

# PROJECT_CONTEXT.md

> **CURRENT FROZEN RESEARCH STATUS -- 2026-08-30.** This section is authoritative for the completed Shape-Aware V2 research chain and the current paper-writing handoff. The former handoff material below is retained for provenance only and is **SUPERSEDED** where it conflicts with this section.

## Current Research Spine

**Physical-scale-driven multi-scale rock segmentation -> 2D-3D association -> ground-referenced 10 mm 2.5D surface reconstruction -> canonical Shape-Aware volume correction -> fragmentation / PSD analysis.**

The study has two deliberately separate evidence domains:

| Domain | Role | Frozen resolution / model use |
| --- | --- | --- |
| External OBJ mesh validation | Methodological validation of the 2.5D-to-volume correction module | 0.5 mm V2 is external-mesh only; scaled 10 mm V2 supplies resolution-matched training evidence |
| DOM2 real-mine application | Operational transfer of the frozen model to accepted DOM2 instances | 10 mm grid (`0.01 m`); no per-rock mine ground-truth volumes |

## Frozen Volume Method

- Canonical schema: 12 ordered features: `C`, `AR`, `solidity`, `compactness`, `eq_diam_ratio`, `H_mean_norm`, `H_std_norm`, `H_p25_norm`, `H_p75_norm`, `H_skew_norm`, `fill_ratio`, `ellipsoid_ratio`.
- `H_skew_norm` is the unnormalised `H_skew` value. The training and production adapters passed a five-sample consistency check with maximum absolute and relative difference of `0`.
- Mine resolution: `10 mm = 0.01 m`, selected from DOM GSD and local point-cloud sampling (`XY P90 = 6.0-6.4 mm`; 3D P90 about `8.5-8.6 mm`).
- Scale adaptation: frozen uniform factor `82.737840`; the 20-object pilot was `20/20` valid. The original-scale 10 mm OBJ dataset is **REJECTED / NOT USED** because it yielded predominantly empty surfaces.
- Frozen model: `research_v2/volume_validation/output_v2_scaled_10mm/shape_aware_model_v2_scaled_10mm.txt`; `best_iteration = 356`.

## Current Evidence

- Scaled 10 mm external dataset: T01 `79/79` + L01 `386/386` successful; split `326/70/69`; no leakage or non-finite values.
- Held-out scaled external test: Shape-Aware volume MAPE `5.82%`, MAE `7,167,711 mm3`, RMSE `11,890,780 mm3`, R2 `0.9838`; raw 2.5D MAPE `54.24%`; train-mean constant correction MAPE `6.99%`.
- DOM2 inventory: `76,407` final instances; `69,911` accepted by the existing 2D-3D association; `6,496` rejected.
- Frozen DOM2 representative sample: `4,000` accepted instances, deterministic size-only stratification. Completed volume pipeline: `3,639/4,000` successes (`90.98%` pipeline success rate); all `361` failures were `empty_2_5d_surface` and concentrated in smaller strata.

## Claim Boundary

The `5.82%` result is a held-out **scaled external mesh** result, not a mine-site accuracy. The `90.98%` result is a real-mine pipeline completion rate, not volume accuracy. Real-mine absolute volume accuracy remains unvalidated because per-rock DOM2 ground-truth volumes are unavailable.

## Paper Writing Handoff -- CURRENT

The project has entered the paper-writing stage. No new experiments, model
training, volume inference, full-population calculation, production-code
changes, or image generation are authorized by the current baseline.

The manuscript method chapter has been reorganized into six main sections:

```text
3.1 Overall framework
3.2 Physical-scale multi-scale DOM instance segmentation
3.3 Duplicate resolution and cascade deduplication
3.4 2D-3D association, ground reference and 2.5D reconstruction
3.5 Shape-aware descriptor, scale adaptation and volume correction
3.6 Representative real-mine application
```

The English draft and Chinese corresponding draft are maintained in
`docs/paper/PAPER_DRAFT.md` and `docs/paper/PAPER_DRAFT_CN.md`. The next
manuscript task is Chapter 4 Results, using the corresponding frozen evidence
in the order 4.1 segmentation/fusion, 4.2 association/filtering, 4.3 external
validation, 4.4 resolution/scale adaptation, and 4.5 real-mine application.
`docs/paper/PAPER_OUTLINE.md` is the controlling chapter architecture.

The paper must distinguish external scaled-mesh validation from real-mine
application. It must not transfer the external MAPE to the mine or describe
the 4,000-rock completion rate as volume accuracy. Missing independent
segmentation, fusion, association, DEM, and mine ground-truth evidence remains
an explicit paper limitation.

## Historical Record (SUPERSEDED; retained for provenance)

> AI Agent 项目交接文档 — 事实状态
> 生成时间: 2026-08-26
> 生成者: TRAE (前序工作 Agent)

---

## 1. 项目名称

**RockSeg: DOM and OSGB Point-Cloud Rock Fragment Measurement**

目标期刊: *Measurement* (Elsevier, IF ~5.1)

## 2. 项目目标

利用无人机正射影像 (DOM) 和摄影测量点云 (OSGB-derived)，对露天矿场爆破石块进行：
1. 自动化个体识别与分割
2. 3D 点云验证过滤误检
3. 石块体积估算
4. 粒径分布 (PSD) 和 P80 指标计算

核心科学问题：**如何对跨粒径范围（0.5m ~ 3.5m+）的爆破石块进行高精度、无重复的个体识别和体积测量。**

## 3. 当前研究问题

- 体积估算的校正模型 (shape-aware) 相对线性校正改进有限（0.3%），需要更强模型或更多训练数据
- 无人工标注 ground truth，检测精度无法量化
- 不确定性分析缺失（Measurement 期刊硬性要求）
- 当前只有 T01 组 79 个外部验证样本（L01/L02 陨石数据不在本地）

## 4. 当前完整技术流程

```
DOM (GeoTIFF, GSD=0.01m)
  → 物理尺度分块 (coarse 10.24m / medium 5.12m / fine 2.56m, N=1024)
  → YOLO11m-seg 实例分割 (imgsz=1024, conf=0.35)
  → 像素→世界坐标映射 (EPSG:4536)
  → 多特征融合去重 (IoU+质心+面积+边界+置信度)
  → 级联去重 (cascade deduplication, 按粒径选主尺度)
  → 3D 点云验证 (GroundDEM + bbox级点云统计)
  → 2.5D 体积估算 (高度图积分 + shape-aware校正)
  → 粒径/体积统计
```

## 5. 当前代码目录结构

```
DOM_Space_message_val/
├── rockseg/                    # 核心包 (V2 多尺度架构)
│   ├── __init__.py             # 公共 API 导出
│   ├── config.py               # PipelineConfig, ScaleConfig, FusionWeights
│   ├── pipeline.py             # MultiScaleRockDetectionPipeline 主流程
│   ├── tiling.py               # 物理尺度感知分块 DOMReader
│   ├── segmentation.py        # YOLO11m-seg 推理封装
│   ├── models.py               # RockInstance, TileMetadata 数据模型
│   ├── fusion.py               # 多特征融合 + 级联去重
│   ├── validation_3d.py        # 3D点云验证 (完整mask级)
│   ├── validation_3d_fast.py   # 3D点云验证 (快速bbox级)
│   ├── validation_3d_batch.py  # 3D点云验证 (批量优化版)
│   └── volume.py               # 体积估算 + shape-aware校正
│
├── experiments/                # V1 实验代码 (历史保留)
│   ├── common/                 # 场景配置 scene_reference.py
│   ├── configs/                # JSON 配置文件
│   │   ├── detection/          # YOLO 参数
│   │   ├── slicing/            # SAHI, Quadtree_DOM 配置
│   │   └── fusion/             # 融合参数
│   ├── slicing/                # 分块实现 (SAHI, Quadtree)
│   ├── detection/              # YOLO 检测 (V1 单尺度)
│   ├── fusion/                 # 融合实现 (V1)
│   ├── evaluation/             # 评估脚本
│   ├── volume/                 # 体积估算 (V1)
│   ├── visualization/          # 可视化
│   ├── reports/                # 报告生成
│   ├── validation/             # 手动标注验证
│   └── site_b_run/            # Site B 独立运行
│
├── research_v2/                # V2 研究 (当前活跃)
│   ├── volume_validation/      # 体积验证实验 (E5)
│   │   ├── config.py           # 验证配置
│   │   ├── mesh_utils.py       # OBJ 网格加载 (需 trimesh)
│   │   ├── simulate_2_5d.py    # 2.5D 模拟 (需 trimesh)
│   │   ├── shape_descriptors.py # 形状特征 (12维增强版)
│   │   ├── volume_estimators.py # 体积估算方法 + LightGBM模型
│   │   ├── metrics.py           # 评估指标
│   │   ├── data_split.py        # 分组感知数据划分
│   │   ├── visualize.py         # 可视化
│   │   ├── run_validation.py    # E5 实验入口
│   │   ├── enhance_shape_aware.py # 增强版训练脚本 (独立, 不需trimesh)
│   │   ├── analyze_features.py  # 特征分析
│   │   ├── output/              # V1 模型输出
│   │   │   └── results/
│   │   │       ├── shape_aware_model.txt  # LightGBM 模型 (V1, 5特征)
│   │   │       ├── metrics_summary.json  # V1 验证指标
│   │   │       └── ...
│   │   └── output_v2_enhanced/  # V2 增强模型输出
│   │       ├── shape_aware_model_v2.txt  # LightGBM 模型 (V2, 12特征)
│   │       └── model_meta_v2.json
│   └── ... (其他规划文档)
│
├── models/
│   └── best.pt                 # YOLO11m-seg 权重 (45.2MB)
│
├── data/
│   ├── dom2/                   # DOM 矿区2
│   │   ├── DOM.tif            # 正射影像 (8783×21713, GSD=0.01m)
│   │   ├── DOM.tfw            # 世界文件
│   │   └── DOM.prj            # 投影文件
│   ├── dom3/                  # DOM 矿区3 (未使用)
│   │   └── DOM.tif
│   ├── experience_rock/       # 外部验证数据 (Čapek et al. 2025)
│   │   └── T01/               # 79 个 OBJ 文件 (tephriphonolite)
│   ├── pointcloud2/           # 点云 矿区2
│   │   └── Data/
│   │       ├── BlockB.laz     # 61.6M 点
│   │       └── BlockY.laz     # 85.1M 点
│   └── pointcloud3/           # 点云 矿区3 (未使用)
│       └── Data/
│           ├── BlockB.laz
│           └── BlockY.laz
│
├── output/                    # 运行结果
│   ├── dom2_cascade_v2/       # V2 多尺度检测 (76,407 实例)
│   ├── dom2_cascade_v2_3d/    # 3D验证 (float32 bug, 96.7%通过率)
│   ├── dom2_cascade_v2_3d_fixed/  # 3D验证 (float64修复, 91.5%通过率)
│   ├── dom2_cascade_v2_volume/    # 体积估算 (float32 bug, 160m³)
│   └── dom2_cascade_v2_volume_fixed/  # 体积估算 (float64修复, 999m³)
│
├── run_rockseg.py              # 检测流水线入口
├── run_3d_validation.py       # 3D验证入口 (完整版)
├── run_3d_validation_fast.py  # 3D验证入口 (快速版)
├── run_volume_estimation.py   # 体积估算入口
├── README.md
└── docs/
    ├── results/
    │   ├── current_results.json   # V1 当前结果快照
    │   ├── current_results.md
    │   └── tables/                # 结果表格
    └── archive/                   # 归档文档
```

## 6. 关键文件作用

| 文件 | 作用 |
|------|------|
| `rockseg/config.py` | 三尺度配置: coarse(10.24m), medium(5.12m), fine(2.56m), GSD=0.01, overlap=20% |
| `rockseg/pipeline.py` | 主流程: 分尺度处理→尺度内融合→跨尺度融合/级联去重 |
| `rockseg/tiling.py` | 物理尺度感知分块: 按地面覆盖尺寸分块, 非固定像素 |
| `rockseg/segmentation.py` | YOLO11m-seg推理, mask→实例转换, 边界完整性评分 |
| `rockseg/fusion.py` | 5特征融合评分 + 并查集聚类 + 级联去重(按粒径选主尺度) |
| `rockseg/validation_3d.py` | GroundDEM(0.5m, P5) + 点云网格索引 + 逐实例mask级验证 |
| `rockseg/validation_3d_fast.py` | bbox级快速验证, 按10m网格批量处理 |
| `rockseg/volume.py` | 2.5D高度图提取 + 形状特征 + LightGBM校正 + SimpleTreeModel fallback |
| `run_rockseg.py` | CLI: --dom --model --output --gsd --scales --cascade |
| `run_3d_validation_fast.py` | CLI: --input --dom --pointcloud --output |
| `run_volume_estimation.py` | CLI: --input --dom --pointcloud --model --output --linear-alpha 0.731 |

## 7. 当前使用的数据

### DOM 数据
- **矿区2** (主场景): `data/dom2/DOM.tif`
  - 尺寸: 8783 × 21713 像素
  - GSD: 0.01 m/pixel
  - 面积: 19,070.53 m²
  - 坐标系: EPSG:4536
- **矿区3** (未使用): `data/dom3/DOM.tif`

### 点云数据
- **矿区2**: `data/pointcloud2/Data/BlockB.laz` (61.6M点) + `BlockY.laz` (85.1M点)
  - 总计: 146.7M 点
  - 坐标: 绝对世界坐标, 无XY偏移
  - 来源: OSGB-derived photogrammetric (非LiDAR)
- **矿区3**: `data/pointcloud3/Data/` (未使用)

### 外部验证数据
- `data/experience_rock/T01/` — 79 个 OBJ 文件 (tephriphonolite, 爆破破碎)
  - 来自 Čapek et al. (2025) 数据集
  - L01 (386个陨石) 和 L02 (403个陨石) 不在本地

### 数据格式
- DOM: GeoTIFF + TFW 世界文件
- 点云: LAZ 格式 (laspy 读取)
- 验证模型: OBJ 格式 (trimesh 或手动解析)
- 模型权重: PyTorch .pt (YOLO11m-seg)
- 校正模型: LightGBM .txt (文本格式决策树)

## 8. 当前模型

### 检测模型
- **YOLO11m-seg** (`models/best.pt`, 45.2MB)
- 训练时间/数据集: UNKNOWN
- 推理参数: imgsz=1024, conf=0.35, max_det=1000

### 体积校正模型
- **V1** (`research_v2/volume_validation/output/results/shape_aware_model.txt`)
  - LightGBM, 1棵树, 6个叶子, 5个特征
  - 校正比: 0.731~0.736 (接近常数)
  - 测试 MAPE: 7.89% (vs 线性 8.05%)
  - 训练: 55样本, 验证: 12, 测试: 12

- **V2 增强版** (`research_v2/volume_validation/output_v2_enhanced/shape_aware_model_v2.txt`)
  - LightGBM, 12个特征, best_iter=27
  - 校正比: ~0.637, 10个唯一值
  - 测试 MAPE: 7.58% (vs 线性 7.57%)
  - 5折CV MAPE: 6.99% ± 1.35%
  - **尚未部署到生产环境** (volume.py 仍使用 V1)

### 模型配置
- 多尺度: coarse(10.24m), medium(5.12m), fine(2.56m), N=1024
- 融合权重: IoU=0.30, centroid=0.20, area=0.20, boundary=0.15, confidence=0.15
- 融合阈值: within_scale=0.50, cross_scale=0.55
- 级联去重边界: coarse/medium=0.50m, medium/fine=0.30m
- 3D验证: min_points=60, min_z_range=0.18m, min_p90_height=0.12m
- 体积: grid_res=0.05m, linear_alpha=0.731

## 9. Python 环境

- **Python**: 3.10 (conda env `rock` at `D:\conda_env\rock`)
- **GPU/CUDA**: UNKNOWN (代码有自动检测, `device=0`)
- **主要依赖**:
  - ultralytics (YOLO11m-seg)
  - rasterio (GeoTIFF 读取)
  - laspy (LAZ 点云读取)
  - numpy, scipy, scikit-image
  - lightgbm 4.7.0
  - matplotlib 3.10.9
  - **未安装**: pandas, trimesh (sandbox 权限限制)

## 10. 已实现的功能

1. ✅ 物理尺度感知多尺度分块 (3尺度)
2. ✅ YOLO11m-seg 实例分割推理
3. ✅ 多特征融合 (5特征加权评分 + 并查集)
4. ✅ 级联去重 (按粒径选主尺度)
5. ✅ 3D点云验证 (3个版本: 完整/快速/批量)
6. ✅ 2.5D体积估算 (高度图积分)
7. ✅ shape-aware LightGBM校正模型 (V1+V2)
8. ✅ SimpleTreeModel (LightGBM fallback, 纯numpy)
9. ✅ V1 实验框架 (experiments/ 目录, SAHI/Quadtree/单尺度对比)

## 11. 已运行的实验及结果

### V1 主场景实验 (quadtree_dom + correlation_clustering)
- 来源: `docs/results/current_results.json`
- 分块: 130 tiles (quadtree_dom), 98 retained
- 检测: 30,993 raw → 7,823 filtered (≥0.5m)
- 融合: 6,933 accepted
- 3D验证: 6,933/7,184 = 96.5%
- 体积: 1,451 m³ (2.5D), 2,222 m³ (2D proxy)

### V2 多尺度级联检测 (DOM2)
- 来源: `output/dom2_cascade_v2/`
- 检测: 76,407 实例 (3尺度级联去重后)

### V2 3D验证 (float32 bug版)
- 来源: `output/dom2_cascade_v2_3d/`
- 通过率: 96.7% (73,896/76,407)
- **此结果受float32精度bug影响, 不准确**

### V2 3D验证 (float64修复版)
- 来源: `output/dom2_cascade_v2_3d_fixed/`
- 通过率: 91.5% (69,911/76,407)
- 拒绝原因: insufficient_p90_height (5,207), insufficient_elevated_ratio (5,847)

### V2 体积估算 (float32 bug版)
- 来源: `output/dom2_cascade_v2_volume/`
- 总体积: 160 m³ (严重低估)
- **此结果受float32精度bug影响, 不可信**

### V2 体积估算 (float64修复版)
- 来源: `output/dom2_cascade_v2_volume_fixed/`
- 总体积: 999 m³ (shape-aware), 1,359 m³ (2.5D raw)
- 有效实例: 65,826
- 校正比: 0.7348

### E5 体积验证实验 (外部OBJ数据)
- V1 (5特征): 测试MAPE 7.89%, 线性8.05%
- V2 (12特征): 测试MAPE 7.58%, 线性7.57%
- V2 5折CV: MAPE 6.99% ± 1.35%
- 改进仅0.3%, shape-aware优势有限

## 12. 当前代码状态

- **rockseg/** 包: 代码完整, 可运行
- **float32→float64 修复**: 已在 validation_3d.py, validation_3d_fast.py, validation_3d_batch.py, volume.py 中完成
- **shape_descriptors.py**: 已扩展为12特征 (V2), 但 volume.py 仍使用5特征 (V1)
- **enhance_shape_aware.py**: 独立训练脚本, 不依赖trimesh, 已产出V2模型
- **V2模型未部署**: volume.py 的 predict_shape_aware 仍使用5特征

## 13. 当前存在的问题

### 已知 bug (已修复)
- **float32精度丢失**: UTM大坐标(4,678,551)在float32下量化为0.5m间隔 → 高度图覆盖降至25% → 体积低估6.2倍。已在4个文件8处修复为float64。

### 已知问题 (未解决)
1. shape-aware模型改进有限 (V2 vs V1 仅改善0.3%)
2. V2增强模型未部署到生产代码
3. 无检测精度评估 (无mAP, 无人工标注)
4. 无不确定性分析
5. V1 和 V2 两套代码并存, 部分功能重叠
6. pandas/trimesh 未安装 (sandbox限制)

### 已尝试但未成功的方法
- 安装 pandas/trimesh: 权限拒绝 (WinError 5)
- 增加LightGBM特征维度: 从5→12特征, 但改进仅0.3%
- 原因: 校正比r的变异系数仅9%, 形状特征能解释的方差有限

## 14. 当前任务状态

- shape-aware模型增强: **进行中** (V2模型已训练但未部署)
- 项目交接文档: **当前正在创建**

## 15. 重要文件路径

| 文件 | 路径 |
|------|------|
| YOLO权重 | `models/best.pt` |
| V1校正模型 | `research_v2/volume_validation/output/results/shape_aware_model.txt` |
| V2校正模型 | `research_v2/volume_validation/output_v2_enhanced/shape_aware_model_v2.txt` |
| V1验证指标 | `research_v2/volume_validation/output/results/metrics_summary.json` |
| V2验证指标 | `research_v2/volume_validation/output_v2_enhanced/model_meta_v2.json` |
| V1结果快照 | `docs/results/current_results.json` |
| 3D验证结果(修复) | `output/dom2_cascade_v2_3d_fixed/validation_summary.json` |
| 体积结果(修复) | `output/dom2_cascade_v2_volume_fixed/volume_summary.json` |
| 场景配置 | `experiments/common/scene_reference.py` |

## 16. 运行命令

```bash
# V2 多尺度检测
python run_rockseg.py --dom data/dom2/DOM.tif --model models/best.pt --output output/dom2_cascade_v2 --gsd 0.01 --scales coarse,medium,fine --cascade

# 3D验证 (快速版)
python run_3d_validation_fast.py --input output/dom2_cascade_v2 --dom data/dom2/DOM.tif --pointcloud data/pointcloud2/Data/BlockB.laz,data/pointcloud2/Data/BlockY.laz --output output/dom2_cascade_v2_3d_fixed

# 体积估算
python run_volume_estimation.py --input output/dom2_cascade_v2_3d_fixed --dom data/dom2/DOM.tif --pointcloud data/pointcloud2/Data/BlockB.laz,data/pointcloud2/Data/BlockY.laz --model research_v2/volume_validation/output/results/shape_aware_model.txt --output output/dom2_cascade_v2_volume_fixed

# E5体积验证
python research_v2/volume_validation/enhance_shape_aware.py
```

## 17. Git 状态

- **分支**: main
- **最新commit**: `711d65a Prepare Measurement pipeline for private publication`
- **工作区状态**: 大量未跟踪文件 (research_v2/, rockseg/, output/, 多个分析脚本)
- **已修改未提交**: README.md, docs/results/*, experiments/configs/*, experiments/volume/run_volume.py

---

## 文档生成信息

- **生成时间**: 2026-08-26
- **当前Git commit**: 711d65a
- **工作区状态**: dirty (大量未跟踪和已修改文件)
- **项目最后状态**: shape-aware模型增强进行中, V2模型已训练但未部署到生产代码

# RockSeg — 航拍岩块检测与点云可视化

从 **OSGB 模型**转换而来的正射影像（DOM）和 LiDAR 点云中，自动检测、分割岩块，并在 3D 点云中可视化。

---

## 完整数据流

```
┌══════════════════════════════════════════════════════════════════════┐
│                       原始数据 (不纳入 Git)                          │
│                                                                     │
│  data/dom3/DOM.tif  ←── OSGB 模型转换的正射影像 (GeoTIFF)          │
│  data/dom3/DOM.tfw  ←── 地理配准文件 (像素→世界坐标)                │
│  data/pointcloud3/Data/BlockB.laz  ←── OSGB 转换的 LiDAR 点云      │
│  data/pointcloud3/Data/BlockY.laz  ←── (与 BlockB 拼接为完整区域)  │
└──────────────────────────────────────────────────────────────────────┘
         │
         │ 读取 DOM.tif + DOM.tfw
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ① SLICING — 切片实验                                               │
│  (experiments/slicing/run_slicing_experiment.py)                    │
│                                                                     │
│  输入: DOM.tif (整张正射影像，可能 7000×23000 px)                   │
│  配置: experiments/configs/slicing/sahi.json 或 quadtree_dom.json   │
│                                                                     │
│  两种方法:                                                          │
│  ┌─ SAHI: 固定 1024×1024 滑窗扫描整个 DOM                          │
│  │   overlap=0.1  → 相邻 tile 重叠 10% (≈1m)                      │
│  │   跳过 黑色无数据区 (>25% 像素为黑 → 跳过)                      │
│  │                                                                  │
│  └─ Quadtree-DOM: 按 Canny 边缘密度自适应切分                       │
│      边缘密度 > 0.15 且 tile > 20m → 四分                          │
│      平坦区域保持大 tile (~23m)，岩石区切到最小 (~10m)              │
│      tile_overlap_m=3.0 → 相邻 tile 扩展 1.5m 避免边界被切断      │
│                                                                     │
│  输出:                                                              │
│    outputs/sahi/tile_stats.json                                     │
│    ├── patches[]: 每个 tile 的信息                                  │
│    │   ├── pixel_origin: [x, y]  ←─ tile 左上角在 DOM 中的像素坐标 │
│    │   ├── pixel_size: 1024       ←─ tile 尺寸 (px)                │
│    │   ├── world_bounds: [xmin,ymin,xmax,ymax]  ←─ 世界坐标范围   │
│    │   ├── content_ratio: 0.85    ←─ 有效像素占比                   │
│    │   └── status: "kept"/"skipped_black"                          │
│    ├── total_patches / kept_patches                                 │
│    └── coverage_ratio  ←─ 有效 tile 覆盖面积 / DOM 总面积           │
│                                                                     │
│    outputs/quadtree_dom/tile_stats.json (同上结构, 用 tiles[] 字段) │
└──────────────────────────────────────────────────────────────────────┘
         │
         │ tile_stats.json 传给检测阶段，按 kept tiles 逐个推理
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ② DETECTION — YOLO 检测实验                                        │
│  (experiments/detection/run_detection_experiment.py)                │
│                                                                     │
│  输入:                                                              │
│    ├─ tile_stats.json (从哪里读 tile 的 pixel_origin 和尺寸)        │
│    ├─ models/best.pt (YOLOv8-seg 模型)                              │
│    └─ DOM.tif (按 tile 坐标裁切原图)                                │
│                                                                     │
│  配置: experiments/configs/detection/default.json                   │
│    ├── imgsz: 1024          ←─ YOLO 输入尺寸                        │
│    ├── conf: 0.35           ←─ 置信度阈值 (低于此值跳过)            │
│    ├── max_det: 1000        ←─ 每张图最多检测数                     │
│    └── min_stone_diameter_m: 1.0  ←─ 最小石头直径 (小于此值跳过)   │
│                                                                     │
│  处理流程:                                                          │
│    1. 读取 tile_stats.json，只处理 kept 状态的 tile                │
│    2. 根据 pixel_origin 从 DOM 裁出 tile 图像                      │
│    3. YOLO 推理 → 得到 masks (每个 mask 是一个二值图像)            │
│    4. 对每个 mask:                                                  │
│       a. 计算面积 → 等效直径                                       │
│       b. 小于 min_stone_diameter_m → 跳过                          │
│       c. 用 cv2.moments 算质心 → 世界坐标                          │
│       d. RLE 压缩 mask (COCO 格式, 无损)                           │
│    5. 输出到 detections.json                                        │
│                                                                     │
│  输出: outputs/sahi/detections.json                                 │
│    [  ←─ 检测列表 (每张 tile 可能有 0~多个)                        │
│      {                                                              │
│        "score": 0.85,              ←─ YOLO 置信度                   │
│        "area_m2": 2.34,            ←─ mask 面积 (m²)               │
│        "equivalent_diameter_m": 1.73,  ←─ 等效圆直径               │
│        "centroid_world": [x, y],   ←─ 质心世界坐标 (供 fusion 用)  │
│        "bbox_world": [xmin,ymin,xmax,ymax],  ←─ 检测框世界坐标     │
│        "pixel_origin": [px, py],   ←─ tile 在 DOM 的起点 (mask→世界│
│        "rle_mask": {"size":[h,w], "counts":[...]},  ←─ RLE mask   │
│        "source_patch_id": "patch_000000"  ←─ 来源 tile ID          │
│      },                                                             │
│      ...                                                             │
│    ]                                                                │
│                                                                     │
│  同时输出: outputs/sahi/detection_stats.json                        │
│    ├── detection_count: 2366  ←─ 检测总数                           │
│    ├── processed_tiles / total_tiles                                │
│    ├── area_m2: {min, max, mean}                                    │
│    └── diameter_m: {min, max, mean}                                 │
└──────────────────────────────────────────────────────────────────────┘
         │
         │ detections.json 传给融合阶段，需要里面的 centroid_world、
         │ bbox_world、rle_mask、source_patch_id 字段
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ③ FUSION — 融合实验                                                │
│  (experiments/fusion/run_fusion_experiment.py)                      │
│                                                                     │
│  输入: outputs/sahi/detections.json (来自检测阶段)                  │
│                                                                     │
│  解决的问题:                                                        │
│    同一个石头可能被多个 tile 检测到 (因为 tile 之间有 overlap)      │
│    → 需要把重复检测合并为一块石头                                    │
│                                                                     │
│  配置: experiments/configs/fusion/                                  │
│    ├── heuristic.json                                               │
│    │   ├── cross_tile_distance_m: 0.8  ←─ 质心距离阈值             │
│    │   └── cross_tile_iou_threshold: 0.3  ←─ bbox IoU 阈值         │
│    │   逻辑: 同 tile 跳过 / 距离>0.8m跳过 / IoU<0.3跳过 → 合并    │
│    │                                                                 │
│    └── correlation_clustering.json                                  │
│        ├── distance_sigma: 0.70     ←─ 高斯权重衰减系数            │
│        ├── positive_weight_threshold: 0.45  ←─ 正边阈值             │
│        ├── iou_weight: 0.3          ←─ IoU 加成系数                │
│        ├── same_tile_boost: 1.20   ←─ 同 tile 亲和加成              │
│        └── max_distance_m: 3.5     ←─ 只计算 3.5m 内的配对          │
│        逻辑: 亲和度 = exp(-d²/2σ²) × (1+0.3×IoU) × tile加成         │
│              亲和度 ≥ 0.45 → 同一块石头                              │
│                                                                     │
│  输出: outputs/sahi/correlation_clustering/fusion_stats.json        │
│    {                                                                 │
│      "source": "sahi",           ←─ 来源切片方法                    │
│      "method": "correlation_clustering",  ←─ 融合方法               │
│      "input_detections": 458,    ←─ 输入检测数                      │
│      "output_stones": 274,       ←─ 融合后石头数                    │
│      "merge_ratio": 0.4017,      ←─ 合并比例 (1 - 石头/检测)        │
│      "stones": [                  ←─ 石头列表 (给 visualize_pc)     │
│        {                                                             │
│          "stone_id": "stone_000000",                                 │
│          "source_detection_count": 2,  ←─ 由几个检测合并而成        │
│          "score_mean": 0.8588,    ←─ 平均置信度                     │
│          "bbox_world": [xmin,ymin,xmax,ymax],  ←─ 石头包围盒        │
│          "detection_indices": [3, 7]  ←─ 在 detections.json 中的索引│
│        },                                                             │
│        ...                                                           │
│      ]                                                               │
│    }                                                                 │
└──────────────────────────────────────────────────────────────────────┘
         │
         │ fusion_stats.json + detections.json 传给可视化
         │ visualize_pc 用 detection_indices 回溯每个石头的 RLE mask，
         │ 用 mask 多边形精确裁剪点云
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ④ VISUALIZE PC — 点云可视化                                        │
│  (experiments/visualize_pc/run_visualize.py)                        │
│                                                                     │
│  输入:                                                              │
│    ├─ fusion_stats.json     ←─ 石头的 bbox + detection_indices      │
│    ├─ detections.json       ←─ 每个检测的 RLE mask + pixel_origin   │
│    └─ BlockB.laz + BlockY.laz  ←─ LiDAR 点云                       │
│                                                                     │
│  配置: 代码中的常量                                                  │
│    LAZ_OFFSET_X = 623499.1061  ←─ LAZ 局部坐标 → DOM 坐标的偏移    │
│    LAZ_OFFSET_Y = 4678587.301  ←─ (通过 _compute_offset.py 算出)   │
│                                                                     │
│  处理流程:                                                          │
│    1. 加载 LAZ 点云 (原始局部坐标, 约 1.87 亿点)                   │
│    2. 降采样到 ~300 万点 (每 N 点取 1)                              │
│    3. 遍历每个石头:                                                  │
│       a. 读取 detection_indices → 找到关联的检测                    │
│       b. 解码 RLE → 二值 mask                                       │
│       c. 提取 mask 轮廓 → 像素坐标 + pixel_origin → 世界坐标       │
│       d. 减去 LAZ_OFFSET → 得到 LAZ 局部坐标多边形                  │
│       e. 对降采样后的点做 point-in-polygon 过滤                     │
│       f. 匹配的点着色为循环颜色 (12 色循环)                         │
│    4. 未匹配任何石头的点 → 灰色                                     │
│    5. Open3D 显示全景点云                                            │
│                                                                     │
│  输出: Open3D 窗口:                                                  │
│    ├─ 非石头区域: 灰色 (0.6, 0.6, 0.6)                             │
│    ├─ 石头区域: 12 色循环 (红绿蓝橙紫青粉...)                       │
│    └─ 操作: 左键旋转 / 滚轮缩放 / 右键平移                          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 1. 环境安装

```bash
# pip
pip install -r requirements.txt

# 或 conda (推荐)
conda env create -f environment.yml
conda activate rock
```

### 2. 数据准备

```
data/
├── dom3/
│   ├── DOM.tif         # 正射影像 (由 OSGB 模型导出)
│   └── DOM.tfw         # 地理配准六参数文件
└── pointcloud3/
    └── Data/
        ├── BlockB.laz  # LiDAR 点云 BlockB (由 OSGB 导出)
        └── BlockY.laz  # LiDAR 点云 BlockY (与 B 拼接为完整区域)
```

> **首次使用时**：如果 LAZ 点云使用局部坐标系（非绝对坐标），需要先计算偏移量：
> ```bash
> python -c "
> import laspy, numpy as np
> # DOM 世界坐标范围
> dom_xmin, dom_xmax = 623422.47, 623495.16
> dom_ymin, dom_ymax = 4678529.33, 4678756.62
> # LAZ 局部坐标范围
> laz = laspy.read('data/pointcloud3/Data/BlockB.laz')
> print(f'OFFSET_X = {dom_xmin - laz.x.min():.6f}')
> print(f'OFFSET_Y = {dom_ymin - laz.y.min():.6f}')
> "
> ```
> 将结果填入 `experiments/visualize_pc/run_visualize.py` 中的 `LAZ_OFFSET_X/Y`。

### 3. 运行完整流程

```bash
# ── 步骤 ①: 切片 ──────────────────────────────────
# 两种方法任选其一 (或跑全部做对比)
python experiments/slicing/run_slicing_experiment.py --method sahi
python experiments/slicing/run_slicing_experiment.py --method quadtree_dom
python experiments/slicing/run_slicing_experiment.py --method all

# 查看切片覆盖效果
python experiments/slicing/visualize_tiles.py --all
# 生成 outputs/comparison_side_by_side.png

# ── 步骤 ②: 检测 ──────────────────────────────────
python experiments/detection/run_detection_experiment.py --source sahi
# 输出: outputs/sahi/detections.json + detection_stats.json

# 快速测试 (只处理前 10 个 tile)
python experiments/detection/run_detection_experiment.py --source sahi --limit 10

# ── 步骤 ③: 融合 ──────────────────────────────────
python experiments/fusion/run_fusion_experiment.py --source sahi --method correlation_clustering
# 输出: outputs/sahi/correlation_clustering/fusion_stats.json

# ── 步骤 ④: 可视化 ────────────────────────────────
python experiments/visualize_pc/run_visualize.py
# 打开 Open3D 窗口: 石头彩色, 非石头灰色
```

---

## 各模块详细说明

### Slicing — 切片 (为什么需要？)

DOM 正射影像可能非常大（7000×23000 px），无法直接送入 YOLO（输入尺寸 1024px）。切片把大图切成小 tile，逐个推理。

**两种方法的区别：**

| | SAHI | Quadtree-DOM |
|--|------|-------------|
| tile 尺寸 | 固定 1024×1024 | 自适应 10~23m (≈1000~2300px) |
| tile 数量 | 较多 (200 个) | 较少 (129 个) |
| 覆盖 | 全覆盖 | 只覆盖有纹理的区域 |
| 速度 | 切片 3.4s / 检测 341s | 切片 1.5s / 检测 440s |
| 适用 | 不漏检为首要目标 | 希望减少检测量 |

**参数说明**（`configs/slicing/sahi.json`）：

```json
{
  "patch_size": 1024,        // tile 像素尺寸 (模型输入尺寸)
  "overlap": 0.1,            // 相邻 tile 重叠比例 (10% = 102px ≈ 1m)
  "min_content_ratio": 0.25, // 有效内容低于 25% 的 tile 跳过 (黑色区域)
  "include_edge_patches": true // 是否在图像边缘补充 tile
}
```

### Detection — 检测 (YOLO 输出了什么？)

每个 tile 经过 YOLOv8-seg 推理，对每个检测到的 mask：

1. **面积过滤**：`area_px × resolution²` → 等效直径 ≥ `min_stone_diameter_m` 才保留
2. **坐标转换**：mask 质心从像素坐标 → 世界坐标 (通过 TFW)
3. **RLE 编码**：二值 mask 用游程编码压缩（COCO 格式），相比存多边形更精确，相比存原始像素更紧凑

**输出数据用途**：

```
detections.json 中的字段：
├── score                → 融合时算平均/最高置信度
├── area_m2              → 统计石头大小分布
├── equivalent_diameter_m → 同上
├── centroid_world       → 融合时算检测间距离
├── bbox_world           → 融合时算 IoU；可视化时快速裁剪
├── pixel_origin         → 将 mask 轮廓从像素→世界坐标
├── rle_mask             → 可视化时精确裁剪点云
└── source_patch_id      → 融合时跳过同 tile 比较
```

### Fusion — 融合 (为什么要合并？)

SAHI 切片有 10% overlap，同一个石头可能出现在两个相邻 tile 中 → 产生重复检测。

**启发式**：简单直接，质心距离 + bbox IoU 都超过阈值就合并。

**相关聚类**：计算每对检测的"亲和度"：

```
亲和度 = exp(-距离² / 2×0.7²) × (1 + 0.3×IoU) × tile_boost
                                                         ↑
                                                  同 tile 1.2x
```

亲和度 ≥ 0.45 → 同一石头。用 Pivot 算法找出所有聚类。

### Visualize PC — 点云可视化 (如何显示？)

```
fusion_stats.json 中的每个 stone:
└── detection_indices: [3, 7]  ←─ 指向 detections.json 中的两条检测

→ 取出 detections[3] 和 detections[7]
→ 解码 RLE mask → 提取轮廓多边形
→ pixel_origin + TFW → DOM 世界坐标
→ 减去 LAZ_OFFSET → LAZ 局部坐标
→ 用多边形裁剪 LAZ 点云 (point-in-polygon)
→ 着色 → Open3D 显示
```

**为什么需要 LAZ_OFFSET？**

OSGB 模型导出的 DOM 使用绝对地理坐标（623xxx, 4678xxx），而 LAZ 点云使用局部坐标（-76~-4, -58~169）。两者跨度相同（72.7m × 227.3m），只是原点不同。通过偏移量将多边形从 DOM 坐标转换到 LAZ 坐标，才能在原始 LAZ 点上做过滤。

---

## 常见问题

### Q: 检测数比预期的少很多
检查 `conf` 阈值是否太高：

```json
// experiments/configs/detection/default.json
"conf": 0.35   // 改低 → 更多检测 (含误检)
"conf": 0.45   // 改高 → 更少检测 (更精确)
```

### Q: 某个大检测框里没有石头
→ 用 visualize_pc 确认是假阳性
→ 加 `--min-z-range 0.3` 自动过滤平坦地面

### Q: 点云可视化窗口打开但看不见石头
→ 确认 LAZ_OFFSET_X/Y 是否正确
→ 运行 `python experiments/visualize_pc/run_visualize.py`
→ 检查终端输出的 `已着色` 点数

### Q: 融合后石头数仍然接近检测数
→ 检查 `heuristic.json` 的 `cross_tile_distance_m` 是否太小
→ 检查 `correlation_clustering.json` 的 `positive_weight_threshold` 是否太高

---

## 项目结构

```
├── experiments/
│   ├── configs/                    # 所有参数配置
│   │   ├── slicing/
│   │   │   ├── sahi.json
│   │   │   └── quadtree_dom.json
│   │   ├── detection/
│   │   │   └── default.json
│   │   └── fusion/
│   │       ├── heuristic.json
│   │       └── correlation_clustering.json
│   ├── slicing/                    # 切片实验
│   │   ├── run_slicing_experiment.py
│   │   └── visualize_tiles.py
│   ├── detection/                  # YOLO 检测实验
│   │   └── run_detection_experiment.py
│   ├── fusion/                     # 融合实验
│   │   ├── run_fusion_experiment.py
│   │   └── visualize_fusion.py
│   ├── visualize_pc/               # 点云可视化 (新增)
│   │   └── run_visualize.py
│   ├── visualization/              # 其他独立可视化工具
│   └── utils/                      # 共享工具函数
├── models/
│   └── best.pt                     # YOLOv8-seg 预训练模型
├── data/                           # 数据 (Git 不跟踪)
├── requirements.txt
├── environment.yml
└── README.md
```

## 引用

项目参考文献见 `rockseg-references/rockseg-references.html`。

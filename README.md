<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8--seg-ultralytics-green" alt="YOLO">
  <img src="https://img.shields.io/badge/Open3D-0.18+-orange" alt="Open3D">
  <img src="https://img.shields.io/badge/status-active-brightgreen" alt="Status">
</p>

<h1 align="center">🪨 RockSeg</h1>
<p align="center"><b>OSGB 航拍岩块检测 · 分割 · 点云可视化</b></p>
<p align="center">从 OSGB 模型导出的正射影像（DOM）中自动检测岩块，并映射到点云中高亮显示。</p>

---

## 整体流程

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                             原始 OSGB 模型                                     │
│   ┌─────────────────┐                     ┌──────────────────────────────┐   │
│   │  正射影像 DOM    │                     │  点云 (LAZ 格式)              │   │
│   │  (data/dom3/)    │                     │  (data/pointcloud3/)         │   │
│   └────────┬────────┘                     └──────────┬───────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
             │                                                  │
             ▼                                                  │
┌─────────────────────────────┐                                │
│  ① 切片 (Slicing)           │                                │
│  SAHI 滑窗 / Quadtree 自适应  │                                │
│  ↓                          │                                │
│  tile 图像 + 元数据           │                                │
└──────────┬──────────────────┘                                │
           ▼                                                   │
┌─────────────────────────────┐                                │
│  ② 检测 (Detection)         │                                │
│  YOLOv8-seg 推理             │                                │
│  ↓                          │                                │
│  检测列表 (含 RLE mask)       │                                │
└──────────┬──────────────────┘                                │
           ▼                                                   │
┌─────────────────────────────┐                                │
│  ③ 融合 (Fusion)            │                                │
│  启发式合并 / 相关聚类         │                                │
│  ↓                          │                                │
│  融合结果 (石头分组)           │                                │
└──────────┬──────────────────┘                                │
           ▼                                                   ▼
┌────────────────────────────────────────────────────────────────┐
│  ④ 可视化 (Visualize PC)                                       │
│  Open3D 点云高亮显示                                             │
│  ↓                                                             │
│  全景点云: 石头彩色 / 非石头灰色                                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 各阶段详解

### 数据准备

| 文件 | 来源 | 格式 | 说明 |
|------|------|------|------|
| `data/dom3/DOM.tif` | OSGB 模型导出 | GeoTIFF | 正射影像，分辨率 0.01m/px |
| `data/dom3/DOM.tfw` | 同上 | 六参数配准 | 像素坐标 ↔ 世界坐标的转换矩阵 |
| `data/pointcloud3/Data/BlockB.laz` | OSGB 模型导出 | LAZ (压缩点云) | 区域 B 的点云 |
| `data/pointcloud3/Data/BlockY.laz` | OSGB 模型导出 | LAZ (压缩点云) | 区域 Y 的点云，与 B 拼接覆盖完整矿区 |

DOM 和点云**来自同一 OSGB 模型**，因此坐标系范围一致。DOM 使用绝对地理坐标（623xxx, 4678xxx），点云使用局部坐标（-76~-4, -58~169），两者跨度相同（72.7m × 227.3m），通过 `LAZ_OFFSET_X/Y` 转换。

---

### ① 切片 (Slicing)

**代码位置**：`experiments/slicing/run_slicing_experiment.py`

**输入**：`DOM.tif`

**配置**：`experiments/configs/slicing/`

**支持两种方法**：

| 方法 | 配置文件 | 策略 | 适用场景 |
|------|---------|------|---------|
| **SAHI** | `sahi.json` | 固定 1024×1024 滑窗扫描全图 | 全覆盖、不漏检为首要目标 |
| **Quadtree-DOM** | `quadtree_dom.json` | 按 Canny 边缘密度自适应四分，纹理丰富区细切、平坦区留大 tile | 节省检测量、加快推理 |

**共同逻辑**：
- 跳过黑色无数据区域（`min_content_ratio`）
- 相邻 tile 设置重叠避免边界被切断（`overlap` / `tile_overlap_m`）
- 输出每个 tile 的 `pixel_origin`（在 DOM 中的像素起点），供检测阶段裁取原图

**输出**：`outputs/{method}/tile_stats.json`

```json
{
  "method": "sahi",
  "total_patches": 320,
  "kept_patches": 235,
  "coverage_ratio": 1.0,
  "patches": [
    {
      "patch_id": "patch_000000",
      "pixel_origin": [0, 0],
      "world_bounds": [623422.47, ..., ...],
      "status": "kept"
    }
  ]
}
```

---

### ② 检测 (Detection)

**代码位置**：`experiments/detection/run_detection_experiment.py`

**输入**：
- `tile_stats.json`（从哪里读 tile 坐标）
- `DOM.tif`（按坐标裁图）
- `models/best.pt`（YOLOv8-seg 模型）

**配置**：`experiments/configs/detection/default.json`

**使用的方法**：
- **YOLOv8-seg**（Ultralytics）— 实例分割模型
- COCO **RLE 游程编码** — 压缩存储二值 mask

**流程**：
1. 遍历每个 kept tile，从 DOM 裁出 tile 图像
2. YOLO 推理 → 得到 masks
3. 对每个 mask 过滤面积（≥ `min_stone_diameter_m`）、计算质心世界坐标、RLE 压缩存储

**输出**：`outputs/{source}/detections.json`

```json
[
  {
    "score": 0.85,
    "area_m2": 2.34,
    "equivalent_diameter_m": 1.73,
    "centroid_world": [623432.12, 4678745.33],
    "bbox_world": [623429.66, 4678743.35, 623434.70, 4678749.39],
    "pixel_origin": [717, 717],
    "rle_mask": {"size": [1024, 1024], "counts": [0, 42, 1, ...]},
    "source_patch_id": "patch_000000"
  }
]
```

| 字段 | 用途 |
|------|------|
| `score` | 融合时统计平均/最高置信度 |
| `centroid_world` | 融合时计算检测间距离 |
| `bbox_world` | 融合时计算 IoU；可视化时快速 bbox 裁剪 |
| `pixel_origin` + `rle_mask` | 转换为世界坐标多边形（用于可视化点云裁剪） |
| `source_patch_id` | 融合时跳过同 tile 比较 |

---

### ③ 融合 (Fusion)

**代码位置**：`experiments/fusion/run_fusion_experiment.py`

**输入**：`detections.json`（来自检测阶段）

**输出**：`outputs/{source}/{method}/fusion_stats.json`

**使用的方法**：

| 方法 | 配置文件 | 原理 |
|------|---------|------|
| **启发式合并** (Heuristic) | `heuristic.json` | 质心距离 + bbox IoU 均超过阈值 → 合并（跳过同 tile） |
| **相关聚类** (Correlation Clustering) | `correlation_clustering.json` | 计算每对检测的亲和度（距离高斯核 × IoU 加成 × 同 tile 加成），阈值化后 Pivot 3-近似算法分组 |

**解决的问题**：SAHI 切片有 10% overlap，同一石头可能出现在两个相邻 tile → 融合将重复检测合并为一块石头。

**输出字段**：

```json
{
  "source": "sahi",
  "method": "correlation_clustering",
  "input_detections": 458,
  "output_stones": 274,
  "stones": [
    {
      "stone_id": "stone_000006",
      "source_detection_count": 2,
      "bbox_world": [623429.66, 4678743.35, 623434.70, 4678749.39],
      "detection_indices": [3, 7]
    }
  ]
}
```

`detection_indices` 是可视化阶段回溯检测 mask 的关键字段。

---

### ④ 可视化 (Visualize PC)

**代码位置**：`experiments/visualize_pc/run_visualize.py`

**输入**：
- `fusion_stats.json`（石头 bbox + detection_indices）
- `detections.json`（RLE mask + pixel_origin）
- `BlockB.laz` + `BlockY.laz`（原始点云）

**使用的方法**：
- **laspy** — 读取 LAZ 点云
- **Open3D** — 3D 点云渲染
- **cv2.pointPolygonTest** — 点云快速多边形裁剪
- **COCO RLE decode** — 从游程编码恢复二值 mask

**流程**：
1. 加载 LAZ 点云（原始局部坐标，约 1.87 亿点）
2. 降采样至约 300 万点
3. 遍历每个石头：
   - 回溯 `detection_indices` → 找到关联检测
   - 解码 RLE → mask → 多边形（像素坐标 → DOM 世界坐标 → LAZ 局部坐标）
   - point-in-polygon 过滤点云 → 匹配的点着色为循环颜色
4. 未匹配任何石头的点 → 灰色

**显示效果**：

| 区域 | 颜色 |
|------|------|
| 非石头区域 | 灰色 `(0.6, 0.6, 0.6)` |
| 石头 1..12 | 🔴🟢🔵🟠🟣 等 12 色循环 |
| 操作 | 左键旋转 · 滚轮缩放 · 右键平移 |

---

## 项目结构

```
├── experiments/
│   ├── configs/                  # 所有参数配置
│   │   ├── slicing/
│   │   │   ├── sahi.json
│   │   │   └── quadtree_dom.json
│   │   ├── detection/
│   │   │   └── default.json
│   │   └── fusion/
│   │       ├── heuristic.json
│   │       └── correlation_clustering.json
│   ├── slicing/                  # 切片实验
│   │   ├── run_slicing_experiment.py
│   │   └── visualize_tiles.py
│   ├── detection/                # YOLO 检测实验
│   │   └── run_detection_experiment.py
│   ├── fusion/                   # 融合实验
│   │   ├── run_fusion_experiment.py
│   │   └── visualize_fusion.py
│   ├── visualize_pc/             # 点云可视化
│   │   └── run_visualize.py
│   └── utils/                    # 共享工具
├── models/
│   └── best.pt                   # YOLOv8-seg 模型
├── data/                         # 数据 (Git 不跟踪)
├── rockseg-references/           # 参考文献
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 环境与运行

```bash
# 安装
conda env create -f environment.yml
conda activate rock

# 切片
python experiments/slicing/run_slicing_experiment.py --method all

# 检测
python experiments/detection/run_detection_experiment.py --source all

# 融合
python experiments/fusion/run_fusion_experiment.py --source all --method all

# 可视化
python experiments/visualize_pc/run_visualize.py
```

---

## 🔮 未来完善计划

| 优先级 | 模块 | 计划 |
|--------|------|------|
| ⭐⭐⭐ | **融合评估** | 引入人工标注真值，量化评估融合效果（准确率/召回率/F1） |
| ⭐⭐⭐ | **假阳性过滤** | 利用点云高度差（Z range）自动标记平坦地面误检 |
| ⭐⭐⭐ | **全量检测** | 配置合理参数后跑满全部切片（当前仅 458 检测），获取完整石头清单 |
| ⭐⭐ | **参数自动调优** | 为 SAHI / Quadtree / Fusion 引入网格搜索或贝叶斯调参 |
| ⭐⭐ | **体积估算** | 基于 mask 裁剪后的点云，用 alpha shape / convex hull 估算石块体积 |
| ⭐⭐ | **检测框标注** | 在 DOM 上绘制检测框 + mask，便于人工校验 |
| ⭐ | **Web 调参器** | 重写切片/融合的实时 Web 调参界面 |
| ⭐ | **多模型支持** | 除 YOLOv8-seg 外支持 SAM / Mask R-CNN 等模型 |

---

## 常见问题

### 检测结果看起来太少？
检查 `configs/detection/default.json` 中的 `conf: 0.35` 是否太高，尝试降低到 0.25 再跑。

### 可视化看不到石头？
确认 `run_visualize.py` 中的 `LAZ_OFFSET_X/Y` 是否正确，终端会输出"已着色"的点数。

### 一块石头的检测框对应明显是空地？
运行 `python experiments/visualize_pc/run_visualize.py`，石头区域会着色显示，如果是平坦地面会自动显示为均匀灰色。

---

<p align="center">
  <sub>GitHub: <a href="https://github.com/Midwinter66/rockseg">Midwinter66/rockseg</a></sub>
</p>

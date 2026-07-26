# RockSeg: 高价值阅读清单与10天计划

> 目标：用 10 天把论文框架写出来（尤其是 Methods/Experiments/Related Work 的骨架）
> 规则：优先读你当前 pipeline 可以直接复用的文章；不要只追求“最近新”，优先“与你的问题一一对应”。

## 一、推荐论文（不局限于本地）

### 1) 大图像切片与小目标检测（与你的DOM切片复用问题最相关）

1. **Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection**（ICIP 2022）
   - 链接: https://arxiv.org/abs/2202.06934
   - 价值：直接回答“切片推理+边界重叠+小目标提升”的工程范式；与你的 SAHI 路径天然对应。

2. **DOTA: A Large-scale Dataset for Object Detection in Aerial Images**（CVPR 2018）
   - 链接: https://arxiv.org/abs/1711.10398
   - 价值：大幅面遥感目标检测的数据特性与评测范式，可用于说明问题定义与基线选择逻辑。

3. **Object Detection in Aerial Images: A Large-scale Benchmark and Challenges**（TPAMI 2023 / arXiv 2102.12219）
   - 链接: https://arxiv.org/abs/2102.12219
   - 价值：DOTA v2 的基准与挑战设置，适合你在 Related Work 中论证“为什么需要适应遥感/地貌场景”。

4. **xView: Objects in Context in Overhead Imagery**（2018）
   - 链接: https://arxiv.org/abs/1802.07856
   - 价值：超大分辨率遥感数据规模与小目标稀疏分布的实际案例，和你们的现场影像边界条件接近。

5. **Towards Large-Scale Small Object Detection: Survey and Benchmarks**（TPAMI 2023）
   - 链接: https://arxiv.org/abs/2207.14096
   - 价值：系统归纳 SOD 评价指标与难点，帮助你写“方法边界/误差来源”时更稳。

6. **Deep Learning-Based Object Detection Techniques for Remote Sensing Images: A Survey**（Remote Sensing 2022）
   - 链接: https://www.mdpi.com/2072-4292/14/10/2385
   - 价值：一篇较新的综述，把小目标检测、遥感对象尺度差异、切片与后处理问题系统放在一起写，适合你 Related Work 的“方法池化”段。

### 2) 实例分割 + 重叠抑制（去重、拼接与重复检测）

7. **Mask R-CNN**
   - 链接: https://arxiv.org/abs/1703.06870
   - 价值：检测+分割联合范式的经典基线；说明为什么你需要“分割结果后融合”。

8. **YOLACT: Real-time Instance Segmentation**
   - 链接: https://arxiv.org/abs/1904.02689
   - 价值：快速实例分割的代表作，能对比速度/精度与后处理复杂度。

9. **PointPainting: Sequential Fusion for 3D Object Detection**
   - 链接: https://arxiv.org/abs/1911.10150
   - 价值：多模态后处理与融合思想可借鉴你们“2D检测-3D世界坐标-聚类融合”的思路。

10. **SOLOv2: Dynamic and Fast Instance Segmentation**（含 Matrix NMS）
   - 链接: https://arxiv.org/abs/2003.10152
   - 价值：Mask/NMS 机制的替代实现，对你“重复检测融合规则”写作很有价值。

11. **Soft-NMS -- Improving Object Detection With One Line of Code**
   - 链接: https://arxiv.org/abs/1704.04503
   - 价值：传统 NMS 改进，适合与你的 heuristic/cluster 方法做“处理流程基线对比”。

### 3) 点云分割与几何重建（你3D世界坐标链路）

12. **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation**
   - 链接: https://arxiv.org/abs/1612.00593
   - 价值：点云学习的经典起点，适合 Related Work 基线列表。

13. **PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space**
   - 链接: https://arxiv.org/abs/1706.02413
   - 价值：分层处理非均匀点云，可用于与随机采样/网格化策略对照。

14. **RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds**
   - 链接: https://arxiv.org/abs/1911.11236
   - 价值：大规模点云语义分割效率策略，和你要写的“工程可用性”非常匹配。

15. **KPConv: Flexible and Deformable Convolution for Point Clouds**
   - 链接: https://arxiv.org/abs/1904.08889
   - 价值：局部几何适配的核点卷积思路，可用于对照几何表征能力。

16. **PointTransformer**
   - 链接: https://arxiv.org/abs/2012.09164
   - 价值：点云中的 Transformer 注意力建模，适合写“方法为何选简单模型/复杂模型”。

17. **PointTransformer V2: Grouped Vector Attention and Partition-based Pooling**
   - 链接: https://arxiv.org/abs/2210.05666
   - 价值：V2 的分组向量注意力，适合讨论更大规模/复杂场景扩展的上限。

18. **Cylinder3D: An Effective 3D Framework for LiDAR Semantic Segmentation**
   - 链接: https://arxiv.org/abs/2008.01550
   - 价值：三维几何结构建模的工程化路线（体素化+柱状编码）。

### 4) 工程测量方向（rock/fragment 与体积）

19. **Rock blasting evaluation - image recognition method based on deep learning**（Scientific Reports, 2025）
   - 链接: https://doi.org/10.1038/s41598-025-09973-1
   - 价值：岩石作业场景下的分割-统计链路与工程可读性写作示例。

20. **Vision-based size distribution analysis of rock fragments using multi-modal deep learning and interactive annotation**（Automation in Construction, 2024）
   - 链接: https://doi.org/10.1016/j.autcon.2024.105276
   - 价值：直接对齐“岩块碎片分布 + 标注效率 + 规模误差控制”。

21. **Automated mapping of rock discontinuities in 3D lidar and photogrammetry models**
   - 链接: https://doi.org/10.1016/j.ijrmms.2012.06.003
   - 价值：3D 点云在岩体结构提取中的工程背景与流程标准，对测量可信度章节有帮助。

22. **Three-dimensional Alpha Shapes**
   - 链接: https://doi.org/10.1145/174462.156635
   - 价值：几何重建/体积估计算法对比时可说明 alpha-shape 与包络边界之间的关系。

23. **Screened Poisson Surface Reconstruction**
   - 链接: https://doi.org/10.1145/2487228.2487237
   - 价值：从点云到曲面重建的经典方法，可作为你几何后处理的“上界/比较对象”。

### 5) 论文写作与规范（Measurement投稿）

24. **COCO dataset paper（Microsoft COCO）**
   - 链接: https://arxiv.org/abs/1405.0312
   - 价值：mAP/AP 及数据集评测口径标准化写法，特别适合 Results 中 metric 说明。

25. **Measurement 官方投稿指南**
   - 链接: https://www.elsevier.com/journals/measurement/0026-1394/guide-for-authors
   - 价值：格式、图注、单位、图表数量和附加材料要求要点。

---

## 二、10天阅读计划（每晚 1.5~2 小时）

### 第1天：确定论文边界（3–4篇）
- 阅读：DOTA、xView、SAHI、COCO
- 产出：
  - 你的任务定义（输入/输出/场景）
  - “为何不是只调参数”一句式问题陈述

### 第2天：切片策略与复用策略（3–4篇）
- 阅读：Uniform Tiling Strategy、DOTA 2.0、SOD Survey、Slicing Aided（重点）
- 产出：
  - 一张“切片策略对比表”（tile size / overlap / 推理成本 / 召回影响 / 重复检测）

### 第3天：检测/分割基线模型（4–5篇）
- 阅读：Mask R-CNN、YOLACT、PointPainting、Soft-NMS、SOLOv2
- 产出：
  - 基线模型列表和为什么选择轻量基线的表述
  - 一段“related work 过渡段”初稿

### 第4天：重复检测合并（你的核心）
- 阅读：Soft-NMS、SOLOv2(Matrix-NMS)、SAHI（复读）
- 产出：
  - 写一版“方法 4.4 融合策略”文本：
    - (A) 启发式当作 baseline
    - (B) 相关聚类作为 proposal
    - (C) 复杂度/稳定性/鲁棒性对比

### 第5天：点云链路（2–3篇）
- 阅读：PointNet、PointNet++、RandLA-Net
- 产出：
  - 一张“点云分割模型对比表”：适用场景/复杂度/精度/工程代价

### 第6天：点云几何与量测（3篇）
- 阅读：KPConv、PointTransformer、Alpha Shapes / Poisson
- 产出：
  - 你方法里“点云到几何参数”的数学口径说明（至少 1 段）

### 第7天：领域论文（2–3篇）
- 阅读：Vision-based size distribution 2024、Rock blasting evaluation 2025
- 产出：
  - 你的论文“应用价值/工程价值”段落（为何对测量有意义）

### 第8天：补充与交叉（2–3篇）
- 阅读：xView challenge page + DOTA 主页说明（若可） + 你本地html里没读的高价值项
- 产出：
  - 相关实验指标清单（mAP、over-merge/under-merge、粒度误差）

### 第9天：相关工作/方法草稿（目标一次过）
- 阅读：按“需要的空位”补齐未完成文献
- 产出：
  - 相关工作 700–1000 字
  - 方法 4.1–4.5 全骨架（包含2种融合算法对比逻辑）

### 第10天：实验设计与图表模板（1天冲刺）
- 产出：
  - 一个完整实验表格（方法 / 基准 / 参数 / 指标）
  - 2张图的文字说明（图注）
  - 按“结果-讨论-局限-改进”写小结

---

## 三、你可以直接使用的阅读记录模板（已分离到模板文件）

### 模板文件
- [Paper_Reading_Notes_Template.md](rockseg-references/Paper_Reading_Notes_Template.md)

我会在模板文件里给你：
1. 论文固定字段（题目/期刊/方法/指标）
2. 你项目映射字段（对应到哪个模块）
3. 可复用到论文中的“可引用句”区
4. 复核项（局限、审稿人可能问题、你要如何先行回应）

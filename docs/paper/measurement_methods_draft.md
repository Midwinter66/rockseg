# 3 Materials and methods

> 写作说明：本章以当前项目实际代码、配置和已生成结果为依据，采用 `Measurement` 论文常见的“总体框架—模块原理—数学定义—实现参数”组织方式。方括号中的“待补充”内容不能在投稿前保留；“图置于此处”用于后续 Word 排版。

## 3.1 Study area and data preparation

本研究以露天矿岩石堆场为研究对象，输入数据包括高分辨率数字正射影像（digital orthophoto map, DOM）和由同一倾斜摄影 OSGB 模型转换得到的三维点云。需要强调的是，本研究所使用的三维数据并非激光雷达（LiDAR）点云，而是摄影测量模型的表面采样结果。DOM 采用 GeoTIFF 格式，影像尺寸为 8783 × 21713 pixels，空间分辨率为 0.01 m/pixel，投影坐标系为 EPSG:4536，覆盖面积约为 19070.53 m²。点云由 `BlockB.laz` 和 `BlockY.laz` 两个空间相邻的数据块组成，分别包含 61,641,369 和 85,080,023 个点，总点数为 146,721,392。

DOM 和点云均采用真实投影坐标，不额外引入经验偏移量。根据 GeoTIFF 配套的 TFW 仿射参数，像素坐标 \((u,v)\) 与世界坐标 \((x,y)\) 的转换关系为

\[
\begin{aligned}
x &= C + Au + Bv,\\
y &= F + Du + Ev,
\end{aligned}
\tag{1}
\]

式中，\(A\) 和 \(E\) 分别表示横、纵方向的像素尺度，\(B\) 和 \(D\) 表示旋转项，\(C\) 和 \(F\) 表示影像原点的投影坐标。当前数据中 \(A=0.01\ \mathrm{m/pixel}\)、\(E=-0.01\ \mathrm{m/pixel}\)，且旋转项为0。该统一坐标关系用于切片定位、实例掩膜回投、跨切片融合、点云裁取和体积计算，从而保证二维检测结果与三维点云在同一空间参考下对应。

【图1置于此处：研究区域与数据对应关系。建议包含：(a) 全场 DOM；(b) 白色背景下的全场 OSGB 转换点云；(c) DOM 与两个 LAZ 数据块的平面范围叠加图。】

## 3.2 Overview of the proposed measurement framework

为实现大范围矿区场景下由二维识别到三维量测的连续处理，本文构建了一套 DOM 与点云协同的岩石块度测量框架。整体流程包括五个主要阶段：首先，根据 DOM 局部纹理复杂度进行自适应四叉树切片；其次，利用 YOLO11m-seg 对有效切片进行单尺度实例分割，并将掩膜转换为具有实际单位的面积、等效粒径和世界坐标；随后，在世界坐标系中建立检测实例之间的空间关联图，通过相关聚类消除重叠切片产生的重复检测；在此基础上，根据融合区域从 OSGB 转换点云中提取对应点集，并利用相对于场景地面的高度特征排除缺乏三维起伏的伪目标；最后，以 GroundDEM 约束的 2.5D 网格积分方法估算逐石体积，并以二维等效球体积作为对比基线。

与仅依赖 DOM 的块度统计不同，点云在本文中承担两个明确作用：一是在融合后利用真实高程起伏验证候选目标是否具有岩石的三维形态；二是为逐石体积估算提供表面高程信息。因而，点云并不替代二维实例分割，而是作为空间验证和量测信息补充，与 DOM 检测形成前后衔接的测量链。

【图2置于此处：完整方法框架图。输入端为 DOM 和 OSGB 转换点云；中间依次展示自适应切片、实例分割、世界坐标映射、相关聚类、三维验证和 2.5D 体积积分；输出端展示逐石位置、面积、等效粒径、体积及粒径—体积分布。应突出“二维识别”和“点云量测”两条信息流在融合阶段汇合。】

## 3.3 Edge-density-guided adaptive quadtree slicing

### 3.3.1 Motivation and baseline

原始 DOM 的长边超过两万像素，无法在保持原始空间分辨率的条件下直接输入实例分割网络。固定窗口切片虽然易于实现，但会在无效背景、低纹理区域和高密度岩石区域使用相同的空间尺度，难以同时兼顾计算量、目标完整性和局部细节。为此，本文以固定滑窗切片作为基线，并提出由 Canny 边缘密度驱动的四叉树自适应切片策略，使切片尺度随岩石纹理复杂度变化。

### 3.3.2 Texture-driven quadtree partition

首先将 DOM 转换为灰度影像，并通过低、高阈值分别为30和90的 Canny 算子提取全局边缘图。对于候选区域 \(R\)，其边缘密度定义为

\[
\rho_e(R)=\frac{N_e(R)}{N_p(R)},
\tag{2}
\]

式中，\(N_e(R)\) 为区域内非零边缘像素数，\(N_p(R)\) 为区域总像素数。为排除影像外部黑色背景，进一步定义有效内容比例

\[
\rho_c(R)=\frac{N\left[I_g(u,v)>T_b\right]}{N_p(R)},
\tag{3}
\]

其中，\(I_g\) 为灰度值，\(T_b=5\) 为黑色像素阈值。当 \(\rho_c<0.05\) 或区域中不存在边缘像素时，该区域被标记为无效切片；否则，当 \(\rho_e\geq0.10\) 且区域最大边长大于5 m时，将其递归划分为四个子区域。初始切片边长为10 m，最小切片边长为5 m。为减轻目标在切片边缘被截断的问题，相邻切片设置0.5 m总重叠宽度，即每个切片边界向外扩展0.25 m，并限制在 DOM 范围内。

该策略在纹理和边缘密集的岩石区域生成较小切片，而在结构简单区域保留较大切片或直接跳过无效区域。每个切片同时记录像素原点、像素范围、世界坐标范围、边缘密度、内容比例和处理状态，用于后续检测结果的空间恢复与审计。

【图3置于此处：自适应四叉树切片原理与结果。建议采用三联图：(a) 原始局部 DOM；(b) Canny 边缘及边缘密度示意；(c) 四叉树切片结果，其中不同颜色区分保留、继续细分和跳过区域。当前 `tile_overlay_paper.png` 可作为全场附图，但正文主图应使用代表性局部放大图，否则图例和切片边界在双栏排版中不清晰。】

## 3.4 Instance segmentation and two-dimensional geometric measurement

### 3.4.1 Rock instance segmentation

本文采用针对单类岩石实例训练的 YOLO11m-seg 模型完成切片级实例分割。该模型以岩石为唯一目标类别，主实验采用单尺度推理，输入尺寸设为1024 pixels，置信度阈值为0.35，单切片最大检测数为1000。为保证后续方法比较中的控制变量一致，主结果不采用多尺度推理；多尺度模式仅作为可选敏感性分析设置。

模型对每个有效切片输出实例置信度、包围框和二值掩膜。掩膜在恢复到原切片尺寸后采用游程编码（run-length encoding, RLE）保存，以避免高分辨率二值矩阵造成过大的存储开销。每个实例同时保存来源切片编号和切片在原始 DOM 中的像素原点，以支持后续从局部像素坐标恢复到全局像素坐标。

模型训练数据、标注规范和训练超参数应在投稿版本中补充说明，包括训练、验证和测试样本数量、数据划分方式、数据增强策略、训练轮数、优化器、学习率及计算硬件。【待补充：上述训练信息目前无法从推理代码和模型权重中完整核验。】

### 3.4.2 Area and equivalent diameter calculation

设实例掩膜内的前景像素数为 \(n_p\)，单像素实际面积为 \(s_xs_y\)，则岩石的二维投影面积为

\[
A_{2D}=n_p s_xs_y.
\tag{4}
\]

当前 DOM 中 \(s_x=s_y=0.01\ \mathrm{m}\)。为使用统一的一维尺度描述不规则岩石轮廓，本文采用等面积圆定义等效粒径：

\[
d_{eq}=\sqrt{\frac{4A_{2D}}{\pi}}.
\tag{5}
\]

当 \(d_{eq}<0.5\ \mathrm{m}\) 时，该实例不进入后续主分析。该阈值并不代表模型无法检测更小目标，而是本文当前实验所定义的目标尺度范围。因此，全文中的岩石数量、粒径分布和体积统计均应注明“最小等效粒径为0.5 m”的实验条件。

实例质心由掩膜的零阶和一阶矩计算，局部像素坐标加上切片原点后，再通过式(1)转换为世界坐标。包围框四角采用相同方式转换，从而得到每个检测实例的 \(A_{2D}\)、\(d_{eq}\)、世界坐标质心和世界坐标包围框。

【图4置于此处：实例分割与二维量测示意。建议四个子图依次展示原始切片、实例掩膜、面积与等效圆、回投到 DOM 世界坐标后的质心和包围框。现有 `stone_005283/dom_mapping.png` 可作为局部映射示例，但建议去掉调试字段并统一为论文图例。】

## 3.5 Cross-tile duplicate fusion by correlation clustering

### 3.5.1 Pairwise association graph

切片之间的重叠能够减轻边界截断，但会使同一岩石在不同切片中产生多个检测实例。常规非极大值抑制主要针对同一图像中的局部包围框竞争，难以显式利用来源切片和世界坐标关系。本文将跨切片重复检测建模为带正关联边的图聚类问题，其中每个检测实例为一个节点，潜在重复实例之间建立关联边。

为降低全连接配对的计算开销，首先按照1.0 m网格对检测质心建立空间索引，仅在最大质心距离3.5 m范围内搜索候选节点。对于来自不同切片的检测实例 \(i\) 和 \(j\)，其关联权重定义为

\[
w_{ij}=\exp\left(-\frac{\lVert \mathbf{c}_i-\mathbf{c}_j\rVert_2^2}{2\sigma^2}\right)
\left(1+\lambda\operatorname{IoU}_{ij}\right),
\tag{6}
\]

式中，\(\mathbf{c}_i\) 和 \(\mathbf{c}_j\) 为世界坐标质心，\(\sigma=0.7\ \mathrm{m}\) 为距离衰减尺度，\(\operatorname{IoU}_{ij}\) 为世界坐标包围框交并比，\(\lambda=0.3\) 为 IoU 权重。只有当两个包围框相交、\(\operatorname{IoU}_{ij}\geq0.15\) 且 \(w_{ij}\geq0.45\) 时，节点之间才建立正关联边。

### 3.5.2 Pivot-based grouping and fused attributes

在关联图上采用 pivot 式相关聚类生成候选岩石组。算法依次选择尚未分组的节点作为 pivot，并将其正关联邻居并入同一组。考虑到同一岩石在同一切片中原则上只应对应一个实例，聚类过程加入“一切片一检测”约束：若同一来源切片存在多个候选邻居，仅保留关联权重更高者；权重相同时优先保留置信度更高的实例。该约束能够抑制局部过分割造成的不合理组内重复。

对于融合组 \(G\)，其等效粒径和二维面积分别取组内检测结果的中位数，以减弱单个边界截断实例的影响；融合包围框取所有成员包围框的并集，置信度采用组内均值。融合结果同时保留全部来源检测索引和切片编号，便于后续点云裁取和结果追溯。

【图5置于此处：跨切片融合机制。建议展示两个重叠切片中的重复掩膜、世界坐标关联图、关联权重筛选以及最终融合实例。当前全场 `fusion_correlation_clustering.png` 更适合放在结果章节展示空间分布，不适合作为方法原理图。】

## 3.6 Point-cloud-assisted three-dimensional validation

### 3.6.1 Extraction of stone point clouds

为将二维实例与三维几何信息对应，首先对融合组内各检测掩膜进行 RLE 解码，并提取掩膜边界。边界点经二维凸包简化后，通过式(1)转换至点云所在的真实世界坐标系。点云裁取采用“包围框粗筛—掩膜多边形精筛”的两阶段策略：首先利用1.0 m分辨率的 XY 网格空间索引查询融合包围框及其0.5 m外扩范围内的候选点；随后通过点在多边形内判定，仅保留落入任一来源掩膜投影区域的点。与对1.47亿点逐石全局扫描相比，该索引显著减少了重复空间查询的计算量，同时保持二维掩膜与三维点集的一致对应。

### 3.6.2 Scene-level ground DEM

由于 OSGB 转换点云主要描述可见表面，逐石点云底部通常不闭合，无法直接获得可靠的完整实体边界。本文首先从全场点云中每隔100个点进行系统下采样，并以0.5 m分辨率建立场景级地面数字高程模型（GroundDEM）。对于网格单元 \(g\)，当其点数不少于3时，以单元内高程的第5百分位数作为初始地面高程：

\[
z_g=Q_{0.05}\left(\{z_k\mid (x_k,y_k)\in g\}\right).
\tag{7}
\]

低百分位高程能够降低岩石上表面点对地面估计的影响。对没有直接观测值的网格，采用邻近有效单元逐层扩展填补；任意位置的地面高程通过相邻四个网格节点双线性插值得到。当前数据的 GroundDEM 原始有效网格覆盖率为41.72%，填补后达到100%。该填补过程保证了全场查询的连续性，但其对局部地面误差的潜在影响需在讨论部分单独说明。

### 3.6.3 Three-dimensional validation rules

对于候选岩石点集 \(P_i\)，定义相对地面高度

\[
h_k=z_k-z_g(x_k,y_k).
\tag{8}
\]

点云三维验证同时考虑点数、整体高程跨度、高位分位数和抬升点比例。候选实例需满足以下条件：点数不少于60；高程极差 \(\Delta z\geq0.18\ \mathrm{m}\)；相对高度的第90百分位数 \(Q_{0.90}(h)\geq0.12\ \mathrm{m}\)；相对地面高度不低于0.08 m的点所占比例不少于0.20。上述指标分别约束点云支持度、整体起伏、主体抬升高度和有效凸起范围。若任一条件不满足，则该候选被记录为三维验证拒绝样本，并保存具体拒绝原因。

该步骤的目的不是再次进行二维语义识别，而是利用点云回答“该二维候选是否具有高于局部地面的三维实体起伏”。因此，它主要用于排除平坦地面纹理、阴影边缘或缺乏点云支持的异常检测。

【图6置于此处：二维掩膜到三维验证的对应过程。建议包含：(a) DOM 掩膜；(b) 掩膜投影范围内的逐石点云；(c) GroundDEM 与相对高度示意；(d) \(\Delta z\)、P90高度和抬升比例的通过/拒绝示例。】

## 3.7 Ground-referenced 2.5D volume estimation

### 3.7.1 2.5D grid integration

通过三维验证的岩石点集被进一步划分为0.05 m × 0.05 m的局部水平网格。对于每个包含岩石点的网格单元，取单元内最大高程作为可见岩石上表面 \(z_i^{top}\)，并在网格中心查询 GroundDEM 高程 \(z_i^{ground}\)。单元有效高度定义为

\[
h_i=\max\left(z_i^{top}-z_i^{ground},0\right).
\tag{9}
\]

岩石的2.5D体积由所有正高度网格进行柱体积分得到：

\[
V_{2.5D}=\sum_{i=1}^{N}h_i\Delta^2,
\tag{10}
\]

式中，\(\Delta=0.05\ \mathrm{m}\) 为局部网格分辨率，\(N\) 为具有有效正高度的网格数。该方法只依赖可见上表面和局部地面基准，不要求 OSGB 点云形成封闭实体，因此比直接对表面壳层点云构建凸包更符合当前数据特征。

### 3.7.2 Two-dimensional proxy baseline

由于当前数据缺少逐石人工体积真值，本文不将任一估计量表述为绝对真实体积。为评估三维高程信息相对于纯二维尺度假设所提供的增益，采用由融合等效粒径构造的等效球体积作为二维代理基线：

\[
V_{2D}=\frac{\pi}{6}d_{eq}^{3}.
\tag{11}
\]

该基线假设岩石为直径等于二维等效粒径的球体，不使用任何点云高度信息。它的作用是提供统一、可重复的二维估算参照，而非替代真实体积标定。后续通过 \(V_{2D}\) 与 \(V_{2.5D}\) 的相关性、比值、中位差异及不同粒径区间的变化，分析二维形状假设可能产生的系统偏差。

### 3.7.3 Post-estimation quality control

体积阶段设置轻量质量控制（quality control, QC），用于识别无法形成稳定2.5D结果的样本，而不是重复执行融合阶段的岩石真实性过滤。论文主分析配置要求逐石点数不少于30、高程极差不小于0.08 m、2.5D体积状态有效且体积大于0。由于融合阶段的三维验证阈值更严格，前两项在完整流程中主要起一致性保护作用；真正导致体积样本失效的通常是地面查询失败、无正高度网格或非正体积等计算状态。所有QC标记均保留在逐石结果中，便于审计和敏感性分析。

【图7置于此处：体积估计原理及方法对比。上排展示二维等效粒径及等效球假设；下排展示逐石点云、GroundDEM、0.05 m网格柱和2.5D积分。现有 `volume_compare.png` 包含已经退出主分析的凸包体积，不应直接用于正文，需重绘为“2D proxy vs 2.5D”版本。】

## 3.8 Implementation and reproducibility

全部流程采用 Python 实现。DOM 读取与空间信息处理使用 Rasterio/Pillow，边缘与掩膜处理使用 OpenCV，实例分割基于 Ultralytics YOLO，LAZ 点云读取使用 laspy，数值计算使用 NumPy，三维交互可视化使用 Open3D。各阶段均由独立配置文件控制并输出 JSON 统计、逐石记录和可视化结果。当前主实验的关键参数汇总见表1。

**表1 当前主实验的方法参数**

| Stage | Parameter | Value |
|---|---|---:|
| DOM | Spatial resolution | 0.01 m/pixel |
| Quadtree | Initial/minimum tile size | 10 m / 5 m |
| Quadtree | Edge-density threshold | 0.10 |
| Quadtree | Overlap width | 0.5 m |
| Detection | Model | YOLO11m-seg |
| Detection | Input size / confidence | 1024 / 0.35 |
| Detection | Minimum equivalent diameter | 0.5 m |
| Fusion | Maximum centroid distance | 3.5 m |
| Fusion | Distance scale / weight threshold | 0.7 m / 0.45 |
| Fusion | Minimum bbox IoU | 0.15 |
| 3D validation | Minimum points / z range | 60 / 0.18 m |
| 3D validation | P90 relative height | 0.12 m |
| 3D validation | Elevated point threshold / ratio | 0.08 m / 0.20 |
| GroundDEM | Resolution / percentile | 0.5 m / 5th |
| Volume | Local grid resolution | 0.05 m |

运行环境还应在投稿版本中补充 Python、CUDA、PyTorch、Ultralytics 和主要依赖版本，以及 GPU、CPU 和内存配置。【待补充：软件版本和硬件型号。】

---

# 方法章节图表安排

| Figure/Table | Placement | Main message | Current material | Action |
|---|---|---|---|---|
| Fig. 1 Data and spatial reference | After Section 3.1 | DOM and OSGB-derived point clouds belong to the same scene and coordinate system | `view_full_pc.py` display; DOM | Need a three-panel composite |
| Fig. 2 Overall framework | After Section 3.2 | DOM detection and point-cloud measurement form one end-to-end chain | Old `01_full_mine_workflow.png` | Must redraw; old counts and 2D volume statement are obsolete |
| Fig. 3 Adaptive slicing | After Section 3.3 | Edge density controls quadtree subdivision | `tile_overlay_paper.png` | Use a representative crop plus a small full-scene inset |
| Fig. 4 Instance mask and 2D geometry | After Section 3.4 | Mask pixels are converted to area, equivalent diameter and world position | `stone_005283/dom_mapping.png` | Remove debug text and add equivalent-circle panel |
| Fig. 5 Correlation-clustering fusion | After Section 3.5 | Cross-tile duplicates become graph nodes and one fused stone | Full-scene fusion overlay | Draw a conceptual four-step schematic; retain full-scene overlay for Results |
| Fig. 6 2D-to-3D validation | After Section 3.6 | Mask projection extracts stone points; relative height rejects flat false positives | Point-cloud visualization and mapping output | Create accepted/rejected case comparison |
| Fig. 7 Volume estimation | After Section 3.7 | 2D equivalent-sphere proxy versus GroundDEM-based 2.5D integration | Old `volume_compare.png` | Must redraw without convex hull |
| Table 1 Main parameters | End of Section 3.8 | Reproducible main experimental configuration | Current configs | Already drafted above |

# Reference-paper style adopted

1. Wang et al., *An intelligent measurement method for the particle distribution of open-pit rock piles with fuzzy boundaries*, `Measurement` 265 (2026) 120351, DOI: 10.1016/j.measurement.2026.120351. The paper introduces the overall architecture first, then explains each module with figures and equations, and finally defines equivalent-diameter measurement separately.
2. Mao et al., *Recognition and statistical method of blast muckpile fragmentation under complex stacking conditions in underground metal mines*, `Measurement` 258 (2026) 119446, DOI: 10.1016/j.measurement.2025.119446. The paper uses a complete algorithm flowchart, decomposes the method into sequential geometric operations, and defines each statistical indicator with equations and variable explanations.

# Information still required before submission

- Study-area description that can be disclosed: mine type, location at an acceptable granularity, rock type, acquisition date and operating condition.
- OSGB/DOM acquisition details: UAV or camera platform, flight height, ground sampling distance, reconstruction software and point-cloud export settings.
- YOLO11m-seg training details: dataset size, annotation rule, split, augmentation, epochs, optimizer, learning rate and training hardware.
- Software versions and computing hardware.
- Formal citation numbers for SAHI, YOLO/Ultralytics, correlation clustering, Canny, GroundDEM/photogrammetric volume measurement and the two local `Measurement` references.
- Whether the 0.5 m minimum equivalent diameter will remain the only main setting or be accompanied by a threshold-sensitivity experiment.

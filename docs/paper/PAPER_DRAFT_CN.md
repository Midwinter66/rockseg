状态：草稿 -- 基于已确认实验事实的中文方法章节对应稿

# 3 方法

## 3.1 总体框架

本研究提出了一条由物理尺度驱动的处理流程，用于基于无人机数字正射影像（digital orthophoto map, DOM）及其关联的 point-cloud data 估计地表可观测石块的体积。该流程将基于影像的实例勾画、以地面为参考的几何重建以及学习得到的 Shape-Aware 校正相结合，并由两个衔接的组成部分构成。首先，对研究区 DOM 依次实施多尺度实例分割、重叠瓦片间的同尺度重复实例消解（within-scale duplicate resolution）和跨尺度 cascade deduplication，得到非重复的候选石块实例清单。随后，将每个实例与局部点云观测相匹配，依据预先定义的三维质量准则进行筛选，并重建为以局部地面为参考的 2.5D 表面。其次，利用具有已知参考体积的外部石块网格数据集，构建用于校正原始 2.5D 体积的 Shape-Aware 模型。

完整处理链如下：

$$
\mathrm{DOM}
\rightarrow
\mathrm{multi\text{-}scale\ instance\ segmentation}
\rightarrow
\mathrm{within\text{-}scale\ duplicate\ resolution}
\rightarrow
\mathrm{cross\text{-}scale\ cascade\ deduplication}
\rightarrow
\mathrm{2D\text{-}3D\ association\ and\ screening}
\rightarrow
\mathrm{ground\text{-}referenced\ 2.5D\ reconstruction}
\rightarrow
\mathrm{shape\ descriptor\ extraction}
\rightarrow
\mathrm{learned\ correction}
\rightarrow
\mathrm{rock\ volume\ estimation}.
$$

DOM 的地面采样距离为 $0.01\ \mathrm{m/pixel}$。该量描述正射影像的空间采样，与 point-cloud spacing 及 2.5D 重建使用的栅格分辨率不同。局部点云间距的第 $90$ 百分位数约为 $6.00$--$6.40\ \mathrm{mm}$（水平面）和 $8.54$--$8.60\ \mathrm{mm}$（三维）。据此，2.5D 表面重建采用 $0.01\ \mathrm{m}$ 的操作栅格。因而，DOM GSD、point-cloud spacing 和 2.5D grid resolution 分别对应影像采样、点云采样和分析参数。

对于具有有效地面参考 2.5D 表面的石块，外部网格数据集中的校正目标定义为

$$
y_{\mathrm{ratio}} =
\frac{V_{\mathrm{true}}}{V_{2.5D}},
$$

其中，$V_{\mathrm{true}}$ 为网格参考体积，$V_{2.5D}$ 为原始地面参考 2.5D 体积。基于 LightGBM 的 Shape-Aware 回归模型以 12 个几何描述符为输入，预测 $y_{\mathrm{pred}}$。最终校正体积为

$$
V_{\mathrm{pred}} =
V_{2.5D} \times y_{\mathrm{pred}}.
$$

地面参考的作用是将绝对点高程转换为相对于局部地面的高度，而非重建遮挡或埋藏部分的石块几何。因此，所得估计代表石块几何中可观测、以地面为参考的组成部分。外部网格验证与真实矿区应用在论文中明确分开：前者以已知网格体积评估校正性能，后者检验该方法对选定真实矿区实例的操作适用性，不能据此获得逐石块真实矿区体积精度。

## 3.2 多尺度 DOM 实例分割

研究区石块 footprint 的物理尺度范围较宽，单一影像上下文难以同时适配所有目标：较大的观测窗口可为大石块提供空间上下文，较小的物理窗口则可提高小石块的有效表征。因此，DOM 按三种物理地面覆盖尺度进行切分：coarse（$10.24\ \mathrm{m}$）、medium（$5.12\ \mathrm{m}$）和 fine（$2.56\ \mathrm{m}$）。三种瓦片均来自同一 DOM，并映射至统一的 $1024\times1024$ 网络输入；其中，medium 和 fine 瓦片分别以两倍和四倍进行重采样。这样，尺度定义始终对应物理地面覆盖范围，而非仅由网络输入像素数决定。

每个尺度的相邻瓦片保留 $20\%$ 重叠，以降低 tile boundary 附近的漏检风险。实例分割在每个瓦片上独立执行，检测置信度阈值设为 $0.25$。每个保留候选实例包含其 mask、bounding box、centroid、confidence、boundary-completeness measure、尺度标签和 footprint 属性。多尺度阶段因此输出的是候选石块实例，而非最终石块清单；由瓦片重叠和不同物理尺度重复观测产生的重复实例，在随后的两个阶段中消解。

这种物理尺度层级将影像观测窗口与石块预期 footprint 尺度联系起来。它不代表三个尺度各自提供独立精度测量，也不构成分割精度评估。原始候选和最终保留实例的数量将在 Results 中报告；独立的 precision、recall 和 mean average precision 评估需要单独的人工标注参考数据集，本文不予报告。

## 3.3 重复实例消解与 cascade deduplication

#### 同尺度重复实例消解

同尺度重复实例消解用于处理同一物理尺度下、重叠瓦片中对同一石块的重复观测；它并不合并单一瓦片内被识别为独立对象的不同石块。对于每个尺度，首先以基于网格的空间索引识别 bounding box 位于相同局部空间邻域的候选实例对。随后，bounding-box intersection over union（IoU）作为计算预筛选。对于候选实例 $i$ 和 $j$，仅当

$$
\mathrm{IoU}_{\mathrm{bbox}}(i,j) \geq 0.05
$$

且 mask overlap 非零时，才进入进一步比较。衡量实际实例轮廓重叠的 mask IoU 定义为

$$
\mathrm{IoU}_{\mathrm{mask}}(i,j) =
\frac{\left|M_i \cap M_j\right|}
{\left|M_i \cup M_j\right|},
$$

其中，$M_i$ 和 $M_j$ 为两个候选实例的二值 mask。因此，bounding-box IoU 用于避免不必要的 mask 比较，mask IoU 则参与重复实例相似性评分。

对每个通过预筛选的候选对，计算加权融合评分：

$$
S_{\mathrm{w}} =
0.30\,S_{\mathrm{m}}
+ 0.20\,S_{\mathrm{c}}
+ 0.20\,S_{\mathrm{a}}
+ 0.15\,S_{\mathrm{b}}
+ 0.15\,S_{\mathrm{p}},
$$

其中，$S_{\mathrm{m}}=\mathrm{IoU}_{\mathrm{mask}}$ 为 mask-overlap similarity，$S_{\mathrm{c}}$ 为 centroid proximity，$S_{\mathrm{a}}$ 为 footprint-area similarity，$S_{\mathrm{b}}$ 为平均 boundary completeness，$S_{\mathrm{p}}$ 为平均 detection confidence。centroid proximity 以 Gaussian score 表示：

$$
S_{\mathrm{c}} =
\exp\left(-\frac{d_{\mathrm{c}}^2}{2\sigma^2}\right),
$$

其中，$d_{\mathrm{c}}$ 为 centroid distance，$\sigma=50$ pixels。面积相似性项定义为

$$
S_{\mathrm{a}} =
\frac{\min(A_i,A_j)}{\max(A_i,A_j)},
$$

其中，$A_i$ 与 $A_j$ 是两个 mask 的面积。$S_{\mathrm{b}}$ 与 $S_{\mathrm{p}}$ 分别为两个实例的 boundary completeness 与 detection confidence 的算术平均值。上述系数在既有处理框架中预先固定，在本文中作为操作权重报告，并不被表述为经独立优化或验证的最优权重。

当 $S_{\mathrm{w}}\geq0.50$ 时，候选对被连接为同一重复实例组。对每个实例组，依据如下质量评分保留一个代表性实例：

$$
Q_i = c_i \times b_i,
$$

其中，$c_i$ 为实例 $i$ 的 detection confidence，$b_i$ 为其 boundary-completeness score。该 best-mask representative 策略优先保留置信度较高且边界更完整的观测，而不是对多个 mask 作无约束的几何并集。该步骤输出同尺度唯一的实例记录，作为后续跨尺度重复实例消解的输入。该流程规定了重叠导致的重复实例处理方式，但尚未利用人工标注的重复实例对独立量化其消解精度。

#### 跨尺度 cascade deduplication

同尺度重复实例消解和跨尺度重复实例消解面向不同来源的重复观测：前者在同一物理尺度的重叠瓦片间进行，后者在不同物理尺度上对同一石块的重复观测间进行。尽管系统实现了基于评分的通用 cross-scale fusion 过程，最终实例清单采用的是 size-aware cascade deduplication，而非跨尺度 mask 的直接融合。

跨尺度候选实例对通过空间检索生成，且仅在同时满足下列条件时被保留：

$$
\mathrm{IoU}_{\mathrm{bbox}}(i,j) \geq 0.05,
$$

$$
\mathrm{IoU}_{\mathrm{mask}}(i,j) > 0,
$$

$$
r_d(i,j) =
\frac{\min(d_{\mathrm{eq},i},d_{\mathrm{eq},j})}
{\max(d_{\mathrm{eq},i},d_{\mathrm{eq},j})}
\geq0.30,
$$

以及

$$
d_{\mathrm{c}}(i,j) \leq
\max\left(r_i,r_j\right),
$$

其中，$d_{\mathrm{c}}(i,j)$ 为 centroid distance，$r_i=d_{\mathrm{eq},i}/2$ 和 $r_j=d_{\mathrm{eq},j}/2$ 为 footprint-equivalent radii，$d_{\mathrm{eq},i}$ 和 $d_{\mathrm{eq},j}$ 为对应的 footprint-equivalent diameters。该比较中的 centroid distance 与半径均在 DOM 像素坐标中计算。对于包含 $N_{\mathrm{pixel}}$ 个像素、GSD 为 $g$ 的二值 DOM mask，其 footprint area 和 equivalent diameter 分别为

$$
A=N_{\mathrm{pixel}}g^2,
$$

和

$$
d_{\mathrm{eq}} =
2\sqrt{\frac{A}{\pi}} =
2\sqrt{\frac{N_{\mathrm{pixel}}g^2}{\pi}}.
$$

该 equivalent diameter 是基于二维 footprint 的量，而不是直接测得的三维粒径。空间、mask、尺度与重心约束共同降低了相邻但物理上不同石块被归入同一重复组的可能性；不过，false-merge rate 和 missed-merge rate 尚未得到独立测量。

对每个跨尺度重复实例组，根据组内最大的 footprint-equivalent diameter 选择优先观测尺度：

$$
\mathrm{primary\ scale} =
\begin{cases}
\mathrm{fine}, & d_{\mathrm{eq}} < 0.30\ \mathrm{m}, \\
\mathrm{medium}, & 0.30\ \mathrm{m} \leq d_{\mathrm{eq}} < 0.50\ \mathrm{m}, \\
\mathrm{coarse}, & d_{\mathrm{eq}} \geq 0.50\ \mathrm{m}.
\end{cases}
$$

在优先尺度成员中，保留 $Q_i$ 最高的实例；若该实例组不含此优先尺度成员，则保留整个组中 $Q_i$ 最高的实例。该 cascade 设计保留单一观测 mask，而不对不同尺度 mask 进行几何并集；同时，它依照估计 footprint 尺度分配优先观测尺度。所得跨尺度唯一记录构成后续 2D--3D association 与几何筛选的 DOM 候选实例清单。该流程规定了可复现的跨尺度重复处理规则，但不提供跨尺度融合精度的独立估计。

## 3.4 2D--3D 关联、地面参考与 2.5D 重建

在进行表面重建之前，先将 DOM 实例记录与点云观测建立关联。对于每个候选实例，利用其影像 footprint 和空间范围，通过可复用的空间索引查询邻近的点云观测。该步骤为候选实例生成局部点集，并保留二维实例与其三维观测之间的对应关系。因此，关联阶段本质上是空间检索过程，不会推断点云中不存在的几何信息。

检索得到的点依据相对地面高度和高程分布特征进行筛选。只有同时满足以下条件的候选实例，才进入后续几何处理：候选点数至少为 $60$；绝对高程范围至少为 $0.18\ \mathrm{m}$；相对地面高度的第 $90$ 百分位数至少为 $0.12\ \mathrm{m}$；elevated-point ratio 至少为 $0.20$。当点的相对地面高度不低于 $0.08\ \mathrm{m}$ 时，将其定义为 elevated point。所有实例均使用相同的固定条件。该筛选用于区分点云支持不足或高于地面的起伏不足的候选区域与可以进入 2.5D 重建的实例；它是质量门控过程，而不是经过独立验证的关联精度测量。

为了获得局部地面参考，将有限点云观测按每 $100$ 个点抽取一个点，并分配至单元大小为 $0.5\ \mathrm{m}$ 的场景级网格。对于包含至少三个点的单元，使用高程的第 $5$ 百分位数作为地面估计；缺失单元由相邻有效地面估计填补。该 GroundDEM 为绝对点高程转换为相对高度提供局部基准。它不用于恢复被石块遮挡的地面、重建埋藏几何，也没有独立的绝对 DEM 精度验证。

#### 以地面为参考的 2.5D 表面重建

对于通过三维质量门控的候选实例，利用局部 GroundDEM 将绝对点高程转换为相对地面高度：

$$
h(x,y)=z(x,y)-z_{\mathrm{ground}}(x,y),
$$

其中，$z(x,y)$ 为观测点高程，$z_{\mathrm{ground}}(x,y)$ 为对应的局部地面参考。随后，将不规则点观测栅格化为正方形操作网格：

$$
\Delta x=\Delta y=0.01\ \mathrm{m}.
$$

对于每个被点云占据的栅格单元，保留观测到的最大相对地面高度作为表面值。令 $\Omega$ 表示 occupied cells 的集合，$h_i$ 表示单元 $i$ 中保留的高度，则可观测 2.5D 体积计算为

$$
V_{2.5D}=
\sum_{i\in\Omega}h_i\Delta x\Delta y.
$$

该表示属于 2.5D，是因为每个水平栅格单元只保留一个高度值。它给出的是观测到的抬高表面的地面参考积分，而不是石块完整物理体积的直接测量。如果关联和筛选后没有有效的 occupied surface，则该候选实例不产生 2.5D 体积。

## 3.5 Shape-Aware 描述符、尺度适配与体积校正

Shape-Aware 校正使用固定顺序的描述符，由五个 footprint 描述符、五个高度分布描述符和两个体积形状比值组成。令 $A$ 为 footprint area，$P$ 为其 perimeter，$L$ 和 $W$ 分别为 footprint 的长轴与短轴尺寸，$H$ 为观测到的最大高度。canonical descriptor 定义如下。

1. Circularity：

$$
C=\min\left(\frac{4\pi A}{P^2},1\right).
$$

2. Aspect ratio：

$$
AR=\frac{L}{W}.
$$

3. Solidity：

$$
\mathrm{solidity}=\min\left(\frac{A}{A_{\mathrm{convex}}},1\right),
$$

其中，$A_{\mathrm{convex}}$ 为 footprint 凸包面积。

4. Compactness：

$$
\mathrm{compactness}=\frac{P}{\sqrt{A}}.
$$

5. Equivalent-diameter ratio：

$$
\mathrm{eq\_diam\_ratio}=\frac{\sqrt{4A/\pi}}{L}.
$$

高度分布描述符定义为

$$
H_{\mathrm{mean,norm}}=\frac{H_{\mathrm{mean}}}{H},
\qquad
H_{\mathrm{std,norm}}=\frac{H_{\mathrm{std}}}{H},
$$

$$
H_{\mathrm{p25,norm}}=\frac{H_{\mathrm{p25}}}{H},
\qquad
H_{\mathrm{p75,norm}}=\frac{H_{\mathrm{p75}}}{H}.
$$

第十个描述符严格采用当前训练与推理定义：

$$
H_{\mathrm{skew,norm}}=H_{\mathrm{skew}}.
$$

尽管其实现名称为 `H_skew_norm`，该描述符表示原始高度偏度，不再除以 $H$。最后两个描述符为

$$
\mathrm{fill\_ratio}=\frac{V_{2.5D}}{V_{\mathrm{box}}},
\qquad
\mathrm{ellipsoid\_ratio}=\frac{V_{2.5D}}{V_{\mathrm{ellipsoid}}},
$$

其中，$V_{\mathrm{box}}$ 和 $V_{\mathrm{ellipsoid}}$ 为根据 footprint 与高度几何得到的参考体积。模型按照固定顺序接收 12 个特征：$C$、$AR$、solidity、compactness、`eq_diam_ratio`、`H_mean_norm`、`H_std_norm`、`H_p25_norm`、`H_p75_norm`、`H_skew_norm`、`fill_ratio` 和 `ellipsoid_ratio`。

外部网格训练目标为已知参考体积与可观测 2.5D 体积之比：

$$
y_{\mathrm{ratio}}=
\frac{V_{\mathrm{true}}}{V_{2.5D}}.
$$

LightGBM 回归模型根据 12-feature descriptor 预测 $y_{\mathrm{pred}}$，最终校正估计为

$$
V_{\mathrm{pred}}=
V_{2.5D}\times y_{\mathrm{pred}}.
$$

因此，该模型是以地面参考 2.5D 体积为几何基础的校正模型，而不是在没有显式几何基础体积时直接预测体积的模型。

#### 分辨率与尺度适配

由于研究区没有逐石块已知参考体积，Shape-Aware 校正模型使用具有已知参考体积的外部石块网格数据集进行开发。外部网格首先以 $0.5\ \mathrm{mm}$ 进行处理，用于方法学验证。原始网格尺度直接在矿区操作尺度 $10\ \mathrm{mm}$ 下进行栅格化时，表面有效性不足，因此该路径不用于训练。

矿区操作栅格的选择基于研究区 DOM GSD、局部点云采样和表面重建尺度之间的关系。DOM GSD 为 $0.01\ \mathrm{m/pixel}$；已有点云统计在报告的第 $90$ 百分位处约为 $6.00$--$6.40\ \mathrm{mm}$（水平面）和 $8.54$--$8.60\ \mathrm{mm}$（三维）。因此，操作栅格设为 $10\ \mathrm{mm}$，同时保持输入采样尺度与表面栅格分辨率之间的概念区分。

为降低外部网格与研究区石块之间的 footprint 尺度差异，根据 footprint-equivalent diameter 分布的独立比较得到统一几何尺度因子：

$$
s=82.737840.
$$

该因子未使用 $V_{\mathrm{true}}$、$y_{\mathrm{ratio}}$、模型预测值或测试误差。随后对外部几何进行统一缩放，并以 $10\ \mathrm{mm}$ 重新栅格化，形成分辨率和尺度匹配的训练数据集。该尺度变换是受控的 domain-adaptation 步骤，可支持操作层面的迁移，但不能单独证明外部网格几何能够代表研究区内的每一个石块。

外部校正模型的 held-out Test 指标将在第 4.3 节报告，分辨率与尺度适配证据将在第 4.4 节单独报告。

## 3.6 真实矿区代表性应用

真实矿区应用限定在 DOM 候选清单中已通过筛选的 accepted instances。体积推理开始前，先基于 footprint-equivalent diameter 采用 `stratified_quantile_systematic` 方法选择固定代表性样本。accepted population 按经验粒径分位点划分为六层：S1（$P0$--$P10$）、S2（$P10$--$P25$）、S3（$P25$--$P50$）、S4（$P50$--$P75$）、S5（$P75$--$P90$）和 S6（$P90$--$P100$）。目标样本数依次为 $400$、$600$、$1{,}000$、$1{,}000$、$600$ 和 $400$。

每个分层内，实例按照 footprint-equivalent diameter 升序排列；当直径相同时，使用唯一实例标识符作为确定性的第二排序键。随后在排序序列中选择固定的等距系统位置。样本选择不使用体积、校正比值、模型预测值、误差或三维参考量。该设计覆盖完整直径范围，包括最大石块分层，同时不依赖随机数，也不进行重复抽样。

对于每个入选实例，处理顺序为：DOM mask 与空间范围、既有关联的 2D--3D association、三维质量筛选、地面参考的 $10\ \mathrm{mm}$ 2.5D 重建、canonical 12-feature 提取、LightGBM 校正以及 $V_{\mathrm{pred}}$ 计算。该应用是针对可观测石块体积的代表性样本分析，并非对全部 accepted inventory 的体积计算；在缺少逐石块参考体积的情况下，也不能据此建立真实矿区绝对体积精度。

相应的完成数量、失败类型和成功估计分布将在第 4.5 节报告；这些结果不属于样本设计本身。

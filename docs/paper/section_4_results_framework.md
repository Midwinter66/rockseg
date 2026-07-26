# 4. Results

> 本章为结果章框架草稿。当前版本先确定章节逻辑、结果表和图件位置；其中带有 `[to be filled]` 的位置需要根据最终实验输出补入数值。若后续更换 `d_min` 或主实验数据块，本章所有统计结果需要同步更新。

## 4.1 Characteristics of DOM Tiling Results

本节用于回答第一个问题：不同切片策略在同一 DOM 场景中会产生怎样的 tile 组织结果。由于后续检测、融合和体积估计均依赖 tile-level 输入，因此切片结果本身需要先被报告，而不是直接跳到检测数量。
建议首先报告固定窗口切片和边缘引导四叉树切片的基本统计，包括生成 tile 数量、保留 tile 数量、有效 tile 比例、覆盖范围和估计推理成本。这里不需要声称某一种切片方法“绝对更好”，而应描述二者在计算分配上的差异：固定窗口切片提供规则覆盖，四叉树切片则倾向于在纹理和边界更复杂的区域生成更细 tile。
对于主实验采用的边缘引导四叉树切片，应进一步说明其 tile 分布是否集中在岩块边界复杂区域。这个结果最好通过空间可视化展示，而不是只放数字。若四叉树切片能够减少无效区域 tile，同时保留岩块堆积区域覆盖，这将支持其作为本文主切片策略的合理性。

**Suggested Table 2. Quantitative comparison of DOM tiling strategies.**

| Tiling strategy | Generated tiles | Valid tiles | Valid tile ratio | Covered area | Relative inference cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed-window tiling | [to be filled] | [to be filled] | [to be filled] | [to be filled] | [to be filled] |
| Edge-guided quadtree tiling | [to be filled] | [to be filled] | [to be filled] | [to be filled] | [to be filled] |

**Suggested Fig. 6. Spatial distribution of generated DOM tiles.**
建议做成两列图：左侧为固定窗口切片结果，右侧为四叉树切片结果。背景使用半透明 DOM，切片边界用细线表示。图中不要同时叠加检测结果，否则读者会分不清这一节讨论的是切片还是检测。

## 4.2 Rock Fragment Detection and Size Distribution

本节报告实例分割模型在 DOM tiles 上得到的候选岩块及其尺度分布。这里要特别注意措辞：tile-level 输出仍是候选结果，不是最终岩块数量。因为重叠切片会造成重复检测，二维纹理也可能形成伪响应。

建议先报告主实验设置下的 tile-level detection 数量、候选掩膜数量、经过最小报告直径 \(d_{min}\) 筛选后的候选数量，以及候选目标的面积和等效直径分布。当前主实验若采用 \(d_{min}=0.5\) m，则本节所有统计都应明确对应于该最小报告尺度，而不是全尺寸岩块统计。

如果后续补充更小尺度测试，可以在本节增加一个小表，比较不同 \(d_{min}\) 下进入融合阶段的候选数量。例如 \(d_{min}=0.3\) m、0.4 m、0.5 m。这样可以说明方法流程并不绑定某一个固定尺寸，同时让读者看到统计尺度变化对候选数量的影响。

**Suggested Table 3. Summary of DOM tile-level detection results.**

| Item | Value |
| --- | ---: |
| Number of processed tiles | [to be filled] |
| Raw mask candidates | [to be filled] |
| Tile-level detections after confidence filtering | [to be filled] |
| Candidates retained under \(d_{min}\) | [to be filled] |
| Mean equivalent diameter | [to be filled] |
| Median equivalent diameter | [to be filled] |

**Optional Table 4. Sensitivity to the minimum reporting diameter.**

| \(d_{min}\) | Candidates entering fusion | Fused candidates | Accepted instances | Notes |
| ---: | ---: | ---: | ---: | --- |
| 0.3 m | [to be filled] | [to be filled] | [to be filled] | optional |
| 0.4 m | [to be filled] | [to be filled] | [to be filled] | optional |
| 0.5 m | [to be filled] | [to be filled] | [to be filled] | main setting |

**Suggested Fig. 7. Examples of DOM tile-level detections.**
建议展示 4 到 6 个代表性 tile。每个 tile 可包含原始 DOM 和半透明实例掩膜叠加。最好包含密集岩块区、边缘截断区、阴影或纹理复杂区。图注中明确这些是 tile-level candidate detections。

**Suggested Fig. 8. Distribution of equivalent diameter for detected candidates.**
建议使用直方图或核密度曲线。若测试多个 \(d_{min}\)，可以用不同颜色显示筛选前后或不同阈值下的分布。图中应标出 \(d_{min}\) 的位置。

## 4.3 Duplicate Resolution and Point-cloud-based Validation

本节是第四章的核心之一，用于展示 tile-level candidate detections 经重复消解和点云验证后转化为可测量岩块实例的过程与结果。第三章说明重复检测如何识别以及点云验证规则如何设置，本节仅报告不同策略的输出差异、各处理阶段的数量变化和代表性案例。

首先比较启发式融合和相关聚类两种重复消解策略。建议报告输入候选数量、处理后的岩块候选数量、合并比例及典型差异案例。这里不要简单写“相关聚类更好”，除非有人工核查或明确错误案例支撑。更稳妥的结果表述是，两种策略对跨 tile 候选的组织方式和最终保留数量存在差异；相关聚类被选作主流程输入的依据，应由数量结果和案例图共同给出。

随后报告点云验证结果。建议以数量链条方式展示：tile-level detections -> fused candidates -> accepted instances -> rejected instances。对于 rejected instances，应尽量给出拒绝原因统计，例如点数不足、高程范围不足、相对地面抬升不足或局部点云缺失。这个结果比单纯说“点云提高可靠性”更有说服力。

若当前主实验采用已有结果，可在草稿中暂时使用如下数量链条，并在最终论文前重新核对：`7349 tile-level detections -> 6258 fused candidates -> 6071 accepted instances -> 187 rejected instances`。这些数值必须与最终运行报告保持一致。

**Suggested Table 5. Comparison of duplicate-resolution strategies.**

| Duplicate-resolution strategy | Input detections | Resolved candidates | Merge ratio | Selected for subsequent measurement |
| --- | ---: | ---: | ---: | --- |
| Heuristic fusion | [to be filled] | [to be filled] | [to be filled] | No, baseline |
| Correlation clustering | [to be filled] | [to be filled] | [to be filled] | Yes, main workflow |

**Suggested Table 6. Summary of point-cloud-based validation results.**

| Stage | Number of instances | Percentage relative to previous stage |
| --- | ---: | ---: |
| Tile-level detections | [to be filled] | - |
| Fused candidates | [to be filled] | [to be filled] |
| Accepted by 3D validation | [to be filled] | [to be filled] |
| Rejected by 3D validation | [to be filled] | [to be filled] |

**Suggested Fig. 9. Changes in the number of rock-fragment candidates across processing stages.**
建议用 Sankey 图或简洁的阶段数量图展示 `tile-level candidates -> resolved candidates -> accepted/rejected instances -> volume QC passed`。图中只表达样本数量如何变化，不重复绘制第三章已经给出的算法流程。

**Suggested Fig. 10. Accepted and rejected examples after point-cloud validation.**
建议做成对比图。每个案例包含 DOM 掩膜、局部点云和高度剖面。accepted case 显示明显三维起伏，rejected case 显示点数不足或高度起伏不足。

## 4.4 Rock Fragment Size and Volume Characteristics

本节报告通过点云验证后岩块实例的粒径与体积特征。它对应整个流程的主要测量输出，重点是呈现当前试验场景中的岩块尺度组成、2.5D 体积分布及不同粒径区间的体积贡献，而不是证明体积估计具有绝对精度。

建议先报告进入体积估计的岩块数量、通过体积质量控制的数量、等效直径统计、总体体积、平均体积、中位数体积以及体积分布范围。由于岩块体积通常呈偏态分布，不能只报告均值，还应报告中位数和四分位区间。

随后按等效直径区间统计数量和体积贡献。这个结果对工程测量最有意义，因为它能回答不同粒径范围对总体体积的贡献。粒径区间可以先沿用你前面讨论过的分组，例如 0.5-0.75 m、0.75-1.0 m、1.0-1.5 m 和 >1.5 m；如果后续 \(d_{min}\) 调整为更小，则分组也应相应修改。

**Suggested Table 7. Summary statistics of the measured rock fragments.**

| Item | Value |
| --- | ---: |
| Instances entering volume estimation | [to be filled] |
| Instances passing volume QC | [to be filled] |
| Mean equivalent diameter | [to be filled] |
| Median equivalent diameter | [to be filled] |
| Total 2.5D volume | [to be filled] |
| Mean 2.5D volume | [to be filled] |
| Median 2.5D volume | [to be filled] |
| 25th-75th percentile | [to be filled] |

**Suggested Table 8. Diameter-binned size and volume statistics.**

| Equivalent diameter bin | Number of stones | Percentage of stones | Total 2.5D volume | Percentage of volume |
| --- | ---: | ---: | ---: | ---: |
| 0.5-0.75 m | [to be filled] | [to be filled] | [to be filled] | [to be filled] |
| 0.75-1.0 m | [to be filled] | [to be filled] | [to be filled] | [to be filled] |
| 1.0-1.5 m | [to be filled] | [to be filled] | [to be filled] | [to be filled] |
| >1.5 m | [to be filled] | [to be filled] | [to be filled] | [to be filled] |

**Suggested Fig. 11. Size and 2.5D volume distributions of the measured rock fragments.**
建议使用直方图或箱线图。若体积分布长尾明显，可以使用 log-scale 横轴或在图注中说明少数大岩块对总体体积的贡献较高。

**Suggested Fig. 12. Diameter-binned volume contribution.**
建议用双轴图或并列柱状图：一个柱表示岩块数量占比，另一个柱表示体积占比。这样能展示“小岩块数量多但体积贡献不一定最大”的工程现象。

## 4.5 Comparison of 2D Proxy and 2.5D Volume Estimates

本节用于展示二维代理体积和点云 2.5D 体积之间的差异。它不是为了证明 2.5D 一定是真值，而是说明仅由二维等效直径推导体积会与考虑实际地面参考和高度起伏的估计结果产生差别。

建议先报告两种体积估计在总体统计上的差异，包括总量、均值、中位数和比例关系。然后绘制散点图，以每个岩块为单位比较 `2D proxy volume` 和 `2.5D volume`。如果二者差异随粒径变大而扩大，可以在图中用颜色表示等效直径区间。

这一节的表述要克制。可以说“2D proxy 与 2.5D volume 存在系统性差异”或“二者在部分粒径区间内差异更明显”，但不要说“2.5D 体积更准确”，除非你有人工体积真值或独立测量数据支撑。

**Suggested Table 9. Comparison between 2D proxy volume and 2.5D volume.**

| Metric | 2D proxy volume | 2.5D volume | Ratio or difference |
| --- | ---: | ---: | ---: |
| Total volume | [to be filled] | [to be filled] | [to be filled] |
| Mean volume | [to be filled] | [to be filled] | [to be filled] |
| Median volume | [to be filled] | [to be filled] | [to be filled] |
| Interquartile range | [to be filled] | [to be filled] | [to be filled] |

**Suggested Fig. 13. Per-instance comparison between 2D proxy and 2.5D volume.**
建议使用散点图，横轴为 2D proxy volume，纵轴为 2.5D volume，并添加 y=x 参考线。点的颜色可表示等效直径区间，点大小可表示掩膜面积。该图能直观展示哪些岩块的二维近似与 2.5D 估计差异较大。

第四章末尾不再单设结果小结。完成二维代理体积与 2.5D 体积对比后，可以用一段简短文字概括两者在总体统计和不同粒径区间中的主要差异，并自然过渡到第五章 Discussion。原因解释、工程意义和局限性均放在第五章展开。

## Notes for Chapter 4

- 所有结果必须绑定最终主实验数据，不要混用不同 run 或不同 DOM/point-cloud 场景的统计值。
- 如果采用 \(d_{min}=0.5\) m，图表标题和正文中应明确这是当前主实验的最小报告直径。
- 如果后续测试更小尺寸，建议加入 Table 4，作为最小报告尺度敏感性分析。
- 没有人工体积真值前，不要使用 `accuracy`, `ground truth`, `absolute error` 这类容易引起误解的词。
- 第四章可以展示误检/漏检案例，但如果没有系统人工标注，不建议报告标准 precision、recall 或 mAP。

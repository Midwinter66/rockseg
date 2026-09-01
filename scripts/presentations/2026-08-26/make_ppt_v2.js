const pptxgen = require("./node_modules/pptxgenjs");
const pres = new pptxgen();

pres.layout = "LAYOUT_16x9";
const SW = 10, SH = 5.625;
const FONT = "Microsoft YaHei";
const BLACK = "000000", GRAY = "595959", LGRAY = "D9D9D9", WHITE = "FFFFFF";
const BLUE = "1F4E79", GREEN = "2E7D32";

// ---- helpers ----
function addTitle(slide, text) {
  slide.addText(text, { x: 0.5, y: 0.2, w: 9.0, h: 0.55, fontFace: FONT, fontSize: 24, bold: true, color: BLUE, align: "left" });
  slide.addShape(pres.ShapeType.line, { x: 0.5, y: 0.78, w: 9.0, h: 0, line: { color: LGRAY, width: 1.2 } });
}

function addNarr(slide, lines, opts = {}) {
  const y0 = opts.y ?? 0.95, x0 = opts.x ?? 0.6, w = opts.w ?? 8.8, fs = opts.fontSize ?? 13;
  let y = y0;
  for (const ln of lines) {
    if (typeof ln === "string") {
      slide.addText(ln, { x: x0, y, w, h: 0.36, fontFace: FONT, fontSize: fs, color: BLACK, align: "left", valign: "top" });
      y += 0.36;
    } else {
      slide.addText(ln.text, { x: x0, y, w, h: ln.h ?? 0.36, fontFace: FONT, fontSize: ln.fontSize ?? fs, color: ln.color ?? BLACK, bold: ln.bold ?? false, align: "left", valign: "top" });
      y += ln.h ?? 0.36;
    }
  }
  return y;
}

function addTable(slide, rows, opts) {
  const o = Object.assign({
    x: 0.6, y: 1.1, w: 8.8,
    fontFace: FONT, fontSize: 12.5, color: BLACK,
    border: { type: "solid", pt: 0.75, color: "BFBFBF" },
    align: "left", valign: "middle",
    fill: { color: WHITE },
    rowH: 0.4,
  }, opts);
  slide.addTable(rows, o);
}

// ========== S1 封面 ==========
{
  const s = pres.addSlide();
  s.addText("RockSeg 研究进展汇报", { x: 1, y: 2.1, w: 8, h: 0.9, fontFace: FONT, fontSize: 38, bold: true, color: BLACK, align: "center" });
  s.addText("2026-08-26", { x: 1, y: 3.2, w: 8, h: 0.5, fontFace: FONT, fontSize: 22, color: GRAY, align: "center" });
}

// ========== S2 整体主线 一图流 ==========
{
  const s = pres.addSlide();
  addTitle(s, "整体研究主线");
  const steps = [
    "无人机影像", "DOM正射影像", "物理尺度分析", "多尺度切片",
    "实例分割", "同Tile融合", "跨尺度融合", "最终岩石实例",
    "2D-3D关联", "三维筛查", "地面构建", "10mm 2.5D表面",
    "12维形状特征", "LightGBM体积校正", "4000块分层估算", "粒径体积统计",
  ];
  const bw = 2.0, bh = 0.55, gx = 0.22, gy = 0.45;
  const x0 = 0.35, y0 = 0.95, cols = 4;
  for (let i = 0; i < 16; i++) {
    const r = Math.floor(i / cols), c = i % cols;
    const x = x0 + c * (bw + gx), y = y0 + r * (bh + gy);
    s.addShape(pres.ShapeType.roundRect, { x, y, w: bw, h: bh, rectRadius: 0.05, line: { color: BLUE, width: 1.2 }, fill: { color: "F2F7FB" } });
    s.addText(`${i+1}. ${steps[i]}`, { x: x+0.05, y, w: bw-0.1, h: bh, fontFace: FONT, fontSize: 12, color: BLACK, align: "center", valign: "middle" });
    if (c < cols - 1) s.addShape(pres.ShapeType.rightArrow, { x: x+bw+0.04, y: y+bh/2-0.08, w: 0.14, h: 0.16, fill: { color: BLUE } });
  }
  for (let r = 0; r < 3; r++) {
    const y = y0 + r*(bh+gy) + bh + 0.06;
    s.addShape(pres.ShapeType.downArrow, { x: x0 + 1.5*(bw+gx) + bw/2 - 0.08, y, w: 0.16, h: 0.32, fill: { color: BLUE } });
  }
}

// ========== S3 DOM与多尺度 ==========
{
  const s = pres.addSlide();
  addTitle(s, "DOM数据与多尺度分割");
  addNarr(s, [
    "采用物理尺度驱动的多尺度方法，使不同大小的石块在网络输入中始终处于合适的像素尺寸。",
    "三档切片覆盖粒径跨度（约2 cm — 3.4 m），解决固定像素尺度下小石块看不清、大石块被切碎的问题。",
  ], { y: 0.95 });
  addTable(s, [
    [{ text: "数据项", options: { bold: true, fill: { color: "EEF2F7" } } }, { text: "内容", options: { bold: true, fill: { color: "EEF2F7" } } }],
    ["DOM数据", "DOM2矿区正射影像"],
    ["影像分辨率", "0.01 米 / 像素（10 mm）"],
    ["粗尺度", "单块覆盖 10.24 m，适用 ≥ 0.5 m 大石块"],
    ["中尺度", "单块覆盖 5.12 m，适用 0.3 – 0.5 m 中石块"],
    ["细尺度", "单块覆盖 2.56 m，适用 < 0.3 m 小石块"],
  ], { y: 2.0, rowH: 0.42, fontSize: 13 });
}

// ========== S4 级联结果（中文叙述+表格）==========
{
  const s = pres.addSlide();
  addTitle(s, "多尺度级联检测结果");
  addNarr(s, [
    "三个尺度独立检测产生大量重复。经过同尺度融合和跨尺度去重两道工序，最终得到唯一的岩石实例。",
    "级联去重先在每个尺度内部合并碎片检测，再在尺度之间合并同一石块的重复检测。",
  ], { y: 0.95 });
  addTable(s, [
    [{ text: "处理阶段", options: { bold: true, fill: { color: "EEF2F7" } } }, { text: "石块数量", options: { bold: true, fill: { color: "EEF2F7" } } }],
    ["粗尺度原始检测", "37,470"],
    ["中尺度原始检测", "101,642"],
    ["细尺度原始检测", "179,286"],
    ["原始检测合计", "318,398"],
    ["同尺度融合后", "112,983"],
    ["跨尺度去重后（最终）", "76,407"],
    ["通过三维筛查", "69,911"],
  ], { x: 0.6, y: 2.05, w: 5.5, rowH: 0.38, fontSize: 12.5, colW: [3.0, 2.5] });
  s.addShape(pres.ShapeType.rect, { x: 6.5, y: 2.2, w: 2.9, h: 1.6, fill: { color: "F2F7FB" }, line: { color: BLUE, width: 1 } });
  s.addText("去重比例", { x: 6.5, y: 2.35, w: 2.9, h: 0.4, fontFace: FONT, fontSize: 14, bold: true, color: BLUE, align: "center" });
  s.addText("318,398 → 76,407", { x: 6.5, y: 2.85, w: 2.9, h: 0.5, fontFace: FONT, fontSize: 18, bold: true, color: BLACK, align: "center" });
  s.addText("重复率约 76%", { x: 6.5, y: 3.35, w: 2.9, h: 0.35, fontFace: FONT, fontSize: 13, color: GRAY, align: "center" });
}

// ========== S5 同Tile融合（数学公式） ==========
{
  const s = pres.addSlide();
  addTitle(s, "同Tile融合");
  addNarr(s, [
    "相邻Tile在重叠区域会重复检测到同一石块的碎片，需要通过多特征评分判断是否为同一石块并合并。",
  ], { y: 0.92 });

  // 候选条件
  addNarr(s, [
    { text: "候选匹配条件（预筛选）", bold: true, fontSize: 13, h: 0.34 },
    "边框交并比 ≥ 0.05，且掩膜交并比 > 0",
  ], { y: 1.35, x: 0.6, w: 5.4, fontSize: 12.5 });

  // 公式框
  s.addShape(pres.ShapeType.rect, { x: 0.6, y: 2.25, w: 5.4, h: 1.05, fill: { color: "FAFBFD" }, line: { color: BLUE, width: 1.2 } });
  s.addText("融合评分公式", { x: 0.6, y: 2.28, w: 5.4, h: 0.32, fontFace: FONT, fontSize: 12, bold: true, color: BLUE, align: "center" });
  s.addText("S = 0.30 · IoU  +  0.20 · exp(−d²/2σ²)  +  0.20 · rA  +  0.15 · B  +  0.15 · C", { x: 0.6, y: 2.6, w: 5.4, h: 0.4, fontFace: "Cambria Math", fontSize: 14, color: BLACK, align: "center", italic: true });
  s.addText("σ = 50 像素（相当于 0.5 米）；rA = 面积较小值 / 面积较大值；B = 边界完整性均值；C = 置信度均值", { x: 0.6, y: 3.0, w: 5.4, h: 0.3, fontFace: FONT, fontSize: 10.5, color: GRAY, align: "center" });

  // 各特征作用（无数值）
  addTable(s, [
    [{ text: "评分项", options: { bold: true, fill: { color: "EEF2F7" } } }, { text: "作用", options: { bold: true, fill: { color: "EEF2F7" } } }],
    ["掩膜交并比 IoU", "衡量两个检测的形状重叠程度，重叠越多越可能是同一块"],
    ["质心距离", "衡量两个检测中心的接近程度，越近越可能是同一块"],
    ["面积相似比", "两块面积差别太大通常不是同一石块"],
    ["边界完整性", "位于Tile边界的检测需要降低权重，避免误合并"],
    ["置信度", "检测置信度高的结果更可靠"],
  ], { x: 0.6, y: 3.45, w: 5.4, rowH: 0.33, fontSize: 11, colW: [1.6, 3.8] });

  // 组内规范 + 图片
  addNarr(s, [
    { text: "合并规则", bold: true, h: 0.32, fontSize: 12.5 },
    "评分 ≥ 0.50 → 判定为同一石块并合并",
    "组内保留置信度与边界完整性乘积最高的实例作为代表",
  ], { x: 6.3, y: 1.35, w: 3.3, fontSize: 12 });
  s.addImage({ path: "output/dom2_full/visualizations/fusion_demo/zoom_within_6_463dets.png", x: 6.3, y: 2.5, w: 3.3, h: 2.6 });
  s.addText("同一石块被多个相邻Tile重复检测（不同颜色），经评分判断后合并为一个实例", { x: 6.3, y: 5.15, w: 3.3, h: 0.35, fontFace: FONT, fontSize: 10, color: GRAY, align: "left" });
}

// ========== S6 跨尺度融合 ==========
{
  const s = pres.addSlide();
  addTitle(s, "跨尺度融合（级联去重）");
  addNarr(s, [
    "同一块石头在粗、中、细三个尺度都会被检测到，需要判断哪些检测属于同一块，并只保留一个最终结果。",
  ], { y: 0.92 });

  // 左：候选条件
  addNarr(s, [
    { text: "候选分组条件", bold: true, h: 0.34, fontSize: 13 },
    "① 边框有一定重叠（IoU ≥ 0.05）",
    "② 掩膜存在重叠（IoU > 0）",
    "③ 尺寸相差不悬殊（等效直径比 ≥ 0.3）",
    "④ 质心距离在较大石块半径以内",
  ], { y: 1.4, x: 0.6, w: 4.8, fontSize: 12.5 });

  // 主尺度规则
  addNarr(s, [
    { text: "主尺度保留规则", bold: true, h: 0.34, fontSize: 13 },
    "等效直径 ≥ 0.50 m → 保留粗尺度",
    "等效直径 0.30 – 0.50 m → 保留中尺度",
    "等效直径 < 0.30 m → 保留细尺度",
    "每个石块只保留其主尺度中质量最高的检测",
    "不做跨尺度掩膜融合，避免误合并相邻石块",
  ], { y: 3.15, x: 0.6, w: 4.8, fontSize: 12.5 });

  // 右：图片+说明
  s.addImage({ path: "0826/fig_cross_scale_prod_1136x591.png", x: 5.8, y: 1.3, w: 3.9, h: 2.0 });
  s.addShape(pres.ShapeType.rect, { x: 5.8, y: 3.9, w: 3.8, h: 1.4, fill: { color: "F5F9FC" }, line: { color: LGRAY, width: 0.8 } });
  s.addText("案例说明", { x: 5.8, y: 3.95, w: 3.8, h: 0.32, fontFace: FONT, fontSize: 12, bold: true, color: BLUE, align: "center" });
  s.addText("左图：融合前——同一块石头被多尺度、多切片重复检测为多个彩色碎片；右图：生产级联去重后——每块石头唯一保留一个实例（本区域共5,540块，含35块一米以上大石块）。", { x: 5.95, y: 4.25, w: 3.5, h: 1.0, fontFace: FONT, fontSize: 11, color: BLACK, align: "left", valign: "top" });
}

// ========== S7 最终实例 ==========
{
  const s = pres.addSlide();
  addTitle(s, "最终岩石实例");
  addNarr(s, [
    "经过多尺度检测、同Tile融合和跨尺度级联去重，最终得到每块岩石唯一的实例结果。",
    "每个实例包含：像素坐标边框、掩膜轮廓、置信度、所属主尺度、边界完整性等信息。",
    "后续所有三维关联和体积计算均基于这 76,407 个唯一实例展开。",
  ], { y: 1.1, fontSize: 14 });
  addTable(s, [
    [{ text: "项目", options: { bold: true, fill: { color: "EEF2F7" } } }, { text: "数值", options: { bold: true, fill: { color: "EEF2F7" } } }],
    ["最终实例总数", "76,407 块"],
    ["通过三维筛查", "69,911 块"],
    ["三维筛查通过率", "91.5%"],
    ["粗尺度主导石块", "约 1,500 块（大石块）"],
    ["中尺度主导石块", "约 8,800 块（中石块）"],
    ["细尺度主导石块", "约 66,000 块（小石块）"],
  ], { y: 2.5, w: 6.0, rowH: 0.42, fontSize: 13, colW: [3.0, 3.0] });
}

// ========== S8 2D-3D关联筛查（全中文） ==========
{
  const s = pres.addSlide();
  addTitle(s, "二维—三维关联筛查");
  addNarr(s, [
    "将二维检测框映射到三维点云空间，逐块统计点云特征。没有真实抬升高度的误检（阴影、纹理等）会被剔除。",
  ], { y: 0.92 });

  // 左：筛选条件（全中文）
  addNarr(s, [
    { text: "四项筛查条件（须全部满足）", bold: true, h: 0.36, fontSize: 13 },
    "① 点云数量 ≥ 60 个",
    "② 高度范围 ≥ 0.18 米",
    "③ 第90百分位高度 ≥ 0.12 米",
    "④ 抬高点占比 ≥ 20%（抬升阈值 0.08 米）",
  ], { y: 1.4, x: 0.6, w: 4.5, fontSize: 12.5 });

  addNarr(s, [
    { text: "剔除原因统计（6,496 块被筛除）", bold: true, h: 0.36, fontSize: 12.5 },
    "抬高点占比不足：5,847 块（最主要）",
    "第90百分位高度不足：5,207 块",
    "高度范围不足：357 块",
    "点云数量不足：184 块",
    "地面点不足：3 块",
  ], { y: 3.35, x: 0.6, w: 4.5, fontSize: 11.5 });

  // 右：对比图
  s.addImage({ path: "0826/fig_2d3d_screening_1340x696.png", x: 5.3, y: 1.15, w: 4.3, h: 2.9 });
  s.addText("左图：筛查前（绿色保留，红色剔除）  右图：筛查后", { x: 5.3, y: 4.15, w: 4.3, h: 0.32, fontFace: FONT, fontSize: 10.5, color: GRAY, align: "center" });
  s.addText("被剔除的红框集中在影像边缘无数据带和阴影区域，说明三维筛查有效消除了误检。", { x: 5.3, y: 4.5, w: 4.3, h: 0.5, fontFace: FONT, fontSize: 11, color: BLACK, align: "left" });
}

// ========== S9 地面构建 ==========
{
  const s = pres.addSlide();
  addTitle(s, "地面构建与地面移除");
  const steps = ["原始点云", "地面DEM", "高度归一化", "岩石相对高度", "2.5D顶面"];
  const bw = 1.6, bh = 0.55, y = 1.05, x0 = 0.6;
  for (let i = 0; i < steps.length; i++) {
    const x = x0 + i * (bw + 0.32);
    s.addShape(pres.ShapeType.roundRect, { x, y, w: bw, h: bh, rectRadius: 0.06, line: { color: BLUE, width: 1.2 }, fill: { color: "F2F7FB" } });
    s.addText(steps[i], { x: x+0.05, y, w: bw-0.1, h: bh, fontFace: FONT, fontSize: 12, color: BLACK, align: "center", valign: "middle" });
    if (i < steps.length - 1) s.addShape(pres.ShapeType.rightArrow, { x: x+bw+0.06, y: y+bh/2-0.08, w: 0.2, h: 0.16, fill: { color: BLUE } });
  }

  addNarr(s, [
    { text: "地面DEM构建方式", bold: true, h: 0.36, fontSize: 13 },
    "将场景点云按 0.5 米网格分箱，每格取高度的第 5 百分位数作为地面高程（低百分位可抑制石块抬升点对地面的高估）。",
    "每格至少 3 个点，不足处通过空洞插值填充。采样步长 100 用于加速场景级构建。",
    { text: "地面移除", bold: true, h: 0.36, fontSize: 13 },
    "逐点查询所在格的地面高程，计算相对高度 h = z − z_ground（截断为非负值）。",
    "岩石 bbox 内的点云按 10 毫米网格取每格最大相对高度，得到地面参考的 2.5D 顶面。",
  ], { y: 1.95, x: 0.6, w: 8.8, fontSize: 12.5 });
}

// ========== S10 外部数据与Shape-Aware ==========
{
  const s = pres.addSlide();
  addTitle(s, "外部数据集与形状校正模型");
  addNarr(s, [
    "真实矿区没有单石块的真值体积，因此用外部三维网格数据集（OBJ格式，每个都有精确体积）来验证体积校正方法。",
    "数据集包含两类共 465 个石块模型，用于方法学验证，不直接用于矿区。",
  ], { y: 0.92 });

  addTable(s, [
    [{ text: "数据集", options: { bold: true, fill: { color: "EEF2F7" } } }, { text: "数量", options: { bold: true, fill: { color: "EEF2F7" } } }, { text: "说明", options: { bold: true, fill: { color: "EEF2F7" } } }],
    ["T01", "79", "火山碎屑岩"],
    ["L01", "386", "另一类岩石样本"],
    ["合计", "465", "均为三维网格模型，有精确体积真值"],
  ], { y: 1.85, rowH: 0.4, fontSize: 12.5, colW: [1.8, 1.5, 5.5] });

  addNarr(s, [
    { text: "尺度适配：外部模型 → 真实矿区分辨率", bold: true, h: 0.38, fontSize: 13 },
    "外部模型尺寸较小，直接用 10 毫米栅格化会产生大量空表面。经尺度审计后统一放大 82.74 倍，使外部模型与矿区石块的footprint尺度可比。",
    "放大后 465/465 全部形成有效 10 毫米表面，按对象分组切分为训练/验证/测试 = 326/70/69。",
    { text: "模型：LightGBM 梯度提升树", bold: true, h: 0.38, fontSize: 13 },
    "输入 12 维无量纲形状特征 → 输出校正比 y = V真实 / V_2.5D → 最终体积 V预测 = V_2.5D × y预测",
  ], { y: 3.25, x: 0.6, w: 8.8, fontSize: 12.5 });
}

// ========== S11 12维特征 ==========
{
  const s = pres.addSlide();
  addTitle(s, "十二维形状感知特征");
  addNarr(s, [
    "从二维轮廓、高度分布和三维形状三个角度提取 12 个无量纲特征，描述岩石的形态特点。",
  ], { y: 0.92 });
  addTable(s, [
    [{ text: "特征类别", options: { bold: true, fill: { color: "EEF2F7" } } }, { text: "特征名称", options: { bold: true, fill: { color: "EEF2F7" } } }, { text: "物理含义", options: { bold: true, fill: { color: "EEF2F7" } } }],
    ["二维形态", "圆度 C", "轮廓接近圆形的程度"],
    ["二维形态", "长宽比 AR", "最大直径与最小直径之比"],
    ["二维形态", "充实度 solidity", "轮廓面积与凸包面积之比"],
    ["二维形态", "紧凑度 compactness", "面积与周长平方的比值"],
    ["二维形态", "等效直径比 eq_diam_ratio", "等效圆直径与最大直径之比"],
    ["高度统计", "高度均值比 H_mean_norm", "平均高度 / 最大高度"],
    ["高度统计", "高度标准差比 H_std_norm", "高度标准差 / 最大高度"],
    ["高度统计", "高度分位数比 H_p25 / H_p75", "第25/75百分位高度 / 最大高度"],
    ["高度统计", "高度偏度 H_skew_norm", "高度分布的偏斜程度"],
    ["三维形状", "填充比 fill_ratio", "2.5D体积与外接棱柱体积之比"],
    ["三维形状", "椭球比 ellipsoid_ratio", "2.5D体积与半椭球体积之比"],
  ], { y: 1.4, rowH: 0.29, fontSize: 11, colW: [1.4, 3.2, 4.2] });
}

// ========== S12 Shape-Aware验证结果 ==========
{
  const s = pres.addSlide();
  addTitle(s, "形状校正模型验证结果");
  addNarr(s, [
    "在缩放至 10 毫米的外部网格测试集上对比三种体积估算方法。",
  ], { y: 0.92 });
  addTable(s, [
    [{ text: "方法", options: { bold: true, fill: { color: "EEF2F7" } } }, { text: "平均绝对误差 (mm³)", options: { bold: true, fill: { color: "EEF2F7" } } }, { text: "平均相对误差", options: { bold: true, fill: { color: "EEF2F7" } } }, { text: "决定系数 R²", options: { bold: true, fill: { color: "EEF2F7" } } }],
    ["原始2.5D体积", "59,903,925", "54.24%", "0.1895"],
    ["全局常数校正", "8,723,245", "6.99%", "0.9705"],
    ["形状感知校正 V2", "7,167,711", "5.82%", "0.9838"],
  ], { y: 1.55, rowH: 0.46, fontSize: 13.5, colW: [2.2, 2.6, 2.0, 2.0] });
  addNarr(s, [
    "最优迭代次数：第 356 轮。测试样本数：69 个（按对象分组切分，无泄漏）。",
    "形状感知校正将原始 2.5D 的相对误差从 54.24% 降至 5.82%，且优于简单的全局常数校正。",
    { text: "注意：5.82% 是外部缩放网格测试集的方法学验证结果，不是真实矿区的体积误差。", color: GRAY, h: 0.4, fontSize: 11.5 },
  ], { y: 3.5, x: 0.6, w: 8.8, fontSize: 12.5 });
}

// ========== S13 4000块体积估算 ==========
{
  const s = pres.addSlide();
  addTitle(s, "四千块真实矿区体积估算");
  addNarr(s, [
    "从 69,911 个通过三维筛查的石块中，按等效直径分层抽取 4,000 块进行体积估算（分层代表性应用，非全量估算）。",
  ], { y: 0.92 });

  // 左：总体结果
  addTable(s, [
    [{ text: "项目", options: { bold: true, fill: { color: "EEF2F7" } } }, { text: "数量", options: { bold: true, fill: { color: "EEF2F7" } } }],
    ["三维筛查通过总数", "69,911 块"],
    ["分层抽样数量", "4,000 块"],
    ["体积估算成功", "3,639 块"],
    ["估算失败", "361 块"],
    ["流程成功率", "90.98%"],
  ], { x: 0.6, y: 1.55, w: 4.3, rowH: 0.42, fontSize: 13, colW: [2.4, 1.9] });

  // 右：分层成功率
  addTable(s, [
    [{ text: "粒径层", options: { bold: true, fill: { color: "EEF2F7" } } }, { text: "抽样数", options: { bold: true, fill: { color: "EEF2F7" } } }, { text: "成功率", options: { bold: true, fill: { color: "EEF2F7" } } }],
    ["S1 最小10%", "400", "84.0%"],
    ["S2 P10–P25", "600", "86.3%"],
    ["S3 P25–P50", "1,000", "87.5%"],
    ["S4 P50–P75", "1,000", "92.3%"],
    ["S5 P75–P90", "600", "98.0%"],
    ["S6 最大10%", "400", "99.8%"],
  ], { x: 5.3, y: 1.55, w: 4.1, rowH: 0.36, fontSize: 12.5, colW: [2.0, 1.0, 1.1] });

  addNarr(s, [
    { text: "失败原因：361 块全部为 2.5D 顶面为空（小石块在 10 毫米网格下没有足够点形成有效表面）。", bold: true, h: 0.4, fontSize: 12 },
    "成功率随粒径单调上升，越大的石块表面越完整、估算越可靠。",
    { text: "90.98% 是体积估算流程的成功率，不是体积精度或模型准确率。", color: GRAY, h: 0.38, fontSize: 11 },
  ], { y: 4.15, x: 0.6, w: 8.8, fontSize: 12.5 });
}

// ========== S14 4000块体积结果分析（统计图） ==========
{
  const s = pres.addSlide();
  addTitle(s, "四千块体积结果分析");

  // 上：体积分布直方图
  s.addImage({ path: "0826/fig_volume_histogram_1067x587.png", x: 0.5, y: 0.95, w: 5.8, h: 3.2 });

  // 右上：关键数据
  addTable(s, [
    [{ text: "关键指标", options: { bold: true, fill: { color: "EEF2F7" } } }, { text: "数值", options: { bold: true, fill: { color: "EEF2F7" } } }],
    ["样本总体积 (V预测)", "约 40.28 立方米"],
    ["单石块体积中位数", "0.00103 立方米"],
    ["第25百分位", "0.00025 立方米"],
    ["第75百分位", "0.00413 立方米"],
    ["总体校正比 V/V_2.5D", "0.681"],
    ["校正比中位数", "0.680"],
    ["粒径范围", "0.023 – 3.45 米"],
  ], { x: 6.6, y: 1.0, w: 2.9, rowH: 0.34, fontSize: 11, colW: [1.5, 1.4] });

  // 下：分层体积图
  s.addImage({ path: "0826/fig_volume_stratum_1428x564.png", x: 0.5, y: 4.25, w: 9.0, h: 1.25 });
}

// ---- 保存 ----
pres.writeFile({ fileName: "0826/0826_RockSeg_Research_Progress_v2.pptx" }).then(() => console.log("PPT v2 saved"));





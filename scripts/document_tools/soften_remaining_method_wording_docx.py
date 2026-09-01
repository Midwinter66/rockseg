from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt


DOCX_PATH = Path(
    r"C:\Users\Administrator\WPSDrive\1714584739\WPS企业云盘\杭州电子科技大学\我的企业文档\Measurement期刊\中文论文第二章_材料与方法与第三章结果框架.docx"
)


REPLACEMENTS = {
    "大范围高分辨率 DOM 不能在保持原始地面分辨率的条件下直接输入实例分割网络。若将整幅影像缩放，岩块边界和小尺度纹理会被压缩；若采用统一小窗口遍历，则低纹理区域和影像外部背景同样产生推理开销。为此，本文以局部边缘分布为依据递归划分 DOM，使切片尺度随局部结构复杂度变化。": (
        "大范围高分辨率 DOM 需要在保持地面分辨率的同时适配实例分割网络的输入尺度。"
        "整幅影像缩放会削弱岩块边界和小尺度纹理，统一小窗口遍历则会增加低纹理区域和影像外部背景的推理开销。"
        "为此，本文以局部边缘分布为依据递归划分 DOM，使切片尺度随局部结构复杂度变化。"
    ),
    "从剔除原因看，矿区 B 中主要问题集中在相对高度分位数不足和抬升点比例不足，分别记录为 334 次和 329 次；局部高程范围不足记录为 16 次，点数不足记录为 1 次。由于同一对象可能同时触发多个筛查条件，各类原因的次数之和不要求等于剔除对象总数。这一结果说明三维筛查主要发挥的是几何一致性约束作用：它可以排除一部分在二维影像中形态相似、但缺少足够三维起伏或点云支撑的候选对象。": (
        "从筛查记录看，矿区 B 中相对高度分位数和抬升点比例是触发候选剔除的主要条件，分别记录为 334 次和 329 次；"
        "局部高程范围和点数条件分别记录为 16 次和 1 次。由于同一对象可能同时触发多个筛查条件，各类原因的次数之和不要求等于剔除对象总数。"
        "这一结果说明三维筛查主要发挥几何一致性约束作用，使进入体积估算的对象具有更明确的局部三维支撑。"
    ),
    "矿区 A 的二维代理总体积为 2221.62 m³，高于对应的 2.5D 总体积 1451.01 m³；矿区 B 的二维代理总体积为 2916.31 m³，同样高于 2.5D 总体积 2090.23 m³。在逐石尺度上，矿区 A 中两种体积指标的 Pearson 相关系数为 0.820，矿区 B 为 0.818。这说明二维尺度信息与 2.5D 体积之间存在较强相关性，但二维代理方法倾向于将等效直径直接映射为规则几何体，无法反映石块可见表面的真实起伏和局部高度变化。": (
        "矿区 A 的二维代理总体积为 2221.62 m³，高于对应的 2.5D 总体积 1451.01 m³；矿区 B 的二维代理总体积为 2916.31 m³，同样高于 2.5D 总体积 2090.23 m³。"
        "在逐石尺度上，矿区 A 中两种体积指标的 Pearson 相关系数为 0.820，矿区 B 为 0.818。"
        "这说明二维尺度信息与 2.5D 体积之间存在较强相关性，同时二维代理方法将等效直径映射为规则几何体，而 2.5D 方法进一步引入了可见表面的局部高度变化。"
    ),
    "因此，二维代理体积更适合作为方法对照，而不是最终测量结果。在矿山现场的石料堆积场景中，石块外形不规则、遮挡和接触关系复杂，仅依赖二维轮廓容易放大或压缩部分对象的体积估计。引入点云高度信息后，2.5D 方法能够在同一二维边界内利用局部 GroundDEM 作为参考面，对可见表面的高度积分进行估算，从而更符合本文面向工程测量流程的研究目标。": (
        "因此，二维代理体积在本文中作为不引入高程信息的对照指标使用。"
        "在矿山现场的石料堆积场景中，石块外形不规则，遮挡和接触关系复杂；引入点云高度信息后，2.5D 方法能够在同一二维边界内利用局部 GroundDEM 作为参考面，对可见表面的高度积分进行估算，从而更符合本文面向工程测量流程的研究目标。"
    ),
}


def set_font(paragraph) -> None:
    for run in paragraph.runs:
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        run.font.size = Pt(10.5)


def main() -> None:
    doc = Document(DOCX_PATH)
    changed = []
    for idx, paragraph in enumerate(doc.paragraphs):
        text = paragraph.text.strip()
        if text in REPLACEMENTS:
            paragraph.text = REPLACEMENTS[text]
            set_font(paragraph)
            changed.append(idx)
    if len(changed) != len(REPLACEMENTS):
        raise RuntimeError(f"Expected {len(REPLACEMENTS)} replacements, changed {changed}")
    doc.save(DOCX_PATH)
    print(DOCX_PATH)
    print("changed_paragraphs", changed)


if __name__ == "__main__":
    main()

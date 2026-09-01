from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt


DOCX_PATH = Path(
    r"C:\Users\Administrator\WPSDrive\1714584739\WPS企业云盘\杭州电子科技大学\我的企业文档\Measurement期刊\中文论文第二章_材料与方法与第三章结果框架.docx"
)


REPLACEMENTS = {
    "切片重叠保留了边界附近岩块的完整信息，也使同一岩块可能在不同切片中产生重复检测。为避免重复计数，掩膜质心和包围框先由式（1）转换至世界坐标，再在来自不同切片的检测之间建立关联。关联同时考虑质心距离、包围框交并比及来源切片约束；满足条件的候选通过相关聚类合并为单一岩块对象。": (
        "切片重叠保留了边界附近岩块的完整信息，也使同一岩块可能在不同切片中产生重复检测。"
        "为避免重复计数，掩膜质心和包围框先由式（1）转换至世界坐标，再在来自不同切片的检测之间建立关联。"
        "关联同时考虑质心距离、包围框交并比及来源切片约束；满足条件的候选通过相关聚类合并为单一岩块对象。"
        "该过程对应图 2-2 中的跨切片重复检测融合环节。"
    ),
    "融合对象的局部点云采用“包围框粗筛—掩膜投影精筛”两步提取。系统先以融合包围框在 XY 空间索引中查询候选点，再仅保留落入来源掩膜投影范围的点。该处理避免逐石扫描全场点云，同时使二维岩块轮廓与三维点集在同一空间参考下对应。": (
        "融合对象的局部点云采用“包围框粗筛—掩膜投影精筛”两步提取。"
        "系统先以融合包围框在 XY 空间索引中查询候选点，再仅保留落入来源掩膜投影范围的点。"
        "该处理避免逐石扫描全场点云，同时使二维岩块轮廓与三维点集在同一空间参考下对应。"
        "该步骤对应图 2-2 中的点云几何筛查环节。"
    ),
    "通过三维筛查的对象进入体积计算。本文采用 GroundDEM 参考的 2.5D 表示，在岩块水平范围内建立局部网格，并以可见表面相对于 GroundDEM 的正高度进行积分。对于网格单元 i，有效高度和 2.5D 体积写为：": (
        "通过三维筛查的对象进入体积计算。本文采用 GroundDEM 参考的 2.5D 表示，在岩块水平范围内建立局部网格，并以可见表面相对于 GroundDEM 的正高度进行积分。"
        "该步骤对应图 2-2 中的 2.5D 体积估计环节。"
        "对于网格单元 i，有效高度和 2.5D 体积写为："
    ),
}

DELETE_TEXTS = {
    "[此处插入图 2-3]",
    "图 2-3 边缘密度引导的四叉树切片与实例掩膜二维量测",
    "图 2-3 DOM-点云协同测量中的关键处理步骤示例：（a）边缘密度引导的自适应切片；（b）实例分割掩膜与二维量测；（c）重叠切片中的重复检测融合；（d）融合对象的点云几何筛查；（e）GroundDEM 参考的 2.5D 体积估算与二维代理体积对照",
    "[此处插入图 4]",
    "图 2-4 重叠切片中重复检测的世界坐标关联与融合",
    "[此处插入图 5]",
    "图 2-5 GroundDEM 参考的 2.5D 网格积分与二维代理体积对照",
}


def set_font(paragraph) -> None:
    for run in paragraph.runs:
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        run.font.size = Pt(10.5)


def delete_paragraph(paragraph) -> None:
    element = paragraph._element
    element.getparent().remove(element)
    paragraph._p = paragraph._element = None


def main() -> None:
    doc = Document(DOCX_PATH)
    changed = []
    to_delete = []
    for idx, paragraph in enumerate(doc.paragraphs):
        text = paragraph.text.strip()
        if text in REPLACEMENTS:
            paragraph.text = REPLACEMENTS[text]
            if paragraph.text.startswith("图 "):
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_font(paragraph)
            changed.append(idx)
        elif text in DELETE_TEXTS:
            to_delete.append(paragraph)
    for paragraph in to_delete:
        delete_paragraph(paragraph)
    doc.save(DOCX_PATH)
    print(DOCX_PATH)
    print("changed", changed)
    print("deleted", len(to_delete))


if __name__ == "__main__":
    main()

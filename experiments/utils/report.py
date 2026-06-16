"""
实验对比报告生成 — 输出 HTML 对比页面

Usage:
    from experiments.utils.report import build_comparison_html
    build_comparison_html(experiments=[...], output_path="comparison.html")
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import json


def build_comparison_html(
    experiments: list[dict[str, Any]],
    output_path: str | Path,
    title: str = "Slicing Method Comparison",
) -> Path:
    """生成单模块多方法的对比 HTML 报告

    experiments: [{
        "method": "sahi_baseline",
        "stats": {...},           # 来自 metrics 的统计 dict
        "overlay_img": "path/to/overlay.png",  # 相对路径
    }, ...]
    """
    rows = ""
    for exp in experiments:
        s = exp["stats"]
        overlay = exp.get("overlay_img", "")
        config = json.dumps(s.get("config", {}), indent=2, ensure_ascii=False)
        rows += f"""
        <div class="exp-card">
            <h2>{exp['method']}</h2>
            <div class="meta">
                <div class="stats">
                    <table>
                        {_stat_row("总切片 / 有效", f"{s.get('total_patches', s.get('total_tiles', '?'))} / {s.get('kept_patches', s.get('kept_tiles', '?'))}")}
                        {_stat_row("跳过数", str(s.get('skipped_patches', s.get('skipped_tiles', '?'))))}
                        {_stat_row("覆盖率", f"{s.get('coverage_ratio', 0):.2%}")}
                        {_stat_row("耗时", f"{s.get('elapsed_seconds', 0):.2f}s")}
                    </table>
                </div>
                <div class="config"><pre>{config}</pre></div>
            </div>
            {f'<div class="overlay"><img src="{overlay}" /></div>' if overlay else ''}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #e94560; }}
.exp-card {{ background: #16213e; border-radius: 8px; padding: 16px; margin: 16px 0; }}
.exp-card h2 {{ color: #0f3460; background: #e94560; display: inline-block; padding: 4px 12px; border-radius: 4px; }}
.meta {{ display: flex; gap: 20px; margin: 12px 0; }}
.stats table {{ border-collapse: collapse; }}
.stats td {{ padding: 4px 12px; border-bottom: 1px solid #333; }}
.stats td:first-child {{ color: #888; }}
.config pre {{ background: #0f3460; padding: 8px; border-radius: 4px; font-size: 12px; max-height: 240px; overflow-y: auto; }}
.overlay img {{ max-width: 100%; border-radius: 4px; margin-top: 12px; }}
</style>
</head>
<body>
<h1>{title}</h1>
{rows}
</body>
</html>"""
    Path(output_path).write_text(html, encoding="utf-8")
    return Path(output_path)


def _stat_row(label: str, value: str) -> str:
    return f"<tr><td>{label}</td><td><strong>{value}</strong></td></tr>"

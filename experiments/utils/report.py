"""
HTML report helpers for experiment comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def _stat_row(label: str, value: str) -> str:
    return f"<tr><td>{label}</td><td><strong>{value}</strong></td></tr>"


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def build_comparison_html(
    experiments: list[dict[str, Any]],
    output_path: str | Path,
    title: str = "Slicing Method Comparison",
) -> Path:
    cards = []
    for exp in experiments:
        stats = exp.get("stats", {})
        overlay = exp.get("overlay_img", "")
        config = json.dumps(stats.get("config", {}), indent=2, ensure_ascii=False)
        total_tiles = stats.get("total_tiles", stats.get("total_patches", "n/a"))
        kept_tiles = stats.get("kept_tiles", stats.get("kept_patches", "n/a"))
        skipped_tiles = stats.get("skipped_tiles", stats.get("skipped_patches", "n/a"))
        coverage = stats.get("coverage_ratio", 0.0)
        elapsed = stats.get("elapsed_seconds", 0.0)

        rows = "\n".join(
            [
                _stat_row("Total tiles", str(total_tiles)),
                _stat_row("Kept tiles", str(kept_tiles)),
                _stat_row("Skipped tiles", str(skipped_tiles)),
                _stat_row("Coverage ratio", f"{float(coverage):.2%}" if isinstance(coverage, (int, float)) else "n/a"),
                _stat_row("Elapsed time", f"{float(elapsed):.2f}s" if isinstance(elapsed, (int, float)) else "n/a"),
            ]
        )

        if "tile_area_m2" in stats:
            area_stats = stats["tile_area_m2"]
            rows += "\n" + _stat_row(
                "Tile area (m²)",
                f"{_fmt_float(area_stats.get('min'), 2)} / {_fmt_float(area_stats.get('mean'), 2)} / {_fmt_float(area_stats.get('max'), 2)}",
            )
        if "tile_size_px" in stats:
            rows += "\n" + _stat_row("Patch size (px)", str(stats.get("tile_size_px", "n/a")))

        overlay_html = f'<div class="overlay"><img src="{overlay}" alt="{exp["method"]} overlay"></div>' if overlay else ""
        cards.append(
            f"""
            <section class="exp-card">
                <h2>{exp['method']}</h2>
                <div class="meta">
                    <div class="stats">
                        <table>{rows}</table>
                    </div>
                    <div class="config"><pre>{config}</pre></div>
                </div>
                {overlay_html}
            </section>
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{
    font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
    margin: 24px;
    background: #f5f7fa;
    color: #1f2933;
}}
h1 {{
    margin-bottom: 20px;
}}
.exp-card {{
    background: #ffffff;
    border: 1px solid #d9e2ec;
    border-radius: 10px;
    padding: 18px;
    margin: 18px 0;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
}}
.exp-card h2 {{
    margin: 0 0 14px 0;
}}
.meta {{
    display: flex;
    gap: 20px;
    align-items: flex-start;
    flex-wrap: wrap;
}}
.stats table {{
    border-collapse: collapse;
    min-width: 280px;
}}
.stats td {{
    padding: 6px 12px;
    border-bottom: 1px solid #e5e7eb;
}}
.stats td:first-child {{
    color: #52606d;
    width: 150px;
}}
.config pre {{
    margin: 0;
    background: #f7fafc;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 12px;
    max-height: 260px;
    overflow-y: auto;
}}
.overlay img {{
    margin-top: 14px;
    max-width: 100%;
    border: 1px solid #d9e2ec;
    border-radius: 8px;
    background: #fff;
}}
</style>
</head>
<body>
<h1>{title}</h1>
{''.join(cards)}
</body>
</html>"""
    Path(output_path).write_text(html, encoding="utf-8")
    return Path(output_path)

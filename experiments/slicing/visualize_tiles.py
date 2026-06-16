"""
切片实验可视化比较 — 读取已运行结果, 生成多方法并排对比图 + HTML 报告

用法:
  cd D:\github_project\image_segment\DOM_Space_message_val

  # 1) 并排对比图 (PNG)
  python experiments/slicing/visualize_tiles.py --side-by-side

  # 2) HTML 报告
  python experiments/slicing/visualize_tiles.py --html

  # 3) 两者都生成
  python experiments/slicing/visualize_tiles.py --all
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.report import build_comparison_html

SELF_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SELF_DIR / "outputs"


def _load_manifest() -> dict:
    mp = OUTPUT_DIR / "results_manifest.json"
    if not mp.exists():
        raise FileNotFoundError(
            f"No results_manifest.json found. Run run_slicing_experiment.py first.\n"
            f"Expected: {mp}"
        )
    return json.loads(mp.read_text(encoding="utf-8"))


def generate_side_by_side(output_path: str | Path | None = None) -> Path:
    """把已运行的所有方法的 tile_overlay.png 拼成一张并排对比图"""
    manifest = _load_manifest()
    if not manifest:
        raise ValueError("No results in manifest. Run slicing experiments first.")

    overlay_paths = {}
    for key, val in manifest.items():
        op = val.get("overlay_img", "")
        if op and Path(op).exists():
            overlay_paths[key] = Path(op)
        else:
            print(f"  [WARN] overlay_img not found for: {key}")

    if not overlay_paths:
        raise ValueError("No overlay images found.")

    images = {}
    for method, p in overlay_paths.items():
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [WARN] Cannot read: {p}")
            continue
        images[method] = img

    if len(images) <= 1:
        print("Only one image available; skipping side-by-side (no comparison needed).")
        return Path("")

    # 统一高度 → 水平拼接
    h = min(img.shape[0] for img in images.values())
    resized = []
    labels = []
    for method, img in images.items():
        scale = h / img.shape[0]
        r = cv2.resize(img, (int(img.shape[1] * scale), h))
        # 加标题条
        bar = np.full((36, r.shape[1], 3), (30, 30, 42), dtype=np.uint8)
        cv2.putText(bar, f" {method}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        resized.append(np.vstack([bar, r]))
        labels.append(method)

    combined = np.hstack(resized)
    out = output_path or str(OUTPUT_DIR / "comparison_side_by_side.png")
    cv2.imwrite(str(out), combined)
    print(f"Side-by-side comparison saved: {out}")
    return Path(out)


def generate_html_report(output_path: str | Path | None = None) -> Path:
    """生成 HTML 对比报告"""
    manifest = _load_manifest()

    experiments = []
    for key, val in manifest.items():
        overlay = val.get("overlay_img", "")
        # 转为相对路径 (HTML 和图片都在 outputs/ 下)
        if overlay:
            overlay_name = Path(overlay).name
        else:
            overlay_name = ""
        experiments.append({
            "method": val["method"],
            "stats": val["stats"],
            "overlay_img": overlay_name,  # 同目录下的文件名
        })

    out = output_path or str(OUTPUT_DIR / "comparison_report.html")
    build_comparison_html(experiments, out, title="Slicing Method Comparison")
    print(f"HTML report saved: {out}")
    return Path(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize slicing experiment results")
    parser.add_argument("--side-by-side", action="store_true", help="Generate side-by-side PNG")
    parser.add_argument("--html", action="store_true", help="Generate HTML comparison report")
    parser.add_argument("--all", action="store_true", help="Generate both")
    args = parser.parse_args()

    if not (args.side_by_side or args.html or args.all):
        # 默认生成两者
        args.all = True

    if args.side_by_side or args.all:
        generate_side_by_side()

    if args.html or args.all:
        generate_html_report()


if __name__ == "__main__":
    main()

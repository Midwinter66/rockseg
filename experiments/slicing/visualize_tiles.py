"""
Build comparison artifacts for slicing outputs.

Examples:
  python experiments/slicing/visualize_tiles.py --side-by-side
  python experiments/slicing/visualize_tiles.py --html
  python experiments/slicing/visualize_tiles.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.report import build_comparison_html

SELF_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SELF_DIR / "outputs"


def _load_manifest() -> dict:
    manifest_path = OUTPUT_DIR / "results_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No results_manifest.json found. Run run_slicing_experiment.py first.\n"
            f"Expected: {manifest_path}"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _record_stats(record: dict) -> dict:
    return record.get("summary") or record.get("stats") or {}


def _record_overlay_path(record: dict) -> Path | None:
    candidates = [
        record.get("overlay_img_paper", ""),
        record.get("overlay_image_paper", ""),
        _record_stats(record).get("overlay_image_paper", ""),
        record.get("overlay_img", ""),
        record.get("overlay_image", ""),
        _record_stats(record).get("overlay_image", ""),
        record.get("overlay_img_audit", ""),
        record.get("overlay_image_audit", ""),
        _record_stats(record).get("overlay_image_audit", ""),
    ]
    for overlay in candidates:
        if not overlay:
            continue
        path = Path(overlay)
        if path.exists():
            return path
    return None


def generate_side_by_side(output_path: str | Path | None = None) -> Path:
    manifest = _load_manifest()
    if not manifest:
        raise ValueError("No results found in results_manifest.json.")

    images: dict[str, np.ndarray] = {}
    for method, record in manifest.items():
        overlay_path = _record_overlay_path(record)
        if overlay_path is None:
            print(f"  [WARN] overlay image not found for: {method}")
            continue
        image = cv2.imread(str(overlay_path))
        if image is None:
            print(f"  [WARN] failed to read image: {overlay_path}")
            continue
        images[method] = image

    if len(images) <= 1:
        print("Only one overlay image available; side-by-side comparison skipped.")
        return Path("")

    target_height = min(image.shape[0] for image in images.values())
    strips = []
    for method, image in images.items():
        scale = target_height / image.shape[0]
        resized = cv2.resize(image, (int(image.shape[1] * scale), target_height))
        bar = np.full((42, resized.shape[1], 3), (245, 247, 250), dtype=np.uint8)
        cv2.putText(
            bar,
            f" {method}",
            (10, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (35, 42, 52),
            2,
            cv2.LINE_AA,
        )
        strips.append(np.vstack([bar, resized]))

    combined = np.hstack(strips)
    out_path = Path(output_path) if output_path else OUTPUT_DIR / "comparison_side_by_side.png"
    cv2.imwrite(str(out_path), combined)
    print(f"Side-by-side comparison saved: {out_path}")
    return out_path


def generate_html_report(output_path: str | Path | None = None) -> Path:
    manifest = _load_manifest()

    experiments = []
    for method, record in manifest.items():
        overlay_path = _record_overlay_path(record)
        experiments.append(
            {
                "method": record.get("method", method),
                "stats": _record_stats(record),
                "overlay_img": overlay_path.name if overlay_path else "",
            }
        )

    out_path = Path(output_path) if output_path else OUTPUT_DIR / "comparison_report.html"
    build_comparison_html(experiments, out_path, title="Slicing Method Comparison")
    print(f"HTML report saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize slicing experiment results")
    parser.add_argument("--side-by-side", action="store_true", help="Generate side-by-side PNG")
    parser.add_argument("--html", action="store_true", help="Generate HTML comparison report")
    parser.add_argument("--all", action="store_true", help="Generate both")
    args = parser.parse_args()

    if not (args.side_by_side or args.html or args.all):
        args.all = True

    if args.side_by_side or args.all:
        generate_side_by_side()

    if args.html or args.all:
        generate_html_report()


if __name__ == "__main__":
    main()

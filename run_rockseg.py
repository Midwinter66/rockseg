"""Run the physical-scale-driven multi-scale rock detection pipeline.

Examples
--------
::

    # Full 3-scale run on DOM2 with GPU
    python run_rockseg.py --dom data/dom2/DOM.tif --model models/best.pt --output output/dom2

    # Coarse-only quick test
    python run_rockseg.py --dom data/dom2/DOM.tif --model models/best.pt --scales coarse

    # Limit tiles for testing
    python run_rockseg.py --dom data/dom2/DOM.tif --model models/best.pt --max-tiles 50
"""

from __future__ import annotations

import argparse
import logging
import sys

from rockseg import create_default_config, ScaleLevel
from rockseg.pipeline import MultiScaleRockDetectionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physical-scale-driven multi-scale rock detection",
    )
    parser.add_argument(
        "--dom", required=True,
        help="Path to DOM GeoTIFF file",
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to YOLO11m-seg weights (.pt)",
    )
    parser.add_argument(
        "--output", default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--gsd", type=float, default=0.01,
        help="Ground sampling distance in m/pixel (default: 0.01)",
    )
    parser.add_argument(
        "--device", default="",
        help="Device for inference, e.g. 'cuda:0' or 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--scales", default="coarse,medium,fine",
        help="Comma-separated scale levels to use (default: coarse,medium,fine)",
    )
    parser.add_argument(
        "--max-tiles", type=int, default=0,
        help="Max tiles per scale for testing (0 = no limit, default: 0)",
    )
    parser.add_argument(
        "--cascade", action="store_true",
        help="Use cascade deduplication instead of cross-scale fusion",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    config = create_default_config(args.dom, args.model, args.output)
    config.gsd = args.gsd
    config.use_cascade = args.cascade

    # Filter scales based on --scales argument
    requested = [s.strip() for s in args.scales.split(",")]
    valid_levels = {ScaleLevel.COARSE, ScaleLevel.MEDIUM, ScaleLevel.FINE}
    selected_levels = set()
    for r in requested:
        try:
            selected_levels.add(ScaleLevel(r))
        except ValueError:
            print(f"Warning: unknown scale '{r}', skipping")
    config.scales = [s for s in config.scales if s.level in selected_levels]
    if not config.scales:
        print("Error: no valid scales selected")
        sys.exit(1)

    pipeline = MultiScaleRockDetectionPipeline(config)
    pipeline.max_tiles_per_scale = args.max_tiles

    # Auto-detect GPU if device not specified
    device = args.device
    if not device:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda:0"
                logging.info("Auto-detected GPU: %s", torch.cuda.get_device_name(0))
        except ImportError:
            pass
    if device:
        pipeline.predictor.device = device

    instances = pipeline.run()

    print(f"\nDone.  {len(instances)} unique rock instances detected.")
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    sys.exit(main())

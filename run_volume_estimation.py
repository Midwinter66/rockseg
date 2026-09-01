"""Run volume estimation on rock detection results.

Usage:
    python run_volume_estimation.py \
        --input output/dom2_cascade_v2_3d \
        --dom data/dom2/DOM.tif \
        --pointcloud "data/pointcloud2/Data/BlockB.laz,data/pointcloud2/Data/BlockY.laz" \
        --model research_v2/volume_validation/output/results/shape_aware_model.txt \
        --output output/dom2_cascade_v2_volume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio

from rockseg.volume import estimate_volumes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Volume estimation for rock detections")
    parser.add_argument("--input", required=True,
                        help="Input directory (rock_instances.json + rock_masks.npz)")
    parser.add_argument("--dom", required=True, help="DOM GeoTIFF path")
    parser.add_argument("--pointcloud", required=True, help="Comma-separated LAZ paths")
    parser.add_argument("--model", required=True, help="Path to LightGBM shape-aware model")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--linear-alpha", type=float, default=0.731,
                        help="Linear correction alpha (default: 0.731)")
    parser.add_argument("--grid-resolution", type=float, default=0.05,
                        help="Height map grid resolution in meters (default: 0.05)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try accepted_instances first, fall back to rock_instances
    inst_path = input_dir / "accepted_instances.json"
    if not inst_path.exists():
        inst_path = input_dir / "rock_instances.json"
    if not inst_path.exists():
        print(f"Error: no instances found in {input_dir}")
        sys.exit(1)

    print(f"Loading instances from {inst_path}...")
    with open(inst_path) as f:
        instances = json.load(f)
    print(f"  {len(instances)} instances")

    # Load masks from the original detection output (input dir may not have masks)
    # Try input dir first, then dom2_cascade_v2
    masks_path = input_dir / "rock_masks.npz"
    if not masks_path.exists():
        masks_path = input_dir / "accepted_masks.npz"
    if not masks_path.exists():
        # Try cascade_v2 as source
        masks_path = Path("output/dom2_cascade_v2/rock_masks.npz")
    if not masks_path.exists():
        print(f"Error: no masks found")
        sys.exit(1)

    print(f"Loading masks from {masks_path}...")
    masks_data = np.load(masks_path, allow_pickle=True)
    masks = []
    missing_count = 0
    for inst in instances:
        key = f"{inst['instance_id']}_mask"
        if key in masks_data.files:
            masks.append(masks_data[key])
        else:
            if inst['instance_id'] in masks_data.files:
                masks.append(masks_data[inst['instance_id']])
            else:
                x1, y1, x2, y2 = inst["bbox"]
                masks.append(np.zeros((y2 - y1, x2 - x1), dtype=bool))
                missing_count += 1
    if missing_count > 0:
        print(f"  WARNING: {missing_count} masks not found, using empty masks (volume will be 0)")
    print(f"  {len(masks)} masks")

    # Get geotransform
    print(f"Reading DOM geotransform...")
    with rasterio.open(args.dom) as src:
        transform = src.transform

    laz_paths = [p.strip() for p in args.pointcloud.split(",")]

    # Run volume estimation
    instances_with_vol, summary = estimate_volumes(
        instances, masks, laz_paths, transform,
        model_path=args.model,
        linear_alpha=args.linear_alpha,
        grid_resolution_m=args.grid_resolution,
    )

    # Save
    print(f"\nSaving results to {output_dir}...")
    with open(output_dir / "stone_volumes.json", "w") as f:
        json.dump(instances_with_vol, f)
    with open(output_dir / "volume_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Volume Summary ===")
    print(f"  Instances:           {summary['total_instances']}")
    print(f"  With volume data:    {summary['instances_with_volume']}")
    print(f"  Total volume (shape-aware): {summary['total_volume_shape_aware_m3']:.2f} m³")
    print(f"  Total volume (2.5D):       {summary['total_volume_2_5d_m3']:.2f} m³")
    print(f"  Mean correction ratio:     {summary['mean_correction_ratio']:.4f}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

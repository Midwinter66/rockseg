"""Run 3D point cloud validation on rock detection results.

Usage:
    python run_3d_validation.py \
        --input output/dom2_cascade_v2 \
        --dom data/dom2/DOM.tif \
        --pointcloud "data/pointcloud2/Data/BlockB.laz,data/pointcloud2/Data/BlockY.laz" \
        --output output/dom2_cascade_v2_3d
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio

from rockseg.validation_3d import Validation3DConfig, run_3d_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D point cloud validation for rock detections")
    parser.add_argument("--input", required=True, help="Input detection directory (rock_instances.json + rock_masks.npz)")
    parser.add_argument("--dom", required=True, help="DOM GeoTIFF path (for geotransform)")
    parser.add_argument("--pointcloud", required=True, help="Comma-separated LAZ file paths")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--min-points", type=int, default=60, help="Min point count (default: 60)")
    parser.add_argument("--min-z-range", type=float, default=0.18, help="Min Z range in meters (default: 0.18)")
    parser.add_argument("--min-p90-height", type=float, default=0.12, help="Min P90 height in meters (default: 0.12)")
    parser.add_argument("--min-elevated-ratio", type=float, default=0.2, help="Min elevated ratio (default: 0.2)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load instances
    instances_path = input_dir / "rock_instances.json"
    masks_path = input_dir / "rock_masks.npz"
    if not instances_path.exists():
        print(f"Error: {instances_path} not found")
        sys.exit(1)
    if not masks_path.exists():
        print(f"Error: {masks_path} not found")
        sys.exit(1)

    print(f"Loading instances from {input_dir}...")
    with open(instances_path) as f:
        instances = json.load(f)
    print(f"  {len(instances)} instances")

    print("Loading masks...")
    masks_data = np.load(masks_path, allow_pickle=True)
    # Masks are stored as "rock_XXXXX_mask" keys, matching instance_id in JSON
    masks = []
    for inst in instances:
        key = f"{inst['instance_id']}_mask"
        masks.append(masks_data[key])
    print(f"  {len(masks)} masks")

    # Get geotransform from DOM
    print(f"Reading DOM geotransform from {args.dom}...")
    with rasterio.open(args.dom) as src:
        transform = src.transform
        print(f"  Transform: {transform}")

    # Point cloud paths
    laz_paths = [p.strip() for p in args.pointcloud.split(",")]
    print(f"Point cloud files: {len(laz_paths)}")
    for p in laz_paths:
        print(f"  {p}")

    # Config
    config = Validation3DConfig(
        min_points=args.min_points,
        min_z_range_m=args.min_z_range,
        min_p90_height_m=args.min_p90_height,
        min_elevated_ratio=args.min_elevated_ratio,
    )

    # Run validation
    accepted, rejected, summary = run_3d_validation(
        instances, masks, laz_paths, transform, config
    )

    # Save results
    print(f"\nSaving results to {output_dir}...")

    with open(output_dir / "accepted_instances.json", "w") as f:
        json.dump(accepted, f)
    with open(output_dir / "rejected_instances.json", "w") as f:
        json.dump(rejected, f)
    with open(output_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save accepted masks
    accepted_rock_ids = {a["rock_id"] for a in accepted}
    accepted_mask_dict = {}
    for i, inst in enumerate(instances):
        if inst["rock_id"] in accepted_rock_ids:
            accepted_mask_dict[f"{inst['instance_id']}_mask"] = masks[i]

    if accepted_mask_dict:
        np.savez_compressed(
            output_dir / "accepted_masks.npz",
            **accepted_mask_dict,
        )

    print(f"\n=== 3D Validation Summary ===")
    print(f"  Total instances:  {summary['total_instances']}")
    print(f"  Accepted:         {summary['accepted']}")
    print(f"  Rejected:         {summary['rejected']}")
    print(f"  Acceptance rate:  {summary['acceptance_rate']*100:.1f}%")
    print(f"  Rejection reasons:")
    for reason, count in summary["rejection_reasons"].items():
        print(f"    {reason}: {count}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

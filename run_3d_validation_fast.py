"""Run fast 3D point cloud validation on rock detection results.

Usage:
    python run_3d_validation_fast.py \
        --input output/dom2_cascade_v2 \
        --dom data/dom2/DOM.tif \
        --pointcloud "data/pointcloud2/Data/BlockB.laz,data/pointcloud2/Data/BlockY.laz" \
        --output output/dom2_cascade_v2_3d
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

from rockseg.validation_3d_fast import Validation3DConfig, run_3d_validation_fast


def parse_args():
    p = argparse.ArgumentParser(description="Fast 3D point cloud validation")
    p.add_argument("--input", required=True)
    p.add_argument("--dom", required=True)
    p.add_argument("--pointcloud", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-points", type=int, default=60)
    p.add_argument("--min-z-range", type=float, default=0.18)
    p.add_argument("--min-p90-height", type=float, default=0.12)
    p.add_argument("--min-elevated-ratio", type=float, default=0.2)
    return p.parse_args()


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args = parse_args()
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    inst_path = in_dir / "rock_instances.json"
    if not inst_path.exists():
        print(f"Error: {inst_path} not found")
        sys.exit(1)

    print(f"Loading {inst_path}...")
    with open(inst_path) as f:
        instances = json.load(f)
    print(f"  {len(instances)} instances")

    print(f"Reading DOM geotransform...")
    with rasterio.open(args.dom) as src:
        transform = src.transform

    laz_paths = [p.strip() for p in args.pointcloud.split(",")]
    config = Validation3DConfig(
        min_points=args.min_points,
        min_z_range_m=args.min_z_range,
        min_p90_height_m=args.min_p90_height,
        min_elevated_ratio=args.min_elevated_ratio,
    )

    accepted, rejected, summary = run_3d_validation_fast(
        instances, laz_paths, transform, config
    )

    print(f"\nSaving to {out_dir}...")
    with open(out_dir / "accepted_instances.json", "w") as f:
        json.dump(accepted, f)
    with open(out_dir / "rejected_instances.json", "w") as f:
        json.dump(rejected, f)
    with open(out_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== 3D Validation ===")
    print(f"  Total:      {summary['total']}")
    print(f"  Accepted:   {summary['accepted']}")
    print(f"  Rejected:   {summary['rejected']}")
    print(f"  Rate:       {summary['rate']*100:.1f}%")
    print(f"  Reasons:")
    for r, c in summary["reasons"].items():
        print(f"    {r}: {c}")
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()

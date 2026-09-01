"""Compute reference (true) volumes for all rock fragments.

Loads each OBJ mesh and computes the exact volume via the divergence theorem.
No 2.5D simulation — this is ground truth from the watertight 3D mesh.

Usage::

    python -m research_v2.volume_validation.compute_true_volumes
    python -m research_v2.volume_validation.compute_true_volumes --data-dir data/experience_rock
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    _project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_project_root))
    from research_v2.volume_validation.config import Config
    from research_v2.volume_validation.mesh_utils import find_obj_files, get_mesh_info
else:
    from .config import Config
    from .mesh_utils import find_obj_files, get_mesh_info

logger = logging.getLogger("compute_volumes")


def main():
    parser = argparse.ArgumentParser(
        description="Compute true volumes for all rock fragments from OBJ meshes."
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing T01/ subfolder with OBJ files.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = Config()
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    config.ensure_output_dir()

    logger.info("=" * 70)
    logger.info("Computing True Volumes from OBJ Meshes")
    logger.info("=" * 70)
    logger.info("Data dir: %s", config.data_dir)

    # Discover OBJ files
    file_list = find_obj_files(config.data_dir, config.groups)
    logger.info("Found %d OBJ files.", len(file_list))

    if len(file_list) == 0:
        logger.error("No OBJ files found. Check data directory: %s", config.data_dir)
        return 1

    # Process each mesh
    results = []
    t0 = time.time()

    try:
        from tqdm import tqdm
        iterator = tqdm(file_list, desc="Computing volumes", unit="rock")
    except ImportError:
        iterator = file_list

    for sample_id, group, obj_path in iterator:
        try:
            info = get_mesh_info(obj_path, sample_id, group)
            results.append({
                "sample_id": sample_id,
                "group": group,
                "filename": obj_path.name,
                "V_true_mm3": info.volume_mm3,
                "V_true_cm3": info.volume_mm3 / 1000.0,
                "L_mm": info.bbox_dims_mm[0],
                "W_mm": info.bbox_dims_mm[1],
                "H_mm": info.bbox_dims_mm[2],
                "surface_area_mm2": info.surface_area_mm2,
                "n_vertices": info.n_vertices,
                "n_faces": info.n_faces,
                "is_watertight": info.is_watertight,
            })
        except Exception as e:
            logger.error("[%s] Failed: %s", sample_id, e)

    t1 = time.time()
    logger.info("Processed %d / %d meshes in %.1f s.",
                len(results), len(file_list), t1 - t0)

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = config.results_path / "true_volumes.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved to %s", csv_path)

    # Print summary
    print("\n" + "=" * 70)
    print("True Volume Summary")
    print("=" * 70)
    print(f"Total samples:  {len(df)}")
    print(f"Watertight:     {df['is_watertight'].sum()} / {len(df)}")

    for g in df["group"].unique():
        sub = df[df["group"] == g]
        print(f"\n--- Group {g} ({len(sub)} samples) ---")
        print(f"  Volume (mm³):  mean={sub['V_true_mm3'].mean():.1f}  "
              f"min={sub['V_true_mm3'].min():.1f}  "
              f"max={sub['V_true_mm3'].max():.1f}  "
              f"std={sub['V_true_mm3'].std():.1f}")
        print(f"  Volume (cm³):  mean={sub['V_true_cm3'].mean():.3f}  "
              f"min={sub['V_true_cm3'].min():.3f}  "
              f"max={sub['V_true_cm3'].max():.3f}")
        print(f"  L (mm):        mean={sub['L_mm'].mean():.2f}  "
              f"range=[{sub['L_mm'].min():.2f}, {sub['L_mm'].max():.2f}]")
        print(f"  W (mm):        mean={sub['W_mm'].mean():.2f}  "
              f"range=[{sub['W_mm'].min():.2f}, {sub['W_mm'].max():.2f}]")
        print(f"  H (mm):        mean={sub['H_mm'].mean():.2f}  "
              f"range=[{sub['H_mm'].min():.2f}, {sub['H_mm'].max():.2f}]")

    # Overall stats
    print(f"\n--- Overall ({len(df)} samples) ---")
    print(f"  Volume (mm³):  mean={df['V_true_mm3'].mean():.1f}  "
          f"min={df['V_true_mm3'].min():.1f}  "
          f"max={df['V_true_mm3'].max():.1f}")
    print(f"  Equivalent diameter (mm):  "
          f"mean={(6*df['V_true_mm3'].mean()/np.pi)**(1/3):.2f}  "
          f"range=[{(6*df['V_true_mm3'].min()/np.pi)**(1/3):.2f}, "
          f"{(6*df['V_true_mm3'].max()/np.pi)**(1/3):.2f}]")

    print(f"\nResults saved to: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

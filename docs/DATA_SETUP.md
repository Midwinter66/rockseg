# Data Setup and Computer Migration

## Data Origin

The current DOM and point cloud are two products derived from the same photogrammetric OSGB reconstruction. The point cloud is not LiDAR data.

The active code uses only this scene pairing:

```text
data/dom2/DOM.tif
data/dom2/DOM.tfw
data/dom2/DOM.prj
data/pointcloud2/Data/BlockB.laz
data/pointcloud2/Data/BlockY.laz
```

`dom3` and `pointcloud3` are different export products and are not part of the current main experiment. Do not substitute one of them into the active scene without first checking spatial alignment and rerunning every downstream stage.

## Spatial Reference

- DOM resolution: 0.01 m/pixel.
- Coordinate reference system: EPSG:4536.
- Coordinate mapping mode: absolute world coordinates.
- Configured XY shift: `(0.0, 0.0)`.
- The two LAZ blocks are loaded in the same coordinate system and treated as one scene.

The authoritative scene definition is `experiments/common/scene_reference.py`.

## Why Data Are Not on GitHub

The raw scene is several gigabytes. Individual files exceed GitHub's normal 100 MB file limit, including the DOM and both LAZ blocks. They must be transferred through an external drive, private object storage, institutional storage, or another large-file service.

GitHub contains code, configuration, documentation, the selected model checkpoint and lightweight result summaries only.

## Moving to a Low-Performance Computer

For a computer intended only for code and result inspection:

1. Clone the GitHub repository.
2. Open `README.md`, `docs/PROJECT_INVENTORY.md`, and `docs/results/current_results.md`.
3. Inspect manuscript tables under `docs/results/tables/`.
4. Raw data and the Python environment are not required unless the pipeline will be executed.

For a computer that will run the pipeline:

1. Clone the repository.
2. Copy the raw files into the exact directory structure shown above.
3. Create the Conda environment with `conda env create -f environment.yml`.
4. Confirm the full scene opens correctly with:

```powershell
python experiments/visualization/view_full_pc.py
```

5. Run the pipeline in the order listed in the root README.

## Pre-Run Checks

Before running a new experiment, verify:

- `DOM.tif`, `DOM.tfw`, and `DOM.prj` belong to the same export.
- Both LAZ files exist and use the same absolute coordinate frame as the DOM.
- The point cloud does not show periodic striping or slicing artifacts in the full-scene viewer.
- `experiments/common/scene_reference.py` points to the intended data pair.
- Old outputs are archived or removed before changing the scene or minimum-diameter setting.

## Result Portability

Full output JSON files may contain absolute paths to the original workstation. They are local audit artifacts and are not suitable as portable repository records. Run:

```powershell
python experiments/reports/generate_measurement_manuscript_assets.py
```

to regenerate sanitized summaries under `docs/results/`.

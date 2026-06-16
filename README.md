# RockSeg — Aerial Rock Detection, Segmentation & Volume Estimation Pipeline

An end-to-end experimental framework for detecting, segmenting, and estimating the volume of rocks from aerial orthophotos (DOM) and LiDAR point clouds.

## Pipeline Overview

```
DOM + LiDAR  ──→  Slicing  ──→  YOLO Detection  ──→  Fusion  ──→  3D Extraction  ──→  Volume & Charts
```

Each stage offers multiple methods for side-by-side comparison:

| Stage | Methods | Description |
|-------|---------|-------------|
| **① Slicing** | SAHI baseline / SAHI dense / Quadtree-DOM / Quadtree-PointCloud | Tiling strategies for large DOM images: regular grid vs. adaptive quadtree |
| **② Detection** | YOLOv8-seg | Instance segmentation on each tile |
| **③ Fusion** | Heuristic / Correlation Clustering | Merge overlapping/cross-tile detections into coherent stone instances |
| **④ 3D Extraction** | Point-in-polygon on LiDAR | Project 2D masks onto LiDAR point cloud to extract 3D stone points |
| **⑤ Volume** | grid_2.5D / convex_hull / alpha_shape | Compute rock volume from extracted point cloud |

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- **Data** (not included in this repo):
  - DOM orthophoto (geoTIFF + TFW)
  - LiDAR point cloud (PLY)

### Installation

```bash
# Option A: pip
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate rock
```

### Prepare Data

Place your data in the following structure:

```
data/
├── dom3/
│   ├── DOM.tif          # Orthophoto
│   └── DOM.tfw          # World file
└── pointcloud2/
    ├── BlockB.ply        # LiDAR point cloud
    └── BlockY.ply
```

### Run the Full Pipeline

```bash
# End-to-end: slice → detect → fuse → 3D extract → volume → charts
python experiments/full_pipeline/run_full_pipeline.py

# Skip completed stages (re-use previous results)
python experiments/full_pipeline/run_full_pipeline.py --skip-slicing --skip-detection

# Use a specific volume method
python experiments/full_pipeline/run_full_pipeline.py --run-id run_20260615_183324 --skip-slicing --skip-detection --volume-method grid_2d5
```

### View Results

```bash
# List all stones by volume
python experiments/full_pipeline/view_stone.py --run-id <run_id> --list

# View a stone in 3D
python experiments/full_pipeline/view_stone.py --run-id <run_id> --stone-id stone_000298

# View top 10 stones together
python experiments/full_pipeline/view_stone.py --run-id <run_id> --top 10

# Show stone location on DOM
python experiments/full_pipeline/view_stone.py --run-id <run_id> --stone-id stone_000276 --dom-view
```

## Comparative Experiments

Run each experimental stage independently to compare methods:

```bash
# Slicing comparison (all 4 methods)
python experiments/slicing/run_slicing_experiment.py

# Detection on sliced tiles
python experiments/detection/run_detection_experiment.py

# Fusion comparison (heuristic vs correlation clustering)
python experiments/fusion/run_fusion_experiment.py
```

## Interactive Web Tuners

Tune slicing and fusion parameters in real-time:

```bash
# Slicing parameter tuning
python experiments/slicing/web_tuner.py

# Fusion parameter tuning
python experiments/fusion/web_tuner.py
```

## Config-Driven Design

All parameters are centralized in JSON configs:

```
experiments/configs/
├── slicing/
│   ├── sahi_baseline.json
│   ├── sahi_dense.json
│   ├── quadtree_dom.json
│   └── quadtree_pointcloud.json
├── detection/
│   └── default.json
├── fusion/
│   ├── heuristic.json
│   └── correlation_clustering.json
└── pipeline_config.json       # Full pipeline defaults
```

## Project Structure

```
├── experiments/
│   ├── configs/               # Parameter configs
│   ├── slicing/               # SAHI & Quadtree tiling
│   ├── detection/             # YOLO inference
│   ├── fusion/                # Detection merging
│   ├── full_pipeline/         # End-to-end pipeline
│   └── utils/                 # Shared helpers
├── models/
│   └── best.pt                # YOLOv8-seg model
├── requirements.txt
├── environment.yml
└── README.md
```

## License

[Your license here]

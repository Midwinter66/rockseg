# Volume Validation (E5)

> **Experiment E5** of the RockSeg V2 research plan.
>
> Validates the 2.5D-to-volume estimation module using 868 external 3D rock
> fragments with watertight meshes (Čapek et al. 2025, A&A 696, A40).

## What This Code Does

```
3D mesh (ground truth)  ──►  simulated 2.5D surface  ──►  shape descriptors
                                                                    │
                                                                    ▼
                                           volume estimation (ablation)
                                           ┌──────────────────────────┐
                                           │ 1. Bounding box  V=LWH   │
                                           │ 2. Ellipsoid  V=π/6·LWH  │
                                           │ 3. 2.5D integration      │
                                           │ 4. Shape-aware (LightGBM)│
                                           └──────────┬───────────────┘
                                                      ▼
                                           compare vs reference volume
```

This is **not** an end-to-end pipeline validation. It validates only the
isolated `2.5D → volume` module, as specified in the V2 master plan
(``00_master_plan.md`` §4) and the leakage controls
(``03_dataset_and_annotation_plan.md`` §6).

## Dataset

| Group | Samples | Material | Fragmentation | Size Range |
|-------|---------|----------|---------------|------------|
| L01 | 386 | L3-6 chondrite (NWA 869) | Hypervelocity impact | 5–24 mm |
| L02 | 403 | L3-6 chondrite (NWA 869) | Explosive charge | 5–24 mm |
| T01 | 79 | Tephriphonolite (terrestrial) | Explosive charge | 5–24 mm |

**Download**: Extract `L01.zip`, `L02.zip`, `T01.zip` into a single directory:

```
data/capek_868/
├── L01/
│   ├── shape_0001.obj
│   ├── shape_0002.obj
│   └── ... (386 files)
├── L02/
│   ├── shape_0001.obj
│   └── ... (403 files)
├── T01/
│   ├── shape_0001.obj
│   └── ... (79 files)
└── shapeList.txt   (optional metadata)
```

**Why T01 matters most**: T01 is the only group with terrestrial rock +
explosive fragmentation, making it the closest analogue to blast rocks in
the field. The group-aware split ensures T01 samples appear in train, val,
and test.

## Installation

```bash
pip install -r research_v2/volume_validation/requirements.txt
```

Required packages beyond the project base:

- `trimesh` — mesh I/O, volume computation, ray casting
- `lightgbm` — shape-aware regression model
- `scikit-learn` — utilities
- `pandas` — result export
- `scipy` — convex hull, morphology

## Usage

### Quick start

```bash
# From the project root
python -m research_v2.volume_validation.run_validation \
    --data-dir data/capek_868 \
    --output-dir research_v2/volume_validation/output
```

### With noise / sparsity simulation

```bash
python -m research_v2.volume_validation.run_validation \
    --data-dir data/capek_868 \
    --noise-std 0.05 \
    --sparsity 0.15
```

### Full options

```
--data-dir        Path to data directory with L01/, L02/, T01/
--output-dir      Output directory (default: research_v2/volume_validation/output)
--grid-resolution Grid cell size in mm (default: 0.5)
--noise-std        Height noise std in mm (default: 0.0)
--sparsity         Fraction of cells removed (default: 0.0)
--seed             Random seed (default: 42)
--no-cache         Disable mesh processing cache
--log-level        DEBUG / INFO / WARNING / ERROR
```

## Output

```
output/
├── results/
│   ├── all_samples.csv          # All 868 samples with descriptors & volumes
│   ├── test_predictions.csv     # Test-set predictions for all 4 methods
│   ├── metrics_summary.json     # Full metrics, acceptance, feature importance
│   └── shape_aware_model.txt     # Saved LightGBM model
├── figures/
│   ├── scatter_bounding_box.png       # Pred vs True scatter (each method)
│   ├── scatter_ellipsoid.png
│   ├── scatter_2.5d_integration.png
│   ├── scatter_shape-aware.png
│   ├── error_boxplot.png             # Per-sample error boxplot
│   ├── ablation_bars.png             # MAPE/SMAPE/R² comparison bars
│   ├── per_group_*.png               # Error by source group
│   └── feature_importance.png        # LightGBM feature importance
└── cache/
    └── processed_data.npz           # Cached mesh + 2.5D data
```

## Method Details

### 1. Stable Orientation

Each mesh is oriented to its most stable resting pose by testing all convex
hull faces as candidate bottoms. The face that maximises
`support_area / com_height` is selected. This mimics how a rock fragment
settles on a flat surface before drone observation.

### 2. 2.5D Surface Simulation

The visible top surface is captured by downward ray casting on a regular XY
grid:

```
For each grid cell (x, y):
    Cast ray from (x, y, z_max) downward (-Z)
    Record the highest mesh intersection as z_top(x, y)
    Ground reference: z_ground = 0 (bottom of the oriented mesh)
    Height: h(x, y) = z_top - z_ground
```

Optional noise and sparsity simulate real-world observation degradation:
- `--noise-std`: Gaussian noise on height measurements
- `--sparsity`: Random cell removal (simulating sparse point clouds)

### 3. Shape Descriptors (13 features)

| Feature | Description | Formula |
|---------|-------------|---------|
| L, W | Footprint dimensions | Bounding box extents (L ≥ W) |
| H | Max height | max(z_top) |
| A | Footprint area | n_valid × Δ² |
| P | Footprint perimeter | Boundary cell count × Δ |
| C | Circularity | 4πA / P² |
| AR | Aspect ratio | L / W |
| H_mean, H_max, H_std | Height statistics | Over valid cells |
| V_box | Box volume | L × W × H |
| V_ellipsoid | Ellipsoid volume | (π/6) × L × W × H |
| V_2.5d | 2.5D integration | Σ max(z_top − z_ground, 0) × Δ² |

### 4. Volume Ablation

| Method | Formula | Role |
|--------|---------|------|
| Bounding box | V = L × W × H | Upper bound (overestimates) |
| Ellipsoid | V = (π/6) × L × W × H | Geometric approximation |
| 2.5D Integration | V = Σ h × Δ² | Deterministic method |
| Shape-aware | LightGBM(13 features) → V | Proposed calibration |

**Acceptance**: E_shape < E_box AND E_shape < E_ellipsoid on held-out test data.

### 5. Data Split

Group-aware stratified split (70/15/15) ensures every source group (L01, L02,
T01) is proportionally represented in train, validation, and test. No sample
appears in more than one split.

### 6. Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error (mm³) |
| RMSE | Root Mean Squared Error (mm³) |
| MAPE | Mean Absolute Percentage Error (%) |
| SMAPE | Symmetric MAPE (%) |
| Median RE | Median Relative Error (%) |
| R² | Coefficient of determination |

## Code Structure

```
volume_validation/
├── __init__.py              # Package init
├── config.py                # Configuration dataclass
├── mesh_utils.py            # Mesh loading, stable orientation, reference volume
├── simulate_2_5d.py         # 2.5D surface simulation (ray casting)
├── shape_descriptors.py     # 13 shape features from 2.5D surface
├── volume_estimators.py     # Box, ellipsoid, 2.5D, shape-aware (LightGBM)
├── metrics.py               # MAE, RMSE, MAPE, SMAPE, R²
├── data_split.py            # Group-aware 70/15/15 stratified split
├── run_validation.py         # Main pipeline (CLI entry point)
├── visualize.py             # Scatter, boxplot, bar chart, importance
├── requirements.txt         # Additional dependencies
└── README.md                # This file
```

## Citation

```
Čapek, D. et al. (2025). 3D shape models of meteoroid and terrestrial rock
fragments. Astronomy & Astrophysics, 696, A40.
```

## Scale Transfer Note

The dataset rocks are 5–24 mm, while blast rocks in the field are 5 cm–3 m.
However, shape features (circularity, aspect ratio, height statistics) are
**scale-invariant** when normalised. This validation confirms the methodology;
absolute scale is handled by unit conversion in the full pipeline.

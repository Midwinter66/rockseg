# Current Main-Scene Results

> Scope: `dom2 + pointcloud2`, minimum equivalent diameter `0.5 m`, quadtree tiling, correlation-clustering fusion, and GroundDEM-based 2.5D volume.

## Dataset

- Data origin: OSGB-derived photogrammetric products (not LiDAR).
- DOM: 8783 x 21713 pixels at 0.01 m/pixel, EPSG:4536.
- Scene area: 19070.53 m2.
- Point-cloud points: 146,721,392.
- Coordinate mapping: absolute world coordinates with zero XY shift.

## Pipeline Counts

| Stage | Count |
|---|---:|
| Quadtree tiles generated | 456 |
| Quadtree tiles retained | 348 |
| Raw mask candidates | 62015 |
| Detections after 0.5 m diameter filter | 7349 |
| Fusion candidates | 6258 |
| 3D-accepted fused stones | 6071 |
| 3D-rejected candidates | 187 |
| Volume QC passed | 6067 |

## Main Measurements

| Metric | Value |
|---|---:|
| Accepted median equivalent diameter | 0.6528 m |
| Accepted mean equivalent diameter | 0.7265 m |
| 2.5D total volume | 1271.5422 m3 |
| 2.5D mean volume | 0.2096 m3 |
| 2.5D median volume | 0.1210 m3 |
| 2D proxy total volume | 1699.2841 m3 |
| Pearson correlation, 2D proxy vs. 2.5D | 0.7733 |
| Median 2D proxy / 2.5D ratio | 1.3565 |

## Interpretation Boundary

The current full-scene pipeline is operational, but the current scene does not yet have a complete manual detection ground truth or per-stone volume ground truth. The 3D acceptance ratio and volume QC pass ratio describe pipeline filtering and numerical validity; they are not precision, recall, or absolute volume accuracy.

Machine-readable values are available in `current_results.json`; manuscript tables are under `tables/`.

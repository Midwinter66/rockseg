# Real Mine-site Point-cloud Spatial Sampling Analysis

Spatial-window LAZ streaming and KD-tree nearest-neighbor analysis; no full cloud loaded, no O(N^2), no volume workflow or model operation.

## 1. Method

Each LAZ file was streamed in 2,000,000-point chunks. Ten fixed-seed spatial windows were selected per file. For each window, only points inside a 1.5 m buffered square were retained; nearest-neighbor queries were evaluated for core points in the central 1.0 m square. This preserves spatial locality without loading the full cloud or calculating all point pairs.

## 2. pointcloud2

Total points: 146,721,392. XYZ range: min [624181.4499003943, 4678356.4470694, 1740.7486543466637]; max [624270.4507863207, 4678570.417302258, 1751.4717265809156].

| File | Points | XY P50 (mm) | XY P90 (mm) | 3D P50 (mm) | 3D P90 (mm) |
|---|---:|---:|---:|---:|---:|
| BlockB.laz | 61,641,369 | 2.24 | 6.00 | 4.58 | 8.54 |
| BlockY.laz | 85,080,023 | 3.16 | 6.40 | 4.58 | 8.60 |

## 2. pointcloud3

Total points: 187,360,460. XYZ range: min [-76.63292761088657, -57.977019331416706, -90.28241560132598]; max [-3.9517982759502646, 169.30342394400523, -79.08325428549725].

| File | Points | XY P50 (mm) | XY P90 (mm) | 3D P50 (mm) | 3D P90 (mm) |
|---|---:|---:|---:|---:|---:|
| BlockB.laz | 97,652,910 | 3.00 | 6.32 | 4.58 | 8.54 |
| BlockY.laz | 89,707,550 | 2.24 | 6.32 | 4.58 | 8.54 |

## 3. DOM and Resolution Decision

The DOM scale is 10 mm/pixel. The point-cloud spacing statistics are reported separately by file because the data sources differ; no artificial average was computed.

**Recommended next Shape-Aware training resolution: 10 mm.** The four LAZ
files have XY P50 of 2.24-3.16 mm and XY P90 of 6.00-6.40 mm; their 3D P90
values are 8.54-8.60 mm. Therefore, 10 mm is above the observed local-spacing
tail and matches the known 10 mm/pixel DOM scale. A 5 mm grid would be below
the XY P90 spacing and more vulnerable to empty or single-point cells.

Before production use, inspect post-ground-removal top-surface cell occupancy
for a small number of real rock masks at 10 mm. If supported, construct and
validate a 10 mm resolution-matched model. Do not apply the existing 0.5 mm
model directly.

## 4. Limitations

- Spatial-window samples estimate raw acquisition density, not density on detected rock tops.
- P50/P90 are provided per LAZ file; pointcloud2 and pointcloud3 are not averaged.
- No Dataset B, model, production code, OBJ, rasterization, or mine-site volume computation was changed.

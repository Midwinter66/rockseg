# Real Mine-site 10 mm 2.5D Feasibility Benchmark

## Status

**BLOCKED: no reusable point-level Site B artifacts are available.** No 10 mm
surface statistics were produced, and no values were inferred from the prior
50 mm aggregate results.

## What Was Checked

The existing Site B volume checkpoint records, for each stone:

- stone ID and world bounding box;
- associated/cropped point count;
- 50 mm grid dimensions, occupancy totals, and height summary;
- 50 mm `V_2.5D`.

For example, `stone_000000` has 61,101 associated points and a successful
50 mm surface with 987 valid cells. This does not provide its point coordinates
or ground heights.

The frozen Site B volume configuration has `save_per_stone: false`. It does
not retain a per-rock point cloud, post-ground-removal point cloud, GroundDEM,
or reusable height map.

## Why 10 mm Cannot Be Calculated From Existing Outputs

A 50 mm cell contains many possible 10 mm height configurations. Its aggregate
height statistics and volume cannot determine 10 mm occupancy, point counts per
cell, height percentiles, skewness, or `V_2.5D`. Reconstructing such values
from the saved 50 mm result would be scientifically invalid.

## Required Inputs That Are Missing

- Associated XYZ points for each selected rock.
- Ground elevation for those points, or the GroundDEM used to obtain it.
- Reusable rock polygon/mask geometry at point level.
- Post-ground-removal rock points.

## Required Next Action

Run the **existing** 2D-3D association and GroundDEM workflow only for a small
saved subset of Site B stones, preserving point-level benchmark intermediates
long enough to measure 5 mm and 10 mm cell occupancy. Do not load the
Shape-Aware V2 model, do not modify `volume.py`, and do not run the full site.

Only after this subset produces valid 10 mm surfaces can the project decide
whether to construct a 10 mm resolution-matched T01+L01 training dataset.

# Measurement 3.1 writing reference notes

Purpose: support Section 3.1 "Study Area and Data" for the RockSeg manuscript. This file records usable literature roles, writing rules, and missing data fields. It is not a full literature review.

## Core writing rule for Section 3.1

3.1 should describe the study scene and data sources only. It should not explain YOLO, fusion, correlation clustering, or volume equations. Those belong to later method subsections.

For a Measurement-style methods paper, the data section should answer:

1. What physical scene is measured?
2. What data were used?
3. What spatial reference/resolution supports metric measurement?
4. How are image and point cloud data related?
5. What reference/ground-truth limitation exists?

## Project facts that can be stated

- Current main scene in code: `dom2_pointcloud2`.
- DOM file: `DOM.tif`.
- Point clouds: `BlockB.laz` and `BlockY.laz`.
- Pixel size from TFW: 0.01 m/pixel.
- Coordinate reference from PRJ: CGCS2000 / 3-degree Gauss-Kruger CM 81E, EPSG:4536.
- The two LAZ files are separate spatial blocks used jointly in one coordinate frame.
- Current workflow uses DOM for 2D detection and point clouds for 3D validation and 2.5D volume estimation.
- No complete stone-by-stone manual volume ground truth is currently available.

## Literature roles

1. Recognition and statistical method of blast muckpile fragmentation under complex stacking conditions in underground metal mines.
   - Role: same Measurement journal and rock fragmentation measurement context.
   - Use in manuscript: support the importance of fragmentation recognition under complex stacking and the need for measurement-oriented workflows.
   - Do not use it as direct evidence for DOM slicing or YOLO-based detection.

2. An intelligent measurement method for the particle distribution of open-pit rock piles with fuzzy boundaries.
   - Role: close application setting, open-pit rock pile particle distribution, fuzzy boundaries.
   - Use in manuscript: motivate image/measurement difficulty caused by fuzzy boundaries and irregular pile scenes.
   - Do not imply it uses the same DOM + point-cloud workflow unless verified from the paper.

3. Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection.
   - Link: https://arxiv.org/abs/2202.06934
   - Role: evidence for slicing-based inference on large images/small objects.
   - Use in manuscript: cite in later tiling section, not in 3.1 except as background in Related Work.

4. Segmentation for High-Resolution Optical Remote Sensing Imagery Using Improved Quadtree and Region Adjacency Graph Technique.
   - Link: https://www.mdpi.com/2072-4292/5/7/3259
   - Role: evidence that quadtree is a recognized spatial partitioning idea in high-resolution remote sensing.
   - Use in manuscript: cite in tiling strategy section, especially when explaining quadtree partitioning. It does not directly justify overlap unless we explain overlap as our engineering adaptation for boundary completeness.

## What 3.1 should not contain

- Local project paths such as `data/pointcloud2/Data/`.
- Unverified acquisition date, UAV model, flight height, camera model, or point density.
- Claims such as "high accuracy" or "complete ground truth" unless backed by results.
- Algorithm details: Canny thresholds, YOLO thresholds, clustering thresholds, volume formulas.
- Final results such as detected stone counts or QC pass counts.

## Missing fields to collect before final submission

- Mine/site name or anonymized location description.
- Data acquisition date or period.
- Data source: UAV photogrammetry, OSGB export, terrestrial/airborne lidar, or reconstructed point cloud.
- Point-cloud density or total point count for the exact scene.
- Whether `BlockB.laz` and `BlockY.laz` overlap, tile adjacent areas, or represent two mine zones.
- Whether DOM and point cloud came from the same OSGB reconstruction pipeline.
- Any manual labels available: number of manually checked stones, empty tiles, or independent survey references.

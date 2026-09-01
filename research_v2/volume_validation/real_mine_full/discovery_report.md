# Real Mine Volume Estimation Discovery

## 1. Input Data

This discovery covers the **entire DOM2 mine area**. It is a read-only inventory of existing detection and 2D-3D validation outputs. No OBJ, rasterization, feature extraction, or volume estimation was run.

| Input | Path | Role |
| --- | --- | --- |
| DOM2 | `data/dom2/DOM.tif` and `data/dom2/DOM.tfw` | DOM coverage and pixel reference |
| Point cloud | `data/pointcloud2/Data/BlockB.laz` | Existing DOM2 3D source |
| Point cloud | `data/pointcloud2/Data/BlockY.laz` | Existing DOM2 3D source |
| Final instances | `output/dom2_cascade_v2/rock_instances.json` | Deduplicated DOM2 instance inventory |
| Masks | `output/dom2_cascade_v2/rock_masks.npz` | One mask entry per final instance |
| Bounding boxes | `output/dom2_cascade_v2/rock_bboxes.npz` | One bbox entry per final instance |
| 3D association | `output/dom2_cascade_v2_3d_fixed/accepted_instances.json` and `rejected_instances.json` | Existing association/validation result |

Frozen downstream configuration recorded for the next phase:

- Shape-Aware V2 model: `research_v2/volume_validation/output_v2_scaled_10mm/shape_aware_model_v2_scaled_10mm.txt`
- Grid resolution: `0.01 m` (10 mm)
- Scale factor: `82.737840`
- Feature schema: canonical 12-feature schema, with `H_skew_norm` implemented as `H_skew`

## 2. Stone Counts

### Current DOM2 instance inventory

| Quantity | Count |
| --- | ---: |
| Raw detections, coarse | 37,470 |
| Raw detections, medium | 101,642 |
| Raw detections, fine | 179,286 |
| Same-scale fused detections | 112,983 |
| Final deduplicated instances / detection records | 76,407 |
| Accepted by existing 3D validation | 69,911 |
| Rejected by existing 3D validation | 6,496 |
| Accepted rate | 91.50% |

The raw, fused, and cascade counts above are the existing DOM2 pipeline summary supplied for this audit. The final `76,407` count was independently confirmed from `rock_instances.json`.

### Final instances by scale level

| Scale level | Final | Accepted | Rejected |
| --- | ---: | ---: | ---: |
| Coarse | 5,925 | 5,838 | 87 |
| Medium | 10,890 | 10,323 | 567 |
| Fine | 59,592 | 53,750 | 5,842 |
| **Total** | **76,407** | **69,911** | **6,496** |

### Existing 3D validation rejection reasons

Reasons are not mutually exclusive.

| Reason | Instances |
| --- | ---: |
| insufficient_p90_height | 5,207 |
| insufficient_elevated_ratio | 5,847 |
| insufficient_z_range | 357 |
| too_few_points | 184 |
| insufficient_ground_points | 3 |

## 3. Scene Statistics

The current DOM2 files do not contain an explicit `scene_id` field. They represent one DOM2 mine-area run, so the only defensible scene summary is:

| Scene/run | Final instances | Accepted | Rejected |
| --- | ---: | ---: | ---: |
| DOM2 full mine area | 76,407 | 69,911 | 6,496 |

The scale-level table in Section 2 is the available per-group breakdown. No additional scene split was inferred.

## 4. Stone Size Statistics

The final instance metadata contains mask area in DOM pixels and pixel bboxes. Using the DOM pixel size `0.01 m`, the area-equivalent footprint diameter was computed as:

`d_eq = 2 * sqrt(area_pixels * 0.01^2 / pi)`

This is a footprint-size proxy, not a recovered 3D rock diameter.

| Population | Min (m) | P10 (m) | P25 (m) | Median (m) | P75 (m) | P90 (m) | Max (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All final instances | 0.0226 | 0.0881 | 0.1106 | 0.1564 | 0.2475 | 0.4123 | 3.4496 |
| Accepted instances | 0.0226 | 0.0903 | 0.1134 | 0.1620 | 0.2571 | 0.4301 | 3.4496 |
| Rejected instances | 0.0319 | 0.0782 | 0.0930 | 0.1178 | 0.1616 | 0.2278 | 1.7748 |

For accepted instances, existing association point counts were:

| Statistic | Points |
| --- | ---: |
| Min | 61 |
| P10 | 17,706 |
| P25 | 20,774 |
| Median | 25,279 |
| P75 | 32,196 |
| P90 | 43,422 |
| Max | 484,533 |

These point counts are from the existing association validation metadata. They are not new point-cloud processing results.

## 5. Data Integrity

| Check | Result |
| --- | --- |
| Final instance IDs | 76,407; unique |
| Accepted IDs | 69,911; unique |
| Rejected IDs | 6,496; unique |
| Accepted/rejected overlap | 0 |
| Final IDs missing from accepted + rejected | 0 |
| IDs outside final inventory | 0 |
| Mask archive entries | 76,407; one matching `rock_######_mask.npy` entry per final ID |
| Bbox archive entries | 76,407; one matching bbox entry per final ID |
| Invalid/non-positive mask area metadata | 0 |
| Invalid bbox shape/order | 0 |
| Bbox outside DOM pixel bounds | 0 |
| Centroid outside DOM pixel bounds or non-finite | 0 |
| Accepted records with zero point count | 0 |
| Accepted records with non-positive z range | 0 |

Existing validation configuration was `min_points=60`, `min_z_range_m=0.18`, `min_p90_height_m=0.12`, and `min_elevated_ratio=0.2`. These are reported as provenance only; they were not changed.

### DOM and point-cloud coverage

The DOM image is `8783 x 21713` pixels with `0.01 m` pixel spacing. Its approximate world extent from the TIFF georeferencing tags is:

- X: `624181.3319` to `624269.1619 m`
- Y: `4678355.3795` to `4678572.5095 m`

The two point-cloud file header extents are:

| File | Point count in header | X range (m) | Y range (m) | Z range (m) |
| --- | ---: | --- | --- | --- |
| BlockB.laz | 61,641,369 | 624181.4499 - 624241.2469 | 4678465.2594 - 4678570.4173 | 1740.7487 - 1750.8898 |
| BlockY.laz | 85,080,023 | 624200.8485 - 624270.4508 | 4678356.4471 - 4678466.6102 | 1741.0209 - 1751.4717 |
| **Combined header total** | **146,721,392** | 624181.4499 - 624270.4508 | 4678356.4471 - 4678570.4173 | 1740.7487 - 1751.4717 |

The existing DOM2 validation metadata separately reports `187,360,460` scene points; this differs from the LAS 1.2 header 32-bit point counts and should be preserved as pipeline provenance rather than silently reconciled here.

## 6. Pilot Consistency

The previously recorded 12-stone pilot uses IDs such as `stone_006809` and is explicitly sourced from `data/dom3` / `pointcloud3` (Site B). DOM2 final instances use IDs `rock_00000` through `rock_76406` and are sourced from `data/dom2` / `pointcloud2`.

Therefore:

- DOM2 pilot IDs found in the current DOM2 inventory: **0 / 12**
- This is an expected dataset identity mismatch, not evidence that DOM2 instances are missing.
- The Site B pilot must not be presented as a DOM2 pilot or used to claim DOM2 full-mine validation.

## 7. Estimated Computational Cost

The formal DOM2 run would need to consider up to `69,911` accepted instances if the existing 3D validation gate is retained, or all `76,407` final instances if rejected records are also sent through a separate failure-reporting path. The current discovery did not perform 10 mm surface construction, canonical feature extraction, model prediction, or volume estimation.

The point-cloud sources contain at least `146,721,392` points by the two file headers. A full run should therefore reuse one process-level spatial index or equivalent existing association index and write per-instance resumable outputs. It should first run a separately approved batch pilot on DOM2 IDs, because the present discovery establishes inventory and association readiness but does not prove 10 mm surface validity for all accepted instances.

## Final Status

**DISCOVERY_WARNING**

The DOM2 whole-mine inventory is complete and internally closed: `76,407` final instances, `69,911` accepted associations, and `6,496` rejected associations, with no ID, mask, bbox, or coordinate-integrity defect found in the inspected metadata. The status remains `DISCOVERY_WARNING` because no full-mine 10 mm surface/feature/prediction run was performed, the prior pilot belongs to DOM3 rather than DOM2, and the existing DOM2 outputs are not Shape-Aware V2 volume results.

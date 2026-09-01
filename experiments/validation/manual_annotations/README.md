# Manual Detection Validation

This folder stores manual validation regions and object-level rock annotations.
The validation script compares manual boxes with fused/passed pipeline outputs
inside the manually annotated regions only.

## What to Validate

Use this validation for object-level rock detection reliability:

- True positive: one manual rock is matched by one predicted rock.
- False negative: one manual rock has no matched prediction.
- False positive: one prediction inside a manually annotated region has no matched manual rock.

This validation supports reporting Precision, Recall, and F1-score for detection
within annotated regions. It does not validate absolute 2.5D volume accuracy.

## Required Files

For each site, fill two CSV files:

- `site_a_validation_regions.csv` / `site_b_validation_regions.csv`
- `site_a_manual_boxes.csv` / `site_b_manual_boxes.csv`

Coordinates must use the same world-coordinate frame as the pipeline output.
For Site B, use the DOM/world coordinates after the saved coordinate transform,
not raw local point-cloud coordinates.

## Validation Regions

Each validation region is a manually reviewed DOM window. Use 3-5 windows per
site as a first paper-ready check:

- one dense pile area;
- one sparse or regular area;
- one edge/shadow/complex boundary area;
- optionally one area with different stone size distribution;
- optionally one difficult area where the model seems weak.

Only predictions whose centroids fall inside these windows are evaluated. This
prevents unannotated full-scene predictions from being counted as false positives.

## Manual Boxes

Each row in `*_manual_boxes.csv` is one manually identified rock instance.
Draw a tight bounding box around the visible rock footprint in DOM/world
coordinates. If an object is too ambiguous to judge, either do not include it or
set `ignore=1`.

The main evaluation only includes manual objects whose equivalent diameter is at
least 0.5 m, which matches the reportable-size boundary used in the manuscript.
Smaller manual objects are retained in the files but excluded from the main
Precision/Recall/F1 calculation. Predictions that match these excluded small
objects are written to `manual_detection_ignored_predictions.csv`.

## Matching Standard

The default rule is:

- match if bbox IoU >= 0.30; or
- match if centroid distance <= 0.30 m.

The IoU threshold is deliberately moderate because rocks are irregular, often
touching, and manual boxes are a coarser reference than true instance masks.
For a stricter sensitivity check, change `iou_threshold` to `0.50` in
`experiments/validation/manual_detection_validation_config.json` and rerun.

## Run

From the project root:

```powershell
python experiments\validation\manual_detection_validation.py
```

Run one site only:

```powershell
python experiments\validation\manual_detection_validation.py --site site_b
```

Outputs are written to:

`experiments/validation/outputs/manual_detection_validation/`

Main outputs:

- `manual_detection_validation_report.md`
- `manual_detection_validation_summary.json`
- `manual_detection_site_summary.csv`
- `manual_detection_region_summary.csv`
- `manual_detection_scene_type_summary.csv`
- `manual_detection_size_bin_summary.csv`
- `manual_detection_matches.csv`
- `manual_detection_false_positives.csv`
- `manual_detection_false_negatives.csv`
- `manual_detection_excluded_small_annotations.csv`
- `manual_detection_ignored_predictions.csv`

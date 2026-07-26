# Project Inventory and Publication Scope

This inventory records the repository state prepared for GitHub publication in July 2026. It separates the active paper pipeline from local data, generated outputs, and historical material.

## Active Main Pipeline

| Directory | Role | Publication status |
|---|---|---|
| `experiments/common/` | Shared scene reference, XY spatial index, mask-to-point-cloud extraction | Keep |
| `experiments/configs/` | Main slicing, detection, and fusion parameters | Keep |
| `experiments/slicing/` | SAHI baseline and edge-guided quadtree slicing | Keep code; ignore outputs |
| `experiments/detection/` | YOLO11m-seg inference, mask geometry, world-coordinate mapping | Keep code; ignore outputs |
| `experiments/fusion/` | Cross-tile fusion, diagnostics, point-cloud 3D validation | Keep code; ignore outputs |
| `experiments/volume/` | GroundDEM, 2.5D integration, 2D proxy and QC | Keep code; ignore outputs |
| `experiments/visualization/` | Full-scene, fused-stone and DOM-to-point-cloud inspection | Keep |
| `experiments/reports/` | Export current result snapshots and manuscript tables | Keep scripts only |
| `models/best.pt` | Selected YOLO11m-seg checkpoint | Keep for current reproducibility |

## Optional Research Utilities

| Directory | Role | Current limitation |
|---|---|---|
| `experiments/tuning/` | Slicing and fusion parameter search | Useful for parameter selection; not a standalone paper contribution |
| `experiments/evaluation/` | Manual review and point-cloud heuristic evaluation | Historical SAHI labels cannot be reused as current quadtree accuracy |
| `experiments/dom_analysis/` | Dataset-oriented batch pipeline | Keep for future multi-scene testing; not used by the current main result snapshot |
| `experiments/utils/` | Shared slicing metrics, reports and drawing helpers | Required by slicing modules |

`experiments/visualization/view_convex_hull_diagnostic.py` is retained only as a clearly named legacy diagnostic. Convex-hull volume is not part of the main paper result chain.

## Documents Retained

- `docs/paper/measurement_methods_draft.md`: current Measurement-style methods draft.
- `docs/paper/section_4_results_framework.md`: planned results chapter structure.
- `docs/results/current_results.md`: readable main-scene result snapshot.
- `docs/results/current_results.json`: machine-readable snapshot without local absolute paths.
- `docs/results/tables/`: current CSV and Markdown manuscript tables.
- `docs/archive/presentations/`: historical group-meeting presentations retained for computer migration, not as current scientific evidence.
- `rockseg-references/Measurement_3_1_writing_reference_notes.md`: reference roles for the data section.
- `rockseg-references/Paper_Reading_Notes_Template.md`: paper-reading template.
- `rockseg-references/Paper_Reading_Plan_and_Template.md`: literature-reading plan.

Publisher PDFs remain local and are excluded from GitHub. This avoids copyright ambiguity and unnecessary repository growth.

## Removed as Obsolete or Regenerable

The cleanup removed the following categories:

- Encoding-corrupted `README.md` content and `result.txt` historical console logs.
- The old `rockseg-paper-analysis` HTML report, which described YOLOv8, old point counts and outdated volume methods.
- Dated `dom1_20260715` experiment runs and 15 July report outputs.
- Dated report-builder scripts tied to obsolete DOM1/DOM2 counts and YOLOv8 text.
- The old `measurement_word_fill_20260722.md` draft that treated convex-hull volume as a main comparison.
- Duplicate early Section 3 drafts superseded by the current complete methods draft.
- The generated workflow image that still labeled the detector as YOLOv8 and used an outdated volume narrative.
- Generated manuscript tables from the old output directory; current tables are regenerated under `docs/results/tables/`.
- Runtime cache files from Ultralytics and local development tools.

## Local-Only Material

The following content is intentionally not deleted from the workstation, but is excluded from Git:

- `data/`: raw DOM, world files, projection files and LAZ point clouds.
- `experiments/**/outputs/`: complete masks, detections, fused-stone records, per-stone volumes and figures.
- `rockseg-references/*.pdf`: publisher article files.
- `Ultralytics/`: local application settings and cache.

## Current Scientific Boundary

The code completes the full `DOM -> detection -> fusion -> point-cloud validation -> 2.5D volume` workflow for one scene and one minimum-diameter setting. It does not yet provide a complete manually labeled accuracy test for the current main run, cross-scene generalization, threshold sensitivity across multiple minimum diameters, or physical per-stone volume ground truth.

These missing experiments are scientific work still required for a strong journal submission; they are not repository-cleanup defects.

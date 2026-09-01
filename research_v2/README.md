# RockSeg V2 Research Workspace

> **FROZEN STATUS -- 2026-08-26.** The active evidence package is complete through the 4,000-rock DOM2 representative application. See [`FINAL_RESEARCH_STATUS.md`](FINAL_RESEARCH_STATUS.md). The planning content below is retained as **SUPERSEDED / ARCHIVED** provenance.

## Current Evidence Package

- External OBJ methodology: T01+L01, `465` objects; the 0.5 mm V2 result is external validation only.
- Mine-resolution matching: DOM `10 mm/pixel`; pointcloud2 XY P90 `6.0-6.4 mm`; mine grid frozen at `10 mm`.
- Scale adaptation: uniform factor `82.737840`; 20-object feasibility pilot `20/20` valid.
- Final model: scaled-10mm 12-feature LightGBM, held-out external MAPE `5.82%`, R2 `0.9838`, best iteration `356`.
- DOM2 application: deterministic 4,000-rock sample from 69,911 accepted instances; `3,639/4,000` pipeline successes, all 361 failures `empty_2_5d_surface`.

**Claim boundary:** external model accuracy and real-mine pipeline success rate are not interchangeable. Mine per-rock absolute accuracy has not been independently validated.

## Historical Planning Record (SUPERSEDED; retained for provenance)

> This folder is the active workspace for the reconstructed RockSeg study.
> Historical manuscript drafts and result snapshots are no longer treated as
> the current plan. They can be consulted only as archived references.

## Current Principle

Do not write the manuscript first. Finish the evidence package first:

```text
baseline freeze
  -> annotation and data audit
  -> physical-scale tiling (input size determined by experiment)
  -> boundary and cross-scale fusion
  -> 2D-3D association and ground removal
  -> 2.5D surface and shape descriptors
  -> external 2.5D-to-volume benchmark (box / ellipsoid / shape-aware ablation)
  -> full-scene volume-weighted P80
  -> manuscript reconstruction
```

## Active Files

| File | Role |
|---|---|
| `00_master_plan.md` | Main planning document and single source of truth for the new study |
| `01_decision_register.md` | Decisions that must be supported by experiments before becoming final |
| `02_experiment_ledger.md` | Experiment queue, status, inputs, outputs, and acceptance gates |
| `03_dataset_and_annotation_plan.md` | Main-scene annotation and external-dataset validation plan |
| `05_method_flow_audit_and_v2_spec.md` | Stage-by-stage audit of current methods and the V2 method spec |
| `04_paper_reconstruction_outline.md` | Deferred manuscript outline; do not expand into full prose yet |
| `archive_index.md` | Map of old Markdown files and why they were archived or left in place |

## Working Rules

1. New experiments are registered in `02_experiment_ledger.md` before results are promoted.
2. New methodological decisions are recorded in `01_decision_register.md`.
3. Manuscript text is not drafted until the corresponding experiment evidence exists.
4. Historical numbers from old runs must not be mixed with V2 results unless they are re-run or explicitly marked as historical baselines.
5. External 3D data validate only the `2.5D -> volume` module, not the full DOM-to-P80 pipeline.

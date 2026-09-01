# Decision Register

> **FROZEN DECISIONS -- 2026-08-26.** The legacy candidate-decision table below is retained as **SUPERSEDED / ARCHIVED** provenance. It must not override the evidence-backed decisions listed here.

| ID | Frozen decision | Evidence | Status |
| --- | --- | --- | --- |
| FD01 | Mine 2.5D grid is `0.01 m` (10 mm). | DOM 10 mm/pixel; pointcloud2 XY P90 6.0-6.4 mm and 3D P90 about 8.5-8.6 mm. | FROZEN |
| FD02 | Canonical feature schema has 12 ordered features; `H_skew_norm = H_skew`. | Five-object training/production consistency check, maximum difference 0. | FROZEN |
| FD03 | Original-scale 10 mm OBJ training is not used. | 63 valid / 465; predominantly empty surfaces. | REJECTED |
| FD04 | External-to-mine scale factor is `82.737840`. | Independent footprint scale audit and 20/20 pilot. | FROZEN |
| FD05 | Final correction model is scaled-10mm LightGBM V2, best iteration 356. | 465-object group-aware dataset; held-out scaled external test MAPE 5.82%, R2 0.9838. | FROZEN |
| FD06 | DOM2 application uses the frozen 4,000-rock deterministic size-only manifest. | Six-stratum systematic sampling from 69,911 accepted rocks. | FROZEN |
| FD07 | Real-mine results are model-applied estimates, not validated absolute accuracy. | No per-rock DOM2 ground-truth volume. | FROZEN CLAIM BOUNDARY |

## Historical Decision Record (SUPERSEDED; retained for provenance)

This file records decisions that are not final until supported by evidence.

| ID | Decision | Current status | Evidence required | Final value |
|---|---|---|---|---|
| D01 | Final physical scales | Candidate: 9.6/4.8/2.4 m (N=960) or 10.24/5.12/2.56 m (N=1024) | E1 and E2 | Pending |
| D02 | Number of scales | Candidate: two or three | E1, E2 runtime, E3 duplicate burden | Pending |
| D03 | Overlap ratio | Candidate: 20%-30% | Boundary recall and duplicate cost | Pending |
| D04 | Size-bin boundaries | Not fixed | Manual annotation diameter distribution | Pending |
| D05 | Annotation format | Mask preferred, box acceptable with limitation | Annotation feasibility check | Pending |
| D06 | Fusion score terms | IoU, centroid, area ratio, boundary, confidence | E3 validation subset | Pending |
| D07 | Fusion threshold and weights | Not fixed | E3 grid search or validation tuning | Pending |
| D08 | Canonical mask rule | Best mask or controlled union | Fusion error audit | Pending |
| D09 | GroundDEM resolution and validity thresholds | Existing settings are baseline | E4 and volume QC | Pending |
| D10 | External dataset subset | 868 rock fragments, not yet audited | Dataset audit | Pending |
| D11 | Reference volume definition | Not fixed | Dataset metadata or watertight mesh calculation | Pending |
| D12 | External split rule | Group-aware split required, 70/15/15 | Dataset source grouping | Pending |
| D13 | Learned volume calibration | Optional, LightGBM candidate | E5 held-out improvement | Pending |
| D14 | Final P80 naming | Use volume-equivalent P80 unless field reference exists | Availability of sieve/reference P80 | Pending |
| D15 | Network input size | Candidate: 960 or 1024 | E1 pixel-size analysis and E2 segmentation performance | Pending |
| D16 | P80 weighting method | Candidate: volume-weighted cumulative passing | Confirmation that field/sieve P80 uses mass weighting | Pending |
| D17 | Ground removal method | Candidate: RANSAC, local surface, DEM, morphological filtering | E4 ground-removal QC | Pending |
| D18 | Volume ablation baselines | Three levels: bounding box, ellipsoid, shape-aware | E5 ablation results | Pending |

## Decision Rules

1. A decision is final only when the experiment ID and result artifact are
   recorded.
2. Visual examples may support a decision but cannot be the only evidence.
3. No threshold is tuned on final held-out test data.
4. Historical run values are not evidence for V2 unless they are re-run or
   explicitly accepted as a historical baseline.
5. The network input size and physical scales are coupled parameters; they must
   be resolved together in E1 and E2.

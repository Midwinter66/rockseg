# Paper Outline

Status: `WRITING OUTLINE BASELINE`

Structure pattern: empirical IMRaD-style research article. The provisional length is approximately 6,000 words excluding abstract, references, data availability, and declarations. Word allocation should be revised after the target journal is fixed.

## Central Argument

The paper presents a single linked framework in which physical-scale selection determines DOM instance segmentation and resolution matching, while a canonical shape descriptor and a frozen LightGBM model correct ground-referenced 2.5D volume estimates. The mine application demonstrates operational transfer on a deterministic, size-stratified sample, but does not establish real-mine absolute volume accuracy.

## 1. Introduction (~800 words)

### Purpose

Define the problem of estimating rock fragmentation and observable individual-rock volume from UAV DOM and point-cloud data, then motivate the chain from physical scale to resolution matching and shape-aware correction.

### Content

#### 1.1 Problem context

- Explain why 2D segmentation alone cannot provide rock volume.
- Explain why raw 2.5D integration is shape-dependent and requires correction.
- Introduce the distinction between visible surface estimates and complete buried geometry.

#### 1.2 Existing methodological limitations

- Multi-scale detection must correspond to physical rock size rather than only image input size.
- Cross-scale duplicate handling is necessary before 2D-3D association.
- External known-volume meshes and real-mine point clouds occupy different scale/resolution domains.

#### 1.3 Research objective and contribution

- Present the integrated research question.
- State the three-part contribution: physical-scale segmentation, resolution-matched Shape-Aware correction, and representative DOM2 application.
- State the scientific boundary: real-mine absolute volume accuracy is not independently validated.

### Evidence

The project evidence supports the research objective and completed workflow. Literature-based claims about prior methods, novelty, and disciplinary context are `EVIDENCE GAP` until a verified literature matrix is supplied.

### Transition

After defining the research problem, Section 2 identifies the imagery, point-cloud, external-mesh, and frozen result assets used to answer it.

## 2. Study Area and Data (~700 words)

### 2.1 DOM2 orthophoto

- Describe DOM2 as the UAV/Digital Orthophoto Map input.
- Report GSD `0.01 m/pixel`.
- Distinguish image GSD from point spacing and the 2.5D grid.
- `EVIDENCE GAP`: geographic setting, acquisition date, sensor, flight parameters, CRS, and orthorectification metadata are not provided in the frozen paper package.

### 2.2 Mine point clouds

- Identify BlockB and BlockY as the existing associated point-cloud inputs.
- Report sampled point spacing: XY P90 6.00/6.40 mm and 3D P90 8.54/8.60 mm.
- Explain that these are sampled local spacing statistics, not a uniform point-density guarantee.
- `EVIDENCE GAP`: acquisition system, date, registration accuracy, absolute vertical accuracy, and complete point-count table are not included in the approved writing inputs.

### 2.3 External OBJ Dataset B

- T01 79 + L01 386 = 465 known-volume external rock meshes.
- Define `V_true`, `V_2.5D`, and `y_ratio = V_true/V_2.5D`.
- Explain that the meshes support methodological training and validation, not mine-site ground truth.

### 2.4 Frozen data partitions and mine population

- Scaled 10 mm external split: 326/70/69 with group-aware separation.
- DOM2 final inventory: 76,407; accepted population: 69,911.
- Frozen representative sample: 4,000 accepted rocks.

### Transition

The different spatial supports of DOM, point clouds, and external meshes motivate the physical-scale and resolution-matching methodology in Section 3.

## 3. Methodology (~1,700 words)

### 3.1 Overall framework

- Present the physical-scale-to-volume argument and the complete linked workflow.
- Separate external mesh model development from real-mine application.
- Define DOM GSD, point spacing, 2.5D grid resolution, and the observable-rock boundary.

### 3.2 Physical-scale multi-scale DOM instance segmentation

- Explain the need for multiple physical observation windows.
- Define coarse, medium, and fine coverage, overlap, common network input, and detection output.
- Keep inventory counts for Section 4.1; do not treat them as segmentation accuracy.

### 3.3 Duplicate resolution and cascade deduplication

- Within-scale duplicate resolution across overlapping tiles: spatial lookup, bbox IoU pre-filter, mask IoU, weighted score, threshold, and representative selection.
- Cross-scale cascade deduplication: spatial/mask/diameter/centroid compatibility and primary-scale selection.
- Define the equivalent-diameter formula and distinguish generic cross-scale fusion from the final cascade path.
- State the absence of an independent duplicate-resolution accuracy benchmark.

### 3.4 2D--3D association, ground reference and 2.5D reconstruction

- Describe instance-to-point-cloud spatial retrieval and fixed 3D quality gates.
- Describe the scene-level GroundDEM and its role in ground-relative height.
- Define the 10 mm height field and $V_{2.5D}$ integration.
- State that the result represents observable ground-referenced geometry.

### 3.5 Shape-aware descriptor, scale adaptation and volume correction

- Define the ordered 12-feature descriptor, including the special $H_{\mathrm{skew,norm}}=H_{\mathrm{skew}}$ rule.
- Define the external correction target and $V_{\mathrm{pred}}$.
- Explain the 0.5 mm external methodological path, the 10 mm resolution choice, and the uniform scale factor 82.737840.
- Keep external test metrics for Section 4.3 and the adaptation evidence for Section 4.4.

### 3.6 Representative real-mine application

- Define the accepted population and deterministic diameter-stratified sample design.
- State the six strata and fixed allocation 400/600/1,000/1,000/600/400.
- Describe the per-rock inference sequence and sample scope.
- Reserve success/failure counts and distributions for Section 4.5.

### Transition

Section 4 reports each stage in the same order. Section 4.1 evaluates the segmentation and duplicate-resolution inventory, Section 4.2 evaluates association and filtering outcomes, Sections 4.3--4.4 evaluate external correction and scale adaptation, and Section 4.5 reports the representative real-mine application. Completion rates are distinguished from independently validated accuracy throughout.

## 4. Results (~1,400 words)

### 4.1 Segmentation and fusion results

- Raw detections: 37,470 coarse, 101,642 medium, 179,286 fine.
- Total raw pool: 318,398.
- Within-scale fused pool: 112,983.
- Final cascade inventory: 76,407.
- Final retained scales: 5,925 coarse, 10,890 medium, 59,592 fine.
- `EVIDENCE GAP`: no precision/recall/mAP; no per-scale post-within-scale counts; no manual fusion-accuracy benchmark.

### 4.2 2D-3D association and filtering results

- Accepted 69,911; rejected 6,496; acceptance rate 91.50%.
- Report rejection-reason counts and note that reasons may co-occur.
- Interpret accepted rate as screening outcome, not accuracy.
- `EVIDENCE GAP`: independent correspondence validation and DEM accuracy.

### 4.3 External Shape-Aware volume validation

- Report Dataset B composition and frozen split.
- Present the ratio metrics on 69 Test meshes.
- Present Raw 2.5D, Constant correction, and Shape-Aware V2 comparison table.
- Primary result: Shape-Aware V2 MAPE 5.82% and R2 0.9838 on scaled external Test data.
- Do not label this as mine-site accuracy.

### 4.4 Resolution and scale adaptation analysis

- Original-scale 10 mm validity: 63/465.
- Spacing audit and 10 mm rationale.
- Scale factor 82.737840 and audit conclusion.
- Pilot 20/20; full dataset 465/465; no non-finite features.
- `EVIDENCE GAP`: independent physical-domain validation of the scale mapping.

### 4.5 Real-mine volume estimation

- Frozen 4,000-rock sample from 69,911 accepted instances.
- 3,639 success and 361 failure; completion rate 90.98%.
- All failures are `empty_2_5d_surface`.
- Report stratum completion rates and increasing success with rock-size stratum.
- Report `V_2.5D`, `y_pred`, and `V_pred` distributions for successful records.
- State explicitly that no real-mine MAE, MAPE, or R2 is available.

### Transition

Section 5 interprets the gains and operational behavior while separating demonstrated results from domain-transfer assumptions.

## 5. Discussion (~900 words)

### 5.1 Physical scale and resolution matching

- Discuss why image scale, point spacing, and reconstruction grid must be coordinated.
- Explain why the original 0.5 mm external model was not directly deployed.
- Treat the uniform scale mapping as a controlled adaptation hypothesis.

### 5.2 Shape-aware correction versus constant correction

- Discuss the substantial improvement over Raw 2.5D.
- Discuss the smaller but measurable improvement over Constant correction.
- Note the ratio-target R2 of 0.3280 alongside volume R2 of 0.9838; avoid equating these metrics.
- `EVIDENCE GAP`: literature-based comparison with other rock-volume methods.

### 5.3 Small-rock surface availability

- Interpret the completion gradient from S1 84.0% to S6 99.75%.
- Relate all 361 failures to empty 10 mm surfaces.
- Do not alter resolution or impute failed volumes retrospectively.

### 5.4 Scientific limitations

- No independent segmentation/fusion/association benchmark.
- No independent GroundDEM accuracy benchmark.
- Scale adaptation is plausible but not physically proven.
- No DOM2 per-rock ground-truth volume.
- 2.5D estimates describe observable surfaces and do not recover buried or occluded geometry.
- 4,000 rocks are a deterministic representative sample, not a full 69,911-rock volume census.

### 5.5 Implications and future validation

- Position the current work as a reproducible end-to-end application framework.
- Restrict future-work statements to reference-volume validation, uncertainty characterization, and evidence-gap closure; do not imply those analyses were completed.

### Transition

Section 6 closes with the demonstrated contribution and its explicit limits.

## 6. Conclusion (~300 words)

### Purpose

Summarize the integrated physical-scale-to-volume framework without adding new numbers or extending the claims.

### Required conclusion points

- 76,407 final DOM2 instances and 69,911 accepted screening records establish the application inventory.
- The scaled external 10 mm Test supports Shape-Aware correction relative to Raw 2.5D and Constant correction.
- The frozen model completed 3,639 of 4,000 representative mine-sample inferences.
- The real-mine output is an application result, not independently validated absolute volume accuracy.
- Observable-surface and scale-adaptation limitations remain explicit.

## Evidence Assignment Summary

| Section | Primary frozen evidence | Status |
| --- | --- | --- |
| 1 Introduction | Research question and claim boundaries | Ready for project-specific framing; external literature is EVIDENCE GAP |
| 2 Study Area and Data | DOM GSD, point spacing, Dataset B, inventory and sample counts | Partly ready; acquisition/site metadata is EVIDENCE GAP |
| 3 Methodology | `PAPER_METHOD_PIPELINE.md`, feature definitions, model and sampling contracts | Ready for drafting with stated limitations |
| 4.1 Segmentation/fusion | Frozen stage counts and implementation | Ready for count reporting; accuracy is EVIDENCE GAP |
| 4.2 Association/filtering | Accepted/rejected totals, thresholds, reasons | Ready for gate results; accuracy is EVIDENCE GAP |
| 4.3 External validation | Frozen Test metrics and comparison table | Ready |
| 4.4 Resolution/scale | Spacing, failure, scale audit/pilot/full dataset | Ready with `NOT PROVEN` qualification |
| 4.5 Mine application | Sampling and 4,000-run QC/distributions | Ready; absolute accuracy is EVIDENCE GAP |
| 5 Discussion | Existing results and limitation register | Ready for internal interpretation; literature comparison is EVIDENCE GAP |
| 6 Conclusion | Frozen demonstrated claims | Ready |

## Provisional Word Count

| Section | Target words |
| --- | ---: |
| 1 Introduction | 800 |
| 2 Study Area and Data | 700 |
| 3 Methodology | 1,900 |
| 4 Results | 1,400 |
| 5 Discussion | 900 |
| 6 Conclusion | 300 |
| Total | 6,000 |

Abstract, references, data availability, author contributions, funding, conflict-of-interest, ethics, and AI-use disclosure are outside this provisional count. Their final wording depends on the target journal and author-supplied metadata.

## Writing Gate

This outline is ready for author review. Formal section drafting should begin only after the author confirms the outline and chooses whether to supply the missing study-area/acquisition metadata and literature matrix. No evidence gap requires an automatic new experiment.

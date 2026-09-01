# Paper Evidence Map

Status labels: `PASS` means a frozen result or implementation record supports
the statement; `IMPLEMENTATION_VERIFIED` means code and outputs establish the
implemented workflow but not an independent accuracy measurement; `PAPER_GAP`
means a claim must not be made as validated.

| Paper part | What must be demonstrated | Current evidence | Status | Paper-safe statement / boundary |
| --- | --- | --- | --- | --- |
| Study area and inputs | DOM and point-cloud provenance | `data/dom2/DOM.tif`; BlockB/BlockY LAZ paths; frozen result metadata | PASS | DOM2 and existing point-cloud association inputs are identified. |
| DOM resolution | Analysis pixel scale | `rockseg/config.py`: `gsd = 0.01` | PASS | DOM2 processing uses 0.01 m/pixel. |
| Multi-scale segmentation | Physical multi-scale detector output | `rockseg/config.py`; `output/dom2_cascade_v2/rock_instances.json` | PASS | Coarse/medium/fine scales produced 76,407 final instances. |
| Segmentation accuracy | Accuracy against independent labels | No frozen manual-label precision/recall/mAP result located | PAPER_GAP | Do not claim segmentation accuracy. |
| Same-scale fusion | Deduplication rule and aggregate output | `rockseg/fusion.py`; `rockseg/config.py`; discovery report | IMPLEMENTATION_VERIFIED | Same-scale fusion uses weighted score >= 0.50; aggregate pool is 112,983. Per-scale post-fusion counts are NOT VERIFIED. |
| Cross-scale cascade | Duplicate grouping and selection rule | `rockseg/fusion.py`; final inventory | IMPLEMENTATION_VERIFIED | Cascade uses spatial/mask/size/centroid compatibility and primary-scale selection. No independent fusion-error annotation is available. |
| Final instance inventory | Final count and IDs | `rock_instances.json`; discovery report | PASS | 76,407 final instance records exist. |
| 2D-3D association | Point-cloud spatial association and screening outcome | `validation_3d_fast.py`; fixed validation output | IMPLEMENTATION_VERIFIED | 69,911 accepted and 6,496 rejected by the frozen float64-fixed screening. Association accuracy against manual correspondences is not established. |
| 3D quality gate | Thresholds and rejection categories | `validation_3d_fast.py`; `validation_summary.json` | PASS | Fixed thresholds and recorded reasons are available. |
| Ground/DEM | Ground-referenced height construction | `GroundDEM` in `validation_3d_fast.py`; `rockseg/volume.py` | IMPLEMENTATION_VERIFIED | A 0.5 m P5 ground reference supports relative observed heights. Independent DEM vertical accuracy is PAPER_GAP. |
| 2.5D volume | Operational surface reconstruction and volume calculation | `rockseg/volume.py`; 4,000-rock results | PASS | Positive observable 2.5D volumes and formula QC exist for successful samples. |
| Occluded/buried geometry | Complete physical rock volume below visibility | No source of subsurface geometry | PAPER_GAP | Do not claim recovery of hidden or buried geometry. |
| External dataset | Training/validation provenance | T01/L01 dataset metadata and scaled dataset | PASS | 465 external OBJ meshes provide known mesh-volume methodology data. |
| Resolution matching | Rationale for 10 mm mine grid | spacing report; resolution study | PASS | DOM GSD and sampled point spacing support the frozen 10 mm operational grid. |
| Scale adaptation | Controlled mapping from external mesh scale to mine scale | scale audit and 20-object pilot | PASS with limitation | It is feasible and pre-registered, but `SCALE_ADAPTATION_PLAUSIBLE_BUT_NOT_PROVEN`. |
| Shape-aware model | Held-out external scaled-mesh correction performance | `training_results_v2_scaled_10mm.json` | PASS | Shape-Aware V2 improves the held-out scaled external test over raw 2.5D. |
| Feature consistency | Identical train/inference schema | `shape_features_v2.py`; feature consistency report; final QC | PASS | The canonical 12-feature order is fixed and was QC-checked. |
| Real-mine transfer | Ability to run inference chain | 12-rock pilot and 4,000-rock summary | PASS | 3,639/4,000 prescribed accepted rocks completed the chain. |
| Real-mine absolute accuracy | Per-rock mine reference comparison | No per-rock DOM2 ground-truth volume | PAPER_GAP | Do not report an absolute mine volume error or transfer 5.82% to DOM2. |
| Representative mine statistics | Transparent selected population and sample | frozen sampling manifest/report | PASS | 4,000 deterministic diameter-stratified accepted rocks are a representative application sample, not the full population. |

## Evidence Sources

| Evidence item | Frozen source |
| --- | --- |
| Configuration and fusion | `rockseg/config.py`, `rockseg/fusion.py` |
| 3D validation and ground reference | `rockseg/validation_3d_fast.py`, `rockseg/validation_3d.py`, `output/dom2_cascade_v2_3d_fixed/validation_summary.json` |
| 2.5D and canonical features | `rockseg/volume.py`, `research_v2/volume_validation/shape_features_v2.py`, `feature_consistency_check.json` |
| Resolution/scale evidence | `research_v2/volume_validation/resolution_study/` |
| Model evidence | `research_v2/volume_validation/output_v2_scaled_10mm/` |
| Sampling and mine application | `research_v2/volume_validation/real_mine_sampling/`, `research_v2/volume_validation/real_mine_full/` |

## Required Claim Discipline

`PAPER_GAP` items are not requests to start new experiments. They identify
claims that must remain omitted or explicitly limited in the manuscript.

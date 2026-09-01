# RockSeg V2 Master Plan

## 1. Core Research Question

How can blast-rock fragments with a wide physical size range be measured from
UAV photogrammetric products by combining physical-scale-aware DOM instance
segmentation, 2D-3D spatial association, and ground-referenced 2.5D volume
estimation?

## 2. Reconstructed Main Line

```text
UAV imagery
  -> DOM and OSGB-derived photogrammetric point cloud
  -> physical-scale analysis
  -> multi-scale DOM tiling by real ground coverage
  -> YOLO11m-seg instance segmentation at each physical scale
  -> within-scale boundary-aware fusion
  -> cross-scale instance fusion
  -> unique rock masks
  -> 2D-3D association with the point cloud
  -> ground point removal
  -> 2.5D rock surface
  -> shape descriptors
  -> shape-aware volume estimation
  -> volume-equivalent diameter distribution
  -> volume-weighted cumulative passing
  -> P80
```

## 3. Three Contributions To Test

### C1. Physical-Scale-Aware Multi-Scale Segmentation

The scale variable is the ground coverage of one network input, not merely the
YOLO `imgsz`.

For physical crop width \(W_k\) and network input width \(N\):

\[
g_k = \frac{W_k}{N},
\qquad
d_{\mathrm{input},k} = \frac{d_{\mathrm{physical}}}{g_k}.
\]

The network input size \(N\) is itself an experimental parameter, not a fixed
constant. Two candidate input sizes are under consideration:

| Candidate N | DOM GSD | Coarse coverage | Medium coverage | Fine coverage |
|---|---|---:|---:|---:|
| 960 px | 0.01 m/px | 9.60 m | 4.80 m | 2.40 m |
| 1024 px | 0.01 m/px | 10.24 m | 5.12 m | 2.56 m |

The physical scales and the network input size are not fixed at this stage.
They become final only after the scale-selection experiment (E1) and the
multi-scale segmentation experiment (E2).

The relationship between rock physical size and pixel size in the input is:

\[
D_{\mathrm{pixel}} = \frac{D_{\mathrm{physical}}}{\mathrm{GSD}}
\]

When a rock's pixel size is insufficient at a given scale, the corresponding
original region can be resampled to the fixed input size so the rock occupies
more pixels. This establishes:

\[
\text{Rock physical size} \rightarrow \text{Optimal image scale}
\]

### C2. Boundary-Aware And Cross-Scale Instance Fusion

The output must satisfy:

```text
one physical rock = one global instance
```

**Within-scale boundary fusion.** Adjacent tiles may predict masks belonging to
the same rock. Given two candidate masks \(M_1\) and \(M_2\), a multi-feature
fusion score is computed:

\[
S_{\mathrm{boundary}} = w_1 \mathrm{IoU} + w_2 S_c + w_3 S_A + w_4 S_b
\]

where \(S_c\) is centroid similarity, \(S_A\) is area-ratio similarity, and
\(S_b\) is boundary/contour similarity. When
\(S_{\mathrm{boundary}} > \tau_{\mathrm{boundary}}\), the two masks are merged:

\[
M_{\mathrm{merge}} = M_1 \cup M_2
\]

**Cross-scale instance matching.** Different scales may detect the same rock
as separate instances. Matching uses the same multi-feature score:

\[
S_{\mathrm{match}} = w_1 \mathrm{IoU}_{\mathrm{mask}} + w_2 S_c + w_3 S_A + w_4 S_b
\]

When \(S_{\mathrm{match}} > \tau_{\mathrm{match}}\), the instances are fused into
one.

Fusion should be evaluated as an object-level problem:

- duplicate reduction;
- boundary recovery;
- over-merge control;
- under-merge control;
- instance count error;
- mask quality against manual reference regions.

Each final instance must retain:

```text
instance_id
mask
centroid
area
bounding box
source scale
confidence
```

### C3. 2D-3D Shape-Aware Volume Estimation

The segmentation result becomes useful for fragmentation only after it is
associated with the point cloud and converted into volume.

**Coordinate unification.** DOM pixel coordinates \((x, y)\) and point-cloud
coordinates \(P_i = (X_i, Y_i, Z_i)\) are unified via CRS, GeoTransform, TFW,
and LAZ/OSGB coordinate information:

\[
(X_i, Y_i) \rightarrow (x_i, y_i)
\]

**Mask-based point extraction.** For rock \(r\) with binary mask
\(M_r(x,y) \in \{0,1\}\), a point \(P_i\) is assigned to the rock if
\(M_r(x_i, y_i) = 1\):

\[
P_r = \{P_i \mid M_r(x_i, y_i) = 1\}
\]

**Ground removal.** Mask projection typically includes ground points beneath
the rock. Candidate ground-estimation methods:

```text
RANSAC plane fitting
local ground surface fitting
DEM reference
morphological filtering
```

If local ground elevation is \(Z_{\mathrm{ground}}(x,y)\), the rock's relative
height is:

\[
H_i = Z_i - Z_{\mathrm{ground}}(x_i, y_i)
\]

Only points satisfying \(Z_i > Z_{\mathrm{ground}}(x_i, y_i) + \delta\) are
retained, where \(\delta\) is the tolerance for ground noise and rock-bottom
error.

**2.5D surface.** After ground removal, the rock is represented as a height
function \(z = f(x,y)\).

**Shape descriptors.** Extracted from the 2.5D surface:

| Category | Features |
|---|---|
| Dimensions | \(L, W, H\) |
| Area and perimeter | \(A, P\) |
| Circularity | \(C = 4\pi A / P^2\) |
| Aspect ratio | \(AR = L / W\) |
| Height statistics | \(H_{\mathrm{mean}}, H_{\mathrm{max}}, H_{\mathrm{std}}\) |

**Volume model ablation.** Three baselines in increasing complexity:

1. **Bounding-box model**: \(V_{\mathrm{box}} = LWH\)
2. **Ellipsoid model**: \(V_{\mathrm{ellipsoid}} = \frac{\pi}{6} LWH\)
3. **Shape-aware model**: uses all geometric features above; initial candidate
   is LightGBM, with optional comparison to XGBoost or MLP.

The bounding-box model overestimates irregular rocks:
\(V_{\mathrm{box}} \gg V_{\mathrm{true}}\). The shape-aware model must demonstrate
\(E_{\mathrm{shape}} < E_{\mathrm{box}}\) and
\(E_{\mathrm{shape}} < E_{\mathrm{ellipsoid}}\) on held-out data to be
justified, where:

\[
E = \frac{|V_{\mathrm{pred}} - V_{\mathrm{true}}|}{V_{\mathrm{true}}}
\]

**Equivalent diameter.** Assuming spherical equivalence:

\[
D_{\mathrm{eq}} = \left(\frac{6V}{\pi}\right)^{1/3}
\]

**P80 computation.** Volume-weighted cumulative passing:

\[
w_i = \frac{V_i}{\sum_{j=1}^{N} V_j}, \qquad
F(D_k) = \sum_{i: D_i \le D_k} w_i
\]

P80 is the diameter satisfying \(F(P_{80}) = 0.80\).

The weighting method (volume/mass vs. count) must be explicitly stated. For
blast fragmentation analysis, volume or mass weighting is standard; count-based
percentiles are not acceptable for P80.

## 4. Data Boundary

### Main Scene

Used for the complete pipeline:

```text
DOM -> masks -> point cloud association -> ground removal -> 2.5D volume -> P80
```

The main scene requires manual validation windows for DOM segmentation,
fusion, and 2D-3D association.

### External 3D Rock Dataset

Used only for the isolated volume module:

```text
3D rock object -> simulated 2.5D observation -> predicted volume -> reference volume
```

It must not be described as validation of DOM segmentation or the end-to-end
RockSeg pipeline.

## 5. Experiment Set

| ID | Experiment | Main question | Gate |
|---|---|---|---|
| E1 | Particle-size and scale selection | Which physical scales and input size are needed? | Final scale set, input size, and overlap selected |
| E2 | Multi-scale segmentation | Does physical multi-scale improve per-size segmentation? | Size-bin metrics support chosen scales |
| E3 | Boundary and cross-scale fusion | Does fusion reduce duplicates without wrong merges? | Fusion metrics outperform baselines |
| E4 | 2D-3D association | Do masks retrieve correct rock point clouds? | Auditable object-level association evidence |
| E5 | External volume validation | Does 2.5D volume estimate reference volume? | Held-out volume errors and ablation reported |
| E6 | Full-scene application | What distribution and P80 does the final pipeline produce? | One frozen run produces final outputs |

## 6. Implementation Phases

### Phase 0. Freeze Current Baseline

Select one authoritative baseline run and reconcile conflicting old documents,
configs, and result snapshots. Historical runs can be referenced but not mixed
with V2 results.

### Phase 1. Prepare Evidence

Complete or refine manual validation windows; define size bins; create fusion
and 2D-3D review subsets; audit the external 3D volume dataset (868 rock
fragments with reference volumes).

### Phase 2. Build Physical Multi-Scale Tiling

Generate tiles by real ground coverage, retain scale provenance, and resample
each tile to the fixed network input size (960 or 1024, determined by E1/E2).

### Phase 3. Build Hierarchical Fusion

First fuse overlap duplicates within each scale using the multi-feature fusion
score, then fuse repeated instances across scales while preserving provenance.

### Phase 4. Select DOM-Side Configuration

Run E1-E3 and freeze scales, input size, overlap, confidence, fusion features,
weights, and thresholds.

### Phase 5. Validate 2D-3D Association

Run reviewed object-level association checks. Validate coordinate
transformation, mask projection, and ground removal. Document accepted,
rejected, and ambiguous cases.

### Phase 6. Validate Volume Module Externally

Build 3D-to-2.5D simulation, evaluate geometric baselines (box, ellipsoid),
and test shape-aware calibration on held-out samples. Perform volume ablation:
bounding box vs. ellipsoid vs. shape-aware model. Report MAE, RMSE, MAPE, R2.

### Phase 7. Run Full Scene

Run the final configuration once, preserve manifests, generate final tables and
figures, and compute volume-equivalent P80 with explicit volume weighting.
Perform end-to-end comparison: single-scale vs. multi-scale vs. +fusion vs.
+2D-3D vs. +shape-aware volume.

### Phase 8. Reconstruct The Paper

Only after Phases 0-7 are complete, write the manuscript from frozen evidence.

## 7. One-Sentence Abstract

Targeting the wide particle-size range of blast rocks and the inconsistency
between 2D observations and 3D volume, this study proposes a
physical-scale-driven multi-scale instance segmentation and 2D-3D association
framework. Boundary-aware and cross-scale instance fusion produces unique rock
instances. The 2D masks are mapped to 3D point clouds, local ground points are
removed, and 2.5D rock surfaces are constructed. Shape descriptors are
extracted and a shape-aware model estimates rock volume. The method is
independently validated by 868 external 3D rock fragments and applied
end-to-end on real DOM and point-cloud data to compute volume-equivalent P80.

## 8. Immediate Next Step

Start with Phase 0:

1. choose the baseline run;
2. list all historical result numbers and their source files;
3. decide which values are historical only and which will be re-generated in
   V2;
4. create a baseline freeze note before implementing new physical multi-scale
   code.

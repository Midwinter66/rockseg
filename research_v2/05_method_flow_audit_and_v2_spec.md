# Method Flow Audit And V2 Specification

> Purpose: inspect each stage of the current pipeline, compare it with the
> earlier experimental path, and state the method that V2 should use when the
> paper is rewritten later.
>
> This is not manuscript prose. It is a method ledger with core ideas,
> equations, current limitations, and the V2 target method for every stage.

## 1. End-To-End Flow

```text
DOM / point cloud / TFW / CRS
  -> spatial reference freeze
  -> physical-scale analysis
  -> physical multi-scale tiling
  -> YOLO11m-seg instance segmentation
  -> within-scale boundary-aware fusion
  -> cross-scale instance fusion
  -> unique rock masks
  -> 2D geometric measurement
  -> 2D-3D spatial association
  -> GroundDEM-referenced 2.5D rock surface
  -> volume estimation
  -> volume-equivalent diameter and P80
  -> external 3D validation of the volume module
```

## 2. Stage-by-Stage Audit

| Stage | Current method in code / past runs | Main limitation | V2 target method |
|---|---|---|---|
| Spatial reference | DOM + TFW + OSGB-derived LAZ; no XY shift for Scene A | Only one scene is fully frozen; Scene B uses a different local shift and must not be mixed | Freeze scene-specific coordinate rules first, then treat all transforms as explicit, versioned inputs |
| Tiling | Edge-density-guided quadtree or fixed SAHI windows | This is not true physical-scale multi-scale tiling; same-tile multi-`imgsz` only changes inference resolution | Build tiles by physical ground coverage \(W_k\) and keep all scale provenance |
| Detection | YOLO11m-seg, usually single-scale `imgsz=1024`, `conf=0.35`, `min_stone_diameter_m=0.5` | Single-scale inference cannot adapt to wide physical size range; current multi-scale mode is only a same-tile `imgsz` sweep | Keep the same model, but run it on physically different crops so that network input pixel size varies by real ground coverage |
| Same-scale fusion | Correlation clustering or heuristic merge using world bbox IoU and centroid distance | Duplicate handling is mainly cross-tile and not fully mask-aware; one-detection-per-tile is only a soft structural rule | Use mask-aware within-scale fusion that prefers boundary completeness and preserves provenance |
| Cross-scale fusion | Not yet a true physical-scale fusion; current multi-scale detection only fuses duplicate `imgsz` outputs inside one crop | No scale hierarchy, no explicit scale provenance, no object-level cross-scale competition | Fuse fine/medium/coarse instances with a hierarchical graph and a canonical mask rule |
| 2D geometry | Area, equivalent diameter, centroid, bbox, confidence | Shape is represented too coarsely for later volume reasoning | Add perimeter, circularity, aspect ratio, bbox asymmetry, and scale-source tags |
| 2D-3D association | Mask boundary -> convex hull -> point-cloud crop; bbox coarse query first, polygon refinement second | Convex-hull fallback can over-expand concave or touching shapes and pull in neighbor points | Use exact mask footprint when possible; keep convex hull only as a fallback or acceleration aid |
| Ground reference | GroundDEM from subsampled point cloud, 5th percentile per cell, hole filling by neighbor expansion | Sensitive to cell resolution, sampling step, and fill behavior; needs explicit ablation | Keep GroundDEM, but treat percentile, resolution, and fill strategy as controlled parameters |
| 3D validity | Point count, z-range, P90 height, elevated ratio | This is screening, not accuracy; it only says whether the candidate looks like a raised object | Recast as association validity and volume-quality control, not detection accuracy |
| 2.5D volume | `sum(max(z_top - z_ground, 0) * Delta^2)` on local grid | Deterministic and interpretable, but depends on surface extraction and ground reference | Keep deterministic 2.5D integration as the core method |
| 2D proxy baseline | `pi/6 * d_eq^3` | Ignores height and overestimates irregular stones | Keep only as a simple lower-complexity comparison baseline |
| External validation | Current site B / A-B summaries are pipeline comparison, not external volume truth | No independent 3D reference volume yet; cannot validate the full pipeline externally | Use external 3D rock data only for isolated `2.5D -> volume` validation |
| P80 | Currently derived from volume-equivalent diameter distribution | Not a sieve-based ground truth unless an external reference exists | Report `volume-equivalent P80` unless a true field P80 reference is available |

## 3. Core Method Specification By Stage

### 3.1 Spatial reference freeze

Current implementation uses:

\[
(x, y) \leftrightarrow (u, v)
\]

through the TFW affine transform, and Scene A uses zero XY shift:

\[
\begin{aligned}
x &= C + Au + Bv,\\
y &= F + Du + Ev.
\end{aligned}
\]

V2 rule:

1. every scene must expose its own coordinate mode;
2. point clouds, masks, tiles, and validation windows must share the same declared reference;
3. Scene B-style local shifts must remain scene-local and never leak into Scene A.

### 3.2 Physical-scale-aware multi-scale tiling

The old path has two types of tiling:

1. quadtree tiling by DOM texture edge density;
2. SAHI fixed windows controlled by overlap.

The V2 method must instead be defined by real ground coverage. For crop width
\(W_k\) and network input width \(N\):

\[
g_k = \frac{W_k}{N},
\qquad
d_{\mathrm{input},k} = \frac{d_{\mathrm{physical}}}{g_k}.
\]

Interpretation:

- \(g_k\) is the effective GSD after resampling;
- \(d_{\mathrm{input},k}\) is how large the same rock appears in the model input;
- scale choice is therefore a physical coverage problem, not a cosmetic `imgsz` choice.

Recommended V2 protocol:

1. choose 2-3 candidate ground coverage ranges;
2. keep input size fixed, e.g. `1024 x 1024`;
3. record physical coverage, resampling factor, overlap, and tile provenance;
4. select the final scale set from the scale-selection experiment, not by intuition.

### 3.3 Instance segmentation

Current code:

- YOLO11m-seg inference;
- mask candidates filtered by minimum equivalent diameter;
- RLE serialization of binary masks;
- centroid and bbox recovered in world coordinates.

Equivalent-diameter rule:

\[
A_{2D} = n_p s_x s_y,
\qquad
d_{eq} = \sqrt{\frac{4A_{2D}}{\pi}}.
\]

V2 method:

1. keep the same model family during tiling experiments;
2. do not attribute backbone changes to the method contribution;
3. run the same model across physically different crops;
4. keep the full binary mask, not only the bbox, as the geometric primitive.

### 3.4 Within-scale boundary-aware fusion

Current code has a light fusion rule using world bbox IoU and centroid distance.
It works, but it is mostly a duplicate-removal heuristic, not a mask-aware
boundary model.

V2 target score:

\[
S_{ij} =
w_1 S_{\mathrm{IoU}} +
w_2 S_{\mathrm{centroid}} +
w_3 S_{\mathrm{area}} +
w_4 S_{\mathrm{boundary}} +
w_5 S_{\mathrm{confidence}}.
\]

Suggested meaning:

- \(S_{\mathrm{IoU}}\): global mask overlap or bbox overlap;
- \(S_{\mathrm{centroid}}\): closeness of centroids;
- \(S_{\mathrm{area}}\): area consistency;
- \(S_{\mathrm{boundary}}\): whether the candidate touches tile borders in a way that suggests truncation;
- \(S_{\mathrm{confidence}}\): model confidence or aggregated confidence.

V2 rule:

1. first fuse duplicates caused by overlap within the same physical scale;
2. prefer the mask with stronger boundary completeness;
3. only merge complementary fragments when border geometry supports a single rock;
4. preserve provenance for every merged instance.

### 3.5 Cross-scale instance fusion

Current pipeline does not yet implement a true physical-scale hierarchy. The
multi-scale option in detection only changes `imgsz` for the same crop and then
fuses duplicate detections from that crop.

V2 method should use a hierarchical graph:

1. nodes are within-scale fused masks;
2. edges connect candidate instances across scales;
3. edge weights combine mask overlap, centroid distance, area ratio, boundary consistency, and confidence;
4. clusters yield unique global rock instances.

Canonical-mask rule:

\[
\text{rock} = \arg\max(\text{boundary completeness, confidence, scale agreement})
\]

or a controlled union only when the masks are complementary and do not violate
geometric plausibility.

### 3.6 2D geometric measurement

Per unique rock mask, current code already measures:

- projected area;
- equivalent diameter;
- centroid;
- bbox;
- confidence;
- source tile / source scale tags.

V2 should also report:

- perimeter \(P\),
- circularity \(C = 4\pi A / P^2\),
- aspect ratio \(AR = L/W\),
- mask compactness or elongation,
- scale provenance.

These features are not the final volume model by themselves; they are shape
priors.

### 3.7 2D-3D spatial association

Current implementation:

1. decode mask RLE;
2. extract boundary;
3. convert boundary pixels to world coordinates;
4. build a 2D polygon or convex hull;
5. query point cloud with bbox + polygon;
6. keep points inside the footprint.

This works, but the convex-hull step can over-expand a concave or touching
stone. In V2:

1. use the exact mask footprint when possible;
2. keep convex hull only as a fallback;
3. if available, use a polygonized mask contour rather than a hull;
4. record failed or ambiguous associations explicitly.

Formal association:

\[
P_r = \{P_i \mid M_r(x_i, y_i) = 1\}
\]

after the point cloud is transformed into the shared world frame.

### 3.8 GroundDEM and ground-referenced validity

Current GroundDEM is built from subsampled OSGB points, using a low percentile
per cell and hole filling by neighbor expansion.

Ground surface rule:

\[
h_i = Z_i - z_{\mathrm{ground}}(X_i, Y_i).
\]

Current 3D screening uses:

- point count;
- z-range;
- P90 relative height;
- elevated-point ratio.

V2 interpretation:

1. this is not segmentation accuracy;
2. it is 3D association validity and volume-quality control;
3. resolution, percentile, subsample step, and hole filling must be studied as explicit factors.

### 3.9 2.5D volume estimation

Current method is already the right core idea:

\[
h_q = \max(z_q^{\mathrm{top}} - z_q^{\mathrm{ground}}, 0),
\qquad
V_{\mathrm{2.5D}} = \sum_q h_q \Delta^2.
\]

Interpretation:

- use the visible top surface on a local grid;
- subtract the local ground reference;
- integrate positive heights only.

V2 should keep this as the deterministic main method because it is interpretable
and compatible with the current data.

Baselines:

\[
V_{\mathrm{box}} = LWH
\]

\[
V_{\mathrm{ellipsoid}} = \frac{\pi}{6}LWH
\]

\[
V_{2D} = \frac{\pi}{6}d_{eq}^3
\]

The main point is comparison, not pretending that any baseline is true volume.

### 3.10 External 3D validation

Current site A/B results are internal pipeline comparisons and 3D screening
statistics. They are not independent volume ground truth.

V2 external benchmark should:

1. take a separate 3D rock dataset;
2. simulate the 2.5D observation;
3. compare predicted volume against reference volume;
4. report MAE, RMSE, MAPE/SMAPE, median relative error, and \(R^2\).

If a learned calibrator is used:

\[
\hat{V} = f\left(A, P, C, AR, h_{\mathrm{mean}}, h_{\mathrm{max}}, h_{\mathrm{std}}, \dots\right)
\]

it should be treated as an optional calibration layer, not the core method.

### 3.11 Volume-equivalent diameter and P80

For each stone volume \(V_i\):

\[
d_{v,i} = \left(\frac{6V_i}{\pi}\right)^{1/3}.
\]

Then sort by \(d_{v,i}\) and compute the cumulative volume-passing curve.
The 80% passing point is the volume-equivalent P80.

If a true sieve or field P80 reference is absent, the label must stay:

```text
volume-equivalent P80
```

## 4. What The Past Experiments Already Give Us

1. The pipeline already runs end-to-end on one scene.
2. The current main scene is internally consistent in coordinate space.
3. The current reportable-size boundary is `0.5 m`.
4. The current quadtree and correlation-clustering settings are usable as a baseline.
5. The existing volume stage is good enough to serve as the deterministic core.

What is still missing for a V2 paper:

1. true physical-scale multi-scale tiling;
2. object-level manual detection ground truth for the current scene;
3. explicit cross-scale fusion evidence;
4. external 3D validation for the volume module;
5. a frozen evidence package before manuscript writing.

## 5. Recommended V2 Writing Order Later

1. freeze baseline;
2. write data and coordinate section;
3. write physical-scale segmentation section;
4. write fusion section;
5. write 2D-3D association section;
6. write 2.5D volume section;
7. write external validation section;
8. write results and discussion only after the evidence is frozen.


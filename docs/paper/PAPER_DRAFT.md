Status: DRAFT -- English methodology manuscript based on established experimental evidence

# 3 Methodology

## 3.1 Overall framework

This study developed a physical-scale-aware workflow for estimating the observable volumes of surface-visible rocks from a UAV-derived digital orthophoto map (DOM) and associated point-cloud data. The workflow integrates image-based instance delineation with ground-referenced geometric reconstruction and a learned shape-aware correction. It comprises two linked components. First, the study-area DOM is processed through multi-scale instance segmentation, within-scale duplicate resolution across overlapping tiles, and cross-scale cascade deduplication to generate a non-redundant inventory of candidate rock instances. Each instance is then associated with local point-cloud observations, screened using predefined three-dimensional (3D) quality criteria, and reconstructed as a ground-referenced two-and-a-half-dimensional (2.5D) surface. Second, an external rock-mesh dataset with known reference volumes is used to develop a shape-aware correction model for the raw 2.5D volume.

The processing chain is summarized as follows:

$$
\mathrm{DOM}
\rightarrow
\mathrm{multi\text{-}scale\ instance\ segmentation}
\rightarrow
\mathrm{within\text{-}scale\ duplicate\ resolution}
\rightarrow
\mathrm{cross\text{-}scale\ cascade\ deduplication}
\rightarrow
\mathrm{2D\text{-}3D\ association\ and\ screening}
\rightarrow
\mathrm{ground\text{-}referenced\ 2.5D\ reconstruction}
\rightarrow
\mathrm{shape\ descriptor\ extraction}
\rightarrow
\mathrm{learned\ correction}
\rightarrow
\mathrm{rock\ volume\ estimation}.
$$

The DOM ground sampling distance (GSD) was $0.01\ \mathrm{m/pixel}$. This quantity characterizes the spatial sampling of the orthophoto and is distinct from point-cloud spacing and from the rasterization resolution used for 2.5D reconstruction. Local point-cloud spacing statistics were approximately $6.00$--$6.40\ \mathrm{mm}$ at the 90th percentile in the horizontal plane and $8.54$--$8.60\ \mathrm{mm}$ in three dimensions. Based on these input-data scales, a $0.01\ \mathrm{m}$ operational grid was used for 2.5D surface reconstruction. The GSD, point-cloud spacing, and 2.5D grid resolution therefore refer, respectively, to image sampling, point-cloud sampling, and an analysis parameter.

For a rock with a valid ground-referenced 2.5D surface, the correction target in the external mesh dataset was defined as

$$
y_{\mathrm{ratio}} =
\frac{V_{\mathrm{true}}}{V_{2.5D}},
$$

where $V_{\mathrm{true}}$ is the reference mesh volume and $V_{2.5D}$ is the raw ground-referenced 2.5D volume. A LightGBM-based shape-aware regression model predicts $y_{\mathrm{pred}}$ from 12 geometric descriptors. The corrected volume is then calculated as

$$
V_{\mathrm{pred}} =
V_{2.5D} \times y_{\mathrm{pred}}.
$$

The ground reference converts absolute point elevations to local ground-relative heights; it does not reconstruct occluded or buried rock geometry. Accordingly, the resulting estimates represent the observable, ground-referenced component of rock geometry. External mesh validation and real-mine application are treated separately: the former evaluates correction performance against known mesh volumes, whereas the latter evaluates operational applicability to selected real-mine instances and does not establish per-rock real-mine volume accuracy.

## 3.2 Multi-scale DOM instance segmentation

Rock footprints in the study area span a wide range of physical sizes. A single image context can therefore be unsuitable for all targets: a broad window supplies contextual information for large rocks, whereas a smaller physical window increases the effective representation of small rocks. The DOM was consequently partitioned at three physical tile coverages: coarse ($10.24\ \mathrm{m}$), medium ($5.12\ \mathrm{m}$), and fine ($2.56\ \mathrm{m}$). The three tile types were derived from the same DOM and were mapped to a common $1024\times1024$ network input. The medium and fine tiles were resampled by factors of two and four, respectively, so that the scale definition remained tied to physical ground coverage rather than being defined only by input-pixel dimensions.

At each scale, tiles overlapped by $20\%$ to reduce missed observations near tile boundaries. Instance segmentation was applied independently to every tile, using a detection-confidence threshold of $0.25$. Each retained candidate carried its mask, bounding box, centroid, confidence, boundary-completeness measure, scale label, and footprint attributes. The multi-scale stage therefore yields candidate rock instances rather than a final rock inventory. Duplicate observations arising from tile overlap and from repeated observations at different physical scales are resolved in the subsequent stages.

The physical-scale hierarchy is intended to link the image observation window to the expected footprint size of a rock. It does not imply that the three scales provide independent accuracy measurements, nor does it itself constitute a segmentation-accuracy assessment. Quantitative counts of raw candidates and retained instances are reported in the Results section; independent precision, recall, and mean average precision measurements require a separately annotated reference dataset and are not reported here.

The corresponding raw-detection, pooled-inventory, and final-inventory outcomes are evaluated in Section 4.1, where inventory scale is kept distinct from segmentation accuracy.

## 3.3 Duplicate resolution and cascade deduplication

#### Within-scale duplicate resolution

Within-scale duplicate resolution addresses repeated observations of the same rock in overlapping tiles at the same physical scale. It does not merge separate rocks detected within a single tile. For each scale, a grid-based spatial index is first used to identify candidate pairs whose bounding boxes occupy common spatial neighborhoods. Bounding-box intersection over union (IoU) is then used as a computational pre-filter. For two candidate instances $i$ and $j$, a pair is retained for detailed comparison only when

$$
\mathrm{IoU}_{\mathrm{bbox}}(i,j) \geq 0.05
$$

and its mask overlap is non-zero. The mask IoU, which measures the actual overlap of the two instance masks, is defined as

$$
\mathrm{IoU}_{\mathrm{mask}}(i,j) =
\frac{\left|M_i \cap M_j\right|}
{\left|M_i \cup M_j\right|},
$$

where $M_i$ and $M_j$ are the binary masks of the two candidate instances. Thus, bounding-box IoU is used to avoid unnecessary mask comparisons, whereas mask IoU contributes to the duplicate-similarity score.

For each retained pair, a weighted fusion score is calculated as

$$
S_{\mathrm{w}} =
0.30\,S_{\mathrm{m}}
+ 0.20\,S_{\mathrm{c}}
+ 0.20\,S_{\mathrm{a}}
+ 0.15\,S_{\mathrm{b}}
+ 0.15\,S_{\mathrm{p}},
$$

where $S_{\mathrm{m}}=\mathrm{IoU}_{\mathrm{mask}}$ is mask-overlap similarity, $S_{\mathrm{c}}$ is centroid proximity, $S_{\mathrm{a}}$ is footprint-area similarity, $S_{\mathrm{b}}$ is mean boundary completeness, and $S_{\mathrm{p}}$ is mean detection confidence. Centroid proximity is represented by a Gaussian score,

$$
S_{\mathrm{c}} =
\exp\left(-\frac{d_{\mathrm{c}}^2}{2\sigma^2}\right),
$$

where $d_{\mathrm{c}}$ is the centroid distance and $\sigma=50$ pixels. The area-similarity term is

$$
S_{\mathrm{a}} =
\frac{\min(A_i,A_j)}{\max(A_i,A_j)},
$$

where $A_i$ and $A_j$ are the two mask areas. The terms $S_{\mathrm{b}}$ and $S_{\mathrm{p}}$ are, respectively, the arithmetic means of boundary completeness and detection confidence for the two instances. The coefficients were fixed in the implemented processing framework. They are reported as predefined operational weights, not as independently optimized or validated weights.

Pairs with $S_{\mathrm{w}}\geq0.50$ are connected into duplicate groups. For each group, one representative instance is retained according to the quality score

$$
Q_i = c_i \times b_i,
$$

where $c_i$ is the detection confidence and $b_i$ is the boundary-completeness score of instance $i$. This best-mask representative strategy preferentially retains the observation with higher confidence and a more complete boundary, rather than constructing an unconstrained union of masks. The output is a set of within-scale unique instance records that is subsequently supplied to cross-scale duplicate resolution. The procedure defines how overlap-related duplicates are processed; its duplicate-removal accuracy has not been independently quantified against manually annotated duplicate pairs.

#### Cross-scale cascade deduplication

Within-scale duplicate resolution and cross-scale duplicate resolution address different sources of repeated observations. The former operates among overlapping tiles at one physical scale. The latter operates among instances from different physical scales that may correspond to the same rock. Although a generic score-based cross-scale fusion procedure was implemented, the final inventory was produced with a size-aware cascade deduplication strategy rather than direct cross-scale mask fusion.

Cross-scale candidate pairs are first identified through spatial lookup and are retained only when all of the following compatibility conditions are met:

$$
\mathrm{IoU}_{\mathrm{bbox}}(i,j) \geq 0.05,
$$

$$
\mathrm{IoU}_{\mathrm{mask}}(i,j) > 0,
$$

$$
r_d(i,j) =
\frac{\min(d_{\mathrm{eq},i},d_{\mathrm{eq},j})}
{\max(d_{\mathrm{eq},i},d_{\mathrm{eq},j})}
\geq0.30,
$$

and

$$
d_{\mathrm{c}}(i,j) \leq
\max\left(r_i,r_j\right),
$$

where $d_{\mathrm{c}}(i,j)$ is the centroid distance, $r_i=d_{\mathrm{eq},i}/2$ and $r_j=d_{\mathrm{eq},j}/2$ are the footprint-equivalent radii, and $d_{\mathrm{eq},i}$ and $d_{\mathrm{eq},j}$ are the corresponding footprint-equivalent diameters. The centroid distances and radii in this comparison are expressed in DOM-pixel coordinates. For a binary DOM mask containing $N_{\mathrm{pixel}}$ pixels with GSD $g$, the footprint area and equivalent diameter are

$$
A=N_{\mathrm{pixel}}g^2,
$$

and

$$
d_{\mathrm{eq}} =
2\sqrt{\frac{A}{\pi}} =
2\sqrt{\frac{N_{\mathrm{pixel}}g^2}{\pi}}.
$$

The equivalent diameter is a two-dimensional footprint-based quantity; it is not a directly measured three-dimensional particle diameter. The spatial, mask, size, and centroid constraints reduce the likelihood that adjacent but physically distinct rocks are grouped as duplicates, although the corresponding false-merge and missed-merge rates were not independently measured.

For each cross-scale duplicate group, the preferred observation scale is selected from the largest footprint-equivalent diameter in the group:

$$
\mathrm{primary\ scale} =
\begin{cases}
\mathrm{fine}, & d_{\mathrm{eq}} < 0.30\ \mathrm{m}, \\
\mathrm{medium}, & 0.30\ \mathrm{m} \leq d_{\mathrm{eq}} < 0.50\ \mathrm{m}, \\
\mathrm{coarse}, & d_{\mathrm{eq}} \geq 0.50\ \mathrm{m}.
\end{cases}
$$

The instance with the highest $Q_i$ is retained among members from the preferred scale. If no member from that scale is present, the instance with the highest $Q_i$ in the full group is retained. This cascade design retains one observed mask rather than taking a geometric union across scales, while assigning the preferred observation scale according to estimated footprint size. The resulting cross-scale unique records form the DOM-based candidate inventory for subsequent 2D--3D association and geometric screening. The procedure specifies a reproducible duplicate-handling rule; it does not provide an independent estimate of cross-scale fusion accuracy.

Section 4.1 reports the resulting inventory counts and retained-scale composition, while keeping the absence of an independently labelled duplicate benchmark explicit.

## 3.4 2D--3D association, ground reference and 2.5D reconstruction

The DOM instance records are associated with point-cloud observations before surface reconstruction. For each candidate instance, its image-space footprint and spatial extent are used to query nearby point-cloud observations through a reusable spatial index. This step produces a local point set for the candidate and retains the correspondence between the two-dimensional (2D) instance and its three-dimensional observations. The association stage is therefore a spatial retrieval operation; it does not infer geometry that is absent from the point cloud.

The retrieved points are screened using ground-relative and elevation-distribution criteria. A candidate is eligible for subsequent geometric processing only when it contains at least $60$ candidate points, has a global elevation range of at least $0.18\ \mathrm{m}$, has a 90th-percentile ground-relative height of at least $0.12\ \mathrm{m}$, and has an elevated-point ratio of at least $0.20$. A point is classified as elevated when its ground-relative height is at least $0.08\ \mathrm{m}$. These fixed criteria are applied uniformly to all instances. The resulting screen separates candidates with insufficient point support or insufficient above-ground relief from those eligible for 2.5D reconstruction; it is a quality gate rather than an independently validated measurement of association accuracy.

To obtain a local ground reference, finite point-cloud observations are subsampled at every 100th point and assigned to a scene-level grid with $0.5\ \mathrm{m}$ cell spacing. For each cell containing at least three points, the fifth percentile of elevation is used as the ground estimate. Missing cells are filled from neighboring valid ground estimates. This GroundDEM supplies a local reference elevation for converting absolute point elevations to relative heights. It is not intended to recover ground beneath an occluded rock, reconstruct buried geometry, or provide an independently validated absolute DEM accuracy.

#### Ground-referenced 2.5D surface reconstruction

For every candidate passing the 3D quality gate, absolute point elevation is converted to ground-relative height using the local GroundDEM:

$$
h(x,y)=z(x,y)-z_{\mathrm{ground}}(x,y),
$$

where $z(x,y)$ is the observed point elevation and $z_{\mathrm{ground}}(x,y)$ is the corresponding local ground reference. The irregular point observations are rasterized on a square operational grid with

$$
\Delta x=\Delta y=0.01\ \mathrm{m}.
$$

For each occupied grid cell, the maximum observed ground-relative height is retained as the surface value. Let $\Omega$ denote the set of occupied cells and $h_i$ the retained height in cell $i$. The observable 2.5D volume is calculated as

$$
V_{2.5D}=
\sum_{i\in\Omega}h_i\Delta x\Delta y.
$$

This representation is 2.5D because it assigns one height value to each occupied horizontal cell. It provides a ground-referenced integral of the observed elevated surface, not a direct measurement of the complete physical volume of a rock. If no valid occupied surface remains after association and filtering, no 2.5D volume is produced for that candidate.

Section 4.2 reports the association and quality-screening outcomes, while Sections 4.3--4.5 report the corresponding external-validation and real-mine volume distributions.

## 3.5 Shape-aware descriptor, scale adaptation and volume correction

The shape-aware correction uses a fixed ordered descriptor comprising five footprint descriptors, five height-distribution descriptors, and two volume-shape ratios. Let $A$ be footprint area, $P$ its perimeter, $L$ and $W$ the major and minor footprint dimensions, and $H$ the maximum observed height. The canonical descriptor is:

1. Circularity:

$$
C=\min\left(\frac{4\pi A}{P^2},1\right).
$$

2. Aspect ratio:

$$
AR=\frac{L}{W}.
$$

3. Solidity:

$$
\mathrm{solidity}=\min\left(\frac{A}{A_{\mathrm{convex}}},1\right),
$$

where $A_{\mathrm{convex}}$ is the area of the footprint convex hull.

4. Compactness:

$$
\mathrm{compactness}=\frac{P}{\sqrt{A}}.
$$

5. Equivalent-diameter ratio:

$$
\mathrm{eq\_diam\_ratio}=\frac{\sqrt{4A/\pi}}{L}.
$$

The height-distribution descriptors are

$$
H_{\mathrm{mean,norm}}=\frac{H_{\mathrm{mean}}}{H},
\qquad
H_{\mathrm{std,norm}}=\frac{H_{\mathrm{std}}}{H},
$$

$$
H_{\mathrm{p25,norm}}=\frac{H_{\mathrm{p25}}}{H},
\qquad
H_{\mathrm{p75,norm}}=\frac{H_{\mathrm{p75}}}{H}.
$$

The tenth descriptor follows the established training and inference definition:

$$
H_{\mathrm{skew,norm}}=H_{\mathrm{skew}}.
$$

Although its implementation name is `H_skew_norm`, this descriptor is the raw height skewness and is not divided by $H$. The remaining two descriptors are

$$
\mathrm{fill\_ratio}=\frac{V_{2.5D}}{V_{\mathrm{box}}},
\qquad
\mathrm{ellipsoid\_ratio}=\frac{V_{2.5D}}{V_{\mathrm{ellipsoid}}},
$$

where $V_{\mathrm{box}}$ and $V_{\mathrm{ellipsoid}}$ are reference volumes derived from the footprint and height geometry. The features are supplied to the regression model in the fixed order $C$, $AR$, solidity, compactness, `eq_diam_ratio`, `H_mean_norm`, `H_std_norm`, `H_p25_norm`, `H_p75_norm`, `H_skew_norm`, `fill_ratio`, and `ellipsoid_ratio`. The feature names in code formatting identify the implementation contract; the mathematical quantities in the equations define the corresponding variables.

The external mesh training target is the ratio between known reference volume and the observable 2.5D volume:

$$
y_{\mathrm{ratio}}=
\frac{V_{\mathrm{true}}}{V_{2.5D}}.
$$

A LightGBM regression model predicts $y_{\mathrm{pred}}$ from the 12-feature descriptor. The final corrected estimate is

$$
V_{\mathrm{pred}}=
V_{2.5D}\times y_{\mathrm{pred}}.
$$

Thus, the model is a correction model for the ground-referenced 2.5D volume rather than a direct model that predicts volume without an explicit geometric base.

#### Resolution and scale adaptation

The correction model was developed using an external rock-mesh dataset with known reference volumes, because such reference volumes are not available for individual rocks in the study area. The external meshes were first processed at $0.5\ \mathrm{mm}$ for methodological validation. Direct rasterization of the original mesh scale at the mine-operational $10\ \mathrm{mm}$ grid was not used for training because the resulting surface availability was insufficient.

The choice of the mine-operational grid was based on the relation between the study-area DOM GSD, local point-cloud sampling, and the desired surface-reconstruction scale. The DOM GSD was $0.01\ \mathrm{m/pixel}$, while the observed point-cloud spacing was approximately $6.00$--$6.40\ \mathrm{mm}$ in the horizontal plane and $8.54$--$8.60\ \mathrm{mm}$ in three dimensions at the reported 90th percentiles. The operational grid was therefore set to $10\ \mathrm{mm}$, while retaining the distinction between input sampling and rasterization resolution.

To reduce the footprint-scale discrepancy between the external meshes and the study-area rocks, a uniform geometric scale factor was derived from an independent comparison of footprint-equivalent diameter distributions:

$$
s=82.737840.
$$

The factor was determined without using $V_{\mathrm{true}}$, $y_{\mathrm{ratio}}$, model predictions, or test errors. The external geometries were then scaled uniformly and rasterized at $10\ \mathrm{mm}$ to form a resolution- and scale-matched training dataset. The scale transformation is a controlled domain-adaptation step that supports operational transfer; it does not by itself establish the absolute validity of external-mesh geometry as a representation of every study-area rock.

The held-out external correction metrics are reported in Section 4.3, and the resolution and scale-adaptation evidence is reported separately in Section 4.4.

## 3.6 Representative real-mine application

The real-mine application was restricted to accepted instances from the DOM-based inventory. A fixed representative sample was selected before volume inference using `stratified_quantile_systematic` sampling based only on the footprint-equivalent diameter. The accepted population was divided by its empirical diameter quantiles into six strata: S1 ($P0$--$P10$), S2 ($P10$--$P25$), S3 ($P25$--$P50$), S4 ($P50$--$P75$), S5 ($P75$--$P90$), and S6 ($P90$--$P100$). The target sample allocations were 400, 600, 1,000, 1,000, 600, and 400, respectively.

Within each stratum, instances were sorted in ascending order of footprint-equivalent diameter, with the unique instance identifier used as a deterministic secondary key. Fixed systematic positions were then selected across the sorted stratum. No volume, correction ratio, model prediction, error, or 3D reference quantity was used for sample selection. This design preserves coverage across the full diameter range, including the largest-rock stratum, while avoiding random-number dependence and repeated sampling.

For each selected instance, the processing sequence was: DOM mask and spatial extent, existing 2D--3D association, 3D quality filtering, ground-referenced $10\ \mathrm{mm}$ 2.5D reconstruction, canonical 12-feature extraction, LightGBM correction, and calculation of $V_{\mathrm{pred}}$. The application is a representative sample-level analysis of observable rock volumes; it is not a volume calculation for the full accepted inventory and cannot establish real-mine absolute volume accuracy without per-rock reference volumes.

The corresponding completion counts, failure modes, and successful-estimate distributions are reported in Section 4.5; they are not part of the methodological definition of the sample.

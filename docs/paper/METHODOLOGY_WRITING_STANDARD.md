# Methodology Writing Standard

## Status and scope

This document defines the manuscript-writing standard for the current study. It is a terminology and claim-discipline guide, not an experimental record, implementation manual, or source of new results. It applies to the complete Methods chapter, Sections 3.1--3.6, and should guide subsequent manuscript sections.

## Core method statement

The paper presents a physical-scale-aware workflow that links multi-scale DOM instance segmentation, duplicate resolution, 2D--3D association, ground-referenced 2.5D surface reconstruction, and shape-aware correction to estimate observable rock volumes across a wide footprint-size range.

## Terminology ledger

| Concept | First-use manuscript form | Use thereafter | Avoid in manuscript prose |
| --- | --- | --- | --- |
| Study orthophoto | UAV-derived digital orthophoto map (DOM) covering the study area | DOM | Internal DOM identifiers |
| Point clouds | associated point-cloud data from two complementary survey blocks | point-cloud data | Internal point-cloud block names |
| Within-scale overlap handling | within-scale duplicate resolution across overlapping tiles | within-scale duplicate resolution | same-tile fusion |
| Cross-scale overlap handling | size-aware cross-scale cascade deduplication | cascade deduplication | generic cross-scale fusion as the final inventory path |
| Representative selection | best-mask representative strategy | representative instance | best rock |
| Mesh dataset | external rock-mesh dataset with known reference volumes | external mesh dataset | Internal dataset or subset identifiers |
| Final model | LightGBM-based shape-aware correction model | shape-aware correction model | frozen model |
| Mine estimate | observable, ground-referenced rock-volume estimate | corrected volume estimate | verified complete rock volume |

## Methods--results separation

Methods report the input data, processing sequence, equations, fixed parameters, and explicit boundaries. Results report outcome counts, completion rates, distribution statistics, and test metrics. Internal file paths, code names, status labels, checkpoints, and processing batches do not belong in manuscript prose.

## Notation and units

All mathematical notation uses LaTeX. Use $V_{2.5D}$, $V_{\mathrm{true}}$, $V_{\mathrm{pred}}$, and $y_{\mathrm{pred}}$ in equations; use code-style identifiers only when discussing an implementation contract in supplementary documentation. Express units as $m$, $m^2$, $m^3$, $mm$, and $mm^3$. Do not use Unicode superscripts or unformatted underscores in mathematical variables.

The canonical descriptor names are circularity, aspect ratio, solidity, compactness, equivalent-diameter ratio, normalized mean height, normalized height standard deviation, normalized height P25, normalized height P75, height skewness, fill ratio, and ellipsoid ratio. In the implementation, `H_skew_norm` is defined as the raw height skewness, $H_{\mathrm{skew,norm}}=H_{\mathrm{skew}}$, and is not divided by total height.

## Claim discipline and scientific boundary

The DOM GSD, local point-cloud spacing, and 2.5D grid resolution are distinct quantities. The ground reference provides local relative heights and does not recover occluded or buried geometry. The 2.5D and corrected volumes therefore describe observable, ground-referenced rock geometry.

The external-mesh test evaluates the shape-aware correction model against known mesh volumes. Real-mine application demonstrates operational execution on a prescribed representative sample, not per-rock real-mine absolute volume accuracy. The manuscript must not claim independently validated segmentation accuracy, duplicate-resolution accuracy, 2D--3D association accuracy, absolute DEM accuracy, hidden-geometry recovery, or real-mine volume accuracy unless separate reference evidence is supplied.

## Fixed fusion definitions

Within-scale duplicate resolution uses bounding-box IoU for candidate pre-filtering, mask IoU for actual mask-overlap similarity, and the following fixed weighted score:

$$
S_{\mathrm{w}} =
0.30\,S_{\mathrm{m}}
+ 0.20\,S_{\mathrm{c}}
+ 0.20\,S_{\mathrm{a}}
+ 0.15\,S_{\mathrm{b}}
+ 0.15\,S_{\mathrm{p}}.
$$

The terms respectively denote mask overlap, centroid proximity, footprint-area similarity, mean boundary completeness, and mean detection confidence. The weights are predefined operational settings, not independently optimized coefficients. For a duplicate group, the retained representative is selected using

$$
Q_i=c_i\times b_i,
$$

where $c_i$ is detection confidence and $b_i$ is boundary completeness.

Cross-scale cascade deduplication uses spatial overlap, non-zero mask overlap, footprint-equivalent diameter compatibility, centroid proximity, and a primary scale selected by footprint-equivalent diameter. Equivalent diameter is a two-dimensional footprint measure:

$$
d_{\mathrm{eq}}=2\sqrt{\frac{A}{\pi}}.
$$

## Writing implementation notes

Describe the method as a computational procedure and provide sufficient definitions for reproduction. Reserve source-code paths, functions, configuration-object names, and output filenames for supplementary materials or a code-availability statement. Do not use internal status labels such as QC outcomes or evidence-map labels in the manuscript body; convert limitations into direct scientific statements.

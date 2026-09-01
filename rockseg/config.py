"""Configuration for physical-scale-driven multi-scale rock detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ScaleLevel(str, Enum):
    """Physical scale levels for multi-scale tiling."""

    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"


@dataclass(frozen=True)
class ScaleConfig:
    """Configuration for a single physical scale.

    The scale variable is the ground coverage of one network input,
    not merely the YOLO ``imgsz``.  For ``N=1024`` and ``GSD=0.01 m/pixel``:

    * Coarse: 10.24 m coverage  (1024 px in DOM, no resampling)
    * Medium:  5.12 m coverage  ( 512 px in DOM, 2x upscale)
    * Fine:    2.56 m coverage  ( 256 px in DOM, 4x upscale)
    """

    level: ScaleLevel
    physical_width_m: float
    network_input_size: int = 1024

    def dom_pixel_size(self, gsd: float) -> int:
        """Original DOM pixel dimension for this scale's physical width."""
        return int(round(self.physical_width_m / gsd))

    def resample_factor(self, gsd: float) -> float:
        """Resample factor from original DOM tile to network input."""
        return self.network_input_size / self.dom_pixel_size(gsd)


@dataclass
class FusionWeights:
    """Weights for the multi-feature fusion score.

    ``S = w_iou * IoU + w_centroid * S_c + w_area * r_A + w_boundary * B + w_conf * C``
    """

    iou: float = 0.30
    centroid: float = 0.20
    area_ratio: float = 0.20
    boundary: float = 0.15
    confidence: float = 0.15


@dataclass
class PipelineConfig:
    """Main configuration for the multi-scale rock detection pipeline.

    Defaults reflect confirmed decisions:
    * 3 scales (coarse / medium / fine)
    * overlap = 20 %
    * canonical mask = best mask (not union)
    * thresholds = conservative
    """

    # ---- input data paths ----
    dom_path: Path
    model_path: Path
    output_dir: Path = Path("./output")

    # ---- DOM spatial resolution ----
    gsd: float = 0.01  # metres per pixel

    # ---- network input ----
    network_input_size: int = 1024

    # ---- tiling ----
    overlap_ratio: float = 0.20

    # ---- three physical scales (N=1024, GSD=0.01 m/pixel) ----
    scales: list[ScaleConfig] = field(default_factory=lambda: [
        ScaleConfig(ScaleLevel.COARSE, 10.24),
        ScaleConfig(ScaleLevel.MEDIUM, 5.12),
        ScaleConfig(ScaleLevel.FINE, 2.56),
    ])

    # ---- segmentation ----
    confidence_threshold: float = 0.25

    # ---- fusion parameters ----
    fusion_weights: FusionWeights = field(default_factory=FusionWeights)
    within_scale_threshold: float = 0.50
    cross_scale_threshold: float = 0.55
    use_cascade: bool = False
    centroid_sigma_px: float = 50.0  # pixels (0.5m at GSD=0.01), for Gaussian distance normalisation

    # ---- canonical mask strategy ----
    # "best_mask"  – pick the instance with highest confidence * boundary_completeness
    # "union"      – controlled union of all merged masks
    canonical_strategy: str = "best_mask"

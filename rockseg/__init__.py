"""RockSeg: physical-scale-driven multi-scale rock detection and fusion.

Public API
-----------
``PipelineConfig``
    Main configuration dataclass.
``ScaleConfig``
    Per-scale physical-coverage definition.
``ScaleLevel``
    Enum: coarse / medium / fine.
``RockInstance``
    Detected rock with mask and metadata.
``TileMetadata``
    Tile provenance record.
``MultiScaleRockDetectionPipeline``
    End-to-end orchestrator.
``create_default_config``
    Convenience factory with 3-scale defaults.
"""

from __future__ import annotations

from .config import FusionWeights, PipelineConfig, ScaleConfig, ScaleLevel
from .models import RockInstance, TileMetadata
from .pipeline import MultiScaleRockDetectionPipeline, create_default_config

__all__ = [
    "FusionWeights",
    "PipelineConfig",
    "ScaleConfig",
    "ScaleLevel",
    "RockInstance",
    "TileMetadata",
    "MultiScaleRockDetectionPipeline",
    "create_default_config",
]

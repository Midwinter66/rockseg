"""Data models for tiles and rock instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TileMetadata:
    """Metadata for a single DOM tile.

    All coordinates are in the **original DOM pixel coordinate system**.
    """

    tile_id: str
    scale_level: str          # ScaleLevel value
    x_start: int              # pixel offset in DOM (top-left x)
    y_start: int              # pixel offset in DOM (top-left y)
    width: int                # pixel width in original DOM
    height: int               # pixel height in original DOM
    resample_factor: float    # factor to resize to network input

    @property
    def x_end(self) -> int:
        return self.x_start + self.width

    @property
    def y_end(self) -> int:
        return self.y_start + self.height


@dataclass
class RockInstance:
    """A detected rock instance.

    The mask is stored **locally** within its bounding box to save memory.
    The bounding box is in **DOM global pixel coordinates**.
    """

    instance_id: str
    mask_local: np.ndarray               # boolean mask, shape (bbox_h, bbox_w)
    bbox: tuple[int, int, int, int]     # (x_min, y_min, x_max, y_max) in DOM coords
    confidence: float
    scale_level: str
    tile_id: str
    boundary_completeness: float = 1.0   # 1.0 = fully inside tile

    @property
    def area(self) -> int:
        """Pixel area (True count in mask)."""
        return int(self.mask_local.sum())

    @property
    def centroid(self) -> tuple[float, float]:
        """Centroid (x, y) in DOM global pixel coordinates."""
        ys, xs = np.where(self.mask_local)
        if len(xs) == 0:
            cx = (self.bbox[0] + self.bbox[2]) / 2.0
            cy = (self.bbox[1] + self.bbox[3]) / 2.0
        else:
            cx = float(xs.mean()) + self.bbox[0]
            cy = float(ys.mean()) + self.bbox[1]
        return (cx, cy)

    def quality_score(self) -> float:
        """Overall quality score for canonical-mask selection.

        Higher is better.  Used when the strategy is ``"best_mask"``.
        """
        return self.confidence * self.boundary_completeness

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict (excludes the mask array)."""
        cx, cy = self.centroid
        return {
            "instance_id": self.instance_id,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "area": self.area,
            "centroid": [cx, cy],
            "scale_level": self.scale_level,
            "boundary_completeness": self.boundary_completeness,
        }

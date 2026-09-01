"""YOLO11m-seg inference wrapper.

A single :class:`SegmentationPredictor` wraps the YOLO model and is used
for **all** scales.  The same backbone works at different physical
coverage ranges — that is the core idea of the multi-scale approach.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .models import RockInstance, TileMetadata


class SegmentationPredictor:
    """Wrapper for YOLO11m-seg model inference.

    Parameters
    ----------
    model_path
        Path to the ``.pt`` weights file.
    network_input_size
        Square input dimension (default 1024).
    """

    def __init__(
        self,
        model_path: str,
        network_input_size: int = 1024,
        device: str = "",
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.network_input_size = network_input_size
        self.device = device or None

    def predict(
        self,
        image: np.ndarray,
        confidence: float = 0.25,
    ) -> list[dict[str, Any]]:
        """Run YOLO segmentation on an image.

        Parameters
        ----------
        image
            RGB image of shape ``(H, W, 3)``, dtype ``uint8``.
        confidence
            Confidence threshold for detections.

        Returns
        -------
        list[dict]
            Each dict has keys: ``mask`` (bool array H×W),
            ``bbox`` (x1, y1, x2, y2), ``confidence`` (float).
        """
        import gc

        results = self.model.predict(
            image,
            imgsz=self.network_input_size,
            conf=confidence,
            device=self.device,
            verbose=False,
        )

        predictions: list[dict[str, Any]] = []

        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()       # (N, H, W)
            boxes = result.boxes
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()

            for i in range(len(masks)):
                predictions.append({
                    "mask": masks[i].astype(bool),
                    "bbox": tuple(int(v) for v in xyxy[i]),
                    "confidence": float(confs[i]),
                })

        del results
        gc.collect()

        return predictions


def predictions_to_instances(
    predictions: list[dict[str, Any]],
    tile: TileMetadata,
) -> list[RockInstance]:
    """Convert YOLO predictions to :class:`RockInstance` objects.

    For each prediction:

    1.  Resize the mask from network-input coordinates back to the
        original tile pixel size (nearest-neighbour to keep it binary).
    2.  Crop the mask to its tight bounding box and convert the
        bounding box to DOM global coordinates.
    3.  Compute boundary completeness (how truncated the rock is by
        the tile edge).
    """
    instances: list[RockInstance] = []

    for i, pred in enumerate(predictions):
        mask_net = pred["mask"]  # (network_input_size, network_input_size)

        # Resize back to original tile pixel size
        mask_tile = cv2.resize(
            mask_net.astype(np.uint8),
            (tile.width, tile.height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        # Tight bounding box in tile-local coordinates
        ys, xs = np.where(mask_tile)
        if len(xs) == 0:
            continue

        x1_local = int(xs.min())
        y1_local = int(ys.min())
        x2_local = int(xs.max()) + 1
        y2_local = int(ys.max()) + 1

        # Convert to DOM global coordinates
        x1_global = tile.x_start + x1_local
        y1_global = tile.y_start + y1_local
        x2_global = tile.x_start + x2_local
        y2_global = tile.y_start + y2_local

        # Crop mask to bbox (copy to release the full tile-size array)
        mask_local = mask_tile[y1_local:y2_local, x1_local:x2_local].copy()

        boundary_completeness = _compute_boundary_completeness(
            x1_local, y1_local, x2_local, y2_local, tile,
        )

        instance_id = f"{tile.tile_id}_{i:03d}"

        instances.append(RockInstance(
            instance_id=instance_id,
            mask_local=mask_local,
            bbox=(x1_global, y1_global, x2_global, y2_global),
            confidence=pred["confidence"],
            scale_level=tile.scale_level,
            tile_id=tile.tile_id,
            boundary_completeness=boundary_completeness,
        ))

    return instances


def _compute_boundary_completeness(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    tile: TileMetadata,
    margin: int = 3,
) -> float:
    """Score how completely the rock appears within the tile.

    A rock whose bounding box touches the tile edge is likely truncated.
    Returns ``1.0`` for fully contained, decreasing with each edge touched.

    Parameters
    ----------
    margin
        Pixels of tolerance before considering the bbox "touching" an edge.
    """
    touches_left = x_min < margin
    touches_right = x_max > tile.width - margin
    touches_top = y_min < margin
    touches_bottom = y_max > tile.height - margin

    touches = sum((touches_left, touches_right, touches_top, touches_bottom))

    penalty = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.1, 4: 0.0}
    return penalty.get(touches, 0.0)

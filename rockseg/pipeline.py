"""Main pipeline orchestrator for multi-scale rock detection and fusion.

Pipeline steps (per scale, then cross-scale):

1.  For each physical scale:
    a.  Generate physical-scale-aware tiles from the DOM.
    b.  Run YOLO11m-seg on each tile.
    c.  Fuse duplicates **within** this scale (boundary-aware).
    d.  Free raw instances to control memory.
2.  Fuse duplicates **across** scales (cross-scale).
3.  Output unique rock instances with masks and provenance.

Memory strategy: each scale is processed independently — raw instances
are freed after within-scale fusion, so only the much-smaller fused set
accumulates across scales.

Usage
-----
::

    from rockseg import create_default_config, MultiScaleRockDetectionPipeline

    config = create_default_config("dom.tif", "model.pt")
    pipeline = MultiScaleRockDetectionPipeline(config)
    instances = pipeline.run()
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import PipelineConfig, ScaleConfig, ScaleLevel
from .fusion import cascade_deduplication, cross_scale_fusion, within_scale_fusion
from .models import RockInstance, TileMetadata
from .segmentation import SegmentationPredictor, predictions_to_instances
from .tiling import DOMReader, generate_tiles

logger = logging.getLogger(__name__)


class MultiScaleRockDetectionPipeline:
    """Physical-scale-driven multi-scale rock detection and fusion pipeline.

    Parameters
    ----------
    config
        :class:`PipelineConfig` with all paths, scales, and fusion settings.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.predictor = SegmentationPredictor(
            str(config.model_path),
            config.network_input_size,
        )
        self.max_tiles_per_scale: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[RockInstance]:
        """Execute the full pipeline and return unique rock instances."""
        logger.info("=== Multi-scale rock detection pipeline ===")

        all_fused: list[RockInstance] = []

        with DOMReader(self.config.dom_path) as dom:
            for scale in self.config.scales:
                all_fused.extend(self._process_scale(dom, scale))

        logger.info("Total fused instances across scales: %d", len(all_fused))

        if self.config.use_cascade:
            final = cascade_deduplication(all_fused, self.config)
            logger.info("After cascade deduplication: %d instances", len(final))
        else:
            final = cross_scale_fusion(all_fused, self.config)
            logger.info("After cross-scale fusion: %d instances", len(final))

        # Assign final IDs and save
        self._assign_final_ids(final)
        self._save_results(final)

        return final

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _process_scale(
        self,
        dom: DOMReader,
        scale: ScaleConfig,
    ) -> list[RockInstance]:
        """Process a single scale: tile, segment, fuse, and free memory."""
        tiles = generate_tiles(
            dom_width=dom.width,
            dom_height=dom.height,
            scale=scale,
            gsd=self.config.gsd,
            overlap_ratio=self.config.overlap_ratio,
        )
        if self.max_tiles_per_scale and len(tiles) > self.max_tiles_per_scale:
            tiles = tiles[:self.max_tiles_per_scale]

        logger.info(
            "  %s: %d tiles  (%.2f m, %d px each)",
            scale.level.value,
            len(tiles),
            scale.physical_width_m,
            scale.dom_pixel_size(self.config.gsd),
        )

        # Segmentation
        raw_instances = self._run_segmentation(dom, tiles)
        logger.info("  %s: %d raw instances", scale.level.value, len(raw_instances))

        # Within-scale fusion
        fused = within_scale_fusion(raw_instances, self.config)
        logger.info(
            "  %s: %d -> %d (after fusion)",
            scale.level.value, len(raw_instances), len(fused),
        )

        # Free raw instances and GPU cache
        del raw_instances
        gc.collect()
        self._clear_gpu_cache()

        return fused

    def _run_segmentation(
        self,
        dom: DOMReader,
        tiles: list[TileMetadata],
    ) -> list[RockInstance]:
        """Run YOLO inference on all tiles and convert to instances."""
        all_instances: list[RockInstance] = []

        for idx, tile in enumerate(
            tqdm(tiles, desc=f"Seg-{tiles[0].scale_level}", unit="tile"),
        ):
            image = dom.read_tile(tile, self.config.network_input_size)
            predictions = self.predictor.predict(
                image,
                confidence=self.config.confidence_threshold,
            )
            instances = predictions_to_instances(predictions, tile)
            all_instances.extend(instances)

            del image, predictions, instances

            if (idx + 1) % 50 == 0:
                gc.collect()
                self._clear_gpu_cache()

        return all_instances

    @staticmethod
    def _clear_gpu_cache() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _assign_final_ids(self, instances: list[RockInstance]) -> None:
        """Assign sequential final IDs: rock_00000, rock_00001, ..."""
        for i, inst in enumerate(instances):
            inst.instance_id = f"rock_{i:05d}"

    def _save_results(self, instances: list[RockInstance]) -> None:
        """Save instance metadata and masks to the output directory."""
        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)

        # Save metadata as JSON
        metadata = [inst.to_dict() for inst in instances]
        json_path = out / "rock_instances.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save masks as compressed npz
        masks_data: dict[str, np.ndarray] = {}
        bboxes_data: dict[str, np.ndarray] = {}
        for inst in instances:
            masks_data[inst.instance_id] = inst.mask_local
            bboxes_data[inst.instance_id] = np.array(inst.bbox)

        np.savez_compressed(
            out / "rock_masks.npz",
            **{f"{k}_mask": v for k, v in masks_data.items()},
        )
        np.savez_compressed(
            out / "rock_bboxes.npz",
            **bboxes_data,
        )

        logger.info("Saved %d instances to %s", len(instances), out)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_default_config(
    dom_path: str,
    model_path: str,
    output_dir: str = "./output",
) -> PipelineConfig:
    """Create a default pipeline configuration with 3 scales.

    Defaults (confirmed by user):

    * 3 scales: coarse (10.24 m), medium (5.12 m), fine (2.56 m)
    * N = 1024, GSD = 0.01 m/pixel
    * overlap = 20 %
    * canonical mask = best mask
    * within-scale threshold = 0.50, cross-scale threshold = 0.45
    """
    return PipelineConfig(
        dom_path=Path(dom_path),
        model_path=Path(model_path),
        output_dir=Path(output_dir),
    )

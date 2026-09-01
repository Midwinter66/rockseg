"""Configuration for external volume validation (Experiment E5).

All paths are relative to the project root:
    d:/github_project/image_segment/DOM_Space_message_val
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """All tunable parameters for the volume validation pipeline."""

    # ── Data paths ────────────────────────────────────────────────
    # Directory containing T01/ (and optionally L01/, L02/) subfolders with OBJ files
    data_dir: Path = Path("data/experience_rock")

    # Output directory for results, figures, and cached data
    output_dir: Path = Path("research_v2/volume_validation/output")

    # shapeList.txt filename (metadata file from the dataset)
    shape_list_filename: str = "shapeList.txt"

    # ── Dataset groups ───────────────────────────────────────────
    # Each group: subfolder name, material, fragmentation mechanism
    groups: dict = field(default_factory=lambda: {
        "L01": {
            "subfolder": "L01",
            "material": "L3-6 chondrite (NWA 869)",
            "fragmentation": "hypervelocity impact",
            "n_samples": 386,
        },
        "L02": {
            "subfolder": "L02",
            "material": "L3-6 chondrite (NWA 869)",
            "fragmentation": "explosive charge",
            "n_samples": 403,
        },
        "T01": {
            "subfolder": "T01",
            "material": "tephriphonolite (terrestrial)",
            "fragmentation": "explosive charge",
            "n_samples": 79,
        },
    })

    # ── 2.5D simulation ──────────────────────────────────────────
    # Grid cell size in mm (rocks are 5–24 mm)
    grid_resolution_mm: float = 0.5

    # Gaussian noise added to height measurements (mm, 0 = clean)
    height_noise_std_mm: float = 0.0

    # Fraction of grid cells randomly removed to simulate sparsity (0–1)
    point_sparsity: float = 0.0

    # Number of viewing directions to test per mesh (1 = top-down only)
    n_viewing_directions: int = 1

    # ── Data split ────────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # ── Shape-aware model (LightGBM) ─────────────────────────────
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "mae",
        "num_leaves": 12,
        "learning_rate": 0.02,
        "n_estimators": 500,
        "verbose": -1,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "min_child_samples": 8,
        "reg_alpha": 0.2,
        "reg_lambda": 0.5,
        "min_gain_to_split": 0.0001,
    })

    # ── Processing ───────────────────────────────────────────────
    # Number of parallel workers for mesh processing (0 = auto)
    n_workers: int = 0

    # Whether to cache processed data (mesh volumes + 2.5D surfaces)
    use_cache: bool = True

    # ── Unit conversion ──────────────────────────────────────────
    # Dataset meshes are in mm; convert to m for final reporting
    mesh_unit: str = "mm"
    report_unit: str = "mm"

    def ensure_output_dir(self) -> None:
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "cache").mkdir(exist_ok=True)

    @property
    def cache_path(self) -> Path:
        return self.output_dir / "cache" / "processed_data.npz"

    @property
    def results_path(self) -> Path:
        return self.output_dir / "results"

    @property
    def figures_path(self) -> Path:
        return self.output_dir / "figures"

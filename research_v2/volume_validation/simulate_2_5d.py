"""3D-to-2.5D surface simulation.

Simulates the observation geometry of a drone looking down at a rock resting
on flat ground. The mesh is first oriented to its stable resting pose, then
the visible top surface is rasterised onto a regular XY grid by ray casting.

The output ``Surface2_5D`` contains:

* ``height_map`` — 2-D array of top-surface heights (NaN = no ray hit).
* ``footprint_mask`` — boolean mask of valid cells (the 2-D projection).
* ``ground_z`` — ground reference elevation (0 after orientation).

This simulates exactly what the RockSeg pipeline receives from a DEM-referenced
point cloud: a height function ``z = f(x, y)`` over the rock's footprint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import trimesh

from .config import Config
from .mesh_utils import find_stable_orientation, load_mesh, orient_mesh

logger = logging.getLogger(__name__)


@dataclass
class Surface2_5D:
    """Result of 2.5-D simulation for a single rock fragment."""

    height_map: np.ndarray  # (nx, ny) float, NaN where no ray hit
    footprint_mask: np.ndarray  # (nx, ny) bool
    grid_x: np.ndarray  # (nx,) x-centre coordinates [mm]
    grid_y: np.ndarray  # (ny,) y-centre coordinates [mm]
    cell_size: float  # mm per grid cell
    ground_z: float  # ground reference elevation [mm]
    n_valid_cells: int


def _build_grid(extents: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Create regular grid centres covering the mesh XY extent.

    Parameters
    ----------
    extents : (3,) array — (x_size, y_size, z_size) of the oriented mesh.
    cell_size : grid spacing in the same units.
    """
    pad = cell_size * 0.5
    x_min, x_max = -extents[0] / 2 - pad, extents[0] / 2 + pad
    y_min, y_max = -extents[1] / 2 - pad, extents[1] / 2 + pad

    nx = int(np.ceil((x_max - x_min) / cell_size))
    ny = int(np.ceil((y_max - y_min) / cell_size))

    grid_x = x_min + (np.arange(nx) + 0.5) * cell_size
    grid_y = y_min + (np.arange(ny) + 0.5) * cell_size
    return grid_x, grid_y


def rasterize_top_surface(
    mesh: trimesh.Trimesh,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    cell_size: float,
) -> np.ndarray:
    """Ray-cast downward to find the visible top surface at each grid cell.

    For each (x, y) grid centre, a ray is cast from above the mesh downward
    (−Z direction). The highest intersection is the top-surface height.
    Cells with no hit are set to NaN.

    Returns
    -------
    height_map : (nx, ny) float array
    """
    nx, ny = len(grid_x), len(grid_y)
    z_above = mesh.vertices[:, 2].max() + cell_size * 4

    xx, yy = np.meshgrid(grid_x, grid_y, indexing="ij")
    origins = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        np.full(nx * ny, z_above),
    ])
    directions = np.tile([0.0, 0.0, -1.0], (nx * ny, 1))

    # Ray casting
    locations, index_ray, _ = mesh.ray.intersects_location(origins, directions)

    # Initialise with NaN
    heights = np.full(nx * ny, np.nan)
    if len(locations) > 0:
        z_values = locations[:, 2]
        # Keep the highest hit per ray
        for z, idx in zip(z_values, index_ray):
            if np.isnan(heights[idx]) or z > heights[idx]:
                heights[idx] = z

    return heights.reshape(nx, ny)


def add_height_noise(
    height_map: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian noise to valid (non-NaN) height cells."""
    if noise_std <= 0:
        return height_map.copy()
    noisy = height_map.copy()
    valid = ~np.isnan(noisy)
    noisy[valid] += rng.normal(0.0, noise_std, size=valid.sum())
    return noisy


def apply_sparsity(
    height_map: np.ndarray,
    sparsity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly remove a fraction of valid cells to simulate sparse sampling."""
    if sparsity <= 0:
        return height_map.copy()
    sparse = height_map.copy()
    valid = ~np.isnan(sparse)
    n_valid = valid.sum()
    n_remove = int(n_valid * sparsity)
    if n_remove > 0:
        valid_indices = np.where(valid.ravel())[0]
        remove_indices = rng.choice(valid_indices, size=n_remove, replace=False)
        sparse.ravel()[remove_indices] = np.nan
    return sparse


def simulate_2_5d_surface(
    obj_path,
    config: Config,
    rng: np.random.Generator | None = None,
) -> Surface2_5D:
    """Full 2.5-D simulation pipeline for one mesh.

    Steps
    -----
    1. Load the OBJ mesh.
    2. Find the stable resting orientation.
    3. Orient the mesh (bottom face down, min-Z = 0).
    4. Build a regular XY grid.
    5. Ray-cast downward to get the top-surface height map.
    6. Optionally add height noise and point sparsity.
    7. Compute the footprint mask (valid cells).

    Parameters
    ----------
    obj_path : path to the .obj file.
    config : pipeline configuration.
    rng : optional NumPy random generator for reproducibility.

    Returns
    -------
    Surface2_5D
    """
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    # 1. Load
    mesh = load_mesh(obj_path)

    # 2-3. Orient
    transform = find_stable_orientation(mesh)
    oriented = orient_mesh(mesh, transform)

    # 4. Grid
    extents = oriented.bounding_box.extents
    grid_x, grid_y = _build_grid(extents, config.grid_resolution_mm)

    # 5. Ray-cast
    height_map = rasterize_top_surface(
        oriented, grid_x, grid_y, config.grid_resolution_mm
    )

    # 6. Noise & sparsity
    height_map = add_height_noise(height_map, config.height_noise_std_mm, rng)
    height_map = apply_sparsity(height_map, config.point_sparsity, rng)

    # 7. Footprint
    footprint_mask = ~np.isnan(height_map)
    n_valid = int(footprint_mask.sum())

    if n_valid == 0:
        logger.warning("No ray hits for %s — check mesh or grid resolution.", obj_path)

    return Surface2_5D(
        height_map=height_map,
        footprint_mask=footprint_mask,
        grid_x=grid_x,
        grid_y=grid_y,
        cell_size=config.grid_resolution_mm,
        ground_z=0.0,
        n_valid_cells=n_valid,
    )

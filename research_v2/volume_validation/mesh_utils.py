"""Mesh loading, stable orientation, and reference volume computation.

The Čapek et al. (2025) dataset provides watertight, two-manifold OBJ meshes.
For such meshes, ``trimesh.Trimesh.volume`` gives the exact signed volume via
the divergence theorem — no approximation needed.

Stable orientation finds the resting pose: the face of the convex hull that
maximises support area while minimising centre-of-mass height. This mimics
how a rock fragment settles on a flat surface before a drone observes it from
above.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError

logger = logging.getLogger(__name__)

_EPS = 1e-9


@dataclass
class MeshInfo:
    """Summary of a loaded mesh."""

    sample_id: str
    group: str
    obj_path: Path
    n_vertices: int
    n_faces: int
    is_watertight: bool
    volume_mm3: float
    bbox_dims_mm: np.ndarray  # (L, W, H) in mm
    surface_area_mm2: float


def load_mesh(obj_path: str | Path) -> trimesh.Trimesh:
    """Load a Wavefront OBJ file as a Trimesh object.

    Parameters
    ----------
    obj_path : path to the .obj file.

    Returns
    -------
    trimesh.Trimesh
    """
    mesh = trimesh.load(str(obj_path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        # Scene with multiple meshes — merge them
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [g for g in mesh.geometry.values()]
            )
        else:
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    return mesh


def compute_reference_volume(mesh: trimesh.Trimesh) -> float:
    """Exact volume of a watertight mesh in mm³.

    Uses the signed-volume-of-tetrahedra method (divergence theorem).
    """
    if not mesh.is_watertight:
        logger.warning("Mesh is not watertight; volume may be approximate.")
    return float(mesh.volume)


def _rotation_align(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Rotation matrix that rotates vector *v_from* onto *v_to*.

    Uses Rodrigues' rotation formula. Both vectors are normalised first.
    """
    a = v_from / (np.linalg.norm(v_from) + _EPS)
    b = v_to / (np.linalg.norm(v_to) + _EPS)

    cross = np.cross(a, b)
    dot = float(np.dot(a, b))

    if np.linalg.norm(cross) < _EPS:
        if dot > 0:
            return np.eye(3)
        # 180° rotation about any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, perp)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis) + _EPS
        # R = 2*n*n^T - I  (180° rotation about axis n)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    s = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + s + s @ s * (1.0 / (1.0 + dot))


def find_stable_orientation(mesh: trimesh.Trimesh) -> np.ndarray:
    """Find the most stable resting pose for the mesh.

    Tests every face of the convex hull as a candidate bottom face.
    For each candidate, the stability score is::

        score = support_area / (com_height + eps)

    A larger support polygon and a lower centre of mass both increase
    stability. Returns a 4×4 homogeneous transform that:

    1. rotates the chosen face to point downward (−Z),
    2. translates the mesh so that the lowest vertex sits at Z = 0.

    Parameters
    ----------
    mesh : trimesh.Trimesh

    Returns
    -------
    transform : (4, 4) ndarray
        Homogeneous transform to apply to the original mesh.
    """
    hull = mesh.convex_hull
    com = mesh.center_mass  # volume-weighted centre of mass
    verts = hull.vertices.copy()

    best_score = -np.inf
    best_R = np.eye(3)

    for normal in hull.face_normals:
        # Rotate so that `normal` points to [0, 0, -1]  (face → bottom)
        R = _rotation_align(normal, np.array([0.0, 0.0, -1.0]))

        rotated = verts @ R.T
        com_rotated = R @ com

        min_z = rotated[:, 2].min()
        z_range = rotated[:, 2].max() - min_z
        # Bottom 5 % of vertices form the support polygon
        bottom_mask = rotated[:, 2] <= min_z + z_range * 0.05
        bottom_xy = rotated[bottom_mask][:, :2]

        if len(bottom_xy) < 3:
            continue

        try:
            support_hull = ConvexHull(bottom_xy)
            support_area = float(support_hull.volume)
        except QhullError:
            continue

        com_height = com_rotated[2] - min_z
        if com_height < _EPS:
            com_height = _EPS

        score = support_area / com_height
        if score > best_score:
            best_score = score
            best_R = R

    # Build 4×4 transform and translate so min-Z = 0
    oriented = mesh.apply_transform(_to_homogeneous(best_R))
    translation = np.eye(4)
    translation[2, 3] = -oriented.vertices[:, 2].min()
    return translation @ _to_homogeneous(best_R)


def _to_homogeneous(R: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    """Convert a 3×3 rotation to a 4×4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def orient_mesh(mesh: trimesh.Trimesh, transform: np.ndarray) -> trimesh.Trimesh:
    """Apply a homogeneous transform to a copy of the mesh."""
    return mesh.apply_transform(transform)


def get_mesh_info(
    obj_path: str | Path,
    sample_id: str,
    group: str,
) -> MeshInfo:
    """Load a mesh and compute all summary information.

    Returns volume in mm³ and bounding-box dimensions sorted as L ≥ W ≥ H.
    """
    mesh = load_mesh(obj_path)
    extents = mesh.bounding_box.extents  # (x, y, z) sizes
    dims = np.sort(extents)[::-1]  # L ≥ W ≥ H

    return MeshInfo(
        sample_id=sample_id,
        group=group,
        obj_path=Path(obj_path),
        n_vertices=len(mesh.vertices),
        n_faces=len(mesh.faces),
        is_watertight=mesh.is_watertight,
        volume_mm3=compute_reference_volume(mesh),
        bbox_dims_mm=dims,
        surface_area_mm2=float(mesh.area),
    )


def find_obj_files(data_dir: Path, groups: dict) -> list[tuple[str, str, Path]]:
    """Discover all OBJ files grouped by source.

    Returns
    -------
    list of (sample_id, group_name, obj_path)
    """
    results = []
    for group_name, info in groups.items():
        subfolder = data_dir / info["subfolder"]
        if not subfolder.exists():
            logger.warning("Group folder not found: %s", subfolder)
            continue
        obj_files = sorted(subfolder.glob("*.obj"))
        for obj_path in obj_files:
            sample_id = f"{group_name}_{obj_path.stem}"
            results.append((sample_id, group_name, obj_path))
    return results

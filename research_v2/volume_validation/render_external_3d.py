"""Render 3D views of external stone OBJ meshes.

Uses trimesh to load OBJ and matplotlib 3D to render.
Shows 6 representative stones from T01 and L01 each.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh

ROOT = Path(__file__).resolve().parent
OBJ_ROOT = ROOT.parents[1] / "data" / "experience_rock"
CACHE_DIR = ROOT / "datasets" / "t01_l01_scaled_10mm" / "cache"
OUT_DIR = ROOT / "output" / "external_stone_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cache(dataset_id):
    folder = CACHE_DIR / dataset_id
    samples = {}
    for p in sorted(folder.glob("*.json")):
        with open(p) as f:
            d = json.load(f)
        if d.get("status") == "success":
            samples[d["sample_id"]] = d
    return samples


def render_mesh_3d(ax, mesh, title, elev=30, azim=-60):
    """Render a trimesh mesh as 3D surface plot."""
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Compute face colors based on face normals (simple shading)
    face_normals = mesh.face_normals
    # Lighting direction
    light = np.array([0.5, 0.3, 0.8])
    light = light / np.linalg.norm(light)
    # Diffuse lighting
    intensity = np.clip(np.dot(face_normals, light), 0.2, 1.0)
    
    # Map intensity to color
    colors = np.zeros((len(faces), 4))
    colors[:, 0] = 0.7 * intensity + 0.2  # R
    colors[:, 1] = 0.6 * intensity + 0.15  # G
    colors[:, 2] = 0.5 * intensity + 0.1  # B
    colors[:, 3] = 1.0
    
    tri = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                          triangles=faces,
                          linewidth=0.1, edgecolor=(0.3, 0.3, 0.3, 0.2),
                          shade=True, cmap="copper", antialiased=True)
    tri.set_facecolors(colors)
    
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 0.7])
    
    # Remove tick labels for cleaner look
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def auto_align_bottom(mesh):
    """Translate mesh so its lowest point is at z=0."""
    min_z = mesh.vertices[:, 2].min()
    mesh.vertices[:, 2] -= min_z
    return mesh


def normalize_size(mesh, target_diameter=1.0):
    """Normalize mesh to unit footprint size for shape comparison."""
    v = mesh.vertices
    x_range = v[:, 0].max() - v[:, 0].min()
    y_range = v[:, 1].max() - v[:, 1].min()
    max_range = max(x_range, y_range)
    if max_range > 0:
        scale = target_diameter / max_range
        mesh.vertices *= scale
    return mesh


def render_dataset(dataset_id, n_stones=6, title=""):
    """Render n representative stones from a dataset."""
    cache = load_cache(dataset_id)
    all_ids = list(cache.keys())
    
    # Select diverse stones by y_ratio and AR
    sorted_by_y = sorted(all_ids, key=lambda sid: cache[sid]["y_ratio"])
    sorted_by_ar = sorted(all_ids, key=lambda sid: cache[sid]["AR"])
    
    picked = []
    # Low y (most conical/spiky)
    picked.append(sorted_by_y[0])
    # Low AR (most round)
    if sorted_by_ar[0] not in picked:
        picked.append(sorted_by_ar[0])
    # Low-mid y
    idx = len(sorted_by_y) // 4
    while sorted_by_y[idx] in picked and idx < len(sorted_by_y) - 1:
        idx += 1
    picked.append(sorted_by_y[idx])
    # Median y
    idx = len(sorted_by_y) // 2
    while sorted_by_y[idx] in picked and idx < len(sorted_by_y) - 1:
        idx += 1
    picked.append(sorted_by_y[idx])
    # High AR (most elongated)
    if sorted_by_ar[-1] not in picked:
        picked.append(sorted_by_ar[-1])
    # High y (most blocky)
    if sorted_by_y[-1] not in picked:
        picked.append(sorted_by_y[-1])
    
    picked = picked[:n_stones]
    
    fig = plt.figure(figsize=(4 * n_stones, 5))
    
    for idx, sid in enumerate(picked):
        ax = fig.add_subplot(1, n_stones, idx + 1, projection='3d')
        
        s = cache[sid]
        obj_path = OBJ_ROOT / dataset_id / f"{s['original_obj_id']}.obj"
        
        try:
            mesh = trimesh.load(str(obj_path), force='mesh')
            mesh = auto_align_bottom(mesh)
            mesh = normalize_size(mesh, target_diameter=1.0)
            
            y = s['y_ratio']
            ar = s['AR']
            c = s['C']
            render_mesh_3d(ax, mesh, f"{sid}\ny={y:.3f}  AR={ar:.2f}  C={c:.2f}")
        except Exception as e:
            ax.text(0.5, 0.5, 0.5, f"Load error:\n{e}",
                    ha='center', va='center', fontsize=8, color='red')
            ax.set_title(sid, fontsize=9)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    out = OUT_DIR / f"{dataset_id.lower()}_3d_render.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    
    # Also render a multi-angle single stone for better 3D perception
    # Pick the median one
    mid_sid = picked[len(picked)//2]
    obj_path = OBJ_ROOT / dataset_id / f"{cache[mid_sid]['original_obj_id']}.obj"
    mesh = trimesh.load(str(obj_path), force='mesh')
    mesh = auto_align_bottom(mesh)
    mesh = normalize_size(mesh, target_diameter=1.0)
    
    angles = [(30, -60), (30, -30), (30, 0), (30, 30), (60, -45), (15, -90)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    for ax, (elev, azim) in zip(axes, angles):
        render_mesh_3d(ax, mesh, f"elev={elev}° azim={azim}°", elev=elev, azim=azim)
    
    s = cache[mid_sid]
    fig.suptitle(f"{dataset_id} — {mid_sid} (multi-angle)\n"
                 f"y={s['y_ratio']:.3f}  AR={s['AR']:.2f}  C={s['C']:.2f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    out = OUT_DIR / f"{dataset_id.lower()}_multiangle_3d.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    print("Rendering T01 stones...")
    render_dataset("T01", n_stones=6, title="T01 — Volcaniclastic Rocks (3D)")

    print("Rendering L01 stones...")
    render_dataset("L01", n_stones=6, title="L01 — External Stones (3D)")
    
    print(f"\nAll 3D renders saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

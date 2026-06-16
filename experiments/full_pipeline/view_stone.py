"""
Standalone stone 3D viewer — directly open an existing pipeline output

Usage:
  python experiments/full_pipeline/view_stone.py --run-id run_20260615_162401 --stone-id stone_000298
  python experiments/full_pipeline/view_stone.py --run-id run_20260615_162401 --list
  python experiments/full_pipeline/view_stone.py --run-id run_20260615_162401 --top 10
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_run(run_dir: Path):
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"report.json not found: {report_path}")
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    return report.get("stones", []), run_dir / "stones"


def _pick_stone(stones, stone_dir, stone_id):
    """Find stone by ID, load its point cloud, return (stone, pcd, pts)."""
    stone = next((s for s in stones if s["stone_id"] == stone_id), None)
    if stone is None:
        print(f"  Stone '{stone_id}' not found")
        return None, None, None
    ply_path = stone_dir / f"{stone_id}.ply"
    if not ply_path.exists():
        print(f"  PLY not found: {ply_path}")
        return None, None, None
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        print(f"  Empty point cloud")
        return None, None, None
    return stone, pcd, pts


def view_stone(stones, stone_dir, stone_id):
    """View a single stone — translates to origin so Open3D camera can focus."""
    stone, pcd, pts = _pick_stone(stones, stone_dir, stone_id)
    if stone is None:
        return

    # ── Translate to origin (coordinates are large CGCS2000 values) ──
    centroid = pts.mean(axis=0)
    pts_origin = pts - centroid
    pcd_origin = o3d.geometry.PointCloud()
    pcd_origin.points = o3d.utility.Vector3dVector(pts_origin)

    geoms = [pcd_origin]

    # BBox (also translated)
    if stone.get("bbox_3d"):
        b = np.array(stone["bbox_3d"])
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=b[:3] - centroid,
            max_bound=b[3:] - centroid,
        )
        bbox.color = (0.0, 0.8, 0.0)
        geoms.append(bbox)

    # Axes
    extent = pts_origin.ptp(axis=0).max()
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=extent * 0.4)
    geoms.append(axes)

    # Info
    print(f"\n{'='*50}")
    print(f"  Stone: {stone_id}")
    print(f"  Points: {len(pts)}")
    print(f"  Size (X×Y×Z): {pts[:,0].ptp():.2f} × {pts[:,1].ptp():.2f} × {pts[:,2].ptp():.2f} m")
    print(f"  Volume: {stone.get('volume_m3', 0):.4f} m3")
    print(f"  Diameter: {stone.get('equivalent_diameter_m', 0):.3f} m")
    print(f"  Surface area: {stone.get('surface_area_m2', 0):.3f} m2")
    print(f"  Method: {stone.get('volume_method', '?')}")
    print(f"  World centroid: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
    print(f"{'='*50}")
    print("  [Open3D] Drag=rotate  Scroll=zoom  Ctrl+drag=pan  Ctrl+R=reset")
    print("  (Point cloud shifted to origin for display)")
    print(f"{'='*50}")

    o3d.visualization.draw_geometries(
        geoms, window_name=f"Stone: {stone_id}",
        width=1024, height=768,
    )


def view_top_n(stones, stone_dir, n=10):
    """Display top N stones together, each translated to origin."""
    sorted_s = sorted(stones, key=lambda s: s.get("volume_m3", 0), reverse=True)[:n]
    geoms = []
    for i, s in enumerate(sorted_s):
        ply_path = stone_dir / f"{s['stone_id']}.ply"
        if not ply_path.exists():
            continue
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points)
        if len(pts) < 4:
            continue
        # Translate each stone to a different position (stagger by 5m in X)
        offset = np.array([i * 5.0, 0.0, 0.0])
        pcd.translate(-pts.mean(axis=0) + offset)
        hue = (i * 0.618033988749895) % 1.0
        import colorsys
        pcd.paint_uniform_color(colorsys.hsv_to_rgb(hue, 0.8, 0.9))
        geoms.append(pcd)
    if not geoms:
        print("  No stones to display")
        return
    print(f"\n  Showing top {len(geoms)} stones (each shifted to origin, spaced 5m apart)")
    o3d.visualization.draw_geometries(
        geoms, window_name=f"Top {len(geoms)} Stones",
        width=1280, height=800,
    )


def list_stones(stones, stone_dir, sort_by="volume", top_n=None):
    key = {"volume": "volume_m3", "diameter": "equivalent_diameter_m",
           "points": "point_count", "id": "stone_id"}.get(sort_by, "volume_m3")
    sorted_s = sorted(stones, key=lambda s: s.get(key, 0), reverse=(sort_by != "id"))
    if top_n:
        sorted_s = sorted_s[:top_n]
    print(f"\n{'Stone ID':<20} {'Volume(m3)':<12} {'Diameter(m)':<12} {'Points':<8} {'Method':<20} {'PLY'}")
    print("-" * 80)
    for s in sorted_s:
        sid = s.get("stone_id", "?")
        vol = s.get("volume_m3", 0)
        diam = s.get("equivalent_diameter_m", 0)
        pts = s.get("point_count", 0)
        method = (s.get("volume_method", "?") or "?")[:18]
        ok = "OK" if (stone_dir / f"{sid}.ply").exists() else "--"
        print(f"  {sid:<18} {vol:<12.4f} {diam:<12.4f} {pts:<8} {method:<18} [{ok}]")


def main():
    parser = argparse.ArgumentParser(description="Stone 3D Viewer")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stone-id", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--sort", default="volume", choices=["volume","diameter","points","id"])
    parser.add_argument("--top", type=int, default=None)
    args = parser.parse_args()

    runs_root = PROJECT_ROOT / "experiments" / "full_pipeline" / "outputs" / "runs"
    run_dir = runs_root / args.run_id
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        if runs_root.exists():
            print("\nAvailable runs:")
            for d in sorted(runs_root.iterdir()):
                if d.is_dir():
                    sc = len(list(d.glob("stones/stone_*.ply")))
                    print(f"    {d.name}  ({sc} stones)")
        return

    try:
        stones, stone_dir = _load_run(run_dir)
    except FileNotFoundError as e:
        print(f"  {e}")
        return

    if args.list:
        list_stones(stones, stone_dir, args.sort)
    elif args.top:
        view_top_n(stones, stone_dir, args.top)
    elif args.stone_id:
        view_stone(stones, stone_dir, args.stone_id)
    else:
        print(f"\n  Run: {args.run_id}  ({len(stones)} stones)")
        list_stones(stones, stone_dir, sort_by="volume", top_n=20)
        if stones:
            print(f"\n  Opening top stone: {stones[0]['stone_id']}")
            view_stone(stones, stone_dir, stones[0]["stone_id"])


if __name__ == "__main__":
    main()

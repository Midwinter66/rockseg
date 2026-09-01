"""Show volume details for the 6 example stones — recompute from visualization results."""
import json
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from experiments.common.scene_reference import CURRENT_SCENE
from experiments.common.stone_region import PointCloudXYGridIndex, _points_in_polygon
from experiments.common.stone_region import pixel_to_world
from experiments.volume.ground_estimator import GroundDEM
from experiments.volume.estimators import estimate_2d5_with_ground

ALPHA = 0.731

def pixel_bbox_to_world(bbox_px, gt):
    x1, y1, x2, y2 = bbox_px
    wx1 = gt[0] + x1 * gt[1] + y1 * gt[2]
    wy1 = gt[3] + x1 * gt[4] + y1 * gt[5]
    wx2 = gt[0] + x2 * gt[1] + y2 * gt[2]
    wy2 = gt[3] + x2 * gt[4] + y2 * gt[5]
    return [min(wx1, wx2), min(wy1, wy2), max(wx1, wx2), max(wy1, wy2)]

def mask_to_polygon(mask, bbox_px, gt, xy_transform):
    if mask.sum() < 3:
        return None
    coords = np.argwhere(mask)
    pts_xy = coords[:, ::-1].astype(np.float64)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts_xy)
        hull_pts = pts_xy[hull.vertices]
    except Exception:
        hull_pts = pts_xy
    ox, oy = bbox_px[0], bbox_px[1]
    polygon = []
    for px, py in hull_pts:
        wx, wy = pixel_to_world(gt, px + ox, py + oy)
        px_pc, py_pc = xy_transform.world_to_point_xy(wx, wy)
        polygon.append([px_pc, py_pc])
    return np.asarray(polygon, dtype=np.float32) if len(polygon) >= 3 else None

def load_point_cloud():
    import laspy
    all_pts = []
    for path in CURRENT_SCENE.pointcloud_paths:
        las = laspy.read(str(path))
        pts = np.column_stack([las.x, las.y, las.z]).astype(np.float64, copy=False)
        all_pts.append(pts)
    return np.vstack(all_pts)

# Load V2 data
with open(PROJECT_ROOT / "output/dom2_full/rock_instances.json") as f:
    rocks = json.load(f)
GSD = 0.01
for r in rocks:
    r["area_m2"] = r["area"] * GSD**2
    r["eq_d_m"] = 2 * np.sqrt(r["area_m2"] / np.pi)

masks_data = np.load(PROJECT_ROOT / "output/dom2_full/rock_masks.npz", allow_pickle=True)
selected_ids = ['rock_02944', 'rock_04180', 'rock_12895', 'rock_09358', 'rock_06902', 'rock_03465']

gt = CURRENT_SCENE.load_gt()
xy_transform = CURRENT_SCENE.xy_transform

print("Loading point cloud...")
pc = load_point_cloud()
print(f"  {len(pc):,} points")
print("Building spatial index...")
pc_index = PointCloudXYGridIndex.build(pc, cell_size=1.0)
print("Building GroundDEM...")
ground_dem = GroundDEM(pc, resolution=0.5, percentile=5, subsample_step=100, min_points_per_cell=3)

print()
print("=" * 90)
print("6 Stone Volume Examples")
print("=" * 90)

for sid in selected_ids:
    rock = next(r for r in rocks if r['instance_id'] == sid)
    mask_key = f"{sid}_mask"
    mask = masks_data[mask_key]

    bbox_world = pixel_bbox_to_world(rock["bbox"], gt)
    x0, y0, x1, y1 = xy_transform.world_bbox_to_point_bbox(bbox_world, pad_m=0.5)
    candidate_indices = pc_index.query_bbox_indices(x0, y0, x1, y1)
    candidates = pc[candidate_indices].copy()

    polygon = mask_to_polygon(mask, rock["bbox"], gt, xy_transform)
    if polygon is not None:
        inside = _points_in_polygon(candidates[:, :2], polygon)
        stone_pts = candidates[inside].copy()
    else:
        stone_pts = candidates

    if len(stone_pts) < 10:
        print(f"\n{sid}: too few points ({len(stone_pts)})")
        continue

    vol = estimate_2d5_with_ground(stone_pts, ground_dem, grid_resolution=0.05)
    hs = vol.get("height_stats", {})
    v_raw = vol.get("volume_m3", 0)
    v_corr = v_raw * ALPHA
    area = vol.get("occupied_area_m2", 0)
    fill = v_raw / (area * hs.get("max_m", 1) + 1e-9) if area > 0 else 0
    z_range = stone_pts[:, 2].max() - stone_pts[:, 2].min()

    print(f"\n--- {sid} ({rock['scale_level']}) ---")
    print(f"  Size:        eq_d={rock['eq_d_m']:.3f}m  mask_area={rock['area_m2']:.3f}m2")
    print(f"  Point cloud: {len(stone_pts):,} pts  z_range={z_range:.3f}m")
    print(f"  Height:      H_mean={hs.get('mean_m',0):.3f}m  H_max={hs.get('max_m',0):.3f}m  H_std={hs.get('std_m',0):.3f}m")
    print(f"  Footprint:   occupied_area={area:.3f}m2  grid={vol.get('grid_nx',0)}x{vol.get('grid_ny',0)}")
    print(f"  Fill ratio:  {fill:.3f}")
    print(f"  Volume:      V_2.5D_raw={v_raw:.4f} m3")
    print(f"               V_corrected={v_corr:.4f} m3  (alpha={ALPHA})")
    print(f"               95% CI: [{v_raw*(ALPHA-0.106):.4f}, {v_raw*(ALPHA+0.106):.4f}] m3")

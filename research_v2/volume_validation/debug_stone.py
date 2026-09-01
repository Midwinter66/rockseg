"""Debug stone structure."""
import json
import numpy as np

with open('experiments/volume/outputs/quadtree_dom/correlation_clustering/stone_volumes.json', 'r') as f:
    data = json.load(f)
stones = data['stones']

s = stones[0]
print('=== First stone keys ===')
print(list(s.keys()))

geo = s.get('geometry', {})
pc = s.get('point_cloud', {})
d25 = s.get('methods', {}).get('2d5', {})
hs = d25.get('height_stats', {})

print('\ngeometry:', json.dumps(geo, indent=2))
print('\npoint_cloud:', json.dumps(pc, indent=2))
print('\n2d5 keys:', list(d25.keys()))
print('  occupied_area_m2:', d25.get('occupied_area_m2'))
print('  surface_area_m2:', d25.get('surface_area_m2'))
print('  height_stats:', json.dumps(hs, indent=2))

# Check 10 stones
print('\n=== 10 stones ===')
for i in range(10):
    s = stones[i]
    d25 = s['methods']['2d5']
    hs = d25.get('height_stats', {})
    occ = d25.get('occupied_area_m2', 0)
    h_mean = hs.get('mean_m', 0)
    h_max = hs.get('max_m', 0)
    v = d25.get('volume_m3', 0)
    pts = s['point_cloud']['point_count']
    # Check if V = occ_area * h_mean (2.5D formula)
    v_check = occ * h_mean if occ > 0 and h_mean > 0 else -1
    print(f'  V={v:.4f}  occ_A={occ:.4f}  h_mean={h_mean:.4f}  h_max={h_max:.4f}  A*h_mean={v_check:.4f}  pts={pts}')

# Compute fill ratio correctly: V / (occupied_area * h_max)
print('\n=== Fill ratio V/(occ_area * h_max) ===')
ratios = []
for s in stones:
    d25 = s.get('methods', {}).get('2d5', {})
    if not isinstance(d25, dict) or d25.get('status') != 'ok':
        continue
    v = d25.get('volume_m3', 0)
    occ = d25.get('occupied_area_m2', 0)
    hs = d25.get('height_stats', {})
    h_max = hs.get('max_m', 0)
    if v > 0 and occ > 0 and h_max > 0:
        r = v / (occ * h_max)
        ratios.append(r)

ratios = np.array(ratios)
print(f'  n={len(ratios)}')
print(f'  mean={np.mean(ratios):.4f}  median={np.median(ratios):.4f}  std={np.std(ratios):.4f}')
print(f'  Q5={np.percentile(ratios, 5):.4f}  Q25={np.percentile(ratios, 25):.4f}  Q75={np.percentile(ratios, 75):.4f}  Q95={np.percentile(ratios, 95):.4f}')
print(f'  min={np.min(ratios):.4f}  max={np.max(ratios):.4f}')

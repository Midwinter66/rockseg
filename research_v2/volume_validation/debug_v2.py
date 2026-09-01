"""Analyze V2 rock_instances to prepare for volume estimation."""
import json
import numpy as np
from collections import Counter

with open('output/dom2_full/rock_instances.json', 'r') as f:
    data = json.load(f)

scales = Counter(d['scale_level'] for d in data)
print('Scale levels:', dict(scales))
print()
for d in data[:5]:
    iid = d['instance_id']
    bb = d['bbox']
    ar = d['area']
    sc = d['scale_level']
    print(f'{iid}: bbox={bb}  area={ar}px  scale={sc}')

areas = np.array([d['area'] for d in data])
GSD = 0.01
areas_m2 = areas * GSD**2
eq_d = 2 * np.sqrt(areas_m2 / np.pi)
print(f'\nArea (px): mean={areas.mean():.0f}  median={np.median(areas):.0f}')
print(f'Eq diameter (m): mean={eq_d.mean():.3f}  median={np.median(eq_d):.3f}')
print(f'  min={eq_d.min():.3f}  max={eq_d.max():.3f}')
print(f'\nStones >= 0.5m: {(eq_d>=0.5).sum()}')
print(f'Stones >= 1.0m: {(eq_d>=1.0).sum()}')

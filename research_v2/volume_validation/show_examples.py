"""Show volume details for the 6 example stones."""
import json
import numpy as np
import pandas as pd

with open('output/dom2_full/rock_instances.json') as f:
    rocks = json.load(f)
GSD = 0.01
for r in rocks:
    r['area_m2'] = r['area'] * GSD**2
    r['eq_d_m'] = 2 * np.sqrt(r['area_m2'] / np.pi)

df = pd.read_csv('research_v2/volume_validation/output/v2_subset_volumes.csv')
df = df[df['status'] == 'ok'].copy()
df['V_corrected'] = df['V_2_5d'] * 0.731

selected_ids = ['rock_02944', 'rock_04180', 'rock_12895', 'rock_09358', 'rock_06902', 'rock_03465']
print('=== 6 Stone Examples ===')
print()
for sid in selected_ids:
    row = df[df['stone_id'] == sid]
    rock = next(rk for rk in rocks if rk['instance_id'] == sid)
    if len(row) == 0:
        print(f'{sid}: scale={rock["scale_level"]}, eq_d={rock["eq_d_m"]:.3f}m, area={rock["area_m2"]:.3f}m2 (not in 200-sample)')
        continue
    r = row.iloc[0]
    fill = r['V_2_5d'] / (r['occupied_area_m2'] * r['h_max_m'] + 1e-9)
    print(f'--- {sid} ({rock["scale_level"]}) ---')
    print(f'  Size:    eq_d={rock["eq_d_m"]:.3f}m  area={rock["area_m2"]:.3f}m2')
    print(f'  Points:  {int(r["point_count"]):,} pts  z_range={r["z_range_m"]:.3f}m')
    print(f'  Height:  H_mean={r["h_mean_m"]:.3f}m  H_max={r["h_max_m"]:.3f}m')
    print(f'  Area:    occupied={r["occupied_area_m2"]:.3f}m2')
    print(f'  Fill:    {fill:.3f}')
    print(f'  Volume:  V_2.5D={r["V_2_5d"]:.4f}m3  V_corrected={r["V_corrected"]:.4f}m3')
    print()

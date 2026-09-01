"""Analyze the latest V2 data: dom2_cascade_v2 (76,407 instances)."""
import json
import numpy as np
from collections import Counter, defaultdict

with open('output/dom2_cascade_v2/rock_instances.json') as f:
    rocks = json.load(f)

GSD = 0.01
for r in rocks:
    r['area_m2'] = r['area'] * GSD**2
    r['eq_d_m'] = 2 * np.sqrt(r['area_m2'] / np.pi)

n = len(rocks)
print("=" * 80)
print(f"Latest V2 Data: dom2_cascade_v2  ({n:,} instances)")
print("=" * 80)

# 1. Scale distribution
scales = Counter(r['scale_level'] for r in rocks)
print(f"\n--- 1. Scale Level Distribution ---")
for s in ['coarse', 'medium', 'fine']:
    c = scales.get(s, 0)
    areas = np.array([r['area_m2'] for r in rocks if r['scale_level'] == s])
    eq_d = np.array([r['eq_d_m'] for r in rocks if r['scale_level'] == s])
    print(f"  {s:<8}: {c:>6,} ({c/n*100:5.1f}%)  eq_d_mean={eq_d.mean():.3f}m  area_sum={areas.sum():.1f}m2")

# 2. Size distribution
eq_d = np.array([r['eq_d_m'] for r in rocks])
print(f"\n--- 2. Size Distribution ---")
bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
labels = ['<5cm', '5-10cm', '10-20cm', '20-30cm', '30-50cm', '50-100cm', '1-2m', '>2m']
hist, _ = np.histogram(eq_d, bins=bins)
for l, h in zip(labels, hist):
    bar = '#' * max(0, int(h / max(hist) * 50))
    print(f"  {l:<12} {h:>6,}  {bar}")
print(f"\n  eq_d: mean={eq_d.mean():.3f}m  median={np.median(eq_d):.3f}m  std={eq_d.std():.3f}m")
print(f"        min={eq_d.min():.3f}m  max={eq_d.max():.3f}m")

# 3. Confidence
confs = np.array([r.get('confidence', 0) for r in rocks])
print(f"\n--- 3. Confidence Distribution ---")
print(f"  mean={confs.mean():.3f}  median={np.median(confs):.3f}")
print(f"  <0.3: {(confs<0.3).sum():>6,}  0.3-0.5: {((confs>=0.3)&(confs<0.5)).sum():>6,}  0.5-0.7: {((confs>=0.5)&(confs<0.7)).sum():>6,}  >0.7: {(confs>=0.7).sum():>6,}")

# 4. Area
areas = np.array([r['area_m2'] for r in rocks])
print(f"\n--- 4. Area ---")
print(f"  sum={areas.sum():.1f} m2  mean={areas.mean():.4f} m2  median={np.median(areas):.4f} m2")

# 5. Volume estimation
fill_ratio = 0.489
h_max_ratio = 0.30
v_raw = fill_ratio * areas * (h_max_ratio * eq_d)
v_corr = v_raw * 0.731
print(f"\n--- 5. Volume Estimation (shape prior) ---")
print(f"  V_2.5D raw:   {v_raw.sum():.1f} m3")
print(f"  V_corrected:  {v_corr.sum():.1f} m3 (alpha=0.731)")

print(f"\n  By scale:")
for sl in ['coarse', 'medium', 'fine']:
    idx = np.array([r['scale_level'] == sl for r in rocks])
    print(f"    {sl:<8}: V_corr={v_corr[idx].sum():>8.1f} m3  ({v_corr[idx].sum()/v_corr.sum()*100:.1f}%)  n={idx.sum():,}")

# 6. Compare all versions
print(f"\n--- 6. Version Comparison ---")
versions = {
    'dom2_full (old)': (24219, '08/24 18:10'),
    'dom2_fixed': (58146, '08/24 19:45'),
    'dom2_v3': (70020, '08/25 13:56'),
    'dom2_cascade': (54660, '08/25 14:48'),
    'dom2_cascade_v2 (LATEST)': (76407, '08/25 15:20'),
}
print(f"  {'Version':<30} {'Count':>8}  {'Time':>12}")
for name, (cnt, t) in versions.items():
    print(f"  {name:<30} {cnt:>8,}  {t:>12}")

# 7. Quality flags
print(f"\n--- 7. Quality Flags ---")
tiny = (eq_d < 0.05).sum()
valid = ((eq_d >= 0.1) & (confs >= 0.35)).sum()
low_conf = (confs < 0.35).sum()
print(f"  Tiny (<5cm):          {tiny:>6,} ({tiny/n*100:.1f}%)")
print(f"  Low conf (<0.35):     {low_conf:>6,} ({low_conf/n*100:.1f}%)")
print(f"  Valid (>=10cm,conf>=0.35): {valid:>6,} ({valid/n*100:.1f}%)")

# 8. Top 10 largest
print(f"\n--- 8. Top 10 Largest Stones ---")
top10 = sorted(rocks, key=lambda r: r['area_m2'], reverse=True)[:10]
print(f"  {'ID':<18} {'Scale':<8} {'Area(m2)':>10} {'Eq_d(m)':>10} {'Conf':>6}")
for r in top10:
    print(f"  {r['instance_id']:<18} {r['scale_level']:<8} {r['area_m2']:>10.3f} {r['eq_d_m']:>10.3f} {r.get('confidence',0):>6.3f}")

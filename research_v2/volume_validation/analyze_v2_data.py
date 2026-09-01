"""Comprehensive analysis of V2 DOM2 latest data."""
import json
import numpy as np
from collections import Counter, defaultdict

# Load V2 data
with open('output/dom2_full/rock_instances.json') as f:
    v2 = json.load(f)

# Load V1 data for comparison
with open('experiments/volume/outputs/quadtree_dom/correlation_clustering/stone_volumes.json') as f:
    v1 = json.load(f)

GSD = 0.01
for r in v2:
    r['area_m2'] = r['area'] * GSD**2
    r['eq_d_m'] = 2 * np.sqrt(r['area_m2'] / np.pi)

v1_stones = v1['stones']
v1_vols = [s['volume_m3'] for s in v1_stones if s.get('volume_m3', 0) > 0]
v1_total = sum(v1_vols)

print("=" * 80)
print("V2 DOM2 Latest Data Analysis")
print("=" * 80)

print("\n--- 1. Overview ---")
print(f"  Total rocks: {len(v2):,}")
print(f"  V1 had:     {len(v1_stones):,} rocks")
print(f"  V2/V1 ratio: {len(v2)/len(v1_stones):.1f}x")

print("\n--- 2. Scale Level Distribution ---")
scales = Counter(r['scale_level'] for r in v2)
for s in ['coarse', 'medium', 'fine']:
    c = scales[s]
    areas = np.array([r['area_m2'] for r in v2 if r['scale_level'] == s])
    eq_d = np.array([r['eq_d_m'] for r in v2 if r['scale_level'] == s])
    print(f"  {s:<8}: {c:>6} ({c/len(v2)*100:5.1f}%)  eq_d mean={eq_d.mean():.3f}m  area sum={areas.sum():.2f}m2")

print("\n--- 3. Size Distribution (eq diameter) ---")
eq_d = np.array([r['eq_d_m'] for r in v2])
bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
labels = ['<5cm', '5-10cm', '10-20cm', '20-30cm', '30-50cm', '50-100cm', '1-2m', '>2m']
hist, _ = np.histogram(eq_d, bins=bins)
for l, h in zip(labels, hist):
    bar = '#' * max(0, int(h / max(hist) * 50))
    print(f"  {l:<12} {h:>6}  {bar}")

print(f"\n  eq_d: mean={eq_d.mean():.3f}m  median={np.median(eq_d):.3f}m  std={eq_d.std():.3f}m")
print(f"        min={eq_d.min():.3f}m  max={eq_d.max():.3f}m")

print("\n--- 4. Confidence Distribution ---")
confs = np.array([r.get('confidence', 0) for r in v2])
print(f"  mean={confs.mean():.3f}  median={np.median(confs):.3f}  std={confs.std():.3f}")
print(f"  <0.3: {(confs<0.3).sum()}  0.3-0.5: {((confs>=0.3)&(confs<0.5)).sum()}  0.5-0.7: {((confs>=0.5)&(confs<0.7)).sum()}  >0.7: {(confs>=0.7).sum()}")

print("\n--- 5. V1 vs V2 Comparison ---")
print(f"  {'Metric':<25} {'V1':>15} {'V2':>15} {'Ratio':>10}")
print(f"  {'-'*65}")

v1_areas = np.array([s.get('occupied_area_m2', 0) for s in v1_stones])
v2_areas = np.array([r['area_m2'] for r in v2])
v1_eq_d = np.array([s.get('eq_diameter_m', 2*np.sqrt(s.get('occupied_area_m2',0)/np.pi)) for s in v1_stones])

print(f"  {'Count':<25} {len(v1_stones):>15,} {len(v2):>15,} {len(v2)/len(v1_stones):>10.1f}x")
print(f"  {'Area sum (m2)':<25} {v1_areas.sum():>15.2f} {v2_areas.sum():>15.2f} {v2_areas.sum()/v1_areas.sum():>10.1f}x")
print(f"  {'Area mean (m2)':<25} {v1_areas.mean():>15.4f} {v2_areas.mean():>15.4f} {v2_areas.mean()/v1_areas.mean():>10.1f}x")
print(f"  {'Area median (m2)':<25} {np.median(v1_areas):>15.4f} {np.median(v2_areas):>15.4f}")
print(f"  {'Eq_d mean (m)':<25} {v1_eq_d.mean():>15.3f} {eq_d.mean():>15.3f}")
print(f"  {'Eq_d median (m)':<25} {np.median(v1_eq_d):>15.3f} {np.median(eq_d):>15.3f}")
print(f"  {'V_2.5D total (m3)':<25} {v1_total:>15.2f} {'(see below)':>15}")
print(f"  {'V_corrected (m3)':<25} {v1_total*0.731:>15.2f} {'(see below)':>15}")

print("\n--- 6. Volume Estimation ---")
# Use shape prior from V1: V = fill_ratio * area * H_max, H_max ~ 0.30 * eq_d
fill_ratio = 0.489
h_max_ratio = 0.30
v2_h_est = h_max_ratio * eq_d
v2_v_raw = fill_ratio * v2_areas * v2_h_est
v2_v_corr = v2_v_raw * 0.731

print(f"  Estimated V_2.5D raw:   {v2_v_raw.sum():.1f} m3")
print(f"  Estimated V_corrected:   {v2_v_corr.sum():.1f} m3 (alpha=0.731)")
print(f"  V1 corrected for ref:    {v1_total*0.731:.1f} m3")

print("\n  By scale level:")
by_scale = defaultdict(list)
for r in v2:
    by_scale[r['scale_level']].append(r)

print(f"  {'Scale':<10} {'Count':>8} {'V_raw(m3)':>12} {'V_corr(m3)':>12} {'% of total':>12}")
for sl in ['coarse', 'medium', 'fine']:
    rs = by_scale[sl]
    areas = np.array([r['area_m2'] for r in rs])
    eq_d = np.array([r['eq_d_m'] for r in rs])
    v_raw = fill_ratio * areas * (h_max_ratio * eq_d)
    v_corr = v_raw * 0.731
    pct = v_corr.sum() / v2_v_corr.sum() * 100
    print(f"  {sl:<10} {len(rs):>8} {v_raw.sum():>12.1f} {v_corr.sum():>12.1f} {pct:>11.1f}%")

print("\n--- 7. Top 10 Largest Stones ---")
top10 = sorted(v2, key=lambda r: r['area_m2'], reverse=True)[:10]
print(f"  {'ID':<16} {'Scale':<8} {'Area(m2)':>10} {'Eq_d(m)':>10} {'Bbox(w x h m)':>16} {'Conf':>6}")
for r in top10:
    bb = r['bbox']
    w = (bb[2]-bb[0]) * GSD
    h = (bb[3]-bb[1]) * GSD
    print(f"  {r['instance_id']:<16} {r['scale_level']:<8} {r['area_m2']:>10.3f} {r['eq_d_m']:>10.3f} {w:.2f}x{h:.2f}{'':>5} {r.get('confidence',0):>6.3f}")

print("\n--- 8. Quality Flags ---")
tiny = (eq_d < 0.05).sum()
small = ((eq_d >= 0.05) & (eq_d < 0.1)).sum()
low_conf = (confs < 0.35).sum()
print(f"  Tiny (<5cm):          {tiny:>6} ({tiny/len(v2)*100:.1f}%)")
print(f"  Small (5-10cm):       {small:>6} ({small/len(v2)*100:.1f}%)")
print(f"  Low confidence (<0.35): {low_conf:>6} ({low_conf/len(v2)*100:.1f}%)")
print(f"  Valid (>=10cm, conf>=0.35): {(eq_d>=0.1).sum() & (confs>=0.35).sum()}")

print("\n--- 9. Key Differences V1 -> V2 ---")
print(f"  V1: quadtree_dom slicing (edge-density driven, 10-20m tiles)")
print(f"      -> {len(v1_stones)} stones, avg area {v1_areas.mean():.4f} m2")
print(f"  V2: multi-scale (coarse/medium/fine, physical scale driven)")
print(f"      -> {len(v2)} stones, avg area {v2_areas.mean():.4f} m2")
print(f"  V2 detects {len(v2)/len(v1_stones):.1f}x more stones")
print(f"  V2 average area is {v2_areas.mean()/v1_areas.mean():.1f}x smaller")
print(f"  => V2 finds many more small stones that V1 missed")

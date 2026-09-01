import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Chinese font
for fn in ['Microsoft YaHei', 'SimHei', 'SimSun']:
    if any(fn in f.name for f in fm.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [fn]
        plt.rcParams['axes.unicode_minus'] = False
        break

# Read CSV
v25, vpred, strata_arr, diam = [], [], [], []
with open(r'research_v2\volume_validation\real_mine_full\real_mine_volume_4000_results.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('status') not in ('PASS', 'success'):
            continue
        try:
            v25.append(float(row['V_2_5D_m3']))
            vpred.append(float(row['V_pred_m3']))
            strata_arr.append(row.get('stratum', ''))
            diam.append(float(row.get('equivalent_diameter_m', 0)))
        except (ValueError, KeyError):
            pass

v25 = np.array(v25)
vpred = np.array(vpred)
strata_arr = np.array(strata_arr)
diam = np.array(diam)
print(f"success records: {len(vpred)}")

# --- Figure 1: Volume histogram (log x axis) ---
fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
bins = np.logspace(-9, 0.5, 36)
ax.hist(vpred, bins=bins, color='steelblue', edgecolor='white', alpha=0.85)
ax.set_xscale('log')
ax.set_xlabel('单石块体积 V (m³)', fontsize=13)
ax.set_ylabel('石块数量', fontsize=13)
ax.set_title('4,000块分层样本体积分布（成功 3,639 块）', fontsize=14, fontweight='bold')
med = np.median(vpred)
ax.axvline(med, color='crimson', linestyle='--', linewidth=1.6, label=f'中位数 = {med:.4f} m³')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
p1 = r'0826\fig_volume_histogram.png'
plt.savefig(p1, dpi=120, bbox_inches='tight')
plt.close()
im1 = Image.open(p1)
print(f'histogram: {im1.size}')

# --- Figure 2: Stratum median + total volume ---
strat_names = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
strat_labels = ['最小10%', 'P10–P25', 'P25–P50', 'P50–P75', 'P75–P90', '最大10%']
medians, totals, counts = [], [], []
for s in strat_names:
    mask = strata_arr == s
    if mask.sum() > 0:
        medians.append(np.median(vpred[mask]))
        totals.append(float(vpred[mask].sum()))
        counts.append(int(mask.sum()))
    else:
        medians.append(0); totals.append(0); counts.append(0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=120)

# Median volume per stratum
colors = ['#6baed6','#4292c6','#2171b5','#08519c','#08306b','#081d58']
bars1 = ax1.bar(strat_labels, medians, color=colors, edgecolor='white')
ax1.set_yscale('log')
ax1.set_ylabel('体积中位数 (m³)', fontsize=12)
ax1.set_title('各粒径层 — 体积中位数', fontsize=13, fontweight='bold')
for b, v, c in zip(bars1, medians, counts):
    ax1.text(b.get_x() + b.get_width()/2, v*1.5, f'{v:.3e}\n(n={c})', ha='center', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Total volume per stratum
colors2 = ['#74c476','#41ab5d','#238b45','#006d2c','#00441b','#00250f']
bars2 = ax2.bar(strat_labels, totals, color=colors2, edgecolor='white')
ax2.set_yscale('log')
ax2.set_ylabel('样本体积总量 (m³)', fontsize=12)
ax2.set_title('各粒径层 — 体积总量（样本内）', fontsize=13, fontweight='bold')
for b, v in zip(bars2, totals):
    ax2.text(b.get_x() + b.get_width()/2, v*1.5, f'{v:.2f}', ha='center', fontsize=10)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
p2 = r'0826\fig_volume_stratum.png'
plt.savefig(p2, dpi=120, bbox_inches='tight')
plt.close()
im2 = Image.open(p2)
print(f'stratum: {im2.size}')

# Rename with dimensions
import shutil
for p in [p1, p2]:
    im = Image.open(p)
    w, h = im.size
    new = p.replace('.png', f'_{w}x{h}.png')
    shutil.copy(p, new)
    print(f'  -> {new.split(chr(92))[-1]}')

# Print key stats
print(f"\n=== Key Stats ===")
print(f"n_success = {len(vpred)}")
print(f"V_pred median = {np.median(vpred):.6f} m³")
print(f"V_pred P25/P75 = {np.percentile(vpred,25):.6f} / {np.percentile(vpred,75):.6f}")
print(f"V_pred total = {vpred.sum():.2f} m³")
print(f"V_2.5D total = {v25.sum():.2f} m³")
print(f"Overall V_pred/V_2.5D = {vpred.sum()/v25.sum():.4f}")
print(f"diameter range: {diam.min():.4f} - {diam.max():.4f} m, median {np.median(diam):.4f} m")

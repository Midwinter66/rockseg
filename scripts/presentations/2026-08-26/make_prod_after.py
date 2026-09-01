"""Regenerate cross-scale fusion demo image using PRODUCTION cascade result.

Before = panel_a_raw_all.png (raw multi-scale detections, same model & tiles)
After  = production cascade_deduplication final instances (rock_instances.json)
"""
from PIL import Image
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

for fn in ['Microsoft YaHei', 'SimHei']:
    if any(fn in f.name for f in fm.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [fn]
        plt.rcParams['axes.unicode_minus'] = False
        break

# Demo region (same as visualize_fusion.py)
RX, RY, RW, RH = 4000, 12000, 2048, 2048

# Load production final instances
with open('output/dom2_cascade_v2/rock_instances.json') as f:
    insts = json.load(f)
masks_npz = np.load('output/dom2_cascade_v2/rock_masks.npz')

region_insts = []
for i in insts:
    cx, cy = i['centroid']
    if RX <= cx <= RX + RW and RY <= cy <= RY + RH:
        region_insts.append(i)
print(f'production final instances in demo region: {len(region_insts)}')

# Diameter stats in region (area px^2, gsd 0.01)
diam = np.array([2 * np.sqrt(i['area'] * 1e-4 / np.pi) for i in region_insts])
print(f'region diameters(m): median={np.median(diam):.3f}  max={diam.max():.3f}')
print(f'  >= 0.5m: {(diam >= 0.5).sum()},  >= 1.0m: {(diam >= 1.0).sum()},  >= 2.0m: {(diam >= 2.0).sum()}')

# Read DOM background via PIL
dom_full = Image.open('data/dom2/DOM.tif')
region_img = dom_full.crop((RX, RY, RX + RW, RY + RH))
img = np.array(region_img)
dom_full.close()

# Draw production instances with distinct colors per instance
rng = np.random.RandomState(42)
after = img.copy()
H, W = RH, RW
for idx, i in enumerate(region_insts):
    bx1, by1, bx2, by2 = i['bbox']
    m = masks_npz[f"{i['instance_id']}_mask"]
    # local position
    lx1 = int(round(bx1)) - RX
    ly1 = int(round(by1)) - RY
    mh, mw = m.shape
    # clip to region
    ix1, iy1 = max(0, lx1), max(0, ly1)
    ix2 = min(W, lx1 + mw)
    iy2 = min(H, ly1 + mh)
    if ix1 >= ix2 or iy1 >= iy2:
        continue
    sub = m[iy1 - ly1:iy2 - ly1, ix1 - lx1:ix2 - lx1]
    color = tuple(int(c) for c in rng.randint(60, 255, 3))
    overlay = after[iy1:iy2, ix1:ix2]
    overlay[sub] = (overlay[sub].astype(np.float32) * 0.45
                    + np.array(color, dtype=np.float32) * 0.55).astype(np.uint8)

# Before image = existing panel_a
panel_a = Image.open('output/dom2_full/visualizations/fusion_demo/panel_a_raw_all.png')
after_img = Image.fromarray(after)

# Same crop as current PPT fig (bigrock_g: panel center 600,200 size 380)
cx, cy, sz = 600, 200, 380
x0, y0 = cx - sz // 2, cy - sz // 2
crop_a = panel_a.crop((x0, y0, x0 + sz, y0 + sz))
crop_c = after_img.crop((x0, y0, x0 + sz, y0 + sz))

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), dpi=130)
axes[0].imshow(crop_a)
axes[0].set_title('融合前（多尺度原始检测）', fontsize=13, fontweight='bold')
axes[0].axis('off')
axes[1].imshow(crop_c)
axes[1].set_title('融合后（生产级联去重结果）', fontsize=13, fontweight='bold')
axes[1].axis('off')
plt.tight_layout()
out = '0826/fig_cross_scale_prod_968x488.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
plt.close()

im = Image.open(out)
print(f'saved: {out} {im.size}')

# Also verify: production instances inside the crop itself
crop_sx, crop_sy = RX + x0, RY + y0
inside = [i for i in region_insts
          if crop_sx <= i['centroid'][0] <= crop_sx + sz
          and crop_sy <= i['centroid'][1] <= crop_sy + sz]
diam_in = [2 * np.sqrt(i['area'] * 1e-4 / np.pi) for i in inside]
print(f'instances inside crop: {len(inside)}')
if inside:
    print('crop diameters(m):', sorted(f'{d:.2f}' for d in diam_in))

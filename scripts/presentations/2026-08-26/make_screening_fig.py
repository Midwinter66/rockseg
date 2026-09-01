import json
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

with open(r'output\dom2_cascade_v2_3d_fixed\rejected_instances.json') as f:
    rej = json.load(f)
with open(r'output\dom2_cascade_v2_3d_fixed\accepted_instances.json') as f:
    acc = json.load(f)

cx, cy = 1314, 6055
W = 900
region_rej = [r for r in rej if abs((r['bbox'][0]+r['bbox'][2])/2-cx) < W and abs((r['bbox'][1]+r['bbox'][3])/2-cy) < W]
region_acc = [r for r in acc if abs((r['bbox'][0]+r['bbox'][2])/2-cx) < W and abs((r['bbox'][1]+r['bbox'][3])/2-cy) < W]
print('in region: rejected', len(region_rej), 'accepted', len(region_acc))

x0, y0 = cx - W, cy - W
dom = Image.open(r'data\dom2\DOM.tif')
img = dom.crop((x0, y0, x0 + 2 * W, y0 + 2 * W)).convert('L')
img = np.asarray(img)
print('crop', img.shape)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 7), dpi=100)
for ax, title in zip(axes, ['Before 2D-3D screening', 'After 2D-3D screening']):
    ax.imshow(img, cmap='gray', vmin=np.percentile(img, 2), vmax=np.percentile(img, 98))
    ax.set_title(title, fontsize=15)
    ax.axis('off')
    for r in region_acc:
        b = r['bbox']
        ax.add_patch(Rectangle((b[0]-x0, b[1]-y0), b[2]-b[0], b[3]-b[1],
                               edgecolor='lime', facecolor='none', lw=1.0))
axes[0].text(0.02, 0.03, 'green = accepted    red = rejected', transform=axes[0].transAxes,
             color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.65))
for r in region_rej:
    b = r['bbox']
    axes[0].add_patch(Rectangle((b[0]-x0, b[1]-y0), b[2]-b[0], b[3]-b[1],
                                edgecolor='red', facecolor='none', lw=2.5))
plt.tight_layout()
plt.savefig(r'0826\fig_2d3d_screening.png', dpi=100, bbox_inches='tight')
print('saved 0826\\fig_2d3d_screening.png')

from PIL import Image as I2
im2 = I2.open(r'0826\fig_2d3d_screening.png')
print('final size:', im2.size)
im2.save(r'0826\fig_2d3d_screening_%dx%d.png' % im2.size)
print('renamed with dims')

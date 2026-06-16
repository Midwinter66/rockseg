"""
切片 Web 调参器 — 实时调整切片参数并观察 tile 覆盖效果

用法:
  cd D:\github_project\image_segment\DOM_Space_message_val
  python experiments/slicing/web_tuner.py

然后在浏览器打开 http://localhost:5000

支持:
  - 下拉菜单切换 4 种切片方法
  - 每种方法有专属参数滑条
  - 实时叠加图 + 统计信息
  - 上传自定义 DOM 图片 (tif/png/jpg)
"""

from __future__ import annotations
import argparse, json, sys, math, time, base64, io, os
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "v2" / "src"))

import cv2, numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000

from flask import Flask, request, jsonify, render_template_string

# ── 默认数据路径 ─────────────────────────────────────────────────────
DEFAULT_DOM_PATH = PROJECT_ROOT / "data" / "dom3" / "DOM.tif"
DEFAULT_TFW_PATH = PROJECT_ROOT / "data" / "dom3" / "DOM.tfw"
POINTCLOUD_DIR = PROJECT_ROOT / "data" / "pointcloud2"

SELF_DIR = Path(__file__).resolve().parent

# ── 全局状态 ─────────────────────────────────────────────────────────
_dom_bgr: np.ndarray | None = None      # 当前 DOM (BGR uint8)
_dom_w: int = 0
_dom_h: int = 0
_dom_path: str = ""                      # 当前 DOM 文件名
_gt: tuple | None = None                 # TFW GeoTransform (如有)
_pc_points: np.ndarray | None = None     # 缓存的点云
_has_pc: bool = False

app = Flask(__name__)


# ══════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════

def _parse_tfw(tfw_path: str | Path) -> tuple:
    lines = [float(l.strip()) for l in Path(tfw_path).read_text("utf-8").splitlines() if l.strip()]
    return (lines[4], lines[0], lines[2], lines[5], lines[1], lines[3])


def _pixel_to_world(gt: tuple, px: float, py: float) -> tuple[float, float]:
    return float(gt[0] + px * gt[1] + py * gt[2]), float(gt[3] + px * gt[4] + py * gt[5])


def _build_positions(limit: int, size: int, stride: int, include_edge: bool = True) -> list[int]:
    if limit <= size:
        return [0]
    pos = list(range(0, limit - size + 1, stride))
    if include_edge and pos[-1] != limit - size:
        pos.append(limit - size)
    return sorted(set(pos))


def _compute_content_ratio(patch: np.ndarray, black_th: int) -> float:
    if patch.ndim == 3:
        gray = patch.mean(axis=2)
    else:
        gray = patch
    return float(np.count_nonzero(gray > black_th) / max(gray.size, 1))


# ── SAHI ─────────────────────────────────────────────────────────────

def _run_sahi(config: dict, dom: np.ndarray, gt: tuple | None) -> dict:
    pc = config["patching"]
    patch_sz = int(pc["patch_size"])
    overlap = float(pc["overlap"])
    stride = max(1, int(round(patch_sz * (1 - overlap))))
    black_th = int(pc["black_pixel_threshold"])
    min_content = float(pc["min_content_ratio"])
    include_edge = bool(pc.get("include_edge_patches", True))

    h, w = dom.shape[:2]
    xs = _build_positions(w, patch_sz, stride, include_edge)
    ys = _build_positions(h, patch_sz, stride, include_edge)

    records = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            patch = dom[y:y + patch_sz, x:x + patch_sz]
            cr = _compute_content_ratio(patch, black_th)
            rec = {
                "patch_id": f"patch_{len(records):06d}",
                "pixel_origin": [x, y],
                "pixel_size": patch_sz,
                "content_ratio": round(cr, 4),
                "status": "kept" if cr >= min_content else "skipped_black",
            }
            if gt is not None:
                wx0, wy0 = _pixel_to_world(gt, x, y)
                wx1, wy1 = _pixel_to_world(gt, x + patch_sz, y + patch_sz)
                rec["world_bounds"] = [min(wx0, wx1), min(wy0, wy1), max(wx0, wx1), max(wy0, wy1)]
            records.append(rec)

    kept = [r for r in records if r["status"] == "kept"]
    return {"records": records, "kept": len(kept), "skipped": len(records) - len(kept)}


# ── 四叉树 (点云密度) ────────────────────────────────────────────────

def _run_quadtree_pc(config: dict, pc_points: np.ndarray, dom_bounds: list[float]) -> dict:
    from nextgen.quadtree import QuadTreeCover  # noqa
    from nextgen.models import Tile  # noqa

    cc = config["cover"]
    qt = QuadTreeCover(dom_bounds, float(cc["base_tile_size_m"]),
                       float(cc["min_tile_size_m"]), float(cc["max_tile_size_m"]))
    tiles = qt.generate(pc_points, float(cc["min_density_points"]))
    records = []
    for t in tiles:
        low_content = False  # 点云法不做内容过滤
        records.append({
            "tile_id": t.tile_id,
            "bounds_m": t.bounds_m,
            "source_points": int(t.source_points),
            "density_score": round(float(t.density_score), 4),
            "content_ratio": 1.0,
            "skipped": bool(t.source_points == 0) or low_content,
            "skip_reason": "no_points" if t.source_points == 0 else ("low_content" if low_content else ""),
        })
    kept = [r for r in records if not r["skipped"]]
    return {"records": records, "kept": len(kept), "skipped": len(records) - len(kept)}


# ── 四叉树 (DOM 纹理) ───────────────────────────────────────────────

class DOMTextureQuadTree:
    def __init__(self, bounds, base_size, min_size, max_size):
        self.bounds = bounds
        self.base_size = base_size
        self.min_size = min_size
        self.max_size = max_size

    def generate(self, dom_image, edge_density_threshold, canny_low=50, canny_high=150,
                 black_threshold=5, min_content_ratio=0.0):
        xmin, ymin, xmax, ymax = self.bounds
        h_img, w_img = dom_image.shape[:2]
        dom_area_m = (xmax - xmin, ymax - ymin)
        gray_full = cv2.cvtColor(dom_image, cv2.COLOR_BGR2GRAY)
        edges_full = cv2.Canny(gray_full, canny_low, canny_high)

        def w2p(wx, wy):
            px = int((wx - xmin) / dom_area_m[0] * w_img)
            py = int((ymax - wy) / dom_area_m[1] * h_img)
            return max(0, min(px, w_img - 1)), max(0, min(py, h_img - 1))

        nx = max(1, int(math.ceil((xmax - xmin) / self.base_size)))
        ny = max(1, int(math.ceil((ymax - ymin) / self.base_size)))
        queue = []
        for ix in range(nx):
            for iy in range(ny):
                tx0 = xmin + ix * self.base_size
                ty0 = ymin + iy * self.base_size
                tx1 = min(xmax, tx0 + self.base_size)
                ty1 = min(ymax, ty0 + self.base_size)
                queue.append({"tile_id": f"tile_{ix}_{iy}", "bounds_m": [tx0, ty0, tx1, ty1]})

        final = []
        while queue:
            tile = queue.pop(0)
            b = tile["bounds_m"]
            w = b[2] - b[0]
            h = b[3] - b[1]
            px0, py0 = w2p(b[0], b[3])
            px1, py1 = w2p(b[2], b[1])
            px0, px1 = sorted([px0, px1])
            py0, py1 = sorted([py0, py1])

            edge_crop = edges_full[py0:py1, px0:px1]
            edge_count = int(np.count_nonzero(edge_crop))
            total_px = max(edge_crop.size, 1)
            edge_density = float(edge_count / total_px)

            gray_crop = gray_full[py0:py1, px0:px1]
            content_ratio = float(np.count_nonzero(gray_crop > black_threshold) / total_px)

            tile["edge_density"] = round(edge_density, 4)
            tile["content_ratio"] = round(content_ratio, 4)

            if content_ratio < min_content_ratio:
                tile["skipped"] = True
                tile["skip_reason"] = "black"
                final.append(tile)
                continue
            if edge_count == 0:
                tile["skipped"] = True
                tile["skip_reason"] = "no_edges"
                final.append(tile)
                continue
            if edge_density >= edge_density_threshold and max(w, h) > self.min_size:
                mx = (b[0] + b[2]) / 2.0
                my = (b[1] + b[3]) / 2.0
                for ci, (dx, dy) in enumerate([(0,0),(1,0),(0,1),(1,1)]):
                    queue.insert(0, {
                        "tile_id": f"{tile['tile_id']}_{ci}",
                        "bounds_m": [b[0]+dx*(mx-b[0]), b[1]+dy*(my-b[1]),
                                     mx+dx*(b[2]-mx), my+dy*(b[3]-my)],
                    })
            else:
                tile["skipped"] = False
                tile["skip_reason"] = ""
                final.append(tile)
        return final


def _run_quadtree_dom(config: dict, dom: np.ndarray, dom_bounds: list[float]) -> dict:
    cc = config["cover"]
    qt = DOMTextureQuadTree(dom_bounds, float(cc["base_tile_size_m"]),
                            float(cc["min_tile_size_m"]), float(cc["max_tile_size_m"]))
    tiles = qt.generate(dom,
                        float(cc["min_edge_density"]),
                        int(cc.get("canny_low", 50)),
                        int(cc.get("canny_high", 150)),
                        black_threshold=int(cc.get("black_pixel_threshold", 5)),
                        min_content_ratio=float(cc.get("min_content_ratio", 0.0)))
    kept = [t for t in tiles if not t["skipped"]]
    return {"records": tiles, "kept": len(kept), "skipped": len(tiles) - len(kept)}


# ── 可视化叠加 ──────────────────────────────────────────────────────

def _draw_sahi_overlay(dom_bgr: np.ndarray, records: list[dict]) -> np.ndarray:
    scale = min(2000 / dom_bgr.shape[1], 1200 / dom_bgr.shape[0], 1.0)
    if scale < 1.0:
        vis = cv2.resize(dom_bgr, (int(dom_bgr.shape[1]*scale), int(dom_bgr.shape[0]*scale)))
    else:
        vis = dom_bgr.copy()
    for r in records:
        x = int(r["pixel_origin"][0] * scale)
        y = int(r["pixel_origin"][1] * scale)
        sz = int(r["pixel_size"] * scale)
        if r["status"] == "kept":
            cv2.rectangle(vis, (x, y), (x+sz, y+sz), (46, 204, 113), 2)
        else:
            cv2.rectangle(vis, (x, y), (x+sz, y+sz), (231, 76, 60), -1)
    return vis


def _draw_quadtree_overlay(dom_bgr: np.ndarray, records: list[dict],
                           dom_bounds: list[float]) -> np.ndarray:
    scale = min(2000 / dom_bgr.shape[1], 1200 / dom_bgr.shape[0], 1.0)
    if scale < 1.0:
        vis = cv2.resize(dom_bgr, (int(dom_bgr.shape[1]*scale), int(dom_bgr.shape[0]*scale)))
    else:
        vis = dom_bgr.copy()
    sw, sh = vis.shape[1], vis.shape[0]
    dw, dh = dom_bgr.shape[1], dom_bgr.shape[0]
    dx, dy = dom_bounds[2]-dom_bounds[0], dom_bounds[3]-dom_bounds[1]
    if dx <= 0 or dy <= 0:
        return vis

    for r in records:
        b = r["bounds_m"]
        x0 = int((b[0]-dom_bounds[0])/dx*dw*scale)
        y0 = int((dom_bounds[3]-b[3])/dy*dh*scale)
        x1 = int((b[2]-dom_bounds[0])/dx*dw*scale)
        y1 = int((dom_bounds[3]-b[1])/dy*dh*scale)
        x0, x1 = sorted([max(0,x0), min(sw,x1)])
        y0, y1 = sorted([max(0,y0), min(sh,y1)])

        if r.get("skipped", False):
            cv2.rectangle(vis, (x0, y0), (x1, y1), (128, 128, 128), -1)
        else:
            density = r.get("edge_density", r.get("density_score", 0))
            intensity = min(255, int(density * 500))
            color = (intensity, 80, 255-intensity)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), 1)

    return vis


# ── DOM 边界计算 ────────────────────────────────────────────────────

def _get_dom_bounds(gt: tuple | None, w: int, h: int) -> list[float]:
    if gt is not None:
        xmin, ymax = _pixel_to_world(gt, 0, 0)
        xmax, ymin = _pixel_to_world(gt, w, h)
        return [min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)]
    return [0.0, 0.0, float(w), float(h)]


# ══════════════════════════════════════════════════════════════════════
#  Flask 路由
# ══════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>切片参数调参器</title>
<style>
* { box-sizing: border-box; }
body { margin:0; padding:12px; font-family:system-ui; background:#1a1a2e; color:#eee; }
.controls { position:sticky; top:0; background:#16213e; padding:12px 16px; border-radius:8px; z-index:10; margin-bottom:12px; }
.row { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
.row label { font-size:12px; color:#888; display:block; }
.row input[type=range] { width:100px; }
.row .val { font-size:11px; color:#e94560; min-width:32px; text-align:center; }
.col { display:flex; flex-direction:column; gap:2px; }
.stat { background:#0f3460; padding:6px 12px; border-radius:4px; font-size:13px; }
.stat strong { color:#e94560; }
.canvas-wrap { overflow:auto; border-radius:4px; border:1px solid #333; }
canvas { display:block; }
select, button { background:#0f3460; color:#eee; border:1px solid #555; padding:5px 12px; border-radius:4px; cursor:pointer; font-size:13px; }
button { background:#e94560; border:none; }
button.secondary { background:#0f3460; }
#upload-area { display:none; background:#0f3460; padding:8px; border-radius:4px; }
</style>
</head>
<body>
<div class="controls">
  <div class="row">
    <div class="col">
      <label>Method</label>
      <select id="method" style="min-width:140px">
        <option value="sahi_baseline">SAHI baseline (10%)</option>
        <option value="sahi_dense">SAHI dense (35%)</option>
        <option value="quadtree_pointcloud">Quadtree (point cloud)</option>
        <option value="quadtree_dom">Quadtree (DOM texture)</option>
      </select>
    </div>
    <div id="params"></div>
    <div class="stat" id="stats">Processing...</div>
    <button onclick="doSlice()">⟳ Refresh</button>
    <button class="secondary" onclick="resetParams()">↺ Reset</button>
    <button class="secondary" onclick="toggleUpload()">📤 Upload</button>
  </div>
  <div id="upload-area">
    <input type="file" id="file-input" accept=".tif,.tiff,.png,.jpg,.jpeg" style="color:#eee;font-size:12px">
    <label style="font-size:11px;color:#888"> (DOM image, optional TFW uploads not supported via browser)</label>
    <button onclick="uploadImage()" style="font-size:12px">Load</button>
    <span id="upload-status" style="font-size:12px;color:#888"></span>
  </div>
</div>
<div class="canvas-wrap">
  <canvas id="canvas"></canvas>
</div>

<script>
const PARAMS = {
  sahi_baseline: {
    patch_size:        {label:'patch_size(px)', min:256, max:2048, step:64, value:1024},
    overlap:           {label:'overlap', min:0, max:0.5, step:0.05, value:0.10},
    black_pixel_threshold:{label:'black_th', min:0, max:50, step:1, value:5},
    min_content_ratio: {label:'min_content', min:0, max:0.8, step:0.05, value:0.25},
    include_edge_patches: {label:'edge_patches', type:'bool', value:true},
  },
  sahi_dense: {
    patch_size:        {label:'patch_size(px)', min:256, max:2048, step:64, value:1024},
    overlap:           {label:'overlap', min:0, max:0.5, step:0.05, value:0.35},
    black_pixel_threshold:{label:'black_th', min:0, max:50, step:1, value:5},
    min_content_ratio: {label:'min_content', min:0, max:0.8, step:0.05, value:0.25},
    include_edge_patches: {label:'edge_patches', type:'bool', value:true},
  },
  quadtree_pointcloud: {
    base_tile_size_m:  {label:'base(m)', min:10, max:80, step:5, value:40},
    min_tile_size_m:   {label:'min(m)', min:5, max:40, step:5, value:20},
    min_density_points:{label:'min_density(pts)', min:100, max:5000, step:100, value:2000},
    tile_overlap_m:    {label:'overlap(m)', min:0, max:10, step:0.5, value:6.0},
    black_pixel_threshold:{label:'black_th', min:0, max:50, step:1, value:5},
    min_content_ratio: {label:'min_content', min:0, max:0.8, step:0.05, value:0.35},
  },
  quadtree_dom: {
    base_tile_size_m:  {label:'base(m)', min:10, max:80, step:5, value:40},
    min_tile_size_m:   {label:'min(m)', min:5, max:40, step:5, value:20},
    min_edge_density:  {label:'edge_density', min:0.01, max:0.5, step:0.01, value:0.15},
    tile_overlap_m:    {label:'overlap(m)', min:0, max:10, step:0.5, value:6.0},
    black_pixel_threshold:{label:'black_th', min:0, max:50, step:1, value:5},
    min_content_ratio: {label:'min_content', min:0, max:0.8, step:0.05, value:0.35},
    canny_low:         {label:'canny_low', min:10, max:200, step:5, value:30},
    canny_high:        {label:'canny_high', min:10, max:400, step:5, value:90},
  },
};

let currentMethod = 'sahi_baseline';

function buildSliders() {
  const div = document.getElementById('params');
  const p = PARAMS[currentMethod];
  div.innerHTML = Object.entries(p).map(([k, v]) => {
    if (v.type === 'bool') {
      return `<div class="col"><label>${v.label}</label>
        <input type="checkbox" ${v.value?'checked':''} onchange="doSlice()" id="p_${k}" style="width:auto"></div>`;
    }
    return `<div class="col"><label>${v.label}</label>
      <input type="range" min="${v.min}" max="${v.max}" step="${v.step}" value="${v.value}" oninput="doSlice()" id="p_${k}">
      <span class="val" id="v_${k}">${v.value}</span></div>`;
  }).join('');
}

function getParams() {
  const p = PARAMS[currentMethod];
  const r = {method: currentMethod};
  Object.keys(p).forEach(k => {
    const el = document.getElementById('p_'+k);
    if (!el) return;
    if (p[k].type === 'bool') { r[k] = el.checked ? 1 : 0; }
    else { r[k] = parseFloat(el.value); document.getElementById('v_'+k).textContent = r[k]; }
  });
  return r;
}

function resetParams() {
  const p = PARAMS[currentMethod];
  Object.entries(p).forEach(([k, v]) => {
    const el = document.getElementById('p_'+k);
    if (!el) return;
    if (v.type === 'bool') { el.checked = v.value; }
    else { el.value = v.value; document.getElementById('v_'+k).textContent = v.value; }
  });
  doSlice();
}

let uploadedImage = null;

async function doSlice() {
  const params = getParams();
  if (uploadedImage) params.uploaded = uploadedImage;
  const resp = await fetch('/slice', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(params)
  });
  const data = await resp.json();
  document.getElementById('stats').innerHTML =
    `<strong>${data.kept}</strong> kept / <strong>${data.total}</strong> total &nbsp;|&nbsp;
     skipped: <strong>${data.skipped}</strong> &nbsp;|&nbsp;
     coverage: <strong>${data.coverage_ratio}%</strong> &nbsp;|&nbsp;
     ${data.elapsed_ms}ms`;
  const canvas = document.getElementById('canvas');
  const img = new Image();
  img.onload = () => { canvas.width = img.width; canvas.height = img.height;
    const ctx = canvas.getContext('2d'); ctx.drawImage(img, 0, 0); };
  img.src = 'data:image/png;base64,' + data.image;
}

function toggleUpload() {
  const el = document.getElementById('upload-area');
  el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

async function uploadImage() {
  const file = document.getElementById('file-input').files[0];
  if (!file) return;
  document.getElementById('upload-status').textContent = 'Uploading...';
  const formData = new FormData();
  formData.append('file', file);
  const resp = await fetch('/upload', { method: 'POST', body: formData });
  const data = await resp.json();
  document.getElementById('upload-status').textContent = data.status === 'ok' ? `Loaded: ${file.name}` : 'Error: '+data.error;
  if (data.status === 'ok') {
    uploadedImage = data.id;
    doSlice();
  }
}

document.getElementById('method').addEventListener('change', function(e) {
  currentMethod = e.target.value;
  buildSliders();
  doSlice();
});

buildSliders();
doSlice();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML


@app.route("/slice", methods=["POST"])
def slice_endpoint():
    global _dom_bgr, _dom_w, _dom_h, _gt, _pc_points

    data = request.get_json()
    method = data.get("method", "sahi_baseline")

    # 检查是否有上传的图片
    uploaded_id = data.get("uploaded")
    if uploaded_id:
        dom_path = SELF_DIR / "uploads" / f"{uploaded_id}.png"
        if dom_path.exists():
            dom = cv2.imread(str(dom_path))
            if dom is None:
                return jsonify({"status": "error", "error": "Failed to load uploaded image"})
            dom_bgr = dom
            gt = None
        else:
            return jsonify({"status": "error", "error": "Uploaded image not found"})
    else:
        if _dom_bgr is None:
            return jsonify({"status": "error", "error": "No DOM loaded"})
        dom_bgr = _dom_bgr
        gt = _gt

    t0 = time.perf_counter()

    dom_bounds = _get_dom_bounds(gt, dom_bgr.shape[1], dom_bgr.shape[0])

    if method in ("sahi_baseline", "sahi_dense"):
        config = {"patching": {
            "patch_size": int(data.get("patch_size", 1024)),
            "overlap": float(data.get("overlap", 0.15)),
            "black_pixel_threshold": int(data.get("black_pixel_threshold", 5)),
            "min_content_ratio": float(data.get("min_content_ratio", 0.25)),
            "include_edge_patches": bool(int(data.get("include_edge_patches", 1))),
        }}
        result = _run_sahi(config, dom_bgr, gt)
        overlay = _draw_sahi_overlay(dom_bgr, result["records"])
    elif method == "quadtree_pointcloud":
        if _pc_points is None:
            return jsonify({"status": "error", "error": "Point cloud not loaded"})
        config = {"cover": {
            "base_tile_size_m": float(data.get("base_tile_size_m", 40)),
            "min_tile_size_m": float(data.get("min_tile_size_m", 20)),
            "max_tile_size_m": float(data.get("base_tile_size_m", 40)),
            "min_density_points": float(data.get("min_density_points", 2000)),
        }}
        result = _run_quadtree_pc(config, _pc_points, dom_bounds)
        overlay = _draw_quadtree_overlay(dom_bgr, result["records"], dom_bounds)
    elif method == "quadtree_dom":
        config = {"cover": {
            "base_tile_size_m": float(data.get("base_tile_size_m", 40)),
            "min_tile_size_m": float(data.get("min_tile_size_m", 20)),
            "max_tile_size_m": float(data.get("base_tile_size_m", 40)),
            "min_edge_density": float(data.get("min_edge_density", 0.15)),
            "tile_overlap_m": float(data.get("tile_overlap_m", 6.0)),
            "black_pixel_threshold": int(data.get("black_pixel_threshold", 5)),
            "min_content_ratio": float(data.get("min_content_ratio", 0.35)),
            "canny_low": int(data.get("canny_low", 30)),
            "canny_high": int(data.get("canny_high", 90)),
        }}
        result = _run_quadtree_dom(config, dom_bgr, dom_bounds)
        overlay = _draw_quadtree_overlay(dom_bgr, result["records"], dom_bounds)
    else:
        return jsonify({"status": "error", "error": f"Unknown method: {method}"})

    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    # 缩小叠加图到适合网络传输
    oh, ow = overlay.shape[:2]
    max_side = 2000
    if max(ow, oh) > max_side:
        sc = max_side / max(ow, oh)
        overlay = cv2.resize(overlay, (int(ow*sc), int(oh*sc)))

    _, buf = cv2.imencode(".png", overlay)
    b64 = base64.b64encode(buf.tobytes()).decode()

    total = len(result["records"])
    kept = result["kept"]
    skipped = result["skipped"]
    coverage = round(kept / max(total, 1) * 100, 1)

    return jsonify({
        "status": "ok",
        "image": b64,
        "kept": kept,
        "skipped": skipped,
        "total": total,
        "coverage_ratio": coverage,
        "elapsed_ms": elapsed,
    })


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"status": "error", "error": "No file"})
    f = request.files["file"]
    upload_dir = SELF_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    fid = str(int(time.time()))
    ext = Path(f.filename).suffix if f.filename else ".png"
    save_path = upload_dir / f"{fid}{ext}"
    f.save(save_path)
    # 转为 PNG
    try:
        img = Image.open(save_path)
        out_path = upload_dir / f"{fid}.png"
        img.save(out_path)
        if ext.lower() not in (".png",):
            save_path.unlink()
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})
    return jsonify({"status": "ok", "id": fid, "name": f.filename})


# ══════════════════════════════════════════════════════════════════════
#  启动
# ══════════════════════════════════════════════════════════════════════

def main():
    global _dom_bgr, _dom_w, _dom_h, _dom_path, _gt, _pc_points, _has_pc

    parser = argparse.ArgumentParser(description="Slicing parameter web tuner")
    parser.add_argument("--port", type=int, default=5000, help="Port (default 5000)")
    parser.add_argument("--dom", default=None, help="Custom DOM path")
    parser.add_argument("--tfw", default=None, help="Custom TFW path")
    args = parser.parse_args()

    # 加载 DOM
    dom_path = Path(args.dom) if args.dom else DEFAULT_DOM_PATH
    if not dom_path.exists():
        print(f"ERROR: DOM not found: {dom_path}"); sys.exit(1)

    dom = Image.open(dom_path)
    _dom_w, _dom_h = dom.size
    _dom_bgr = np.array(dom)
    if _dom_bgr.ndim == 3 and _dom_bgr.shape[2] == 3:
        _dom_bgr = cv2.cvtColor(_dom_bgr, cv2.COLOR_RGB2BGR)
    _dom_path = dom_path.name

    # 加载 TFW
    tfw_path = Path(args.tfw) if args.tfw else DEFAULT_TFW_PATH
    if tfw_path.exists():
        _gt = _parse_tfw(tfw_path)
        print(f"  TFW loaded: {tfw_path.name}")
    else:
        _gt = None
        print(f"  No TFW found, using pixel coordinates")

    # 加载点云 (四叉树点云法需要)
    ply_files = sorted(POINTCLOUD_DIR.glob("*.ply"))
    if ply_files:
        import open3d as o3d
        print("  Loading point cloud...")
        all_pts = []
        for pf in ply_files:
            pcd = o3d.io.read_point_cloud(str(pf))
            pts = np.asarray(pcd.points)
            if len(pts) > 0:
                all_pts.append(pts)
        if all_pts:
            _pc_points = np.vstack(all_pts)
            _has_pc = True
            print(f"  Point cloud loaded: {len(_pc_points)} points")
    else:
        print(f"  No point cloud found in {POINTCLOUD_DIR}")

    print(f"\n{'='*60}")
    print(f"  Slicing Web Tuner")
    print(f"  DOM: {_dom_path} ({_dom_w}x{_dom_h})")
    print(f"  Point cloud: {'✓' if _has_pc else '✗'} (quadtree_pointcloud {'可用' if _has_pc else '不可用'})")
    print(f"  Open: http://localhost:{args.port}")
    print(f"{'='*60}\n")

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()

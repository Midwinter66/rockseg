"""
交互式融合参数调参器 - Web 版 (基于 Flask)

用法:
  cd D:\github_project\image_segment\DOM_Space_message_val
  python experiments/fusion/web_tuner.py --source sahi_baseline

然后在浏览器打开 http://localhost:5000

拖动滑条实时看 bbox 变化, 浏览器直接渲染, 不会变形
"""

from __future__ import annotations
import argparse, json, sys, math, io
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2, numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000

from flask import Flask, request, jsonify, send_file

DOM_PATH = PROJECT_ROOT / "data" / "dom3" / "DOM.tif"
DETECTION_OUTPUTS = PROJECT_ROOT / "experiments" / "detection" / "outputs"
SELF_DIR = Path(__file__).resolve().parent

SOURCES = ["sahi_baseline", "sahi_dense", "quadtree_pointcloud", "quadtree_dom"]

app = Flask(__name__)

# ── 全局状态 ────────────────────────────────────────────────────────
_detections: list[dict] = []
_gt: tuple = ()
_bg_bgr: np.ndarray | None = None
_scale: float = 1.0


def _parse_tfw(tfw_path):
    lines = [float(l.strip()) for l in Path(tfw_path).read_text("utf-8").splitlines() if l.strip()]
    return (lines[4], lines[0], lines[2], lines[5], lines[1], lines[3])

def _bbox_intersects(a, b, pad=0.0):
    return not (a[2]+pad<b[0]-pad or b[2]+pad<a[0]-pad or a[3]+pad<b[1]-pad or b[3]+pad<a[1]-pad)

def _bbox_iou(a, b):
    ix0, iy0 = max(a[0],b[0]), max(a[1],b[1])
    ix1, iy1 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0.0, ix1-ix0)*max(0.0, iy1-iy0)
    if inter == 0: return 0.0
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter/(area_a+area_b-inter)

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False
        if self.rank[ra] < self.rank[rb]: ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]: self.rank[ra] += 1
        return True
    def groups(self):
        g = defaultdict(list)
        for i in range(len(self.parent)):
            g[self.find(i)].append(i)
        return g

def heuristic_fuse(detections, same_th, cross_th, same_iou, cross_iou):
    n = len(detections)
    if n == 0: return []
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i+1, n):
            a, b = detections[i], detections[j]
            c1 = a.get("centroid_world", [0,0])
            c2 = b.get("centroid_world", [0,0])
            dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
            same_tile = a.get("source_patch_id") == b.get("source_patch_id")
            th = same_th if same_tile else cross_th
            iou_th = same_iou if same_tile else cross_iou
            if dist > th: continue
            b1 = a.get("bbox_world", [0,0,0,0])
            b2 = b.get("bbox_world", [0,0,0,0])
            if not _bbox_intersects(b1,b2): continue
            if iou_th > 0 and _bbox_iou(b1,b2) < iou_th: continue
            uf.union(i,j)
    return list(uf.groups().values())

def correlation_fuse(detections, sigma, pos_th, tile_boost, max_dist, require_bbox):
    n = len(detections)
    if n <= 1: return [list(range(n))] if n==1 else []
    edges = {}
    for i in range(n):
        c1 = detections[i].get("centroid_world", [0,0])
        bb1 = detections[i].get("bbox_world", [0,0,0,0])
        for j in range(i+1, n):
            c2 = detections[j].get("centroid_world", [0,0])
            dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
            if dist > max_dist: continue
            if require_bbox:
                bb2 = detections[j].get("bbox_world", [0,0,0,0])
                if not _bbox_intersects(bb1, bb2): continue
            w_dist = math.exp(-dist**2 / (2*sigma**2))
            boost = tile_boost if detections[i].get("source_patch_id") == detections[j].get("source_patch_id") else 1.0
            w = w_dist * boost
            edges[(i,j)] = w
    remaining = set(range(n))
    clusters = []
    while remaining:
        pivot = min(remaining)
        cluster = [pivot]
        remaining.remove(pivot)
        to_del = []
        for v in list(remaining):
            edge = (min(pivot,v), max(pivot,v))
            w = edges.get(edge, 0.0)
            if w >= pos_th:
                cluster.append(v)
                to_del.append(v)
        for v in to_del:
            remaining.discard(v)
        clusters.append(cluster)
    return clusters


# ══════════════════════════════════════════════════════════════════════
#  Flask 路由
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Fusion Tuner</title>
<style>
body { margin:0; padding:12px; font-family:system-ui; background:#1a1a2e; color:#eee; }
.controls { position:sticky; top:0; background:#16213e; padding:12px 16px; border-radius:8px; display:flex; flex-wrap:wrap; gap:12px; align-items:center; z-index:10; margin-bottom:12px; }
.controls label { font-size:12px; color:#888; display:block; }
.controls input[type=range] { width:120px; }
.controls .val { font-size:11px; color:#e94560; min-width:36px; text-align:center; }
.stat { background:#0f3460; padding:8px 14px; border-radius:4px; }
.stat strong { color:#e94560; }
.canvas-wrap { overflow:auto; border-radius:4px; }
canvas { display:block; }
.panel { display:flex; gap:8px; margin-bottom:8px; }
.legend { display:flex; gap:10px; align-items:center; }
.legend-item { display:flex; align-items:center; gap:4px; font-size:12px; }
.legend-swatch { width:14px; height:14px; border-radius:2px; }
button { background:#e94560; color:#fff; border:none; padding:6px 16px; border-radius:4px; cursor:pointer; font-size:13px; }
button:hover { background:#c23152; }
button.secondary { background:#0f3460; }
</style>
</head>
<body>
<div class="controls">
  <div>
    <label>Method</label>
    <select id="method" style="background:#0f3460;color:#eee;border:1px solid #333;padding:4px 8px;border-radius:4px">
      <option value="heuristic">Heuristic (2-layer)</option>
      <option value="correlation" selected>Correlation Clustering</option>
    </select>
  </div>
  <div id="param-sliders"></div>
  <div class="stat">
    处理中...
  </div>
  <button onclick="saveConfig()">💾 Save Config</button>
  <button class="secondary" onclick="resetParams()">↺ Reset</button>
</div>
<div class="canvas-wrap">
  <canvas id="canvas"></canvas>
</div>

<script>
const PARAMS = {
  heuristic: {
    same_th: {label:'same_th (m)', min:0.05, max:1.5, step:0.05, value:0.5},
    cross_th: {label:'cross_th (m)', min:0.1, max:2.0, step:0.05, value:0.8},
    same_iou: {label:'same_iou', min:0, max:0.8, step:0.05, value:0.5},
    cross_iou: {label:'cross_iou', min:0, max:0.8, step:0.05, value:0.3},
  },
  correlation: {
    sigma:     {label:'sigma', min:0.05, max:1.0, step:0.05, value:0.30},
    pos_th:    {label:'pos_th', min:0.1, max:0.9, step:0.05, value:0.55},
    tile_boost:{label:'tile_boost', min:1.0, max:2.5, step:0.05, value:1.10},
    max_dist:  {label:'max_dist (m)', min:1.0, max:8.0, step:0.5, value:5.0},
    require_bbox:{label:'require_bbox', min:0, max:1, step:1, value:1},
  }
};

let method = 'correlation';

function buildSliders() {
  const div = document.getElementById('param-sliders');
  const p = PARAMS[method];
  div.innerHTML = Object.entries(p).map(([k, v]) => {
    const checked = k === 'require_bbox' && v.value ? 'checked' : '';
    if (k === 'require_bbox') {
      return `<div><label>${v.label}</label><input type="checkbox" ${checked} onchange="onChange()" id="slider_${k}" style="width:auto"></div>`;
    }
    return `<div><label>${v.label}</label>
      <input type="range" min="${v.min}" max="${v.max}" step="${v.step}" value="${v.value}" oninput="onChange()" id="slider_${k}">
      <span class="val" id="val_${k}">${v.value}</span></div>`;
  }).join('');
}

function getParams() {
  const p = PARAMS[method];
  const result = {method};
  Object.keys(p).forEach(k => {
    if (k === 'require_bbox') {
      result[k] = document.getElementById(`slider_${k}`).checked ? 1 : 0;
    } else {
      result[k] = parseFloat(document.getElementById(`slider_${k}`).value);
      document.getElementById(`val_${k}`).textContent = result[k].toFixed(2);
    }
  });
  return result;
}

async function onChange() {
  const params = getParams();
  const resp = await fetch('/fuse', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(params)
  });
  const data = await resp.json();
  document.querySelector('.stat').innerHTML =
    `<strong>${data.stones}</strong> stones &nbsp;|&nbsp;
     <strong>${data.single}</strong> single &nbsp;|&nbsp;
     <strong>${data.merged}</strong> merged &nbsp;|&nbsp;
     <strong>${data.elapsed_ms}ms</strong>`;
  drawBboxes(data.bboxes);
}

function resetParams() {
  const p = PARAMS[method];
  Object.entries(p).forEach(([k, v]) => {
    const el = document.getElementById(`slider_${k}`);
    if (el) {
      if (k === 'require_bbox') { el.checked = v.value === 1; }
      else { el.value = v.value; document.getElementById(`val_${k}`).textContent = v.value; }
    }
  });
  onChange();
}

async function saveConfig() {
  const params = getParams();
  const resp = await fetch('/save', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(params)
  });
  const data = await resp.json();
  alert('Saved: ' + data.path);
}

document.getElementById('method').addEventListener('change', function(e) {
  method = e.target.value;
  buildSliders();
  onChange();
});

// ── Canvas 绘制 ──────────────────────────────────────────────
let bgImg = null;
fetch('/bg')
  .then(r => r.json())
  .then(data => {
    console.log('BG loaded, detections:', data.n_dets);
    bgImg = new Image();
    bgImg.onload = () => {
      console.log('Image drawn, size:', bgImg.width, 'x', bgImg.height);
      onChange();
    };
    bgImg.onerror = (e) => { console.error('Image load error:', e); };
    bgImg.src = data.bg_url;
    document.querySelector('.stat').innerHTML = '<strong>' + data.n_dets + '</strong> detections loaded';
  })
  .catch(e => console.error('Fetch /bg error:', e));

function drawBboxes(bboxes) {
  const canvas = document.getElementById('canvas');
  if (!bgImg) return;
  canvas.width = bgImg.width;
  canvas.height = bgImg.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bgImg, 0, 0);

  const colors = ['#2ecc71','#3498db','#9b59b6','#f1c40f','#e67e22','#e74c3c'];
  bboxes.forEach(b => {
    const color = colors[Math.min(b.sz-1, 5)];
    ctx.strokeStyle = color;
    ctx.lineWidth = b.sz === 1 ? 1 : 2;
    ctx.strokeRect(b.x0, b.y0, b.x1-b.x0, b.y1-b.y0);
    if (b.sz >= 2) {
      ctx.fillStyle = color;
      ctx.font = '12px system-ui';
      ctx.fillText(b.sz, b.x0+2, b.y0+14);
    }
  });

  // legend
  const legendData = [[1,'1 det'],[2,'2 dets'],[3,'3 dets'],[4,'4-5 dets'],[5,'6+ dets']];
  let lx = 10, ly = canvas.height - 40;
  legendData.forEach(([sz, label]) => {
    const color = colors[sz-1];
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly-8, 14, 14);
    ctx.fillText(label, lx+18, ly+5);
    lx += ctx.measureText(label).width + 28;
  });
}

buildSliders();
onChange();
</script>
</body>
</html>"""


@app.route("/bg")
def bg():
    global _bg_bgr, _detections
    # Encode background to PNG base64 for canvas
    import base64
    _, buf = cv2.imencode(".png", _bg_bgr)
    b64 = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
    return jsonify({
        "bg_url": b64,
        "n_dets": len(_detections),
    })


@app.route("/fuse", methods=["POST"])
def fuse():
    data = request.get_json()
    method = data.pop("method", "correlation")

    t0 = __import__("time").perf_counter()

    if method == "heuristic":
        groups = heuristic_fuse(_detections, data["same_th"], data["cross_th"],
                                data["same_iou"], data["cross_iou"])
    else:
        require_bbox = int(data.get("require_bbox", 1)) > 0
        groups = correlation_fuse(_detections, data["sigma"], data["pos_th"],
                                   data["tile_boost"], data["max_dist"], require_bbox)

    elapsed_ms = round((__import__("time").perf_counter() - t0) * 1000, 1)

    # 转像素坐标
    origin_x, res_x, _, origin_y, _, res_y = _gt

    def w2p_x(wx): return int((wx - origin_x) / abs(res_x) * _scale)
    def w2p_y(wy): return int((origin_y - wy) / abs(res_y) * _scale)

    bboxes = []
    for indices in groups:
        members = [_detections[i] for i in indices]
        bs = [m.get("bbox_world", [0,0,0,0]) for m in members]
        x0 = w2p_x(min(b[0] for b in bs))
        y0 = w2p_y(max(b[3] for b in bs))
        x1 = w2p_x(max(b[2] for b in bs))
        y1 = w2p_y(min(b[1] for b in bs))
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        bboxes.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "sz": len(indices)})

    n_single = sum(1 for g in groups if len(g) == 1)
    return jsonify({
        "stones": len(groups),
        "single": n_single,
        "merged": len(groups) - n_single,
        "elapsed_ms": elapsed_ms,
        "bboxes": bboxes,
    })


@app.route("/save", methods=["POST"])
def save():
    data = request.get_json()
    method = data.pop("method", "correlation")

    source = app.config.get("SOURCE", "unknown")
    if method == "heuristic":
        cfg = {
            "_comment": f"Web tuner - {source}",
            "method": "heuristic",
            "association": {
                "same_tile_distance_m": round(data["same_th"], 2),
                "cross_tile_distance_m": round(data["cross_th"], 2),
                "same_tile_iou_threshold": round(data["same_iou"], 2),
                "cross_tile_iou_threshold": round(data["cross_iou"], 2),
                "boundary_margin_m": 0.3,
                "boundary_bbox_padding_m": 0.25,
            }
        }
    else:
        cfg = {
            "_comment": f"Web tuner - {source}",
            "method": "correlation_clustering",
            "correlation": {
                "max_distance_m": data["max_dist"],
                "distance_sigma": round(data["sigma"], 2),
                "positive_weight_threshold": round(data["pos_th"], 2),
                "use_iou": True, "iou_weight": 0.3,
                "use_tile_affinity": True,
                "same_tile_boost": round(data["tile_boost"], 2),
                "require_bbox_intersect": int(data.get("require_bbox", 1)) > 0,
            }
        }
    path = PROJECT_ROOT / "experiments" / "configs" / "fusion" / f"{method}_tuned.json"
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return jsonify({"status": "ok", "path": str(path)})


def main():
    global _detections, _gt, _bg_bgr, _scale

    parser = argparse.ArgumentParser(description="Web fusion parameter tuner")
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if not DOM_PATH.exists():
        print(f"DOM not found: {DOM_PATH}"); sys.exit(1)

    # 加载检测
    det_path = DETECTION_OUTPUTS / args.source / "detection_stats.json"
    _detections = json.loads(det_path.read_text(encoding="utf-8"))["detections"]
    _gt = _parse_tfw(PROJECT_ROOT / "data" / "dom3" / "DOM.tfw")

    # 加载背景 (缩小)
    dom = Image.open(DOM_PATH)
    w, h = dom.size
    _scale = 2000 / w  # 宽 2000, base64 体积小加载快
    nw, nh = int(w * _scale), int(h * _scale)
    dom_small = dom.resize((nw, nh), Image.LANCZOS)
    bg = np.array(dom_small)
    if bg.ndim == 3 and bg.shape[2] == 3:
        bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
    _bg_bgr = bg

    app.config["SOURCE"] = args.source

    print(f"\n{'='*60}")
    print(f"  Fusion Web Tuner")
    print(f"  Source: {args.source}  ({len(_detections)} detections)")
    print(f"  Open: http://localhost:{args.port}")
    print(f"{'='*60}\n")

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()

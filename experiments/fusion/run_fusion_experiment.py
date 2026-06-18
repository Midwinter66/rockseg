"""
融合实验 — 两种方法: heuristic（阈值合并） vs correlation_clustering（相关聚类）

用法:
  cd D:\github_project\image_segment\DOM_Space_message_val

  # 先跑检测
  python experiments/detection/run_detection_experiment.py --source all

  # 单源单方法
  python experiments/fusion/run_fusion_experiment.py --source sahi --method heuristic
  python experiments/fusion/run_fusion_experiment.py --source quadtree_dom --method correlation_clustering

  # 全部对比
  python experiments/fusion/run_fusion_experiment.py --source all --method all
"""

from __future__ import annotations
import argparse, json, sys, time, math
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

SELF_DIR = Path(__file__).resolve().parent
DETECTION_OUTPUTS = PROJECT_ROOT / "experiments" / "detection" / "outputs"
SOURCES = ["sahi", "quadtree_dom"]
FUSION_METHODS = ["heuristic", "correlation_clustering"]


# ══════════════════════════════════════════════════════════════════════
#  工具
# ══════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_fusion_config(method: str) -> dict:
    return _load_json(PROJECT_ROOT / "experiments" / "configs" / "fusion" / f"{method}.json")


def _load_detections(source: str) -> list[dict]:
    path = DETECTION_OUTPUTS / source / "detections.json"
    if not path.exists():
        raise FileNotFoundError(f"No detections for {source}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_output_dir(source: str, method: str) -> Path:
    d = SELF_DIR / "outputs" / source / method
    d.mkdir(parents=True, exist_ok=True)
    return d


# ══════════════════════════════════════════════════════════════════════
#  BBox 工具
# ══════════════════════════════════════════════════════════════════════

def _bbox_intersects(a: list[float], b: list[float], pad: float = 0.0) -> bool:
    return not (a[2] + pad < b[0] - pad or b[2] + pad < a[0] - pad or
                a[3] + pad < b[1] - pad or b[3] + pad < a[1] - pad)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# ══════════════════════════════════════════════════════════════════════
#  Union-Find
# ══════════════════════════════════════════════════════════════════════

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def groups(self) -> dict[int, list[int]]:
        g: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            g[self.find(i)].append(i)
        return g


# ══════════════════════════════════════════════════════════════════════
#  方法 A: 启发式融合
# ══════════════════════════════════════════════════════════════════════

def _heuristic_fuse(detections: list[dict], config: dict) -> list[list[int]]:
    """仅合并跨切片的重复检测（同一切片内 YOLO NMS 已处理）"""
    ac = config["association"]
    dist_th = float(ac["cross_tile_distance_m"])
    iou_th = float(ac["cross_tile_iou_threshold"])

    n = len(detections)
    if n == 0:
        return []
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = detections[i], detections[j]
            if a.get("source_patch_id") == b.get("source_patch_id"):
                continue
            c1, c2 = a.get("centroid_world", [0, 0]), b.get("centroid_world", [0, 0])
            dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            if dist > dist_th:
                continue
            b1, b2 = a.get("bbox_world", [0, 0, 0, 0]), b.get("bbox_world", [0, 0, 0, 0])
            if not _bbox_intersects(b1, b2):
                continue
            if iou_th > 0 and _bbox_iou(b1, b2) < iou_th:
                continue
            uf.union(i, j)

    return list(uf.groups().values())


# ══════════════════════════════════════════════════════════════════════
#  方法 B: 相关性聚类
# ══════════════════════════════════════════════════════════════════════

def _correlation_clustering_fuse(detections: list[dict], config: dict) -> list[list[int]]:
    """Pivot-based 3-approximation 相关聚类"""
    cc = config["correlation"]
    sigma = float(cc["distance_sigma"])
    pos_th = float(cc["positive_weight_threshold"])
    iou_w = float(cc.get("iou_weight", 0.3))
    tile_boost = float(cc.get("same_tile_boost", 1.5))
    use_iou = bool(cc.get("use_iou", True))
    max_dist = float(cc.get("max_distance_m", 5.0))

    n = len(detections)
    if n <= 1:
        return [list(range(n))] if n == 1 else []

    # 构建稀疏边
    edges: dict[tuple[int, int], float] = {}

    for i in range(n):
        c1 = detections[i].get("centroid_world", [0, 0])
        bbox1 = detections[i].get("bbox_world", [0, 0, 0, 0])
        for j in range(i + 1, n):
            c2 = detections[j].get("centroid_world", [0, 0])
            dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            if dist > max_dist:
                continue
            if cc.get("require_bbox_intersect"):
                bbox2 = detections[j].get("bbox_world", [0, 0, 0, 0])
                if not _bbox_intersects(bbox1, bbox2):
                    continue

            w_dist = math.exp(-dist**2 / (2 * sigma**2))
            iou = _bbox_iou(bbox1, detections[j].get("bbox_world", [0, 0, 0, 0])) if use_iou and _bbox_intersects(bbox1, detections[j].get("bbox_world", [0, 0, 0, 0])) else 0.0
            boost = tile_boost if tile_boost > 1.0 and detections[i].get("source_patch_id") == detections[j].get("source_patch_id") else 1.0
            edges[(i, j)] = w_dist * (1 + iou_w * iou) * boost

    # Pivot 算法
    remaining = set(range(n))
    clusters: list[list[int]] = []

    while remaining:
        pivot = min(remaining)
        cluster = [pivot]
        remaining.remove(pivot)
        to_remove = []
        for v in list(remaining):
            if edges.get((min(pivot, v), max(pivot, v)), 0.0) >= pos_th:
                cluster.append(v)
                to_remove.append(v)
        for v in to_remove:
            remaining.discard(v)
        clusters.append(cluster)

    return clusters


# ══════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════

def _run_fusion(source: str, method: str) -> dict:
    detections = _load_detections(source)
    config = _load_fusion_config(method)
    out_dir = _resolve_output_dir(source, method)

    t0 = time.perf_counter()
    if method == "heuristic":
        groups = _heuristic_fuse(detections, config)
    else:
        groups = _correlation_clustering_fuse(detections, config)
    elapsed = time.perf_counter() - t0

    stones = []
    for gi, indices in enumerate(groups):
        members = [detections[i] for i in indices]
        src_ids = list({m.get("source_patch_id", "") for m in members})
        bboxes = [m.get("bbox_world", [0, 0, 0, 0]) for m in members]
        scores = [m.get("score", 0) for m in members]
        stones.append({
            "stone_id": f"stone_{gi:06d}",
            "merge_method": method,
            "source_detection_count": len(members),
            "source_patches_span": len(src_ids),
            "score_mean": round(float(np.mean(scores)) if scores else 0, 4),
            "score_max": round(float(max(scores)) if scores else 0, 4),
            "bbox_world": [
                round(min(b[0] for b in bboxes), 4),
                round(min(b[1] for b in bboxes), 4),
                round(max(b[2] for b in bboxes), 4),
                round(max(b[3] for b in bboxes), 4),
            ],
            "detection_indices": indices,
        })

    stats = {
        "source": source,
        "method": method,
        "input_detections": len(detections),
        "output_stones": len(stones),
        "merge_ratio": round(1 - len(stones) / max(len(detections), 1), 4),
        "elapsed_seconds": round(elapsed, 4),
        "stones": stones,
    }

    (out_dir / "fusion_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    sizes = [len(g) for g in groups]
    print(f"  [{source}/{method}] {len(detections)} dets → {len(stones)} stones ({stats['merge_ratio']:.1%} merged) in {elapsed:.2f}s")
    print(f"    merge groups: min={min(sizes) if sizes else 0}  max={max(sizes) if sizes else 0}  mean={sum(sizes)/len(sizes):.1f}")

    return stats


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Run fusion experiment")
    parser.add_argument("--source", choices=["all"] + SOURCES, default="all",
                        help="切片方法")
    parser.add_argument("--method", choices=["all"] + FUSION_METHODS, default="all",
                        help="融合方法")
    args = parser.parse_args()

    methods = FUSION_METHODS if args.method == "all" else [args.method]
    sources = SOURCES if args.source == "all" else [args.source]

    print(f"\n{'='*60}")
    print(f"  Fusion Experiment")
    print(f"  Sources: {sources}")
    print(f"  Methods: {methods}")
    print(f"{'='*60}\n")

    all_results: dict[str, dict[str, dict]] = defaultdict(dict)
    for source in sources:
        for method in methods:
            try:
                stats = _run_fusion(source, method)
                all_results[source][method] = stats
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()

    (SELF_DIR / "outputs" / "fusion_manifest.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n结果: {SELF_DIR / 'outputs' / 'fusion_manifest.json'}")


if __name__ == "__main__":
    main()

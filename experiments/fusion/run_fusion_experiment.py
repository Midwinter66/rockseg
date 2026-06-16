"""
融合阶段实验 — 两种方法: 启发式双层融合 vs 相关性聚类

用法:
  cd D:\github_project\image_segment\DOM_Space_message_val

  # 先确保检测结果存在
  python experiments/detection/run_detection_experiment.py --source all

  # 单源对比
  python experiments/fusion/run_fusion_experiment.py --source sahi_baseline --method heuristic
  python experiments/fusion/run_fusion_experiment.py --source sahi_baseline --method correlation

  # 全部对比 (所有切片源 × 两种融合方法)
  python experiments/fusion/run_fusion_experiment.py --source all --method all

  # 生成对比报告
  python experiments/fusion/run_fusion_experiment.py --source all --method all --report
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

SOURCES = ["sahi_baseline", "sahi_dense", "quadtree_pointcloud", "quadtree_dom"]
FUSION_METHODS = ["heuristic", "correlation_clustering"]

# ...
# 然后映射: CLI 用 short name, 内部转真实文件名
METHOD_MAP = {
    "heuristic": "heuristic",
    "correlation": "correlation_clustering",
    "correlation_clustering": "correlation_clustering",
}


# ══════════════════════════════════════════════════════════════════════
#  工具
# ══════════════════════════════════════════════════════════════════════

def _load_fusion_config(method: str) -> dict:
    path = PROJECT_ROOT / "experiments" / "configs" / "fusion" / f"{method}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_detections(source: str) -> list[dict]:
    path = DETECTION_OUTPUTS / source / "detection_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"No detections for {source}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["detections"]


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
#  方法 A: 启发式双层融合
# ══════════════════════════════════════════════════════════════════════

def _heuristic_fuse(detections: list[dict], config: dict) -> list[list[int]]:
    ac = config["association"]
    same_th = float(ac["same_tile_distance_m"])
    cross_th = float(ac["cross_tile_distance_m"])
    same_iou = float(ac["same_tile_iou_threshold"])
    cross_iou = float(ac["cross_tile_iou_threshold"])

    n = len(detections)
    if n == 0:
        return []
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = detections[i], detections[j]
            c1 = a.get("centroid_world", [0, 0])
            c2 = b.get("centroid_world", [0, 0])
            dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

            same_tile = a.get("source_patch_id") == b.get("source_patch_id")
            threshold = same_th if same_tile else cross_th
            iou_th = same_iou if same_tile else cross_iou

            if dist > threshold:
                continue
            b1 = a.get("bbox_world", [0, 0, 0, 0])
            b2 = b.get("bbox_world", [0, 0, 0, 0])
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
    """Pivot-based 3-approximation for correlation clustering.

    边权重:
      w_ij = gauss_distance * (1 + iou_weight * iou) * tile_boost
      正边: w_ij >= positive_weight_threshold
      负边: w_ij <  positive_weight_threshold
    """
    cc = config["correlation"]
    sigma = float(cc["distance_sigma"])
    pos_th = float(cc["positive_weight_threshold"])
    iou_w = float(cc.get("iou_weight", 0.3))
    tile_boost = float(cc.get("same_tile_boost", 1.5))
    use_iou = bool(cc.get("use_iou", True))

    n = len(detections)
    if n <= 1:
        return [list(range(n))] if n == 1 else []

    # 构建稀疏邻接 (max_distance_m = 5m → 只连近距离的)
    max_dist = float(cc.get("max_distance_m", 5.0))
    neighbors: dict[int, set[int]] = defaultdict(set)
    edges: dict[tuple[int, int], float] = {}

    for i in range(n):
        c1 = detections[i].get("centroid_world", [0, 0])
        bbox1 = detections[i].get("bbox_world", [0, 0, 0, 0])
        for j in range(i + 1, n):
            c2 = detections[j].get("centroid_world", [0, 0])
            dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            if dist > max_dist:
                continue

            # 强制 bbox 相交的要求 (如果配置了)
            if cc.get("require_bbox_intersect"):
                bbox2 = detections[j].get("bbox_world", [0, 0, 0, 0])
                if not _bbox_intersects(bbox1, bbox2):
                    continue

            # 高斯距离权重: exp(-dist^2 / 2*sigma^2)
            w_dist = math.exp(-dist**2 / (2 * sigma**2))

            # IoU 加成
            iou = 0.0
            if use_iou:
                bbox2 = detections[j].get("bbox_world", [0, 0, 0, 0])
                if _bbox_intersects(bbox1, bbox2):
                    iou = _bbox_iou(bbox1, bbox2)

            # tile 亲和加成
            boost = 1.0
            if tile_boost > 1.0 and detections[i].get("source_patch_id") == detections[j].get("source_patch_id"):
                boost = tile_boost

            weight = w_dist * (1 + iou_w * iou) * boost
            edges[(i, j)] = weight
            neighbors[i].add(j)
            neighbors[j].add(i)

    # Pivot 算法
    remaining = set(range(n))
    clusters: list[list[int]] = []

    while remaining:
        pivot = min(remaining)  # 确定性: 取最小索引
        cluster = [pivot]
        remaining.remove(pivot)

        # 找与 pivot 有正边关系的节点
        to_remove = []
        for v in list(remaining):
            edge = (min(pivot, v), max(pivot, v))
            w = edges.get(edge, 0.0)
            if w >= pos_th:
                cluster.append(v)
                to_remove.append(v)

        for v in to_remove:
            remaining.discard(v)

        # 去掉与 cluster 内其他成员有负边的节点 (Pivot 的 refine)
        # 简化: 直接输出
        clusters.append(cluster)

    return clusters


# ══════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════

def _run_fusion(source: str, method: str) -> dict:
    detections = _load_detections(source)
    config = _load_fusion_config(METHOD_MAP.get(method, method))
    out_dir = _resolve_output_dir(source, method)

    t0 = time.perf_counter()

    if method == "heuristic":
        groups = _heuristic_fuse(detections, config)
    else:
        groups = _correlation_clustering_fuse(detections, config)

    elapsed = time.perf_counter() - t0

    # 汇总
    stones = []
    for gi, indices in enumerate(groups):
        members = [detections[i] for i in indices]
        bboxes = [m.get("bbox_world", [0, 0, 0, 0]) for m in members]
        scores = [m.get("score", 0) for m in members]
        src_ids = list({m.get("source_patch_id", "") for m in members})

        stone = {
            "stone_id": f"stone_{gi:06d}",
            "merge_method": method,
            "source_detection_count": len(members),
            "source_patch_ids": src_ids,
            "source_patches_span": len(src_ids),
            "score_mean": round(float(np.mean(scores)) if scores else 0, 4),
            "score_max": round(float(max(scores)) if scores else 0, 4),
            "bbox_world": [
                round(min(b[0] for b in bboxes), 4),
                round(min(b[1] for b in bboxes), 4),
                round(max(b[2] for b in bboxes), 4),
                round(max(b[3] for b in bboxes), 4),
            ],
            "diameters": [m.get("equivalent_diameter_m", 0) for m in members],
        }
        stones.append(stone)

    stats = {
        "source": source,
        "method": method,
        "config": config,
        "input_detections": len(detections),
        "output_stones": len(stones),
        "merge_ratio": round(1 - len(stones) / max(len(detections), 1), 4),
        "elapsed_seconds": round(elapsed, 4),
        "stones": stones,
    }

    (out_dir / "fusion_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # 分布
    sizes = [len(g) for g in groups]
    dist_lines = ["    size  count"]
    for k in sorted(set(sizes)):
        cnt = sizes.count(k)
        dist_lines.append(f"    {k:4d}  {cnt:5d}  {'█' * min(cnt, 40)}")
    print(f"  [{source}/{method}] {len(detections)} dets → {len(stones)} stones "
          f"in {elapsed:.2f}s")
    for line in dist_lines:
        print(line)

    return stats


def _build_report(results: dict[str, dict[str, dict]]) -> None:
    """生成简单的文本对比报告"""
    lines = []
    lines.append("Fusion Comparison Report")
    lines.append("=" * 60)
    for source, methods in results.items():
        lines.append(f"\n{source}:")
        for m, s in methods.items():
            lines.append(f"  {m}: {s['input_detections']} dets → "
                         f"{s['output_stones']} stones "
                         f"(merge_ratio={s['merge_ratio']:.2%}, "
                         f"elapsed={s['elapsed_seconds']:.3f}s)")
    text = "\n".join(lines)
    print(text)
    (SELF_DIR / "outputs" / "comparison_report.txt").write_text(text, encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Run fusion experiment")
    parser.add_argument("--source", choices=["all"] + SOURCES, default="all")
    parser.add_argument("--method", choices=["all"] + FUSION_METHODS, default="all")
    parser.add_argument("--report", action="store_true")
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
            print(f"── {source} / {method} ──")
            try:
                stats = _run_fusion(source, method)
                all_results[source][method] = stats
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()

    # manifest
    (SELF_DIR / "outputs" / "fusion_manifest.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nFusion manifest saved")

    if len(sources) > 1 or len(methods) > 1:
        _build_report(all_results)


if __name__ == "__main__":
    main()

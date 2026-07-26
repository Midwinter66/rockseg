"""
切片参数网格搜索

对 SAHI / Quadtree-DOM 的参数组合逐一跑切片，用切片质量指标（覆盖率、
效率、均匀性）评分，找最优参数。

用法:
  cd D:\github_project\image_segment\DOM_Space_message_val

  # 搜索 SAHI 切片参数
  python experiments/tuning/search_slicing.py --method sahi

  # 搜索 Quadtree 参数
  python experiments/tuning/search_slicing.py --method quadtree_dom

  # 只搜索指定参数（覆盖网格部分）
  python experiments/tuning/search_slicing.py --method sahi --fast
"""

from __future__ import annotations
import argparse, json, sys, itertools, math, time, copy
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.slicing.run_slicing_experiment import (
    _run_sahi, _run_quadtree_dom, _load_config, _resolve_output_dir,
)

SELF_DIR = Path(__file__).resolve().parent


def _summarize_numeric(values: list[float]) -> dict:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(arr.std(ddof=0)), 4),
    }


def _tile_uniformity(stats: dict) -> float:
    tile_sizes = stats.get("tile_sizes") or stats.get("patch_sizes") or []
    if tile_sizes:
        arr = np.asarray([float(v) for v in tile_sizes if float(v) > 0], dtype=np.float64)
        if arr.size >= 2:
            cv = float(arr.std(ddof=0) / max(arr.mean(), 1e-9))
            size_balance = 1.0 / (1.0 + cv)
        else:
            size_balance = 0.5
    else:
        size_balance = 0.5

    kept = float(stats.get("kept_patches", 0) or stats.get("kept_tiles", 0))
    total = float(stats.get("total_patches", 0) or stats.get("total_tiles", 0))
    if total > 0:
        keep_ratio = kept / total
        balance = 1.0 - abs(keep_ratio - 0.75) / 0.75
        balance = max(0.0, min(1.0, balance))
    else:
        balance = 0.0

    return round(0.6 * size_balance + 0.4 * balance, 4)


def _grid_values(param: dict) -> list[float]:
    """从 {min, max, steps} 生成等间隔值"""
    mn, mx, steps = float(param["min"]), float(param["max"]), int(param["steps"])
    if steps <= 1:
        return [(mn + mx) / 2]
    step = (mx - mn) / (steps - 1)
    return [round(mn + i * step, 4) for i in range(steps)]


def _score_slicing(stats: dict, weights: dict) -> float:
    """切片质量评分（无需检测，快速可得）

    指标：
      - 覆盖率 (coverage_ratio)：接近 1.0 最好
      - 效率 (kept/total)：高 → 不浪费算力在黑色区域
      - 均匀性 (CV of tile sizes)：CV 小 → 切分均匀
    """
    cov = stats.get("coverage_ratio", 0)
    kept = stats.get("kept_patches", 0) or stats.get("kept_tiles", 0)
    total = stats.get("total_patches", 0) or stats.get("total_tiles", 0)
    eff = kept / max(total, 1)

    # 覆盖率误差（偏离 1.0 的程度）
    cov_penalty = max(0, 1 - abs(1.0 - cov))
    # 覆盖率超 1 也要惩罚（重叠过多）
    if cov > 1.2:
        cov_penalty *= max(0, 1 - (cov - 1.2) * 2)

    w_cov = weights.get("slicing_coverage_weight", 0.5)
    w_eff = weights.get("slicing_efficiency_weight", 0.3)
    w_uni = weights.get("slicing_uniformity_weight", 0.2)
    uniformity = _tile_uniformity(stats)

    score = w_cov * cov_penalty + w_eff * eff + w_uni * uniformity
    return round(score, 4)


def _search_sahi(config: dict, params: dict, fast: bool = False) -> list[dict]:
    """SAHI 网格搜索"""
    overlap_vals = _grid_values(params["slicing"]["sahi"]["overlap"])
    ratio_vals = _grid_values(params["slicing"]["sahi"]["min_content_ratio"])

    if fast:
        overlap_vals = [0.05, 0.2, 0.35]
        ratio_vals = [0.15, 0.35]

    results = []
    total = len(overlap_vals) * len(ratio_vals)
    print(f"\n  SAHI 搜索: {len(overlap_vals)}×{len(ratio_vals)} = {total} 组合\n")

    for i, (ol, rc) in enumerate(itertools.product(overlap_vals, ratio_vals)):
        # 修改配置
        cfg = copy.deepcopy(config)
        cfg["patching"]["overlap"] = ol
        cfg["patching"]["min_content_ratio"] = rc

        t0 = time.perf_counter()
        try:
            out_dir = _resolve_output_dir(f"_tune_{i:03d}")
            stats = _run_sahi(cfg, out_dir)
            elapsed = time.perf_counter() - t0
            score = _score_slicing(stats, params.get("scoring", {}))
            results.append({
                "overlap": ol,
                "min_content_ratio": rc,
                "score": score,
                "coverage_ratio": stats.get("coverage_ratio", 0),
                "uniformity": _tile_uniformity(stats),
                "kept": stats.get("kept_patches", 0),
                "total": stats.get("total_patches", 0),
                "elapsed_s": round(elapsed, 2),
            })
            bar = "█" * int(score * 40) + "░" * (40 - int(score * 40))
            print(f"  [{i+1:3d}/{total}] overlap={ol:.2f} ratio={rc:.2f}  "
                  f"cov={stats.get('coverage_ratio',0):.3f}  score={score:.3f}  {bar}")
        except Exception as e:
            print(f"  [{i+1:3d}/{total}] overlap={ol:.2f} ratio={rc:.2f}  FAILED: {e}")

    # 排序
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _search_quadtree(config: dict, params: dict, fast: bool = False) -> list[dict]:
    """Quadtree-DOM 网格搜索"""
    density_vals = _grid_values(params["slicing"]["quadtree_dom"]["min_edge_density"])
    overlap_vals = _grid_values(params["slicing"]["quadtree_dom"]["tile_overlap_m"])
    ratio_vals = _grid_values(params["slicing"]["quadtree_dom"]["min_content_ratio"])

    if fast:
        density_vals = [0.05, 0.15, 0.3, 0.5]
        overlap_vals = [1.0, 3.0, 5.0]
        ratio_vals = [0.15, 0.3]

    results = []
    total = len(density_vals) * len(overlap_vals) * len(ratio_vals)
    print(f"\n  Quadtree 搜索: {len(density_vals)}×{len(overlap_vals)}×{len(ratio_vals)} = {total} 组合\n")

    for i, (den, ol, rc) in enumerate(itertools.product(density_vals, overlap_vals, ratio_vals)):
        cfg = copy.deepcopy(config)
        target = cfg.get("config", cfg)
        target["min_edge_density"] = den
        target["tile_overlap_m"] = ol
        target["min_content_ratio"] = rc

        t0 = time.perf_counter()
        try:
            out_dir = _resolve_output_dir(f"_tune_{i:03d}")
            stats = _run_quadtree_dom(cfg, out_dir)
            elapsed = time.perf_counter() - t0
            score = _score_slicing(stats, params.get("scoring", {}))
            results.append({
                "min_edge_density": den,
                "tile_overlap_m": ol,
                "min_content_ratio": rc,
                "score": score,
                "coverage_ratio": stats.get("coverage_ratio", 0),
                "uniformity": _tile_uniformity(stats),
                "kept": stats.get("kept_tiles", 0),
                "total": stats.get("total_tiles", 0),
                "elapsed_s": round(elapsed, 2),
            })
            bar = "█" * int(score * 40) + "░" * (40 - int(score * 40))
            print(f"  [{i+1:3d}/{total}] den={den:.2f} ol={ol:.1f} rc={rc:.2f}  "
                  f"cov={stats.get('coverage_ratio',0):.3f}  score={score:.3f}  {bar}")
        except Exception as e:
            print(f"  [{i+1:3d}/{total}] den={den:.2f} ol={ol:.1f} rc={rc:.2f}  FAILED: {e}")

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _report(results: list[dict], method: str, config_name: str) -> None:
    """输出搜索报告"""
    print(f"\n{'='*55}")
    print(f"  {method} 最佳参数 (Top 5)")
    print(f"{'='*55}")
    print(f"  {'#':>3}  score  参数                                            cov    kept/total")
    print(f"  {'-'*60}")

    for rank, r in enumerate(results[:5], 1):
        # 提取除 score/coverage/kept/total/elapsed_s 外的字段作为参数
        params_str = "  ".join(f"{k}={v}" for k, v in r.items()
                               if k not in ("score", "coverage_ratio", "kept", "total", "elapsed_s"))
        print(f"  {rank:3d}  {r['score']:.3f}  {params_str}")
        print(f"          cov={r['coverage_ratio']:.3f}  kept={r['kept']}/{r['total']}")

    # 保存
    out = {
        "method": method,
        "task": "slicing",
        "config_name": config_name,
        "best": results[0],
        "summary": {
            "score": _summarize_numeric([r["score"] for r in results]),
            "coverage_ratio": _summarize_numeric([r.get("coverage_ratio", 0.0) for r in results]),
            "uniformity": _summarize_numeric([r.get("uniformity", 0.0) for r in results]),
            "keep_ratio": _summarize_numeric([r.get("kept", 0) / max(r.get("total", 1), 1) for r in results]),
        },
        "top5": results[:5],
        "all": results,
    }
    out_path = SELF_DIR / "outputs" / f"slicing_search_{method}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  输出: {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="切片参数网格搜索")
    parser.add_argument("--method", choices=["sahi", "quadtree_dom", "both"], default="both")
    parser.add_argument("--fast", action="store_true", help="快速模式（少采样点）")
    args = parser.parse_args()

    config = json.loads((SELF_DIR / "config.json").read_text(encoding="utf-8"))

    methods = ["sahi", "quadtree_dom"] if args.method == "both" else [args.method]

    for method in methods:
        slicing_cfg = _load_config(method)
        if method == "sahi":
            results = _search_sahi(slicing_cfg, config, args.fast)
        else:
            results = _search_quadtree(slicing_cfg, config, args.fast)
        _report(results, method, slicing_cfg.get("method", method))


if __name__ == "__main__":
    main()

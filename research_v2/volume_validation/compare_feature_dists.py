"""Compare shape feature distributions: external training set vs DOM2 mine.

Reads cached external features + 4000-rock CSV, prints numerical comparison.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "datasets" / "t01_l01_scaled_10mm" / "cache"
MINE_CSV = ROOT / "real_mine_full" / "real_mine_volume_4000_results.csv"

FEATURES = ["C", "AR", "solidity", "compactness", "eq_diam_ratio",
            "H_mean_norm", "H_std_norm", "H_p25_norm", "H_p75_norm",
            "H_skew_norm", "fill_ratio", "ellipsoid_ratio", "y_pred"]


def load_external(dataset_id):
    data = {f: [] for f in FEATURES}
    folder = CACHE_DIR / dataset_id
    for p in sorted(folder.glob("*.json")):
        with open(p) as f:
            d = json.load(f)
        if d.get("status") == "success":
            for feat in FEATURES:
                key = feat if feat != "y_pred" else "y_ratio"
                if key in d:
                    data[feat].append(d[key])
    return data


def load_mine():
    data = {f: [] for f in FEATURES}
    n_success = 0
    with open(MINE_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"].upper() in ("SUCCESS", "PASS", "OK"):
                n_success += 1
                for feat in FEATURES:
                    try:
                        data[feat].append(float(row[feat]))
                    except (ValueError, KeyError):
                        pass
    return data, n_success


def main():
    print("=== Feature Distribution Comparison ===")
    print()

    t01 = load_external("T01")
    l01 = load_external("L01")
    mine, n_mine = load_mine()

    n_t01 = len(t01["C"])
    n_l01 = len(l01["C"])
    n_train = n_t01 + n_l01
    print(f"Training set: T01={n_t01}, L01={n_l01}, total={n_train}")
    print(f"Mine (DOM2 4000-sample, success): {n_mine}")
    print()

    # Combined training set
    train = {}
    for feat in FEATURES:
        train[feat] = t01[feat] + l01[feat]

    # Header
    header = (f"{'Feature':<20} {'Train mean':>10} {'Train std':>10} "
              f"{'Mine mean':>10} {'Mine std':>10} "
              f"{'Diff(σ)':>8} {'Overlap':>10}")
    print(header)
    print("-" * len(header))

    shifts = []
    for feat in FEATURES:
        t = np.array(train[feat])
        m = np.array(mine[feat])

        t_mean, t_std = np.mean(t), np.std(t)
        m_mean, m_std = np.mean(m), np.std(m)

        diff_sigma = (m_mean - t_mean) / t_std if t_std > 0 else 0

        # Overlap: what fraction of mine 90% range falls in training 90% range
        t_p5, t_p95 = np.percentile(t, [5, 95])
        m_p5, m_p95 = np.percentile(m, [5, 95])
        overlap_lo = max(t_p5, m_p5)
        overlap_hi = min(t_p95, m_p95)
        overlap_w = max(0, overlap_hi - overlap_lo)
        mine_w = m_p95 - m_p5
        overlap_pct = overlap_w / mine_w * 100 if mine_w > 0 else 0

        if abs(diff_sigma) < 0.5 and overlap_pct > 80:
            tag = "✅ 高"
        elif abs(diff_sigma) < 1.0 and overlap_pct > 60:
            tag = "⚠️ 中"
        else:
            tag = "❌ 低"

        print(f"{feat:<20} {t_mean:>10.4f} {t_std:>10.4f} "
              f"{m_mean:>10.4f} {m_std:>10.4f} "
              f"{diff_sigma:>+8.2f} {overlap_pct:>9.0f}% {tag}")

        shifts.append((abs(diff_sigma), feat, diff_sigma, overlap_pct))

    print()

    # Summary by category
    print("=== 分类总结 ===")
    print()

    cats = {
        "二维形态": ["C", "AR", "solidity", "compactness", "eq_diam_ratio"],
        "高度分布": ["H_mean_norm", "H_std_norm", "H_p25_norm", "H_p75_norm", "H_skew_norm"],
        "三维形状": ["fill_ratio", "ellipsoid_ratio"],
        "校正比": ["y_pred"],
    }

    for cat, feats in cats.items():
        avg_shift = np.mean([abs((np.mean(mine[f]) - np.mean(train[f])) / np.std(train[f]))
                             for f in feats])
        avg_overlap = np.mean([min(100, _overlap_pct(train[f], mine[f])) for f in feats])
        print(f"  {cat} ({len(feats)}项): 平均偏移 {avg_shift:.2f}σ, 平均重叠 {avg_overlap:.0f}%")

    print()

    # Top shifted features
    print("=== 偏移最大的5个特征 ===")
    shifts.sort(reverse=True)
    for abs_shift, feat, diff, overlap in shifts[:5]:
        direction = "矿区更高" if diff > 0 else "矿区更低"
        print(f"  {feat}: {abs_shift:.2f}σ ({direction}), 重叠 {overlap:.0f}%")

    print()

    # y_pred deep dive
    print("=== 校正比 y (最关键) ===")
    t_y = np.array(train["y_pred"])
    m_y = np.array(mine["y_pred"])
    print(f"  训练集: mean={np.mean(t_y):.4f}, std={np.std(t_y):.4f}, "
          f"P5={np.percentile(t_y,5):.4f}, P95={np.percentile(t_y,95):.4f}")
    print(f"  矿区 : mean={np.mean(m_y):.4f}, std={np.std(m_y):.4f}, "
          f"P5={np.percentile(m_y,5):.4f}, P95={np.percentile(m_y,95):.4f}")
    diff_y = np.mean(m_y) - np.mean(t_y)
    print(f"  均值差: {diff_y:+.4f} (= {diff_y/np.std(t_y):+.2f}σ)")

    # Fraction of mine within training range
    in_range = np.sum((m_y >= np.percentile(t_y, 2.5)) &
                      (m_y <= np.percentile(t_y, 97.5)))
    print(f"  矿区y落在训练集95%范围内: {in_range}/{len(m_y)} = {in_range/len(m_y)*100:.1f}%")

    print()
    print("=== 结论 ===")


def _overlap_pct(train_vals, mine_vals):
    t = np.array(train_vals)
    m = np.array(mine_vals)
    t_p5, t_p95 = np.percentile(t, [5, 95])
    m_p5, m_p95 = np.percentile(m, [5, 95])
    overlap_lo = max(t_p5, m_p5)
    overlap_hi = min(t_p95, m_p95)
    overlap_w = max(0, overlap_hi - overlap_lo)
    mine_w = m_p95 - m_p5
    return overlap_w / mine_w * 100 if mine_w > 0 else 100


if __name__ == "__main__":
    main()

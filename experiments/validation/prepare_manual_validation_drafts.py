from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = ROOT / "experiments" / "validation" / "manual_annotations"
WINDOW_SIZE_M = 15.0


SITES = [
    {
        "site_id": "site_a",
        "predictions_json": ROOT / "experiments" / "fusion" / "outputs" / "quadtree_dom" / "correlation_clustering" / "accepted_stones.json",
        "regions_csv": ANNOTATION_DIR / "site_a_validation_regions.csv",
        "boxes_csv": ANNOTATION_DIR / "site_a_manual_boxes.csv",
    },
    {
        "site_id": "site_b",
        "predictions_json": ROOT / "experiments" / "site_b_run" / "outputs" / "p1" / "fusion" / "quadtree_dom" / "correlation_clustering" / "accepted_stones.json",
        "regions_csv": ANNOTATION_DIR / "site_b_validation_regions.csv",
        "boxes_csv": ANNOTATION_DIR / "site_b_manual_boxes.csv",
    },
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_predictions(path: Path) -> list[dict]:
    payload = load_json(path)
    records = payload["stones"] if isinstance(payload, dict) and "stones" in payload else payload
    predictions = []
    for item in records:
        if "centroid_world" not in item or "bbox_world" not in item:
            continue
        predictions.append(item)
    return predictions


def cell_key(prediction: dict, cell_size: float) -> tuple[int, int]:
    x, y = prediction["centroid_world"]
    return (math.floor(float(x) / cell_size), math.floor(float(y) / cell_size))


def window_from_center(cx: float, cy: float, size: float) -> tuple[float, float, float, float]:
    half = size / 2.0
    return (round(cx - half, 4), round(cy - half, 4), round(cx + half, 4), round(cy + half, 4))


def point_in_window(prediction: dict, window: tuple[float, float, float, float]) -> bool:
    x, y = [float(v) for v in prediction["centroid_world"]]
    x0, y0, x1, y1 = window
    return x0 <= x <= x1 and y0 <= y <= y1


def overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def choose_regions(predictions: list[dict]) -> list[dict]:
    grid: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for pred in predictions:
        grid[cell_key(pred, WINDOW_SIZE_M)].append(pred)

    cells = []
    for key, items in grid.items():
        xs = [float(item["centroid_world"][0]) for item in items]
        ys = [float(item["centroid_world"][1]) for item in items]
        cells.append(
            {
                "key": key,
                "count": len(items),
                "cx": sum(xs) / len(xs),
                "cy": sum(ys) / len(ys),
                "window": window_from_center(sum(xs) / len(xs), sum(ys) / len(ys), WINDOW_SIZE_M),
            }
        )
    if not cells:
        return []

    chosen = []

    dense = max(cells, key=lambda row: row["count"])
    chosen.append({**dense, "scene_type": "dense_pile"})

    usable = [row for row in cells if not any(overlap(row["window"], c["window"]) for c in chosen)]
    if usable:
        positive = [row for row in usable if row["count"] >= 3]
        sparse_pool = positive or usable
        target = sorted(sparse_pool, key=lambda row: row["count"])[0]
        chosen.append({**target, "scene_type": "sparse_pile"})

    usable = [row for row in cells if not any(overlap(row["window"], c["window"]) for c in chosen)]
    if usable:
        min_x = min(float(item["centroid_world"][0]) for item in predictions)
        max_x = max(float(item["centroid_world"][0]) for item in predictions)
        min_y = min(float(item["centroid_world"][1]) for item in predictions)
        max_y = max(float(item["centroid_world"][1]) for item in predictions)

        def edge_score(row: dict) -> float:
            return min(row["cx"] - min_x, max_x - row["cx"], row["cy"] - min_y, max_y - row["cy"])

        edge = min(usable, key=edge_score)
        chosen.append({**edge, "scene_type": "edge_or_boundary"})

    return chosen[:3]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_site(site: dict) -> None:
    site_id = site["site_id"]
    predictions = read_predictions(site["predictions_json"])
    regions = choose_regions(predictions)

    region_rows = []
    box_rows = []
    for idx, region in enumerate(regions, start=1):
        region_id = f"{site_id.upper()}_R{idx:02d}"
        x0, y0, x1, y1 = region["window"]
        in_window = [pred for pred in predictions if point_in_window(pred, region["window"])]
        region_rows.append(
            {
                "site_id": site_id,
                "region_id": region_id,
                "x_min": x0,
                "y_min": y0,
                "x_max": x1,
                "y_max": y1,
                "scene_type": region["scene_type"],
                "ignore": 0,
                "notes": f"Auto-selected draft window; review all rocks >= 0.5 m. Prediction count in window: {len(in_window)}",
            }
        )
        for box_idx, pred in enumerate(in_window, start=1):
            bx0, by0, bx1, by1 = [round(float(v), 4) for v in pred["bbox_world"]]
            box_rows.append(
                {
                    "site_id": site_id,
                    "region_id": region_id,
                    "ann_id": f"{region_id}_DRAFT_{box_idx:04d}",
                    "x_min": bx0,
                    "y_min": by0,
                    "x_max": bx1,
                    "y_max": by1,
                    "label": "rock",
                    "ignore": 1,
                    "notes": "DRAFT_FROM_PREDICTION_REVIEW_REQUIRED",
                    "source_pred_id": pred.get("stone_id", pred.get("detection_id", "")),
                    "pred_score": pred.get("score_mean", pred.get("score", "")),
                    "pred_diameter_m": pred.get("equivalent_diameter_m", ""),
                }
            )

    write_csv(
        site["regions_csv"],
        ["site_id", "region_id", "x_min", "y_min", "x_max", "y_max", "scene_type", "ignore", "notes"],
        region_rows,
    )
    write_csv(
        site["boxes_csv"],
        [
            "site_id",
            "region_id",
            "ann_id",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "label",
            "ignore",
            "notes",
            "source_pred_id",
            "pred_score",
            "pred_diameter_m",
        ],
        box_rows,
    )
    print(site["regions_csv"])
    print(site["boxes_csv"])
    print(f"{site_id}: regions={len(region_rows)}, draft_boxes={len(box_rows)}")


def main() -> None:
    for site in SITES:
        write_site(site)


if __name__ == "__main__":
    main()

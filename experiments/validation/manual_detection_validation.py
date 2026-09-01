from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments" / "validation" / "manual_detection_validation_config.json"


@dataclass(frozen=True)
class Box:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "Box":
        x0, y0, x1, y1 = [float(value) for value in values]
        return cls(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @property
    def area(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def equivalent_diameter(self) -> float:
        return math.sqrt(4.0 * self.area / math.pi) if self.area > 0 else 0.0

    def contains_point(self, x: float, y: float, *, pad: float = 0.0) -> bool:
        return self.xmin - pad <= x <= self.xmax + pad and self.ymin - pad <= y <= self.ymax + pad

    def to_list(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass(frozen=True)
class Region:
    site_id: str
    region_id: str
    bbox: Box
    scene_type: str
    notes: str


@dataclass(frozen=True)
class Annotation:
    ann_id: str
    site_id: str
    region_id: str
    bbox: Box
    label: str
    notes: str

    @property
    def equivalent_diameter_m(self) -> float:
        return self.bbox.equivalent_diameter


@dataclass(frozen=True)
class Prediction:
    pred_id: str
    site_id: str
    bbox: Box
    centroid: tuple[float, float]
    score: float | None
    equivalent_diameter_m: float | None
    source: dict


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "ignore", "ignored"}


def has_box(row: dict[str, str]) -> bool:
    return all(row.get(key, "").strip() for key in ["x_min", "y_min", "x_max", "y_max"])


def iou(a: Box, b: Box) -> float:
    ix0 = max(a.xmin, b.xmin)
    iy0 = max(a.ymin, b.ymin)
    ix1 = min(a.xmax, b.xmax)
    iy1 = min(a.ymax, b.ymax)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def metric_row(tp: int, fp: int, fn: int) -> dict[str, float | int | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and precision + recall else None
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def fmt_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def size_bin(value: float, bins: list[dict]) -> str:
    for item in bins:
        lo = float(item["min_m"])
        hi_raw = item.get("max_m")
        hi = float("inf") if hi_raw is None else float(hi_raw)
        if lo <= value < hi:
            return str(item["label"])
    return "out_of_bins"


def pair_passes(
    ann_box: Box,
    pred_box: Box,
    pred_centroid: tuple[float, float],
    *,
    iou_threshold: float,
    center_distance_m: float,
) -> tuple[bool, float, float]:
    pair_iou = iou(ann_box, pred_box)
    center_dist = distance(ann_box.centroid, pred_centroid)
    return pair_iou >= iou_threshold or center_dist <= center_distance_m, pair_iou, center_dist


def read_regions(path: Path, site_id: str) -> dict[str, Region]:
    regions: dict[str, Region] = {}
    for row in read_csv_rows(path):
        if row.get("site_id", "").strip() != site_id or parse_bool(row.get("ignore")) or not has_box(row):
            continue
        region_id = row["region_id"].strip()
        regions[region_id] = Region(
            site_id=site_id,
            region_id=region_id,
            bbox=Box.from_values([row["x_min"], row["y_min"], row["x_max"], row["y_max"]]),
            scene_type=row.get("scene_type", "").strip() or "unspecified",
            notes=row.get("notes", "").strip(),
        )
    return regions


def read_annotations(path: Path, site_id: str, regions: dict[str, Region]) -> list[Annotation]:
    annotations: list[Annotation] = []
    for idx, row in enumerate(read_csv_rows(path), start=1):
        if row.get("site_id", "").strip() != site_id or parse_bool(row.get("ignore")) or not has_box(row):
            continue
        region_id = row["region_id"].strip()
        if region_id not in regions:
            raise ValueError(f"Annotation {row.get('ann_id') or idx} references unknown region_id={region_id!r}")
        ann_id = row.get("ann_id", "").strip() or f"{site_id}_ann_{idx:05d}"
        annotations.append(
            Annotation(
                ann_id=ann_id,
                site_id=site_id,
                region_id=region_id,
                bbox=Box.from_values([row["x_min"], row["y_min"], row["x_max"], row["y_max"]]),
                label=row.get("label", "rock").strip() or "rock",
                notes=row.get("notes", "").strip(),
            )
        )
    return annotations


def read_predictions(path: Path, site_id: str) -> list[Prediction]:
    payload = load_json(path)
    if isinstance(payload, dict) and "stones" in payload:
        records = payload["stones"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"Unsupported prediction JSON structure: {path}")

    predictions: list[Prediction] = []
    for idx, item in enumerate(records):
        bbox_values = item.get("bbox_world")
        if not bbox_values:
            continue
        bbox = Box.from_values(bbox_values)
        centroid_values = item.get("centroid_world") or bbox.centroid
        pred_id = item.get("stone_id") or item.get("detection_id") or f"{site_id}_pred_{idx:05d}"
        score = item.get("score_mean", item.get("score"))
        diameter = item.get("equivalent_diameter_m")
        predictions.append(
            Prediction(
                pred_id=str(pred_id),
                site_id=site_id,
                bbox=bbox,
                centroid=(float(centroid_values[0]), float(centroid_values[1])),
                score=float(score) if score is not None else None,
                equivalent_diameter_m=float(diameter) if diameter is not None else bbox.equivalent_diameter,
                source=item,
            )
        )
    return predictions


def prediction_region_id(prediction: Prediction, regions: dict[str, Region], pad_m: float) -> str | None:
    hits = [
        region_id
        for region_id, region in regions.items()
        if region.bbox.contains_point(prediction.centroid[0], prediction.centroid[1], pad=pad_m)
    ]
    if not hits:
        return None
    return hits[0] if len(hits) == 1 else min(hits, key=lambda rid: regions[rid].bbox.area)


def make_summary_rows(rows: list[dict], group_key: str) -> list[dict]:
    groups: dict[str, dict[str, int]] = {}
    for row in rows:
        key = str(row[group_key])
        group = groups.setdefault(key, {"tp": 0, "fp": 0, "fn": 0})
        group["tp"] += int(row.get("tp", 0))
        group["fp"] += int(row.get("fp", 0))
        group["fn"] += int(row.get("fn", 0))
    result = []
    for key, counts in sorted(groups.items()):
        result.append({group_key: key, **metric_row(counts["tp"], counts["fp"], counts["fn"])})
    return result


def match_site(
    *,
    site_id: str,
    predictions: list[Prediction],
    annotations: list[Annotation],
    regions: dict[str, Region],
    min_report_diameter_m: float,
    iou_threshold: float,
    center_distance_m: float,
    region_pad_m: float,
    diameter_bins: list[dict],
) -> dict:
    region_predictions: dict[str, list[Prediction]] = {rid: [] for rid in regions}
    for pred in predictions:
        region_id = prediction_region_id(pred, regions, region_pad_m)
        if region_id is not None:
            region_predictions[region_id].append(pred)

    evaluable_annotations: list[Annotation] = []
    excluded_annotations: list[dict] = []
    excluded_by_region: dict[str, list[Annotation]] = {rid: [] for rid in regions}
    for ann in annotations:
        if ann.equivalent_diameter_m >= min_report_diameter_m:
            evaluable_annotations.append(ann)
        else:
            excluded_annotations.append(
                {
                    "site_id": site_id,
                    "region_id": ann.region_id,
                    "ann_id": ann.ann_id,
                    "equivalent_diameter_m": round(ann.equivalent_diameter_m, 4),
                    "bbox_world": ann.bbox.to_list(),
                    "reason": "below_min_report_diameter",
                }
            )
            excluded_by_region[ann.region_id].append(ann)

    region_annotations: dict[str, list[Annotation]] = {rid: [] for rid in regions}
    for ann in evaluable_annotations:
        region_annotations[ann.region_id].append(ann)

    matches: list[dict] = []
    false_negatives: list[dict] = []
    false_positives: list[dict] = []
    ignored_predictions: list[dict] = []
    region_rows: list[dict] = []
    size_rows_events: list[dict] = []
    scene_rows_events: list[dict] = []

    for region_id, region in regions.items():
        anns = region_annotations[region_id]
        preds = region_predictions[region_id]
        candidates: list[tuple[float, float, Annotation, Prediction]] = []
        for ann in anns:
            for pred in preds:
                passed, pair_iou, center_dist = pair_passes(
                    ann.bbox,
                    pred.bbox,
                    pred.centroid,
                    iou_threshold=iou_threshold,
                    center_distance_m=center_distance_m,
                )
                if passed:
                    score = pair_iou + max(0.0, 1.0 - center_dist / max(center_distance_m, 1e-9)) * 0.01
                    candidates.append((score, pair_iou, ann, pred))
        candidates.sort(key=lambda item: item[0], reverse=True)

        used_anns: set[str] = set()
        used_preds: set[str] = set()

        for _, pair_iou, ann, pred in candidates:
            if ann.ann_id in used_anns or pred.pred_id in used_preds:
                continue
            center_dist = distance(ann.bbox.centroid, pred.centroid)
            bin_label = size_bin(ann.equivalent_diameter_m, diameter_bins)
            matches.append(
                {
                    "site_id": site_id,
                    "region_id": region_id,
                    "scene_type": region.scene_type,
                    "ann_id": ann.ann_id,
                    "pred_id": pred.pred_id,
                    "manual_diameter_m": round(ann.equivalent_diameter_m, 4),
                    "diameter_bin": bin_label,
                    "iou": round(pair_iou, 4),
                    "center_distance_m": round(center_dist, 4),
                    "pred_score": pred.score,
                    "pred_diameter_m": pred.equivalent_diameter_m,
                }
            )
            size_rows_events.append({"diameter_bin": bin_label, "tp": 1, "fp": 0, "fn": 0})
            scene_rows_events.append({"scene_type": region.scene_type, "tp": 1, "fp": 0, "fn": 0})
            used_anns.add(ann.ann_id)
            used_preds.add(pred.pred_id)

        for ann in anns:
            if ann.ann_id not in used_anns:
                bin_label = size_bin(ann.equivalent_diameter_m, diameter_bins)
                false_negatives.append(
                    {
                        "site_id": site_id,
                        "region_id": region_id,
                        "scene_type": region.scene_type,
                        "ann_id": ann.ann_id,
                        "manual_diameter_m": round(ann.equivalent_diameter_m, 4),
                        "diameter_bin": bin_label,
                        "bbox_world": ann.bbox.to_list(),
                    }
                )
                size_rows_events.append({"diameter_bin": bin_label, "tp": 0, "fp": 0, "fn": 1})
                scene_rows_events.append({"scene_type": region.scene_type, "tp": 0, "fp": 0, "fn": 1})

        for pred in preds:
            if pred.pred_id in used_preds:
                continue
            matched_excluded = False
            for ann in excluded_by_region[region_id]:
                passed, pair_iou, center_dist = pair_passes(
                    ann.bbox,
                    pred.bbox,
                    pred.centroid,
                    iou_threshold=iou_threshold,
                    center_distance_m=center_distance_m,
                )
                if passed:
                    ignored_predictions.append(
                        {
                            "site_id": site_id,
                            "region_id": region_id,
                            "scene_type": region.scene_type,
                            "pred_id": pred.pred_id,
                            "matched_excluded_ann_id": ann.ann_id,
                            "manual_diameter_m": round(ann.equivalent_diameter_m, 4),
                            "iou": round(pair_iou, 4),
                            "center_distance_m": round(center_dist, 4),
                            "reason": "matched_below_min_report_diameter_annotation",
                        }
                    )
                    matched_excluded = True
                    break
            if matched_excluded:
                continue
            false_positives.append(
                {
                    "site_id": site_id,
                    "region_id": region_id,
                    "scene_type": region.scene_type,
                    "pred_id": pred.pred_id,
                    "pred_diameter_m": pred.equivalent_diameter_m,
                    "bbox_world": pred.bbox.to_list(),
                    "centroid_world": list(pred.centroid),
                    "pred_score": pred.score,
                }
            )

        region_tp = sum(1 for row in matches if row["region_id"] == region_id and row["site_id"] == site_id)
        region_fp = sum(1 for row in false_positives if row["region_id"] == region_id and row["site_id"] == site_id)
        region_fn = sum(1 for row in false_negatives if row["region_id"] == region_id and row["site_id"] == site_id)
        region_rows.append(
            {
                "site_id": site_id,
                "region_id": region_id,
                "scene_type": region.scene_type,
                "region_area_m2": round(region.bbox.area, 4),
                "manual_total": sum(1 for ann in annotations if ann.region_id == region_id),
                "manual_evaluable": len(anns),
                "manual_excluded_small": len(excluded_by_region[region_id]),
                "predictions_in_region": len(preds),
                "ignored_predictions": sum(1 for row in ignored_predictions if row["region_id"] == region_id and row["site_id"] == site_id),
                **metric_row(region_tp, region_fp, region_fn),
            }
        )

    tp = len(matches)
    fp = len(false_positives)
    fn = len(false_negatives)
    site_summary = {
        "site_id": site_id,
        "regions": len(regions),
        "manual_total": len(annotations),
        "manual_evaluable": len(evaluable_annotations),
        "manual_excluded_small": len(excluded_annotations),
        "predictions_in_regions": sum(len(items) for items in region_predictions.values()),
        "ignored_predictions": len(ignored_predictions),
        **metric_row(tp, fp, fn),
    }

    return {
        "site_summary": site_summary,
        "region_summary": region_rows,
        "scene_type_summary": make_summary_rows(scene_rows_events, "scene_type"),
        "size_bin_summary": make_summary_rows(size_rows_events, "diameter_bin"),
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "excluded_annotations": excluded_annotations,
        "ignored_predictions": ignored_predictions,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict) -> None:
    matching = summary["matching"]
    evaluation = summary["evaluation"]
    lines = [
        "# Manual Detection Validation Report",
        "",
        f"Created UTC: {summary['created_utc']}",
        "",
        f"Main evaluation objects: manual annotations with equivalent diameter >= {evaluation['min_report_diameter_m']} m.",
        f"Matching rule: IoU >= {matching['iou_threshold']} or centroid distance <= {matching['center_distance_m']} m.",
        "",
        "## Site-Level Results",
        "",
        "| Site | Regions | Manual total | Manual evaluated | Small excluded | Predictions in regions | Ignored predictions | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["site_results"]:
        lines.append(
            "| {site_id} | {regions} | {manual_total} | {manual_evaluable} | {manual_excluded_small} | {predictions_in_regions} | {ignored_predictions} | {true_positive} | {false_positive} | {false_negative} | {precision} | {recall} | {f1} |".format(
                **{
                    **row,
                    "precision": fmt_metric(row["precision"]),
                    "recall": fmt_metric(row["recall"]),
                    "f1": fmt_metric(row["f1"]),
                }
            )
        )
    overall = summary["overall"]
    lines.extend(
        [
            "| Overall | - | - | - | - | - | - | {true_positive} | {false_positive} | {false_negative} | {precision} | {recall} | {f1} |".format(
                **{
                    **overall,
                    "precision": fmt_metric(overall["precision"]),
                    "recall": fmt_metric(overall["recall"]),
                    "f1": fmt_metric(overall["f1"]),
                }
            ),
            "",
            "## Interpretation",
            "",
            "- Metrics are computed only within manually annotated validation regions.",
            "- Manual annotations below the minimum reportable diameter are excluded from the main recall denominator.",
            "- Predictions matching excluded small annotations are listed as ignored predictions rather than false positives.",
            "- This is object-level detection validation; it does not validate absolute 2.5D volume accuracy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fused rock detections against manual annotations.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--site", action="append", dest="sites", help="Site id to validate. Can be used multiple times.")
    args = parser.parse_args()

    config = load_json(args.config)
    output_dir = (ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matching = config["matching"]
    evaluation = config["evaluation"]
    diameter_bins = evaluation["diameter_bins_m"]
    selected_sites = set(args.sites or [site["site_id"] for site in config["sites"]])

    site_results: list[dict] = []
    region_rows: list[dict] = []
    scene_rows: list[dict] = []
    size_rows: list[dict] = []
    all_matches: list[dict] = []
    all_fp: list[dict] = []
    all_fn: list[dict] = []
    all_excluded: list[dict] = []
    all_ignored: list[dict] = []

    for site in config["sites"]:
        site_id = site["site_id"]
        if site_id not in selected_sites:
            continue
        regions = read_regions(ROOT / site["regions_csv"], site_id)
        annotations = read_annotations(ROOT / site["annotations_csv"], site_id, regions)
        predictions = read_predictions(ROOT / site["predictions_json"], site_id)
        result = match_site(
            site_id=site_id,
            predictions=predictions,
            annotations=annotations,
            regions=regions,
            min_report_diameter_m=float(evaluation["min_report_diameter_m"]),
            iou_threshold=float(matching["iou_threshold"]),
            center_distance_m=float(matching["center_distance_m"]),
            region_pad_m=float(matching.get("region_pad_m", 0.0)),
            diameter_bins=diameter_bins,
        )
        site_results.append(result["site_summary"])
        region_rows.extend(result["region_summary"])
        scene_rows.extend(result["scene_type_summary"])
        size_rows.extend(result["size_bin_summary"])
        all_matches.extend(result["matches"])
        all_fp.extend(result["false_positives"])
        all_fn.extend(result["false_negatives"])
        all_excluded.extend(result["excluded_annotations"])
        all_ignored.extend(result["ignored_predictions"])

    tp = sum(int(row["true_positive"]) for row in site_results)
    fp = sum(int(row["false_positive"]) for row in site_results)
    fn = sum(int(row["false_negative"]) for row in site_results)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "matching": matching,
        "evaluation": evaluation,
        "site_results": site_results,
        "region_summary": region_rows,
        "scene_type_summary": scene_rows,
        "size_bin_summary": size_rows,
        "overall": metric_row(tp, fp, fn),
        "paper_scope": {
            "main_claim": "Object-level detection performance within locally complete manual annotation windows.",
            "not_supported": "Absolute 2.5D volume accuracy without external volume reference.",
        },
    }

    (output_dir / "manual_detection_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir / "manual_detection_validation_report.md", summary)
    write_csv(output_dir / "manual_detection_site_summary.csv", site_results)
    write_csv(output_dir / "manual_detection_region_summary.csv", region_rows)
    write_csv(output_dir / "manual_detection_scene_type_summary.csv", scene_rows)
    write_csv(output_dir / "manual_detection_size_bin_summary.csv", size_rows)
    write_csv(output_dir / "manual_detection_matches.csv", all_matches)
    write_csv(output_dir / "manual_detection_false_positives.csv", all_fp)
    write_csv(output_dir / "manual_detection_false_negatives.csv", all_fn)
    write_csv(output_dir / "manual_detection_excluded_small_annotations.csv", all_excluded)
    write_csv(output_dir / "manual_detection_ignored_predictions.csv", all_ignored)

    print(output_dir / "manual_detection_validation_report.md")


if __name__ == "__main__":
    main()

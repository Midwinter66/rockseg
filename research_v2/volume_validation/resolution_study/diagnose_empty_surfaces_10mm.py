"""Small diagnostic for 10 mm empty mesh surfaces; does not alter datasets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "research_v2/volume_validation/datasets/t01_l01_v2_10mm"
OUT_JSON = ROOT / "research_v2/volume_validation/resolution_study/empty_surface_diagnostic_10mm.json"
OUT_MD = ROOT / "research_v2/volume_validation/resolution_study/empty_surface_diagnostic_10mm.md"

import sys
sys.path.insert(0, str(ROOT / "research_v2/volume_validation"))
from enhance_shape_aware import load_obj_simple, simulate_2_5d_surface


def select_records() -> list[dict]:
    selected = []
    for dataset_id in ("T01", "L01"):
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((DATASET / "cache" / dataset_id).glob("*.json"))
        ]
        errors = [record for record in records if record.get("error") == "empty_2_5d_surface"][:3]
        successes = [record for record in records if record.get("status") == "success"][:3]
        selected.extend(errors + successes)
    return selected


def triangle_scale(vertices: np.ndarray, faces: np.ndarray) -> dict:
    triangles = vertices[faces]
    xy = triangles[:, :, :2]
    span = np.ptp(xy, axis=1)
    edge = np.stack([
        np.linalg.norm(xy[:, 1] - xy[:, 0], axis=1),
        np.linalg.norm(xy[:, 2] - xy[:, 1], axis=1),
        np.linalg.norm(xy[:, 0] - xy[:, 2], axis=1),
    ], axis=1).reshape(-1)
    projected_area = np.abs(
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
    ) * 0.5
    return {
        "face_count": int(len(faces)),
        "edge_length_mm": {"p50": float(np.percentile(edge, 50)), "p90": float(np.percentile(edge, 90))},
        "xy_triangle_span_mm": {"p50": float(np.percentile(span, 50)), "p90": float(np.percentile(span, 90))},
        "projected_area_mm2": {"p50": float(np.percentile(projected_area, 50)), "p90": float(np.percentile(projected_area, 90))},
        "face_area_below_10mm_cell_ratio": float(np.mean(projected_area < 100.0)),
    }


def main() -> None:
    samples = []
    for record in select_records():
        vertices, faces = load_obj_simple(ROOT / record["obj_path"])
        surfaces = {}
        for resolution_mm in (2.5, 5.0, 10.0):
            surface = simulate_2_5d_surface(vertices, faces, grid_resolution=resolution_mm)
            surfaces[f"{resolution_mm:g}mm"] = {
                "status": "ok" if surface is not None else "empty",
                "valid_cells": int(surface["n_valid_cells"]) if surface is not None else 0,
            }
        samples.append({
            "sample_id": record["sample_id"],
            "dataset_id": record["dataset_id"],
            "cached_10mm_status": record["status"],
            "cached_10mm_error": record.get("error", ""),
            "mesh": triangle_scale(vertices, faces),
            "surfaces": surfaces,
        })

    empty_at_10 = [sample for sample in samples if sample["surfaces"]["10mm"]["status"] == "empty"]
    recovered_at_5 = [sample for sample in empty_at_10 if sample["surfaces"]["5mm"]["status"] == "ok"]
    result = {
        "purpose": "Diagnose empty 10 mm mesh surfaces without changing rasterization or datasets.",
        "sample_selection": "First 3 cached empty and first 3 cached successful samples from each of T01 and L01.",
        "sample_count": len(samples),
        "empty_at_10mm": len(empty_at_10),
        "empty_at_10mm_recovered_at_5mm": len(recovered_at_5),
        "interpretation": (
            "The current rasterizer accepts a face only when a grid-cell centre falls inside its XY projection. "
            "At coarse grids, sub-cell triangles can therefore leave no accepted cell even for a non-empty mesh."
        ),
        "samples": samples,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# 10 mm Empty-Surface Diagnostic",
        "",
        f"Sampled meshes: {len(samples)}; empty at 10 mm: {len(empty_at_10)}; recovered at 5 mm: {len(recovered_at_5)}.",
        "",
        "The current mesh rasterizer tests grid-cell centres against triangle XY projections. Thus a coarse grid can return an empty surface when no centre lands within any projected triangle, even though the mesh itself is valid.",
        "",
        "| Dataset | Sample | Cached 10 mm | 2.5 mm cells | 5 mm cells | 10 mm cells | Face area < 100 mm2 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for sample in samples:
        surfaces = sample["surfaces"]
        lines.append(
            f"| {sample['dataset_id']} | {sample['sample_id']} | {sample['cached_10mm_status']} | "
            f"{surfaces['2.5mm']['valid_cells']} | {surfaces['5mm']['valid_cells']} | {surfaces['10mm']['valid_cells']} | "
            f"{sample['mesh']['face_area_below_10mm_cell_ratio']:.3f} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        "Do not train from the 10 mm cache. A revised mesh-to-grid coverage rule must first be defined and validated separately against the existing 0.5 mm method; changing it would create a different scientific measurement pipeline.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

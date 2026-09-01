"""Isolated Mine Site B runner.

This runner reuses the established slicing, segmentation, fusion and volume
implementations, but redirects their module-level paths to this directory.
It never edits ``CURRENT_SCENE`` or writes to any Mine Site A output folder.

P0 = DOM slicing -> segmentation -> 2D fusion.
P1 = DOM slicing -> segmentation -> fusion -> 3D geometric screening.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = RUN_ROOT / "config"
OUTPUT_ROOT = RUN_ROOT / "outputs"
MANIFEST_PATH = RUN_ROOT / "site_b_run_manifest.json"
SOURCE = "quadtree_dom"
FUSION_METHOD = "correlation_clustering"

sys.path.insert(0, str(PROJECT_ROOT))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, include_hash: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if include_hash:
        result["sha256"] = _sha256(path)
    return result


def _scene_reference():
    from experiments.common.scene_reference import SceneReference, XYCoordinateTransform

    config = _load_json(CONFIG_DIR / "scene_b.json")
    transform = config["xy_transform"]
    scene = SceneReference(
        name=config["scene_name"],
        dom_path=PROJECT_ROOT / config["dom_path"],
        tfw_path=PROJECT_ROOT / config["tfw_path"],
        pointcloud_paths=tuple(PROJECT_ROOT / item for item in config["pointcloud_paths"]),
        xy_transform=XYCoordinateTransform(
            mode=transform["mode"],
            x_shift=float(transform["x_shift_m"]),
            y_shift=float(transform["y_shift_m"]),
        ),
    )
    return scene, config


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _write_manifest(*, scene, scene_config: dict[str, Any], stage: str, status: str, details: dict[str, Any]) -> None:
    previous: dict[str, Any] = {}
    if MANIFEST_PATH.exists():
        previous = _load_json(MANIFEST_PATH)
    previous.setdefault("runner", "experiments/site_b_run/run_site_b.py")
    previous.setdefault("scope_guard", "Mine Site B only. No Mine Site A configuration or output path is written.")
    previous["scene"] = scene.to_dict()
    previous["scene_coordinate_qc"] = {
        "status": scene_config["coordinate_qc_status"],
        "note": scene_config["coordinate_qc_note"],
    }
    previous["frozen_config_files"] = {
        path.name: _file_record(path, include_hash=True)
        for path in sorted(CONFIG_DIR.glob("*.json"))
    }
    events = previous.setdefault("events", [])
    events.append(
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "status": status,
            "details": details,
        }
    )
    MANIFEST_PATH.write_text(json.dumps(previous, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def prepare() -> tuple[Any, dict[str, Any]]:
    scene, scene_config = _scene_reference()
    _require_file(scene.dom_path, "Mine Site B DOM")
    _require_file(scene.tfw_path, "Mine Site B world file")
    for path in scene.pointcloud_paths:
        _require_file(path, "Mine Site B point cloud")
    _require_file(PROJECT_ROOT / "models" / "best.pt", "segmentation model")
    for path in CONFIG_DIR.glob("*_frozen.json"):
        _require_file(path, "frozen configuration")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        scene=scene,
        scene_config=scene_config,
        stage="prepare",
        status="ready",
        details={
            "dom": _file_record(scene.dom_path),
            "pointclouds": [_file_record(path) for path in scene.pointcloud_paths],
            "model": _file_record(PROJECT_ROOT / "models" / "best.pt", include_hash=True),
        },
    )
    return scene, scene_config


def run_slicing(scene) -> dict[str, Any]:
    import experiments.slicing.run_slicing_experiment as slicing

    out_dir = OUTPUT_ROOT / "slicing" / SOURCE
    out_dir.mkdir(parents=True, exist_ok=True)
    slicing.DOM_PATH = scene.dom_path
    slicing.DOM_WORLD_PATH = scene.tfw_path
    config = _load_json(CONFIG_DIR / "quadtree_dom_frozen.json")
    stats = slicing._run_quadtree_dom(config, out_dir)
    slicing._write_method_summary(SOURCE, stats, out_dir)
    return {"output_dir": str(out_dir), "kept_tiles": int(stats.get("kept_tiles", 0)), "total_tiles": int(stats.get("total_tiles", 0))}


def run_detection(scene, limit: int | None) -> dict[str, Any]:
    import experiments.detection.run_detection_experiment as detection

    out_dir = OUTPUT_ROOT / "detection" / SOURCE
    out_dir.mkdir(parents=True, exist_ok=True)
    detection.DOM_PATH = scene.dom_path
    detection.DOM_WORLD_PATH = scene.tfw_path
    detection.SLICING_OUTPUTS = OUTPUT_ROOT / "slicing"
    detection.DETECTION_CONFIG_PATH = CONFIG_DIR / "detection_frozen.json"
    detection._resolve_output_dir = lambda method: out_dir
    stats = detection._run_detection(SOURCE, limit=limit, multi_scale=False, scales=None)
    (OUTPUT_ROOT / "detection" / "detection_manifest.json").write_text(
        json.dumps({SOURCE: stats}, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return {"output_dir": str(out_dir), "detections": int(stats.get("detection_count", 0)), "processed_tiles": int(stats.get("processed_tiles", 0))}


def run_fusion(scene, pipeline: str) -> dict[str, Any]:
    import experiments.fusion.run_fusion_experiment as fusion

    if pipeline not in {"p0", "p1"}:
        raise ValueError(f"Unsupported pipeline: {pipeline}")
    config_path = CONFIG_DIR / f"fusion_{pipeline}_frozen.json"
    config = _load_json(config_path)
    out_dir = OUTPUT_ROOT / pipeline / "fusion" / SOURCE / FUSION_METHOD
    out_dir.mkdir(parents=True, exist_ok=True)
    fusion.CURRENT_SCENE = scene
    fusion.DETECTION_OUTPUTS = OUTPUT_ROOT / "detection"
    fusion._CACHED_PC = None
    fusion._CACHED_PC_INDEX = None
    fusion._load_fusion_config = lambda method: config
    fusion._resolve_output_dir = lambda source, method: out_dir
    summary = fusion._run_fusion(SOURCE, FUSION_METHOD)
    manifest_path = OUTPUT_ROOT / pipeline / "fusion" / "fusion_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({SOURCE: {FUSION_METHOD: summary}}, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return {
        "output_dir": str(out_dir),
        "input_detections": int(summary.get("input_detections", 0)),
        "candidate_stones": int(summary.get("candidate_stones", 0)),
        "accepted_stones": int(summary.get("output_stones", 0)),
        "three_dimensional_screening": bool(config.get("validation_3d", {}).get("enabled", False)),
    }


def run_volume(scene) -> dict[str, Any]:
    import experiments.volume.run_volume as volume

    expected_fusion = OUTPUT_ROOT / "p1" / "fusion" / SOURCE / FUSION_METHOD / "fusion_stats.json"
    if not expected_fusion.exists():
        raise FileNotFoundError("P1 fusion result is required before volume estimation. Run --stage p1 first.")
    volume.CURRENT_SCENE = scene
    volume.FUSION_ROOT = OUTPUT_ROOT / "p1" / "fusion"
    volume.DETECTION_ROOT = OUTPUT_ROOT / "detection"
    volume.SELF_DIR = OUTPUT_ROOT / "p1" / "volume"
    volume._CACHED_PC_INDEX = None
    summary = volume._run_single_case(
        SOURCE,
        FUSION_METHOD,
        _load_json(CONFIG_DIR / "volume_frozen.json"),
        progress_every=200,
    )
    return {
        "output_dir": str(OUTPUT_ROOT / "p1" / "volume" / "outputs" / SOURCE / FUSION_METHOD),
        "qc_passed": int(summary["scene"].get("qc_passed", 0)),
        "processed_stones": int(summary["scene"].get("processed_stones", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mine Site B without changing Mine Site A.")
    parser.add_argument("--stage", choices=["prepare", "slicing", "detection", "p0", "p1", "volume", "all"], default="prepare")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of tiles for a segmentation smoke test.")
    parser.add_argument("--dry-run", action="store_true", help="Validate files and write the B manifest without executing a stage.")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    scene, scene_config = prepare()
    if args.dry_run or args.stage == "prepare":
        print(f"Mine Site B runner is ready. Manifest: {MANIFEST_PATH}")
        return

    stages = ["slicing", "detection", "p0", "p1", "volume"] if args.stage == "all" else [args.stage]
    actions = {
        "slicing": lambda: run_slicing(scene),
        "detection": lambda: run_detection(scene, args.limit),
        "p0": lambda: run_fusion(scene, "p0"),
        "p1": lambda: run_fusion(scene, "p1"),
        "volume": lambda: run_volume(scene),
    }
    for stage in stages:
        try:
            print(f"\n=== Mine Site B: {stage} ===")
            details = actions[stage]()
            _write_manifest(scene=scene, scene_config=scene_config, stage=stage, status="complete", details=details)
            print(json.dumps(details, ensure_ascii=False, indent=2))
        except Exception as exc:
            _write_manifest(scene=scene, scene_config=scene_config, stage=stage, status="failed", details={"error": repr(exc)})
            raise


if __name__ == "__main__":
    main()

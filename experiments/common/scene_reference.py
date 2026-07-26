from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_tfw(path: Path) -> tuple[float, float, float, float, float, float]:
    lines = [float(line.strip()) for line in path.read_text("utf-8").splitlines() if line.strip()]
    if len(lines) < 6:
        raise ValueError(f"Invalid TFW file: {path}")
    return (lines[4], lines[0], lines[2], lines[5], lines[1], lines[3])


@dataclass(frozen=True, slots=True)
class XYCoordinateTransform:
    mode: str = "absolute_world"
    x_shift: float = 0.0
    y_shift: float = 0.0

    def world_to_point_xy(self, x_world: float, y_world: float) -> tuple[float, float]:
        return float(x_world - self.x_shift), float(y_world - self.y_shift)

    def point_to_world_xy(self, x_point: float, y_point: float) -> tuple[float, float]:
        return float(x_point + self.x_shift), float(y_point + self.y_shift)

    def world_bbox_to_point_bbox(
        self,
        bbox_world: list[float] | tuple[float, float, float, float],
        pad_m: float = 0.0,
    ) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = [float(v) for v in bbox_world[:4]]
        px0, py0 = self.world_to_point_xy(x0, y0)
        px1, py1 = self.world_to_point_xy(x1, y1)
        return (
            min(px0, px1) - float(pad_m),
            min(py0, py1) - float(pad_m),
            max(px0, px1) + float(pad_m),
            max(py0, py1) + float(pad_m),
        )

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "x_shift": float(self.x_shift),
            "y_shift": float(self.y_shift),
        }


@dataclass(frozen=True, slots=True)
class SceneReference:
    name: str
    dom_path: Path
    tfw_path: Path
    pointcloud_paths: tuple[Path, ...]
    xy_transform: XYCoordinateTransform

    def load_gt(self) -> tuple[float, float, float, float, float, float]:
        return parse_tfw(self.tfw_path)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dom_path": str(self.dom_path),
            "tfw_path": str(self.tfw_path),
            "pointcloud_paths": [str(path) for path in self.pointcloud_paths],
            "xy_transform": self.xy_transform.to_dict(),
        }


DOM2_POINTCLOUD2_SCENE = SceneReference(
    name="dom2_pointcloud2",
    dom_path=PROJECT_ROOT / "data" / "dom2" / "DOM.tif",
    tfw_path=PROJECT_ROOT / "data" / "dom2" / "DOM.tfw",
    pointcloud_paths=(
        PROJECT_ROOT / "data" / "pointcloud2" / "Data" / "BlockB.laz",
        PROJECT_ROOT / "data" / "pointcloud2" / "Data" / "BlockY.laz",
    ),
    xy_transform=XYCoordinateTransform(mode="absolute_world", x_shift=0.0, y_shift=0.0),
)


CURRENT_SCENE = DOM2_POINTCLOUD2_SCENE

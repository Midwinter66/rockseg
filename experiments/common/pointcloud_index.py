from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PointCloudXYGridIndex:
    points: np.ndarray
    cell_size: float
    xmin: float
    ymin: float
    nx: int
    ny: int
    ordered_indices: np.ndarray
    unique_cell_ids: np.ndarray
    cell_starts: np.ndarray
    cell_counts: np.ndarray

    @classmethod
    def build(cls, points: np.ndarray, cell_size: float = 1.0) -> "PointCloudXYGridIndex":
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must be an Nx3 array")
        if len(pts) == 0:
            raise ValueError("points must not be empty")
        if cell_size <= 0:
            raise ValueError("cell_size must be > 0")

        xmin = float(np.min(pts[:, 0]))
        ymin = float(np.min(pts[:, 1]))
        xmax = float(np.max(pts[:, 0]))
        ymax = float(np.max(pts[:, 1]))

        nx = max(1, int(np.floor((xmax - xmin) / cell_size)) + 1)
        ny = max(1, int(np.floor((ymax - ymin) / cell_size)) + 1)

        xi = np.floor((pts[:, 0] - xmin) / cell_size).astype(np.int32)
        yi = np.floor((pts[:, 1] - ymin) / cell_size).astype(np.int32)
        xi = np.clip(xi, 0, nx - 1)
        yi = np.clip(yi, 0, ny - 1)
        flat = yi.astype(np.int64) * np.int64(nx) + xi.astype(np.int64)

        order = np.argsort(flat, kind="mergesort")
        flat_sorted = flat[order]
        unique_ids, starts, counts = np.unique(
            flat_sorted,
            return_index=True,
            return_counts=True,
        )

        return cls(
            points=pts,
            cell_size=float(cell_size),
            xmin=xmin,
            ymin=ymin,
            nx=int(nx),
            ny=int(ny),
            ordered_indices=order.astype(np.int32, copy=False),
            unique_cell_ids=unique_ids.astype(np.int64, copy=False),
            cell_starts=starts.astype(np.int64, copy=False),
            cell_counts=counts.astype(np.int64, copy=False),
        )

    def _bbox_cell_ids(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        gx0 = int(np.floor((x0 - self.xmin) / self.cell_size))
        gy0 = int(np.floor((y0 - self.ymin) / self.cell_size))
        gx1 = int(np.floor((x1 - self.xmin) / self.cell_size))
        gy1 = int(np.floor((y1 - self.ymin) / self.cell_size))

        gx0 = max(0, min(self.nx - 1, gx0))
        gy0 = max(0, min(self.ny - 1, gy0))
        gx1 = max(0, min(self.nx - 1, gx1))
        gy1 = max(0, min(self.ny - 1, gy1))

        if gx1 < gx0 or gy1 < gy0:
            return np.empty(0, dtype=np.int64)

        xs = np.arange(gx0, gx1 + 1, dtype=np.int64)
        ys = np.arange(gy0, gy1 + 1, dtype=np.int64)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        return (grid_y.reshape(-1) * np.int64(self.nx) + grid_x.reshape(-1)).astype(np.int64, copy=False)

    def query_bbox_indices(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        cell_ids = self._bbox_cell_ids(x0, y0, x1, y1)
        if cell_ids.size == 0:
            return np.empty(0, dtype=np.int32)

        pos = np.searchsorted(self.unique_cell_ids, cell_ids)
        valid = (
            (pos >= 0)
            & (pos < len(self.unique_cell_ids))
            & (self.unique_cell_ids[np.clip(pos, 0, len(self.unique_cell_ids) - 1)] == cell_ids)
        )
        if not np.any(valid):
            return np.empty(0, dtype=np.int32)

        parts: list[np.ndarray] = []
        for idx in pos[valid]:
            start = int(self.cell_starts[idx])
            count = int(self.cell_counts[idx])
            parts.append(self.ordered_indices[start : start + count])

        if not parts:
            return np.empty(0, dtype=np.int32)

        candidates = np.concatenate(parts)
        pts = self.points[candidates]
        mask = (
            (pts[:, 0] >= x0)
            & (pts[:, 0] <= x1)
            & (pts[:, 1] >= y0)
            & (pts[:, 1] <= y1)
        )
        return candidates[mask]

    def query_bbox_points(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        indices = self.query_bbox_indices(x0, y0, x1, y1)
        if len(indices) == 0:
            return np.empty((0, 3), dtype=self.points.dtype)
        return self.points[indices]

    def to_dict(self) -> dict:
        return {
            "cell_size": self.cell_size,
            "xmin": self.xmin,
            "ymin": self.ymin,
            "nx": self.nx,
            "ny": self.ny,
            "point_count": int(len(self.points)),
            "indexed_cell_count": int(len(self.unique_cell_ids)),
        }

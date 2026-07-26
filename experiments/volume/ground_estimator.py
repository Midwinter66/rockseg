from __future__ import annotations

import numpy as np


class GroundDEM:
    """Scene-level ground DEM built from a subsampled OSGB point cloud."""

    def __init__(
        self,
        pc: np.ndarray,
        resolution: float = 0.5,
        percentile: int = 5,
        subsample_step: int = 100,
        min_points_per_cell: int = 3,
    ) -> None:
        pts = np.asarray(pc, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("pc must be an Nx3 array")
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        if subsample_step <= 0:
            raise ValueError("subsample_step must be positive")

        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            raise RuntimeError("GroundDEM: no valid input points")

        self.resolution = float(resolution)
        self.percentile = int(percentile)
        self.subsample_step = int(subsample_step)
        self.min_points_per_cell = int(min_points_per_cell)

        sub = pts[:: self.subsample_step]
        print(f"  GroundDEM: full points {len(pts):,} -> sampled {len(sub):,}")

        xmin = float(sub[:, 0].min())
        ymin = float(sub[:, 1].min())
        xmax = float(sub[:, 0].max())
        ymax = float(sub[:, 1].max())
        nx = max(2, int(np.ceil((xmax - xmin) / self.resolution)))
        ny = max(2, int(np.ceil((ymax - ymin) / self.resolution)))

        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmin + nx * self.resolution
        self.ymax = ymin + ny * self.resolution
        self.nx = int(nx)
        self.ny = int(ny)

        print(f"  GroundDEM grid: {self.nx} x {self.ny} = {self.nx * self.ny:,} cells @ {self.resolution} m")

        xi = np.floor((sub[:, 0] - xmin) / self.resolution).astype(np.int32)
        yi = np.floor((sub[:, 1] - ymin) / self.resolution).astype(np.int32)
        xi = np.clip(xi, 0, self.nx - 1)
        yi = np.clip(yi, 0, self.ny - 1)
        flat_idx = yi * self.nx + xi

        order = np.argsort(flat_idx)
        sorted_idx = flat_idx[order]
        sorted_z = sub[order, 2]
        unique_idx, starts, counts = np.unique(
            sorted_idx, return_index=True, return_counts=True
        )

        self.grid = np.full((self.ny, self.nx), np.nan, dtype=np.float64)
        for uid, start, cnt in zip(unique_idx, starts, counts):
            if cnt < self.min_points_per_cell:
                continue
            cell_z = sorted_z[start : start + cnt]
            self.grid.flat[int(uid)] = float(np.percentile(cell_z, self.percentile))

        self.raw_valid_cell_count = int(np.sum(~np.isnan(self.grid)))
        self.raw_coverage_ratio = float(self.raw_valid_cell_count / (self.nx * self.ny))
        self.hole_fill_iterations = int(self._fill_holes())

        self.valid_cell_count = int(np.sum(~np.isnan(self.grid)))
        self.coverage_ratio = float(self.valid_cell_count / (self.nx * self.ny))
        print(
            "  GroundDEM coverage: "
            f"raw {self.raw_valid_cell_count}/{self.nx * self.ny} ({self.raw_coverage_ratio:.1%})"
            f" -> filled {self.valid_cell_count}/{self.nx * self.ny} ({self.coverage_ratio:.1%})"
        )

    def _fill_holes(self) -> int:
        """Fill NaN cells by nearest-neighbor expansion."""

        valid = ~np.isnan(self.grid)
        if valid.sum() == 0:
            raise RuntimeError("GroundDEM: no valid cells available")
        if valid.all():
            return 0

        grid = self.grid.copy()
        ny, nx = grid.shape
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        ]

        iterations = 0
        for _ in range(max(nx, ny)):
            mask = np.isnan(grid)
            if not mask.any():
                break
            updated = False
            for dy, dx in offsets:
                src = np.roll(grid, shift=(-dy, -dx), axis=(0, 1))
                fill = mask & ~np.isnan(src)
                if dy > 0:
                    fill[-dy:, :] = False
                elif dy < 0:
                    fill[: -dy, :] = False
                if dx > 0:
                    fill[:, -dx:] = False
                elif dx < 0:
                    fill[:, : -dx] = False
                if fill.any():
                    grid[fill] = src[fill]
                    updated = True
            if not updated:
                break
            iterations += 1

        self.grid = grid
        return iterations

    def get_ground_z(self, x, y):
        """Bilinear interpolation over the ground grid."""

        scalar = np.ndim(x) == 0
        xa = np.atleast_1d(np.asarray(x, dtype=np.float64))
        ya = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if xa.shape != ya.shape:
            raise ValueError("x and y must have the same shape")

        fx = (xa - self.xmin) / self.resolution
        fy = (ya - self.ymin) / self.resolution

        x0 = np.floor(fx).astype(np.int32)
        y0 = np.floor(fy).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        outside = (fx < 0) | (fy < 0) | (fx > self.nx - 1) | (fy > self.ny - 1)

        x0 = np.clip(x0, 0, self.nx - 1)
        x1 = np.clip(x1, 0, self.nx - 1)
        y0 = np.clip(y0, 0, self.ny - 1)
        y1 = np.clip(y1, 0, self.ny - 1)

        wx = fx - x0
        wy = fy - y0

        z = (
            self.grid[y0, x0] * (1 - wx) * (1 - wy)
            + self.grid[y0, x1] * wx * (1 - wy)
            + self.grid[y1, x0] * (1 - wx) * wy
            + self.grid[y1, x1] * wx * wy
        ).astype(np.float64)

        z[outside] = np.nan
        return float(z[0]) if scalar else z

    def to_dict(self) -> dict:
        return {
            "resolution_m": self.resolution,
            "percentile": self.percentile,
            "subsample_step": self.subsample_step,
            "min_points_per_cell": self.min_points_per_cell,
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
            "nx": self.nx,
            "ny": self.ny,
            "raw_valid_cell_count": self.raw_valid_cell_count,
            "raw_coverage_ratio": round(self.raw_coverage_ratio, 6),
            "valid_cell_count": self.valid_cell_count,
            "coverage_ratio": round(self.coverage_ratio, 6),
            "hole_fill_iterations": self.hole_fill_iterations,
        }

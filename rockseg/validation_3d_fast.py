"""Fast 3D point cloud screening using bbox-based validation.

For screening purposes, bbox-level point cloud statistics are sufficient
to reject obvious false detections (flat ground, shadows, etc.).
This is much faster than per-pixel mask testing.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Validation3DConfig:
    min_points: int = 60
    min_z_range_m: float = 0.18
    elevated_height_m: float = 0.08
    min_p90_height_m: float = 0.12
    min_elevated_ratio: float = 0.2
    dem_resolution_m: float = 0.5
    dem_percentile: int = 5
    dem_subsample_step: int = 100
    dem_min_points: int = 3
    bbox_pad_m: float = 0.5


def load_point_cloud(laz_paths):
    import laspy
    all_pts = []
    for p in laz_paths:
        p = Path(p)
        if not p.exists():
            logger.warning("Missing: %s", p)
            continue
        logger.info("Loading %s", p.name)
        las = laspy.read(str(p))
        pts = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
        all_pts.append(pts)
        logger.info("  %d points", len(pts))
    if not all_pts:
        raise RuntimeError("No point cloud files")
    result = np.vstack(all_pts)
    logger.info("Total: %d points", len(result))
    return result


class GroundDEM:
    def __init__(self, pc, resolution=0.5, percentile=5, subsample_step=100, min_points=3):
        pts = np.asarray(pc, dtype=np.float64)
        pts = pts[np.isfinite(pts).all(axis=1)]
        sub = pts[::subsample_step]
        self.res = float(resolution)
        self.xmin = float(sub[:, 0].min())
        self.ymin = float(sub[:, 1].min())
        self.xmax = float(sub[:, 0].max())
        self.ymax = float(sub[:, 1].max())
        nx = max(2, int(np.ceil((self.xmax - self.xmin) / self.res)))
        ny = max(2, int(np.ceil((self.ymax - self.ymin) / self.res)))
        self.nx, self.ny = nx, ny
        self.xmax = self.xmin + nx * self.res
        self.ymax = self.ymin + ny * self.res

        xi = np.clip(np.floor((sub[:, 0] - self.xmin) / self.res).astype(np.int32), 0, nx - 1)
        yi = np.clip(np.floor((sub[:, 1] - self.ymin) / self.res).astype(np.int32), 0, ny - 1)
        flat = yi * nx + xi
        order = np.argsort(flat)
        sf = flat[order]
        sz = sub[order, 2]
        uid, starts, counts = np.unique(sf, return_index=True, return_counts=True)

        self.grid = np.full((ny, nx), np.nan, dtype=np.float64)
        for u, s, c in zip(uid, starts, counts):
            if c >= min_points:
                self.grid.flat[int(u)] = float(np.percentile(sz[s:s+c], percentile))
        self._fill_holes()
        logger.info("GroundDEM: %dx%d = %d cells, %.0f%% coverage",
                     nx, ny, nx*ny, np.sum(~np.isnan(self.grid))/(nx*ny)*100)

    def _fill_holes(self):
        grid = self.grid.copy()
        ny, nx = grid.shape
        offs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for _ in range(max(nx,ny)):
            mask = np.isnan(grid)
            if not mask.any(): break
            upd = False
            for dy, dx in offs:
                src = np.roll(grid, (-dy, -dx), (0, 1))
                f = mask & ~np.isnan(src)
                if dy>0: f[-dy:,:]=False
                elif dy<0: f[:-dy,:]=False
                if dx>0: f[:,-dx:]=False
                elif dx<0: f[:,:-dx]=False
                if f.any(): grid[f]=src[f]; upd=True
            if not upd: break
        self.grid = grid

    def get_ground_z(self, x, y):
        xa = np.atleast_1d(np.asarray(x, np.float64))
        ya = np.atleast_1d(np.asarray(y, np.float64))
        fx = (xa - self.xmin) / self.res
        fy = (ya - self.ymin) / self.res
        x0 = np.clip(np.floor(fx).astype(np.int32), 0, self.nx-1)
        y0 = np.clip(np.floor(fy).astype(np.int32), 0, self.ny-1)
        x1 = np.clip(x0+1, 0, self.nx-1)
        y1 = np.clip(y0+1, 0, self.ny-1)
        wx = fx - x0; wy = fy - y0
        z = (self.grid[y0,x0]*(1-wx)*(1-wy) + self.grid[y0,x1]*wx*(1-wy)
             + self.grid[y1,x0]*(1-wx)*wy + self.grid[y1,x1]*wx*wy)
        z[(fx<0)|(fy<0)|(fx>self.nx-1)|(fy>self.ny-1)] = np.nan
        return z


class PCIndex:
    def __init__(self, points, cell_size=1.0):
        pts = np.asarray(points, np.float64)
        self.pts = pts
        self.cs = float(cell_size)
        self.xmin = float(pts[:,0].min())
        self.ymin = float(pts[:,1].min())
        self.xmax = float(pts[:,0].max())
        self.ymax = float(pts[:,1].max())
        nx = max(1, int(np.floor((self.xmax-self.xmin)/self.cs))+1)
        ny = max(1, int(np.floor((self.ymax-self.ymin)/self.cs))+1)
        self.nx, self.ny = nx, ny
        xi = np.clip(np.floor((pts[:,0]-self.xmin)/self.cs).astype(np.int32), 0, nx-1)
        yi = np.clip(np.floor((pts[:,1]-self.ymin)/self.cs).astype(np.int32), 0, ny-1)
        flat = yi.astype(np.int64)*np.int64(nx) + xi.astype(np.int64)
        order = np.argsort(flat, kind="mergesort")
        sf = flat[order]
        uid, starts, counts = np.unique(sf, return_index=True, return_counts=True)
        self._order = order.astype(np.int32)
        self._uid = uid.astype(np.int64)
        self._starts = starts.astype(np.int64)
        self._counts = counts.astype(np.int64)

    def query(self, x0, y0, x1, y1):
        gx0 = max(0, min(self.nx-1, int(np.floor((x0-self.xmin)/self.cs))))
        gy0 = max(0, min(self.ny-1, int(np.floor((y0-self.ymin)/self.cs))))
        gx1 = max(0, min(self.nx-1, int(np.floor((x1-self.xmin)/self.cs))))
        gy1 = max(0, min(self.ny-1, int(np.floor((y1-self.ymin)/self.cs))))
        if gx1 < gx0 or gy1 < gy0:
            return np.empty(0, np.int32)
        xs = np.arange(gx0, gx1+1, dtype=np.int64)
        ys = np.arange(gy0, gy1+1, dtype=np.int64)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        cids = (gy.reshape(-1)*np.int64(self.nx) + gx.reshape(-1)).astype(np.int64)
        pos = np.searchsorted(self._uid, cids)
        valid = ((pos>=0)&(pos<len(self._uid))&
                 (self._uid[np.clip(pos,0,len(self._uid)-1)]==cids))
        if not valid.any():
            return np.empty(0, np.int32)
        parts = [self._order[int(self._starts[i]):int(self._starts[i])+int(self._counts[i])]
                 for i in pos[valid]]
        if not parts:
            return np.empty(0, np.int32)
        cands = np.concatenate(parts)
        p = self.pts[cands]
        m = (p[:,0]>=x0)&(p[:,0]<=x1)&(p[:,1]>=y0)&(p[:,1]<=y1)
        return cands[m]


def run_3d_validation_fast(
    instances,
    laz_paths,
    transform,
    config=None,
):
    """Fast bbox-based 3D validation.

    Uses bbox point cloud statistics instead of per-pixel mask testing.
    Much faster and still effective for screening false detections.
    """
    if config is None:
        config = Validation3DConfig()

    logger.info("=== 3D Validation (fast, bbox-based) ===")
    pc = load_point_cloud(laz_paths)
    dem = GroundDEM(pc, config.dem_resolution_m, config.dem_percentile,
                    config.dem_subsample_step, config.dem_min_points)
    idx = PCIndex(pc, cell_size=1.0)

    n = len(instances)
    passed = np.ones(n, dtype=bool)
    reasons: list[list[str]] = [[] for _ in range(n)]
    pt_count = np.zeros(n, dtype=np.int32)
    z_range = np.zeros(n, dtype=np.float32)
    p90_h = np.zeros(n, dtype=np.float32)
    elev_r = np.zeros(n, dtype=np.float32)

    # Process in batches by spatial grid
    batch_cell = 10.0  # 10m cells
    groups = defaultdict(list)
    for i, inst in enumerate(instances):
        x1, y1, x2, y2 = inst["bbox"]
        cx_px = (x1 + x2) / 2
        cy_px = (y1 + y2) / 2
        cx_w, cy_w = transform * (cx_px, cy_px)
        gx = int(np.floor((cx_w - idx.xmin) / batch_cell))
        gy = int(np.floor((cy_w - idx.ymin) / batch_cell))
        groups[(gx, gy)].append(i)

    logger.info("Grouped %d instances into %d cells (%.0fm grid)",
                 n, len(groups), batch_cell)

    done = 0
    for (gx, gy), idxs in groups.items():
        # Compute combined bbox
        xmins, ymins, xmaxs, ymaxs = [], [], [], []
        for i in idxs:
            x1, y1, x2, y2 = instances[i]["bbox"]
            wx0, wy0 = transform * (x1 - config.bbox_pad_m / transform[0],
                                     y1 - config.bbox_pad_m / abs(transform[4]))
            wx1, wy1 = transform * (x2 + config.bbox_pad_m / transform[0],
                                     y2 + config.bbox_pad_m / abs(transform[4]))
            if wx0 > wx1: wx0, wx1 = wx1, wx0
            if wy0 > wy1: wy0, wy1 = wy1, wy0
            xmins.append(wx0); ymins.append(wy0); xmaxs.append(wx1); ymaxs.append(wy1)

        qx0, qy0 = min(xmins), min(ymins)
        qx1, qy1 = max(xmaxs), max(ymaxs)

        pidx = idx.query(qx0, qy0, qx1, qy1)
        if len(pidx) == 0:
            for i in idxs:
                passed[i] = False
                reasons[i] = ["no_points_in_bbox"]
            done += len(idxs)
            continue

        pts = idx.pts[pidx]  # (M, 3)
        gz = dem.get_ground_z(pts[:, 0], pts[:, 1])
        gvalid = ~np.isnan(gz)

        for j, i in enumerate(idxs):
            wx0, wy0 = xmins[j], ymins[j]
            wx1, wy1 = xmaxs[j], ymaxs[j]

            in_bbox = ((pts[:,0]>=wx0)&(pts[:,0]<=wx1)&
                       (pts[:,1]>=wy0)&(pts[:,1]<=wy1))
            cnt = int(in_bbox.sum())
            pt_count[i] = cnt

            if cnt < config.min_points:
                passed[i] = False
                reasons[i].append("too_few_points")
                continue

            if gvalid.sum() == 0:
                passed[i] = False
                reasons[i].append("no_ground_data")
                continue

            local_gvalid = gvalid & in_bbox
            if local_gvalid.sum() < max(3, cnt // 10):
                passed[i] = False
                reasons[i].append("insufficient_ground_points")
                continue

            rel_h = np.maximum(pts[local_gvalid, 2] - gz[local_gvalid], 0.0)
            zr = float(np.max(pts[local_gvalid, 2]) - np.min(pts[local_gvalid, 2]))
            p90 = float(np.percentile(rel_h, 90))
            er = float(np.mean(rel_h >= config.elevated_height_m))

            z_range[i] = zr
            p90_h[i] = p90
            elev_r[i] = er

            if zr < config.min_z_range_m:
                passed[i] = False
                reasons[i].append("insufficient_z_range")
            if p90 < config.min_p90_height_m:
                passed[i] = False
                reasons[i].append("insufficient_p90_height")
            if er < config.min_elevated_ratio:
                passed[i] = False
                reasons[i].append("insufficient_elevated_ratio")

        done += len(idxs)
        if done % 10000 < len(idxs) or done == n:
            logger.info("  %d/%d (%.0f%%)", done, n, done/n*100)

    # Build output
    accepted, rejected = [], []
    rc: dict[str, int] = {}
    for i, inst in enumerate(instances):
        out = dict(inst)
        out["validation_3d"] = {
            "passed": bool(passed[i]),
            "reasons": reasons[i],
            "point_count": int(pt_count[i]),
            "z_range_m": round(float(z_range[i]), 4),
            "p90_height_m": round(float(p90_h[i]), 4),
            "elevated_ratio": round(float(elev_r[i]), 4),
        }
        if passed[i]:
            accepted.append(out)
        else:
            rejected.append(out)
            for r in reasons[i]:
                rc[r] = rc.get(r, 0) + 1

    summary = {
        "total": n, "accepted": len(accepted), "rejected": len(rejected),
        "rate": len(accepted)/n if n else 0,
        "reasons": rc,
        "config": {"min_points": config.min_points,
                   "min_z_range_m": config.min_z_range_m,
                   "min_p90_height_m": config.min_p90_height_m,
                   "min_elevated_ratio": config.min_elevated_ratio,
                   "elevated_height_m": config.elevated_height_m},
    }

    logger.info("Done: %d accepted, %d rejected (%.1f%%)",
                 len(accepted), len(rejected), len(accepted)/n*100 if n else 0)
    logger.info("Reasons: %s", rc)

    return accepted, rejected, summary

"""RX position sampling: street grid points inside scene bbox, exclude buildings."""

from __future__ import annotations

import numpy as np
import drjit as dr
import mitsuba as mi
import sionna.rt as srt


def _ray_test_inside_building(scene: srt.Scene, positions_xyz: np.ndarray,
                              up_max: float = 200.0) -> np.ndarray:
    """For each (x,y,z), cast a ray straight upward; if it does NOT escape the
    scene (i.e. hits geometry within `up_max`), the point is under/inside a
    building. Returns a boolean array `is_inside` of length N.
    """
    o = mi.Point3f(positions_xyz[:, 0].astype(np.float32),
                   positions_xyz[:, 1].astype(np.float32),
                   positions_xyz[:, 2].astype(np.float32))
    d = mi.Vector3f(0.0, 0.0, 1.0)
    ray = mi.Ray3f(o, d)  # default time=0, mono variant ignores wavelengths
    si = scene.mi_scene.ray_intersect(ray)
    t = np.array(si.t)  # distance along ray to first hit; inf if miss
    inside = np.isfinite(t) & (t < up_max)
    return inside


def _find_terrain_z(scene: srt.Scene, pts_xy: np.ndarray, z_top: float,
                    max_iters: int = 8) -> np.ndarray:
    """For each (x, y) cast a downward ray from `z_top` repeatedly, advancing
    past each hit to find the LAST surface below.  In Sionna RT scenes this
    bottom-most surface is the street / terrain (above it sit walls + roofs).

    Returns z_terrain of shape (N,); NaN where no surface was found
    (point is off the scene / outside any ground geometry).
    """
    pts_xy = np.asarray(pts_xy, dtype=np.float32)
    N = pts_xy.shape[0]
    last_z = np.full(N, np.nan, dtype=np.float64)
    cur_z = np.full(N, float(z_top), dtype=np.float64)
    active = np.ones(N, dtype=bool)

    for _ in range(max_iters):
        idx = np.where(active)[0]
        if idx.size == 0:
            break
        o = mi.Point3f(pts_xy[idx, 0], pts_xy[idx, 1],
                       cur_z[idx].astype(np.float32))
        d = mi.Vector3f(0.0, 0.0, -1.0)
        si = scene.mi_scene.ray_intersect(mi.Ray3f(o, d))
        t = np.array(si.t)
        hit = np.isfinite(t)
        # Update terrain estimate for points that hit, and continue ray below.
        hit_global = idx[hit]
        last_z[hit_global] = cur_z[hit_global] - t[hit]
        cur_z[hit_global] = last_z[hit_global] - 0.5
        # Points that didn't hit are done.
        active[idx[~hit]] = False
    return last_z


def grid_candidates(xy_min: np.ndarray, xy_max: np.ndarray,
                    step_m: float, rx_height: float) -> np.ndarray:
    """Regular xy-grid at fixed height.  Returns (N, 3) numpy float64."""
    xs = np.arange(xy_min[0], xy_max[0] + 1e-6, step_m, dtype=np.float64)
    ys = np.arange(xy_min[1], xy_max[1] + 1e-6, step_m, dtype=np.float64)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xv.ravel(), yv.ravel(),
                    np.full(xv.size, rx_height, dtype=np.float64)], axis=1)
    return pts


def random_candidates(xy_min: np.ndarray, xy_max: np.ndarray, n: int,
                      rx_height: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform random xy samples inside scene bbox."""
    xy = rng.uniform(low=xy_min, high=xy_max, size=(n, 2))
    z = np.full((n, 1), rx_height, dtype=np.float64)
    return np.concatenate([xy, z], axis=1)


def _find_first_hit_z(scene: srt.Scene, pts_xy: np.ndarray,
                       z_top: float = 700.0) -> np.ndarray:
    """Single downward ray from `z_top`; return first-hit z per (x,y).

    Used for the new range-based outdoor detection (rx_outdoor_z_range).
    For scenes WITH an artificial back-up plane below the real ground
    (e.g. SHENZHEN_with_ground), `_find_terrain_z` would descend past the
    real city ground to the back-up plane and report z = -0.2; this single
    cast keeps the FIRST surface, which is the rooftop or city ground.
    """
    pts = np.asarray(pts_xy, dtype=np.float32)
    o = mi.Point3f(pts[:, 0], pts[:, 1],
                   np.full(len(pts), float(z_top), dtype=np.float32))
    si = scene.mi_scene.ray_intersect(mi.Ray3f(o, mi.Vector3f(0., 0., -1.)))
    t = np.array(si.t)
    z_hit = z_top - t
    z_hit = np.where(np.isfinite(z_hit), z_hit, np.nan)
    return z_hit


def sample_street_positions(scene: srt.Scene,
                            n_target: int,
                            rx_height: float = 1.5,
                            grid_step_m: float = 1.0,
                            mode: str = "grid",
                            rng_seed: int = 0,
                            tx_xy: np.ndarray | None = None,
                            max_radius: float | None = None,
                            outdoor_z_range: tuple[float, float] | None = None
                            ) -> np.ndarray:
    """Generate up to `n_target` candidate RX positions outside buildings.

    `mode="grid"` uses a regular grid then shuffles + truncates.
    `mode="random"` uniformly samples the bbox.

    If `max_radius` is given (with `tx_xy`), only points whose xy distance
    to `tx_xy` is <= `max_radius` are kept -- this constrains the dataset
    to a single ~cell-sized region around the TX.

    Returns: (M, 3) array with M <= n_target outdoor positions.
    """
    from .scenes import scene_xy_bbox
    rng = np.random.default_rng(rng_seed)
    xy_min, xy_max = scene_xy_bbox(scene)

    # When a TX-centred radius is given, restrict candidate-generation
    # bbox to a tight square around the TX (or union square covering all
    # TXs in multi-BS mode) so we don't waste compute on cells we'd
    # discard anyway.  `tx_xy` may be shape (2,) or (K, 2).
    tx_xy_arr = None
    if max_radius is not None and tx_xy is not None:
        tx_xy_arr = np.atleast_2d(np.asarray(tx_xy, dtype=np.float64))
        bbox_min = np.array([tx_xy_arr[:, 0].min() - max_radius,
                             tx_xy_arr[:, 1].min() - max_radius])
        bbox_max = np.array([tx_xy_arr[:, 0].max() + max_radius,
                             tx_xy_arr[:, 1].max() + max_radius])
        xy_min = np.maximum(xy_min, bbox_min)
        xy_max = np.minimum(xy_max, bbox_max)

    if mode == "grid":
        pts = grid_candidates(xy_min, xy_max, grid_step_m, rx_height)
        # shuffle once so that later truncation is spatially diverse
        idx = rng.permutation(pts.shape[0])
        pts = pts[idx]
    elif mode == "random":
        # oversample 2x to absorb building-pruning loss
        pts = random_candidates(xy_min, xy_max,
                                int(n_target * 2.5), rx_height, rng)
    else:
        raise ValueError(f"Unknown sampling mode {mode!r}")

    # Disk filter -- in multi-BS mode keep points within `max_radius` of
    # ANY TX (union of disks).
    if max_radius is not None and tx_xy_arr is not None:
        # shape (N, K): distance from each candidate to each TX.
        dxy = pts[:, None, :2] - tx_xy_arr[None, :, :]
        dmin = np.sqrt(np.sum(dxy * dxy, axis=-1)).min(axis=1)
        pts = pts[dmin <= max_radius]

    # Ground-plane scenes (e.g. SHENZHEN with_ground) can have ray-marching
    # land on a rooftop when the building has no floor mesh below.  We want
    # pedestrians on the road, so reject obvious outliers: keep only RX
    # within ~5 m of the median candidate height (street level).
    if pts.shape[0] > 50:
        z_med = float(np.median(pts[:, 2]))
        ok = np.abs(pts[:, 2] - z_med) < 5.0
        pts = pts[ok]

    # Determine ray-cast starting altitude (top of bbox + buffer).
    bbox = scene.mi_scene.bbox()
    z_top = float(bbox.max[2]) + 50.0

    # New path: explicit outdoor z range. Single downward ray + accept only
    # candidates whose first hit lands inside [z_min, z_max].  This skips
    # under-building cells (first hit at rooftop is above z_max) and
    # back-up-plane cells (first hit at street above z_min, but the actual
    # ground we want is the first hit, so we don't ray-march past it).
    use_first_hit = outdoor_z_range is not None
    z_lo, z_hi = (outdoor_z_range if outdoor_z_range is not None
                  else (-np.inf, np.inf))

    keep_pts = []
    kept = 0
    chunk = 100_000
    for s in range(0, pts.shape[0], chunk):
        sub = pts[s:s + chunk]
        if use_first_hit:
            z_first = _find_first_hit_z(scene, sub[:, :2], z_top=z_top)
            valid = np.isfinite(z_first) & (z_first >= z_lo) & (z_first <= z_hi)
            if not valid.any():
                continue
            sub_lifted = sub[valid].copy()
            sub_lifted[:, 2] = z_first[valid] + rx_height
        else:
            z_terrain = _find_terrain_z(scene, sub[:, :2], z_top)
            valid = np.isfinite(z_terrain)
            if not valid.any():
                continue
            sub_lifted = sub[valid].copy()
            sub_lifted[:, 2] = z_terrain[valid] + rx_height
        # Building-cover test: upward ray from RX; if first hit is within
        # ~3 m above, RX is under cover (canopy/balcony) -> drop.
        inside = _ray_test_inside_building(scene, sub_lifted, up_max=3.0)
        outdoor = sub_lifted[~inside]
        keep_pts.append(outdoor)
        kept += outdoor.shape[0]
        if kept >= n_target:
            break
    if not keep_pts:
        return np.zeros((0, 3), dtype=np.float64)
    out = np.concatenate(keep_pts, axis=0)
    if out.shape[0] > n_target:
        out = out[:n_target]
    return out.astype(np.float64)

"""Auto-pick K BS positions for a scene.

Strategy: scan the scene's xy plane on a regular grid, compute building
height at every cell (first downward hit minus terrain), then greedily
pick the tallest cells while enforcing a minimum separation between
chosen BS locations.

The result is a list of `(x, y, z)` triples (absolute world coords) ready
to drop into the SceneConfig.tx_positions field.
"""

from __future__ import annotations

import numpy as np
import mitsuba as mi
import sionna.rt as srt

from .sampler import _find_terrain_z


def _grid_building_heights(scene: srt.Scene, step_m: float = 5.0
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray]:
    """Return (X, Y, z_first_hit, z_terrain) on a regular xy grid covering
    the scene bbox.  Outputs are 2-D arrays in xy-grid order."""
    bbox = scene.mi_scene.bbox()
    bmin, bmax = np.array(bbox.min), np.array(bbox.max)
    xs = np.arange(bmin[0] + 5.0, bmax[0] - 5.0, step_m)
    ys = np.arange(bmin[1] + 5.0, bmax[1] - 5.0, step_m)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts_xy = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    z_top = float(bmax[2]) + 50.0

    o = mi.Point3f(pts_xy[:, 0], pts_xy[:, 1],
                   np.full(len(pts_xy), float(z_top), dtype=np.float32))
    si = scene.mi_scene.ray_intersect(mi.Ray3f(o,
                                                mi.Vector3f(0.0, 0.0, -1.0)))
    z_first = (z_top - np.array(si.t)).reshape(X.shape)
    z_terr = _find_terrain_z(scene, pts_xy,
                              z_top=z_top).reshape(X.shape)
    return X, Y, z_first, z_terr


def pick_bs_positions(scene: srt.Scene, num_bs: int,
                      min_separation_m: float = 200.0,
                      mount_above_roof_m: float = 2.5,
                      grid_step_m: float = 5.0,
                      min_building_height_m: float = 8.0,
                      max_building_height_m: float | None = None,
                      restrict_to_bbox: tuple[float, float, float,
                                              float] | None = None,
                      ) -> list[tuple[float, float, float]]:
    """Greedy-pick `num_bs` BS positions.

    Each chosen BS sits on the roof of a tall building, with mount height
    `mount_above_roof_m` above the roof.  Subsequent picks must be at
    least `min_separation_m` (xy distance) away from all previous picks.

    `restrict_to_bbox`, if provided, is (x_min, x_max, y_min, y_max) -- only
    candidates inside this rectangle are considered (useful for scenes
    like SUTD where Singapore Expo halls or non-campus tall buildings
    would otherwise dominate).
    """
    X, Y, z_first, z_terr = _grid_building_heights(scene, step_m=grid_step_m)
    bldg = (z_first - z_terr).clip(0)

    valid = np.isfinite(bldg) & (bldg > min_building_height_m)
    if max_building_height_m is not None:
        valid &= bldg < max_building_height_m
    if restrict_to_bbox is not None:
        x_min, x_max, y_min, y_max = restrict_to_bbox
        valid &= (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)

    candidates = np.argsort(-np.where(valid, bldg, 0.0).ravel())
    chosen: list[tuple[float, float, float]] = []
    chosen_xy = np.zeros((0, 2), dtype=np.float64)

    for flat_idx in candidates:
        iy, ix = np.unravel_index(flat_idx, X.shape)
        if not valid[iy, ix]:
            break
        x, y = float(X[iy, ix]), float(Y[iy, ix])
        if chosen_xy.size:
            d = np.sqrt(np.sum((chosen_xy - np.array([x, y])) ** 2, axis=1))
            if d.min() < min_separation_m:
                continue
        roof_z = float(z_first[iy, ix])
        chosen.append((x, y, roof_z + mount_above_roof_m))
        chosen_xy = np.vstack([chosen_xy, [x, y]])
        if len(chosen) >= num_bs:
            break
    return chosen

"""Scene loading + TX/array configuration for Sionna RT built-in cities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import os

import numpy as np
import sionna.rt as srt

from .paths import repository_root

_REPO_ROOT = str(repository_root())


BUILTIN_SCENES = {
    "munich": srt.scene.munich,
    "etoile": srt.scene.etoile,
    "florence": srt.scene.florence,
    "simple_street_canyon": srt.scene.simple_street_canyon,
    # Custom scenes added under `scenes/<name>/<name>.xml`.
    "ZJU":  os.path.join(_REPO_ROOT, "scenes", "ZJU",  "ZJU.xml"),
    "SUTD": os.path.join(_REPO_ROOT, "scenes", "SUTD", "SUTD.xml"),
    # Shenzhen Futian CBD (Ping An tower 637m cluster); use the
    # _with_ground variant so the streets between super-tall towers
    # behave correctly for terrain-aware RX placement.
    "SHENZHEN": os.path.join(_REPO_ROOT, "scenes", "SHENZHEN",
                              "SHENZHEN_SKYLINE_with_ground.xml"),
    # Alias for the test-only / all-BS variant (same scene, different
    # `configs/SHENZHEN_test.yaml`).
    "SHENZHEN_test": os.path.join(_REPO_ROOT, "scenes", "SHENZHEN",
                                   "SHENZHEN_SKYLINE_with_ground.xml"),
    # Alias for few-shot training-data generation with a different RX seed.
    "SHENZHEN_train": os.path.join(_REPO_ROOT, "scenes", "SHENZHEN",
                                    "SHENZHEN_SKYLINE_with_ground.xml"),
}


@dataclass
class SceneConfig:
    name: str
    frequency_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    num_subcarriers: int = 1024

    # BS array
    bs_rows: int = 1
    bs_cols: int = 32
    bs_vertical_spacing: float = 0.5
    bs_horizontal_spacing: float = 0.5
    bs_pattern: str = "iso"
    bs_polarization: str = "V"

    # UE array
    ue_rows: int = 1
    ue_cols: int = 1
    ue_pattern: str = "iso"
    ue_polarization: str = "V"

    # Single TX position (legacy single-BS path).  If `tx_positions` is
    # given, this is ignored.  If both are None, auto-place above scene
    # xy-center at `tx_height` (single-BS mode).
    tx_position: tuple[float, float, float] | None = None
    # Multi-BS placements as a list of absolute (x, y, z) triples.  Takes
    # precedence over `tx_position` and disables auto-selection.
    tx_positions: list[tuple[float, float, float]] | None = None
    # Number of BSes to auto-pick when neither `tx_position` nor
    # `tx_positions` is set.  1 -> backward compatible single-BS behaviour.
    num_bs: int = 1
    # Minimum xy separation between auto-picked BSes [m].  Tuned per
    # scene size (canyon ~50 m, city ~250 m).
    min_bs_separation_m: float = 250.0
    # Mount height above the roof of the chosen building [m].
    bs_mount_above_roof_m: float = 2.5
    # Restrict auto BS picking to a (x_min, x_max, y_min, y_max) rectangle.
    # Useful for scenes where unwanted geometry (e.g. Singapore Expo halls
    # in SUTD or Yuquan Mountain residential towers in ZJU) would otherwise
    # be picked as tall buildings.
    bs_pick_bbox: tuple[float, float, float, float] | None = None
    # Restrict auto BS picking to rooftops whose height (above terrain)
    # falls within this [min, max] range, in metres.  Useful for CBD scenes
    # like Shenzhen where we want realistic macro-cell heights (~30 m) and
    # the super-tall towers (Ping An 637m, KK100 442m, etc.) should be
    # *scatterers*, not BS sites.  None disables the filter.
    bs_pick_height_range_m: tuple[float, float] | None = None
    tx_height: float = 25.0
    rx_height: float = 1.5
    grid_step_m: float = 1.0
    # Acceptable street/terrain z range (m) for placing RX.
    # If set, the sampler casts a single downward ray from above and only
    # accepts candidates whose FIRST hit z is in [zmin, zmax].  This
    # prevents two bugs:
    #   1) SHENZHEN with_ground has a back-up plane at z = -0.2 that
    #      sits BELOW the actual city street at z ~= 35.  Without this
    #      range filter the ray-marching descends to -0.2 and places RX
    #      at z = 1.3 (in the basement, under the city floor).
    #   2) In any scene, a candidate under a building gets the first hit
    #      at the rooftop; without this filter we'd ray-march past it and
    #      place RX under the building.  The range cap catches that too.
    # None = legacy behaviour (use _find_terrain_z, return last hit).
    rx_outdoor_z_range: tuple[float, float] | None = None
    # Cap the RX sampling region to a disk of this radius around the TX
    # (xy distance, in metres).  Keeps each scene at a single ~cell scale
    # so delays fit in the 32-tap budget at 20 MHz BW.  `None` = no cap.
    max_radius_from_tx: float | None = 500.0

    # PathSolver
    max_depth: int = 5
    los: bool = True
    specular_reflection: bool = True
    refraction: bool = True
    diffraction: bool = True
    diffuse_reflection: bool = False
    # Override scattering coefficient on all scene radio materials when > 0.
    # Sionna's built-in ITU materials default scattering_coefficient = 0,
    # which means no diffuse reflection paths even with diffuse_reflection=True.
    # Set 0.3-0.5 for realistic urban (concrete/glass) multipath.
    scattering_coefficient: float = 0.0
    # Lambertian scattering pattern's α_R parameter when scattering>0.
    # Higher α_R = more forward-scattered (mirror-like), lower = more diffuse.
    scattering_alpha_r: float = 1.0
    synthetic_array: bool = True
    seed: int = 42

    # Multi-BS serving:
    #   "argmax" -- per RX keep only the strongest BS's channel (1 sample/RX)
    #   "all_bs" -- per RX keep ALL K BS channels (up to K samples/RX); each
    #               sample tagged with bs_id 0..K-1; better realism for
    #               heterogeneous cellular / cross-BS evaluation.
    serving_mode: str = "argmax"

    # Sampling
    target_samples: int = 40_000
    oversample: float = 1.5
    batch_size: int = 1000

    # Splits
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    # test_ratio = 1 - train_ratio - val_ratio


def get_scene_xml(name: str) -> str:
    if name not in BUILTIN_SCENES:
        raise ValueError(
            f"Unknown scene '{name}'. Available: {sorted(BUILTIN_SCENES.keys())}"
        )
    return BUILTIN_SCENES[name]


def scene_xy_bbox(scene: srt.Scene) -> Tuple[np.ndarray, np.ndarray]:
    """Return (xy_min, xy_max) numpy float arrays of shape (2,)."""
    bbox = scene.mi_scene.bbox()
    bmin = np.array([float(bbox.min[0]), float(bbox.min[1])], dtype=np.float64)
    bmax = np.array([float(bbox.max[0]), float(bbox.max[1])], dtype=np.float64)
    return bmin, bmax


def _apply_scattering_to_materials(scene: srt.Scene, coeff: float,
                                   alpha_r: float = 1.0) -> int:
    """Override scattering_coefficient on every radio material in the scene.

    ITU materials default to 0 (no diffuse paths even with the flag), so we
    force it to coeff so diffuse_reflection=True actually produces scattered
    rays. Returns the number of materials touched.
    """
    if coeff <= 0:
        return 0
    n = 0
    # Sionna RT scene exposes `radio_materials` dict (name → RadioMaterial).
    materials = getattr(scene, "radio_materials", None) or {}
    for mat in materials.values():
        try:
            mat.scattering_coefficient = float(coeff)
            # LambertianPattern is the default; some Sionna versions need it
            # re-set when scattering_coefficient changes.
            if hasattr(mat, "scattering_pattern") and hasattr(srt, "LambertianPattern"):
                if mat.scattering_pattern is None:
                    mat.scattering_pattern = srt.LambertianPattern()
            n += 1
        except Exception:
            pass
    return n


def build_scene(cfg: SceneConfig) -> srt.Scene:
    """Load the built-in scene, set frequency + arrays + TX."""
    scene = srt.load_scene(get_scene_xml(cfg.name))
    scene.frequency = cfg.frequency_hz
    if cfg.diffuse_reflection and cfg.scattering_coefficient > 0:
        n_mats = _apply_scattering_to_materials(
            scene, cfg.scattering_coefficient, cfg.scattering_alpha_r)
        print(f"[{cfg.name}] applied scattering_coefficient={cfg.scattering_coefficient:.2f} "
              f"to {n_mats} radio materials (Lambertian pattern)")

    scene.tx_array = srt.PlanarArray(
        num_rows=cfg.bs_rows,
        num_cols=cfg.bs_cols,
        vertical_spacing=cfg.bs_vertical_spacing,
        horizontal_spacing=cfg.bs_horizontal_spacing,
        pattern=cfg.bs_pattern,
        polarization=cfg.bs_polarization,
    )
    scene.rx_array = srt.PlanarArray(
        num_rows=cfg.ue_rows,
        num_cols=cfg.ue_cols,
        pattern=cfg.ue_pattern,
        polarization=cfg.ue_polarization,
    )

    tx_list = _resolve_tx_positions(scene, cfg)
    for i, tx_pos in enumerate(tx_list):
        # Backward-compat name 'tx' for single-BS mode; 'tx_<i>' for
        # multi-BS so all the downstream code that does
        # `scene.transmitters[...]` keeps working.
        name = "tx" if len(tx_list) == 1 else f"tx_{i}"
        scene.add(srt.Transmitter(name=name, position=list(tx_pos)))
    return scene


def _resolve_tx_positions(scene: srt.Scene, cfg: SceneConfig
                          ) -> list[tuple[float, float, float]]:
    """Decide where to place TX(s) based on the config."""
    # Manual multi-BS list takes precedence.
    if cfg.tx_positions is not None:
        return [tuple(float(v) for v in p) for p in cfg.tx_positions]

    # Manual single-BS position (legacy).
    if cfg.tx_position is not None:
        return [tuple(float(v) for v in cfg.tx_position)]

    # Auto single-BS = scene xy-center + local terrain (legacy default).
    if cfg.num_bs <= 1:
        xy_min, xy_max = scene_xy_bbox(scene)
        cx, cy = 0.5 * (xy_min + xy_max)
        z_terrain = _find_terrain_z_single(scene, float(cx), float(cy))
        if not np.isfinite(z_terrain):
            z_terrain = 0.0
        return [(float(cx), float(cy),
                 float(z_terrain) + float(cfg.tx_height))]

    # Auto multi-BS via the bs_selector module.
    from .bs_selector import pick_bs_positions
    h_lo, h_hi = (None, None)
    if cfg.bs_pick_height_range_m is not None:
        h_lo, h_hi = cfg.bs_pick_height_range_m
    chosen = pick_bs_positions(
        scene,
        num_bs=int(cfg.num_bs),
        min_separation_m=float(cfg.min_bs_separation_m),
        mount_above_roof_m=float(cfg.bs_mount_above_roof_m),
        restrict_to_bbox=cfg.bs_pick_bbox,
        min_building_height_m=h_lo if h_lo is not None else 8.0,
        max_building_height_m=h_hi,
    )
    if len(chosen) < int(cfg.num_bs):
        print(f"[WARN] only {len(chosen)} BS positions could be found "
              f"(requested {cfg.num_bs}); reduce min_bs_separation_m or "
              f"relax bs_pick_bbox.")
    return chosen


def _find_terrain_z_single(scene: srt.Scene, x: float, y: float) -> float:
    """Single-point variant of sampler._find_terrain_z; advances downward
    rays through all upper surfaces to find the ground/terrain z. Returns
    NaN if nothing is found (point is off the scene)."""
    import mitsuba as mi
    bbox = scene.mi_scene.bbox()
    z = float(bbox.max[2]) + 50.0
    last_z = float("nan")
    for _ in range(8):
        si = scene.mi_scene.ray_intersect(mi.Ray3f(
            mi.Point3f(float(x), float(y), float(z)),
            mi.Vector3f(0.0, 0.0, -1.0)))
        t = float(np.asarray(si.t).reshape(-1)[0])
        if not np.isfinite(t):
            break
        last_z = z - t
        z = last_z - 0.5
    return last_z

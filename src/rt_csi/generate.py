"""Generate one RT-CSI scene or all five Mix5 scenes."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config_loader import load_scene_config
from .dataset import generate_scene_dataset


MIX5_SCENES = ("etoile", "ZJU", "SUTD", "florence", "munich")
SCENES = (*MIX5_SCENES, "SHENZHEN_test", "SHENZHEN_train")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=(*SCENES, "all"),
                        help="scene name, or all five Mix5 scenes")
    parser.add_argument("--out", type=Path,
                        help="NPZ path for one scene; defaults under --out-dir")
    parser.add_argument("--out-dir", type=Path, default=Path("regenerated"))
    parser.add_argument("--target-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grid-step-m", type=float)
    parser.add_argument("--rx-height", type=float)
    parser.add_argument("--tx-height", type=float)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--frequency", type=float, dest="frequency_hz")
    parser.add_argument("--bandwidth", type=float, dest="bandwidth_hz")
    parser.add_argument("--num-subcarriers", type=int)
    parser.add_argument("--oversample", type=float)
    parser.add_argument("--max-radius-from-tx", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.scene == "all" and args.out is not None:
        parser.error("--out is only valid for a single scene; use --out-dir")

    fields = (
        "target_samples", "batch_size", "grid_step_m", "rx_height",
        "tx_height", "max_depth", "frequency_hz", "bandwidth_hz",
        "num_subcarriers", "oversample", "max_radius_from_tx", "seed",
    )
    overrides = {field: getattr(args, field) for field in fields
                 if getattr(args, field) is not None}
    scene_names = MIX5_SCENES if args.scene == "all" else (args.scene,)
    for scene_name in scene_names:
        cfg = load_scene_config(scene_name, overrides=overrides)
        default_name = (f"csi_{scene_name}_{int(cfg.frequency_hz / 1e9)}GHz_"
                        f"32x{cfg.num_subcarriers}.npz")
        out = args.out or args.out_dir / default_name
        generate_scene_dataset(cfg, str(out), verbose=not args.quiet)


if __name__ == "__main__":
    main()

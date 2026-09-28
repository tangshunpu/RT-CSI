"""Rebuild the Mix5 archive from the five published per-scene archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCENES = ("etoile", "ZJU", "SUTD", "florence", "munich")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path,
                        default=Path("regenerated/csi_combined5_3GHz_32x1024.npz"))
    args = parser.parse_args(argv)

    out: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        xs, positions, ids = [], [], []
        for scene_id, scene in enumerate(SCENES):
            path = args.scene_dir / f"csi_{scene}_3GHz_32x1024.npz"
            with np.load(path, allow_pickle=False) as source:
                x = source[f"x_{split}"]
                xs.append(x)
                positions.append(source[f"pos_{split}"])
                ids.append(np.full(len(x), scene_id, dtype=np.int32))
        out[f"x_{split}"] = np.concatenate(xs)
        out[f"pos_{split}"] = np.concatenate(positions)
        out[f"scene_id_{split}"] = np.concatenate(ids)
    out["scenes"] = np.array(json.dumps(SCENES))
    out["scale"] = np.float32(np.nan)
    out["config"] = np.array(json.dumps({"combined": SCENES,
                                        "norm": "per-sample max-abs"}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(args.out)


if __name__ == "__main__":
    main()

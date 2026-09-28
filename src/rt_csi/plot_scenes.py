"""Plot one angular-delay CSI example from each of the six benchmark scenes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


SCENES = (
    ("munich", "Munich"),
    ("etoile", "Paris Étoile"),
    ("florence", "Florence"),
    ("ZJU", "ZJU"),
    ("SUTD", "SUTD"),
    ("SHENZHEN_test", "Shenzhen Futian CBD"),
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path,
                        default=Path("assets/angular_delay_six_scenes.png"))
    parser.add_argument("--sample-index", type=int, default=100)
    parser.add_argument("--floor-db", type=float, default=-50.0)
    args = parser.parse_args(argv)
    if args.sample_index < 0:
        parser.error("--sample-index must be nonnegative")
    if args.floor_db >= 0:
        parser.error("--floor-db must be negative")

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.4), layout="constrained")
    fig.patch.set_facecolor("white")
    colors = LinearSegmentedColormap.from_list(
        "white_blue", ("#ffffff", "#deebf7", "#9ecae1", "#3182bd", "#08519c"))
    image = None
    for ax, (filename, title) in zip(axes.flat, SCENES):
        path = args.data_dir / f"csi_{filename}_3GHz_32x1024.npz"
        with np.load(path, allow_pickle=False) as ds:
            samples = ds["x_test"]
            if args.sample_index >= len(samples):
                raise IndexError(f"{path}: test index {args.sample_index} out of range")
            sample = samples[args.sample_index].copy()
            del samples
        channel = (sample[0] - 0.5) + 1j * (sample[1] - 0.5)
        magnitude = np.abs(channel)
        relative_db = 20 * np.log10(np.maximum(
            magnitude / magnitude.max(), 10 ** (args.floor_db / 20)))
        image = ax.imshow(relative_db, origin="lower", aspect="equal",
                          vmin=args.floor_db, vmax=0, cmap=colors,
                          interpolation="nearest")
        ax.set_box_aspect(1)
        ax.set_facecolor("white")
        ax.set_title(title)
        ax.set_xticks((0, 8, 16, 24, 31))
        ax.set_yticks((0, 8, 16, 24, 31))
    fig.supxlabel("Delay tap (first 32 of 1,024)")
    fig.supylabel("Angular bin (32-antenna DFT)")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.76, pad=0.015)
    colorbar.set_label("Relative magnitude (dB)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor="white")
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()

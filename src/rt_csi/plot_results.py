"""Plot the manuscript's Sionna-RT-Mix5 NMSE comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


SCENES = (("Paris", "paris"), ("Florence", "florence"),
          ("Munich", "munich"), ("ZJU", "zju"), ("SUTD", "sutd"))
RATIOS = (("1/4", "1_4"), ("1/8", "1_8"), ("1/16", "1_16"))
METHODS = (
    ("LRP", "LRP", "#9B9B9B"),
    ("CRNet", "CRNet", "#4C78A8"),
    ("CLNet", "CLNet", "#72B7B2"),
    ("DCRNet-1x", "DCRNet-1×", "#8C9B48"),
    ("TransNet", "TransNet", "#E2A84B"),
    ("DCRNetV2-unified", "DCRNetV2-unified", "#C74764"),
)


def read_results(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {name for name, _, _ in METHODS}
    if {row["method"] for row in rows} != expected or len(rows) != len(expected):
        raise ValueError(f"{path}: expected one row per method")
    results = {}
    for row in rows:
        values = {}
        for _, scene in SCENES:
            for _, ratio in RATIOS:
                key = f"{scene}_{ratio}"
                value = float(row[key])
                if not np.isfinite(value) or value >= 0:
                    raise ValueError(f"{path}: expected a negative NMSE dB value at {row['method']}, {key}")
                values[key] = value
        results[row["method"]] = values
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("docs/mix5-nmse.csv"))
    parser.add_argument("--out", type=Path, default=Path("assets/mix5_nmse_bars.png"))
    args = parser.parse_args(argv)
    results = read_results(args.csv)

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 11.2), sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Sionna-RT-Mix5 NMSE by scene and compression ratio",
                 fontsize=17, y=0.995)
    fig.text(0.5, 0.966,
             "Bar height = −NMSE (dB); taller is better. Dark outline marks the best result in each scene.",
             ha="center", va="top", fontsize=10.5, color="#3B4651")

    x = np.arange(len(SCENES))
    width = 0.13
    offset = (len(METHODS) - 1) * width / 2
    for ax, (ratio_label, ratio_key) in zip(axes, RATIOS):
        best = {
            scene: min(METHODS, key=lambda method: results[method[0]][f"{scene}_{ratio_key}"])[0]
            for _, scene in SCENES
        }
        for method_index, (name, _, color) in enumerate(METHODS):
            heights = [-results[name][f"{scene}_{ratio_key}"] for _, scene in SCENES]
            edgecolors = ["#18212B" if best[scene] == name else "white"
                          for _, scene in SCENES]
            linewidths = [1.4 if best[scene] == name else 0.5
                          for _, scene in SCENES]
            ax.bar(x - offset + method_index * width, heights, width,
                   color=color, edgecolor=edgecolors, linewidth=linewidths,
                   zorder=3)
        ax.set_title(f"Compression ratio η = {ratio_label}", loc="left",
                     fontsize=12, fontweight="semibold", pad=8)
        ax.set_xticks(x, [label for label, _ in SCENES], fontsize=11)
        ax.set_ylim(0, 27)
        ax.set_yticks((0, 5, 10, 15, 20, 25))
        ax.set_ylabel("−NMSE (dB)", fontsize=10.5)
        ax.grid(axis="y", color="#DDE2E7", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#7A858F")
    handles = [Patch(facecolor=color, edgecolor="none", label=label)
               for _, label, color in METHODS]
    handles = [handles[i] for i in (0, 3, 1, 4, 2, 5)]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.944),
               frameon=False, fontsize=10.5, columnspacing=2.2)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.865, bottom=0.065, hspace=0.43)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor="white")
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()

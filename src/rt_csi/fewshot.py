"""Remove UE overlap with Shenzhen test and split remaining UEs 80/20.

The paper-experiment files are preserved separately. This creates a corrected
alternative without altering their CSI values or the fixed test archive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def position_keys(positions: np.ndarray) -> list[bytes]:
    return [row.tobytes() for row in np.asarray(positions, dtype=np.float32)]


def convert(source_path: Path, test_path: Path, output_path: Path) -> tuple[int, int, int]:
    with np.load(test_path, allow_pickle=False) as test:
        test_positions = set(position_keys(test["pos_test"]))
    with np.load(source_path, allow_pickle=False) as source:
        train_positions = source["pos_train"]
        val_positions = source["pos_val"]
        all_positions = np.concatenate((train_positions, val_positions))
        keys = position_keys(all_positions)
        eligible = np.array([key not in test_positions for key in keys])
        unique_keys = list(dict.fromkeys(key for key, keep in zip(keys, eligible) if keep))
        if len(unique_keys) < 2:
            raise ValueError("Need at least two distinct eligible UE positions")
        groups = {key: i for i, key in enumerate(unique_keys)}
        counts = np.bincount([groups[key] for key, keep in zip(keys, eligible) if keep],
                             minlength=len(unique_keys))
        rng = np.random.default_rng(20260928)
        order = rng.permutation(len(unique_keys))
        target = round(int(eligible.sum()) * 0.8)
        cumulative = np.cumsum(counts[order])
        cut = min(range(1, len(order)), key=lambda i: abs(int(cumulative[i - 1]) - target))
        train_groups = set(order[:cut].tolist())
        train_mask = np.array([keep and groups[key] in train_groups
                               for key, keep in zip(keys, eligible)])
        val_mask = eligible & ~train_mask

        out: dict[str, np.ndarray] = {}
        handled: set[str] = set()
        for name in source.files:
            if name in handled:
                continue
            counterpart = name.replace("_train", "_val", 1)
            if "_train" in name and counterpart in source.files:
                all_values = np.concatenate((source[name], source[counterpart]))
                out[name] = all_values[train_mask]
                out[counterpart] = all_values[val_mask]
                handled.add(counterpart)
            elif name == "config":
                cfg = json.loads(str(source[name]))
                cfg.update(n_train=int(train_mask.sum()), n_val=int(val_mask.sum()),
                           n_test=0, release_disjoint_by_ue=True,
                           release_source_file=source_path.name,
                           release_removed_test_ue_links=int((~eligible).sum()))
                out[name] = np.array(json.dumps(cfg))
            else:
                out[name] = source[name]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out)
    return int(train_mask.sum()), int(val_mask.sum()), int((~eligible).sum())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("data"))
    parser.add_argument("--test", type=Path,
                        default=Path("data/csi_SHENZHEN_test_3GHz_32x1024.npz"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("regenerated/disjoint-fewshot"))
    args = parser.parse_args(argv)
    for source in sorted(args.source_dir.glob("csi_SHENZHEN_train*_3GHz_32x1024.npz")):
        output = args.out_dir / source.name.replace(".npz", "_disjoint.npz")
        n_train, n_val, removed = convert(source, args.test, output)
        print(f"{output.name}: train={n_train}, val={n_val}, removed={removed}")


if __name__ == "__main__":
    main()

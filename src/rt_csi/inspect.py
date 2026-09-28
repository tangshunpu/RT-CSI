"""Print shapes, value ranges, and metadata of a generated RT-CSI archive."""

from __future__ import annotations

import argparse
import json

import numpy as np


def summarize(path: str, name: str = "") -> None:
    d = np.load(path, allow_pickle=False)
    print(f"\n=== {name or path} ===")
    for k in d.files:
        v = d[k]
        if k == "config":
            try:
                cfg = json.loads(str(v))
                print(f"  config (json): {len(cfg)} keys")
                for ck, cv in cfg.items():
                    print(f"    {ck} = {cv}")
            except Exception:
                print(f"  {k}: <opaque> {v.shape if hasattr(v,'shape') else v}")
            continue
        if v.ndim == 0:
            print(f"  {k:14s}: scalar = {v}")
            continue
        arr = v
        if arr.size == 0:
            print(f"  {k:14s}: shape={arr.shape} dtype={arr.dtype} empty")
            continue
        if not np.issubdtype(arr.dtype, np.number):
            print(f"  {k:14s}: shape={arr.shape} dtype={arr.dtype} value={arr}")
            continue
        print(f"  {k:14s}: shape={arr.shape} dtype={arr.dtype} "
              f"min={float(arr.min()):.4f} max={float(arr.max()):.4f} "
              f"mean={float(arr.mean()):.4f} std={float(arr.std()):.4f}")


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    a = p.parse_args(argv)
    for path in a.paths:
        summarize(path)


if __name__ == "__main__":
    main()

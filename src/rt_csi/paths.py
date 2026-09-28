"""Locate the dataset checkout containing configs and custom scenes."""

from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    candidates = []
    explicit = os.environ.get("RT_CSI_ROOT")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend((Path.cwd(), Path(__file__).resolve().parents[2]))
    for root in candidates:
        if (root / "configs" / "default.yaml").is_file() and (root / "scenes").is_dir():
            return root.resolve()
    raise FileNotFoundError(
        "RT-CSI configs/scenes not found. Run from the repository root or set RT_CSI_ROOT."
    )

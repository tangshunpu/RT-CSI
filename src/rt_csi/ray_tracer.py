"""Batched ray tracing: chunks of RX -> CFR (frequency-domain H)."""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import sionna.rt as srt

from .scenes import SceneConfig


def _add_receivers(scene: srt.Scene, rx_xyz: np.ndarray) -> list[str]:
    """Add a batch of receivers to the scene. Returns the names added."""
    names = []
    for i, (x, y, z) in enumerate(rx_xyz):
        n = f"rx_{i}"
        scene.add(srt.Receiver(name=n, position=[float(x), float(y), float(z)]))
        names.append(n)
    return names


def _remove_receivers(scene: srt.Scene, names: list[str]) -> None:
    for n in names:
        scene.remove(n)


def trace_batch(scene: srt.Scene, rx_xyz: np.ndarray, cfg: SceneConfig,
                solver: srt.PathSolver) -> np.ndarray:
    """Place `rx_xyz` receivers, trace paths, compute CFR.

    Returns H of shape:
      * single-BS scene:  (N, num_tx_ant, num_subcarriers) complex64
      * multi-BS scene:   (N, num_tx, num_tx_ant, num_subcarriers) complex64

    Rows with no valid path are returned as all-zero (caller can filter).

    On Dr.Jit overflow (cfr's broadcast > 2^32 entries) the batch is split
    in half and recursed -- this happens automatically when path density
    is higher than expected.
    """
    names = _add_receivers(scene, rx_xyz)
    try:
        paths = solver(
            scene,
            max_depth=cfg.max_depth,
            los=cfg.los,
            specular_reflection=cfg.specular_reflection,
            diffuse_reflection=cfg.diffuse_reflection,
            refraction=cfg.refraction,
            diffraction=cfg.diffraction,
            synthetic_array=cfg.synthetic_array,
            seed=cfg.seed,
        )

        freqs = np.linspace(-cfg.bandwidth_hz / 2.0,
                            cfg.bandwidth_hz / 2.0,
                            cfg.num_subcarriers,
                            endpoint=False).astype(np.float32)

        try:
            H = paths.cfr(frequencies=freqs,
                          normalize_delays=True,
                          normalize=False,
                          out_type="numpy")
        except RuntimeError as exc:
            # Dr.Jit 2^32-entry overflow -- recurse on halves of the
            # chunk so each call sees smaller `num_paths * batch`.
            if "exceeds the limit" not in str(exc):
                raise
            if rx_xyz.shape[0] <= 1:
                raise RuntimeError(
                    f"single RX still overflowed cfr; reduce K or freq: {exc}"
                ) from exc
            _remove_receivers(scene, names)
            names = []   # parent finally won't double-remove
            mid = rx_xyz.shape[0] // 2
            print(f"  [overflow] retrying chunk {rx_xyz.shape[0]}"
                  f" -> {mid}+{rx_xyz.shape[0]-mid}")
            h1 = trace_batch(scene, rx_xyz[:mid], cfg, solver)
            h2 = trace_batch(scene, rx_xyz[mid:], cfg, solver)
            return np.concatenate([h1, h2], axis=0)
        # Native shape: [num_rx, num_rx_ant=1, num_tx, num_tx_ant, num_time=1, num_freq]
        H = np.asarray(H).squeeze((1, 4))   # -> (N, num_tx, num_tx_ant, num_freq)
        if H.dtype != np.complex64:
            H = H.astype(np.complex64)
        if H.shape[1] == 1:
            # Backward compatible single-BS layout.
            H = H[:, 0]                     # -> (N, num_tx_ant, num_freq)
        return H
    finally:
        _remove_receivers(scene, names)


def iter_traced_batches(scene: srt.Scene, rx_xyz: np.ndarray, cfg: SceneConfig
                        ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (positions_chunk, H_chunk) batches over `rx_xyz`."""
    solver = srt.PathSolver()
    n = rx_xyz.shape[0]
    B = cfg.batch_size
    for s in range(0, n, B):
        chunk = rx_xyz[s:s + B]
        H = trace_batch(scene, chunk, cfg, solver)
        yield chunk, H

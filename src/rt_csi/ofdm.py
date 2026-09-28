"""Frequency-spatial H  ->  angular-delay (2, 32, 32) tensor."""

from __future__ import annotations

import numpy as np


def to_angular_delay(H: np.ndarray, n_delay_keep: int = 32) -> np.ndarray:
    """
    H: complex array of shape (..., N_ant, N_sc) frequency-spatial channel.

    Pipeline for the angular-delay CSI tensor:
      1. DFT along antenna axis (-2)   -> angular domain
      2. IFFT along subcarrier axis (-1) -> delay domain
      3. Keep the first `n_delay_keep` delay taps
      4. Stack [Re, Im] along a new channel axis at position -3

    Returns float32 array of shape (..., 2, N_ant, n_delay_keep).
    """
    H_ad = np.fft.ifft(np.fft.fft(H, axis=-2, norm="ortho"),
                       axis=-1, norm="ortho")
    H_ad = H_ad[..., :n_delay_keep]
    out = np.stack([H_ad.real, H_ad.imag], axis=-3).astype(np.float32)
    return out


def filter_valid(H: np.ndarray, positions: np.ndarray,
                 power_floor: float = 1e-12
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Drop RX entries whose total channel energy is below `power_floor`.

    Returns (H_kept, positions_kept).
    """
    energy = np.mean(np.abs(H) ** 2, axis=tuple(range(1, H.ndim)))
    mask = energy > power_floor
    return H[mask], positions[mask]

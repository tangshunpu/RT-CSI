"""End-to-end scene -> npz dataset orchestration."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import numpy as np
from tqdm import tqdm

from .scenes import SceneConfig, build_scene
from .sampler import sample_street_positions
from .ray_tracer import iter_traced_batches
from .ofdm import to_angular_delay, filter_valid


def _split_indices(n: int, train_ratio: float, val_ratio: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = rng.permutation(n)
    n_tr = int(round(n * train_ratio))
    n_va = int(round(n * val_ratio))
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def _bs_xyz_from_scene(scene) -> np.ndarray:
    """Read configured TX positions back from the scene in deterministic
    name order (`tx` for single-BS or `tx_0, tx_1, ...` for multi-BS)."""
    names = list(scene.transmitters.keys())
    if names == ["tx"]:
        order = ["tx"]
    else:
        order = sorted(names, key=lambda n: int(n.split("_")[1]))
    out = []
    for n in order:
        p = np.array(scene.transmitters[n].position).reshape(-1).astype(float)
        out.append(p)
    return np.stack(out, axis=0)


def _select_serving_bs(H_multi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pick the strongest-signal serving BS per RX.

    H_multi: (N, K, A, F) complex64.  Returns
        H_serving: (N, A, F) complex64 -- channel of the chosen BS
        bs_id:     (N,) int32 -- index of the chosen BS for each sample
    """
    # Per-BS total channel power across antennas + subcarriers.
    power = np.sum(np.abs(H_multi) ** 2, axis=(2, 3))   # (N, K)
    bs_id = np.argmax(power, axis=1).astype(np.int32)   # (N,)
    H_serving = np.take_along_axis(
        H_multi, bs_id[:, None, None, None], axis=1).squeeze(1)
    return H_serving, bs_id


def generate_scene_dataset(cfg: SceneConfig, out_path: str,
                           verbose: bool = True) -> dict:
    """Run full pipeline for one scene and save an npz at `out_path`.

    npz keys:
      x_train, x_val, x_test  : float32 (N, 2, 32, 32) in [0, 1]
      pos_train, pos_val, pos_test : float32 (N, 3) RX positions
      bs_id_train, bs_id_val, bs_id_test : int32 (N,) serving BS index
        (always 0 for single-BS scenes; 0..K-1 for multi-BS).
      bs_positions : float32 (K, 3) absolute BS positions in world coords
      scale : float32 scalar (NaN with per-sample normalisation)
      config : json-encoded SceneConfig + counts
    """
    if verbose:
        print(f"[{cfg.name}] building scene...")
    scene = build_scene(cfg)
    bs_positions = _bs_xyz_from_scene(scene)
    num_bs = bs_positions.shape[0]
    if verbose:
        print(f"[{cfg.name}] {num_bs} BS placed:")
        for i, p in enumerate(bs_positions):
            print(f"  BS{i}: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")

    if verbose:
        radius_msg = (f", radius={cfg.max_radius_from_tx:.0f} m / BS"
                      if cfg.max_radius_from_tx is not None else "")
        print(f"[{cfg.name}] sampling candidate RX positions "
              f"(target={cfg.target_samples}, oversample={cfg.oversample}"
              f"{radius_msg})...")
    n_candidates = int(cfg.target_samples * cfg.oversample)
    candidates = sample_street_positions(
        scene,
        n_target=n_candidates,
        rx_height=cfg.rx_height,
        grid_step_m=cfg.grid_step_m,
        mode="grid",
        rng_seed=cfg.seed,
        tx_xy=bs_positions[:, :2],
        max_radius=cfg.max_radius_from_tx,
        outdoor_z_range=cfg.rx_outdoor_z_range,
    )
    if verbose:
        print(f"[{cfg.name}] outdoor candidates: {candidates.shape[0]}")

    # Ray-trace in batches, accumulating angular-delay tensors and positions.
    all_x: list[np.ndarray] = []
    all_pos: list[np.ndarray] = []
    all_bs: list[np.ndarray] = []
    pbar_total = (candidates.shape[0] + cfg.batch_size - 1) // cfg.batch_size
    kept_count = 0
    t0 = time.time()
    serving_mode = getattr(cfg, "serving_mode", "argmax")
    for chunk_pos, H_chunk in tqdm(
            iter_traced_batches(scene, candidates, cfg),
            total=pbar_total,
            disable=not verbose,
            desc=f"[{cfg.name}] tracing"):
        if num_bs > 1 and serving_mode == "all_bs":
            # H_chunk: (N, K, A, F).  Flatten (RX, BS) -> samples and tag
            # each by bs_id.  Filter out (RX, BS) pairs with no energy.
            N_rx, K, A, F = H_chunk.shape
            H_flat = H_chunk.reshape(N_rx * K, A, F)
            bs_id_flat = np.tile(np.arange(K, dtype=np.int32), N_rx)
            pos_flat = np.repeat(chunk_pos, K, axis=0)
            energy = np.mean(np.abs(H_flat) ** 2, axis=(1, 2))
            keep = energy > 1e-12
            if not keep.any():
                continue
            H_valid = H_flat[keep]
            pos_valid = pos_flat[keep]
            bs_id_valid = bs_id_flat[keep]
        else:
            if num_bs > 1:
                # H_chunk: (N, K, A, F).  Pick the strongest BS per RX.
                H_serving, bs_id = _select_serving_bs(H_chunk)
            else:
                H_serving = H_chunk
                bs_id = np.zeros(H_chunk.shape[0], dtype=np.int32)
            H_valid, pos_valid = filter_valid(H_serving, chunk_pos)
            if H_valid.shape[0] == 0:
                continue
            energy = np.mean(np.abs(H_serving) ** 2,
                             axis=tuple(range(1, H_serving.ndim)))
            keep = energy > 1e-12
            bs_id_valid = bs_id[keep]

        x_ad = to_angular_delay(H_valid, n_delay_keep=32)  # (n, 2, 32, 32)
        all_x.append(x_ad)
        all_pos.append(pos_valid)
        all_bs.append(bs_id_valid)
        kept_count += x_ad.shape[0]
        if kept_count >= cfg.target_samples:
            break

    if not all_x:
        raise RuntimeError(f"[{cfg.name}] no valid samples generated")

    x_all = np.concatenate(all_x, axis=0)[:cfg.target_samples]
    pos_all = np.concatenate(all_pos, axis=0)[:cfg.target_samples]
    bs_all = np.concatenate(all_bs, axis=0)[:cfg.target_samples]
    if verbose:
        print(f"[{cfg.name}] kept {x_all.shape[0]} samples "
              f"in {time.time()-t0:.1f}s")
        if num_bs > 1:
            uniq, cnts = np.unique(bs_all, return_counts=True)
            dist = ", ".join(f"BS{int(u)}:{int(c)}"
                              for u, c in zip(uniq, cnts))
            print(f"[{cfg.name}] serving-BS distribution: {dist}")

    # Split into train/val/test (spatially shuffled).
    rng = np.random.default_rng(cfg.seed + 1)
    tr_idx, va_idx, te_idx = _split_indices(
        x_all.shape[0], cfg.train_ratio, cfg.val_ratio, rng)
    x_train, x_val, x_test = x_all[tr_idx], x_all[va_idx], x_all[te_idx]
    pos_train, pos_val, pos_test = pos_all[tr_idx], pos_all[va_idx], pos_all[te_idx]
    bs_train,  bs_val,  bs_test  = bs_all[tr_idx],  bs_all[va_idx],  bs_all[te_idx]

    # Keep raw angular-delay tensors (before per-sample norm) so any
    # renormalisation scheme can be derived later without re-tracing.
    x_train_raw = x_train.astype(np.float32).copy()
    x_val_raw   = x_val.astype(np.float32).copy()
    x_test_raw  = x_test.astype(np.float32).copy()

    # Per-sample max-abs normalization for CSI-feedback tensor storage:
    # each H scaled so max(|Re|, |Im|) -> 0.5, then +0.5 -> [0, 1].
    def _per_sample_norm(x):
        if len(x) == 0:
            return x.astype(np.float32), np.empty((0,), dtype=np.float32)
        flat = x.reshape(len(x), 2, -1)
        s = np.maximum(np.abs(flat[:, 0]).max(axis=1),
                       np.abs(flat[:, 1]).max(axis=1))
        s = np.maximum(s, 1e-12)
        s4 = s[:, None, None, None]
        return ((x / s4) * 0.5 + 0.5), s.astype(np.float32)

    x_train, s_train = _per_sample_norm(x_train)
    x_val,   s_val   = _per_sample_norm(x_val)
    x_test,  s_test  = _per_sample_norm(x_test)
    scale = float("nan")
    if verbose:
        if len(x_train):
            r = f"x_train range=[{x_train.min():.4f}, {x_train.max():.4f}]"
        elif len(x_test):
            r = f"x_test range=[{x_test.min():.4f}, {x_test.max():.4f}]"
        else:
            r = "no samples in any split"
        print(f"[{cfg.name}] per-sample max-abs norm applied; {r}")

    cfg_dict = asdict(cfg)
    cfg_dict["scene_name"] = cfg.name
    cfg_dict["n_train"] = int(x_train.shape[0])
    cfg_dict["n_val"] = int(x_val.shape[0])
    cfg_dict["n_test"] = int(x_test.shape[0])
    cfg_dict["num_bs_actual"] = int(num_bs)
    cfg_dict["bs_positions"] = bs_positions.tolist()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        # Per-sample max-abs normalised version (drop-in for current loader):
        x_train=x_train.astype(np.float32),
        x_val=x_val.astype(np.float32),
        x_test=x_test.astype(np.float32),
        # Raw angular-delay tensors (before per-sample norm). Sufficient to
        # reconstruct any other normalisation scheme (global max-abs,
        # Frobenius, dataset-wide etc.) without re-tracing.
        x_train_raw=x_train_raw,
        x_val_raw=x_val_raw,
        x_test_raw=x_test_raw,
        # Per-sample max-abs scales used for normalisation. Equivalent to
        # x_*_raw modulo numerical precision: x_raw = (x_norm - 0.5) * 2 * scale.
        scale_train=s_train,
        scale_val=s_val,
        scale_test=s_test,
        pos_train=pos_train.astype(np.float32),
        pos_val=pos_val.astype(np.float32),
        pos_test=pos_test.astype(np.float32),
        bs_id_train=bs_train.astype(np.int32),
        bs_id_val=bs_val.astype(np.int32),
        bs_id_test=bs_test.astype(np.int32),
        bs_positions=bs_positions.astype(np.float32),
        scale=np.float32(scale),
        config=np.array(json.dumps(cfg_dict)),
    )
    if verbose:
        print(f"[{cfg.name}] wrote {out_path}")
    return cfg_dict

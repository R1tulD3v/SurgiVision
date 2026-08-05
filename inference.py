"""Shared inference helpers for SurgiVision.

This module centralizes logic that was previously copy-pasted across several
scripts (preprocess -> apply spleen mask -> tensor -> reconstruct -> error).
Keeping it in one place makes the behavior consistent across the detector, the
evaluation code and the Streamlit demo, and makes it unit-testable.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import config
from spleen_3d_model import Spleen3DAutoencoder


def set_global_seed(seed: int = config.RANDOM_SEED) -> None:
    """Seed Python, NumPy and torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA device when available (and preferred), else CPU."""
    return torch.device("cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu")


def normalize_hu(
    volume: np.ndarray, window: tuple[int, int] = config.HU_WINDOW
) -> np.ndarray:
    """Clip a CT volume to a Hounsfield-unit window and scale to ``[0, 1]``.

    ``lo`` maps to 0.0 and ``hi`` maps to 1.0. This is the single source of
    truth for intensity normalization.
    """
    lo, hi = window
    if hi <= lo:
        raise ValueError(f"HU window must have hi > lo, got {window!r}")
    clipped = np.clip(volume, lo, hi).astype(np.float32)
    return (clipped - lo) / float(hi - lo)


def apply_spleen_mask(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out everything outside the spleen mask (the model sees spleen only)."""
    masked = volume.copy()
    masked[mask <= 0] = 0
    return masked


def to_model_tensor(masked_volume: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a ``[D, H, W]`` array to a ``[1, 1, D, H, W]`` float tensor."""
    return torch.as_tensor(
        masked_volume[np.newaxis, np.newaxis, ...], dtype=torch.float32, device=device
    )


def reconstruction_error_map(
    model: torch.nn.Module,
    masked_volume: np.ndarray,
    device: Optional[torch.device] = None,
) -> tuple[float, np.ndarray]:
    """Run the autoencoder and return ``(scalar_mse, per_voxel_abs_error_map)``.

    The per-voxel error map is a genuine reconstruction-error signal (not random
    noise) and is suitable to render as an anomaly heatmap.
    """
    device = device or get_device()
    model.eval()
    tensor = to_model_tensor(masked_volume, device)
    with torch.no_grad():
        recon = model(tensor)
        error_map = torch.abs(tensor - recon).squeeze().cpu().numpy()
        mse = float(torch.mean((tensor - recon) ** 2).item())
    return mse, error_map


def reconstruction_error(
    model: torch.nn.Module,
    masked_volume: np.ndarray,
    device: Optional[torch.device] = None,
) -> float:
    """Return only the scalar mean-squared reconstruction error."""
    return reconstruction_error_map(model, masked_volume, device)[0]


def decide_anomaly(error: float, threshold: float) -> dict:
    """Shared decision rule: flag anomaly when error exceeds threshold."""
    threshold = float(threshold)
    is_anomaly = error > threshold
    confidence = (error / threshold) if threshold > 0 else 0.0
    return {
        "is_anomaly": bool(is_anomaly),
        "confidence": float(confidence),
        "threshold": threshold,
    }


def load_autoencoder(
    model_path: Optional[Path | str] = None,
    device: Optional[torch.device] = None,
) -> Spleen3DAutoencoder:
    """Securely load the 3D autoencoder checkpoint.

    Uses ``weights_only=True`` so loading a checkpoint cannot execute arbitrary
    code (the old ``torch.load`` default deserializes pickles, which is an RCE
    risk if the file is untrusted).
    """
    device = device or get_device()
    path = Path(model_path) if model_path is not None else config.AUTOENCODER_PATH
    if not path.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {path}")
    model = Spleen3DAutoencoder()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

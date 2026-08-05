"""Central configuration for SurgiVision.

All filesystem paths and tunable constants live here so the rest of the
codebase contains no hardcoded paths or scattered "magic numbers". Any path can
be overridden with an environment variable (handy for Docker, CI, or pointing
at a dataset stored elsewhere).

This module intentionally depends only on the standard library so it is cheap
to import from anywhere (no torch/numpy import side effects).
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root is the directory that contains this file. Paths are resolved
# relative to it, so the project no longer assumes a particular working dir
# (the old code hardcoded "../data/..." which only worked from a src/ folder).
REPO_ROOT = Path(__file__).resolve().parent


def _path_from_env(var: str, default: Path) -> Path:
    """Return an absolute Path from env var ``var`` or ``default``."""
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


# --- Data & model locations (override with environment variables) ---
DATA_ROOT: Path = _path_from_env(
    "SURGIVISION_DATA_ROOT", REPO_ROOT / "data" / "Task09_Spleen"
)
MODELS_DIR: Path = _path_from_env("SURGIVISION_MODELS_DIR", REPO_ROOT / "models")
AUTOENCODER_PATH: Path = _path_from_env(
    "SURGIVISION_AUTOENCODER_PATH", MODELS_DIR / "best_spleen_3d_autoencoder.pth"
)
HYBRID_MODEL_PATH: Path = _path_from_env(
    "SURGIVISION_HYBRID_MODEL_PATH", MODELS_DIR / "best_hybrid_detector.pth"
)
# MONAI pretrained spleen segmentation weights (UNet). Enables mask-free
# inference on raw uploads; if absent, the API falls back to the coarse heuristic.
SEGMENTATION_MODEL_PATH: Path = _path_from_env(
    "SURGIVISION_SEGMENTATION_MODEL_PATH", MODELS_DIR / "spleen_ct_segmentation.pt"
)

# --- Preprocessing constants ---
TARGET_SIZE: tuple[int, int, int] = (64, 64, 64)  # volume size fed to the model
HU_WINDOW: tuple[int, int] = (-200, 300)          # Hounsfield-unit soft-tissue clip window

# --- Anomaly-detection defaults ---
DEFAULT_THRESHOLD: float = 0.015   # reconstruction-error decision threshold
THRESHOLD_SIGMA: float = 3.0       # mean + k*std when calibrating on normal volumes

# Heuristic threshold multipliers used when the spleen segmentation mask is not
# available (mixed-tissue volume or a 2D image). These are crude fallbacks, not
# clinically validated values.
MIXED_TISSUE_THRESHOLD_MULTIPLIER: float = 5.0
IMAGE_2D_THRESHOLD_MULTIPLIER: float = 10.0

# --- Database ---
# Defaults to a local SQLite file; set SURGIVISION_DATABASE_URL to a Postgres
# URL in production, e.g. postgresql+psycopg://user:pass@host:5432/surgivision
DATABASE_URL: str = os.environ.get(
    "SURGIVISION_DATABASE_URL",
    f"sqlite:///{(REPO_ROOT / 'surgivision.db').as_posix()}",
)

# --- Reproducibility ---
RANDOM_SEED: int = 42

# --- Upload limits (used by the Streamlit demo) ---
MAX_UPLOAD_MB: int = 200
ALLOWED_UPLOAD_EXTENSIONS: tuple[str, ...] = (
    ".nii",
    ".nii.gz",
    ".png",
    ".jpg",
    ".jpeg",
)

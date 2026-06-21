"""Shared pytest fixtures.

These intentionally avoid the dataset and any nibabel-dependent modules so the
suite runs anywhere (CI included) with only torch/numpy installed.
"""
import numpy as np
import pytest
import torch

import inference
from spleen_3d_model import Spleen3DAutoencoder


@pytest.fixture(scope="session")
def device() -> torch.device:
    # Tests run on CPU for determinism and portability.
    return torch.device("cpu")


@pytest.fixture(scope="session")
def model(device):
    """A freshly-initialized (untrained) autoencoder in eval mode."""
    inference.set_global_seed(0)
    net = Spleen3DAutoencoder().to(device)
    net.eval()
    return net


@pytest.fixture
def synthetic_volume() -> np.ndarray:
    """A deterministic 64^3 volume in [0, 1] standing in for a preprocessed CT."""
    rng = np.random.default_rng(0)
    return rng.random((64, 64, 64)).astype(np.float32)


@pytest.fixture
def synthetic_mask() -> np.ndarray:
    """A small cubic 'spleen' mask inside the volume."""
    mask = np.zeros((64, 64, 64), dtype=np.float32)
    mask[20:40, 20:40, 20:40] = 1.0
    return mask

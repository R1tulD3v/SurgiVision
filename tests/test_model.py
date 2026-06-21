"""Tests for the 3D autoencoder architecture."""
import torch

from spleen_3d_model import Spleen3DAutoencoder, count_parameters


def test_forward_preserves_shape(model, device):
    x = torch.randn(1, 1, 64, 64, 64, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 64, 64, 64)


def test_output_in_unit_range(model, device):
    # Decoder ends in a Sigmoid, so reconstructions must lie in [0, 1].
    x = torch.randn(1, 1, 64, 64, 64, device=device)
    with torch.no_grad():
        y = model(x)
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0


def test_encode_returns_latent_vector(model, device):
    x = torch.randn(1, 1, 64, 64, 64, device=device)
    with torch.no_grad():
        feats = model.encode(x)
    assert feats.shape == (1, model.latent_dim)


def test_parameter_counts():
    total, trainable = count_parameters(Spleen3DAutoencoder())
    assert total > 0
    assert trainable == total  # nothing frozen in the base model

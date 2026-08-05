"""Tests for the FastAPI inference service (api.py), via Starlette's TestClient.

Uses the shared ``api_client`` fixture (temp SQLite DB) from conftest.
"""
import os
import tempfile

import numpy as np
import pytest

import api
import config
from spleen_3d_model import Spleen3DAutoencoder


def _inject_untrained_model():
    """Override the model dependency with a fresh (untrained) autoencoder."""
    api.app.dependency_overrides[api.get_model] = lambda: Spleen3DAutoencoder().eval()


def _nifti_bytes(nib, volume):
    img = nib.Nifti1Image(volume, affine=np.eye(4))
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        path = tmp.name
    try:
        nib.save(img, path)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_healthz(api_client):
    r = api_client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info(api_client):
    r = api_client.get("/api/v1/model")
    assert r.status_code == 200
    body = r.json()
    assert body["target_size"] == list(config.TARGET_SIZE)
    assert body["default_threshold"] == config.DEFAULT_THRESHOLD
    assert isinstance(body["model_available"], bool)


def test_predict_returns_503_without_model(api_client):
    # No checkpoint in the test environment -> the dependency yields 503.
    files = {"file": ("scan.nii.gz", b"irrelevant", "application/gzip")}
    r = api_client.post("/api/v1/predict", files=files)
    assert r.status_code == 503


def test_predict_rejects_unsupported_extension(api_client):
    _inject_untrained_model()
    files = {"file": ("scan.txt", b"hello", "text/plain")}
    r = api_client.post("/api/v1/predict", files=files)
    assert r.status_code == 415


def test_predict_happy_path(api_client):
    nib = pytest.importorskip("nibabel")  # skipped in minimal CI without nibabel
    _inject_untrained_model()
    volume = (np.random.default_rng(0).random((40, 40, 30)).astype(np.float32) * 400 - 100)
    payload = _nifti_bytes(nib, volume)

    files = {"file": ("scan.nii", payload, "application/octet-stream")}
    r = api_client.post("/api/v1/predict", data={"threshold": "0.02"}, files=files)

    assert r.status_code == 200, r.text
    body = r.json()
    assert {"is_anomaly", "confidence", "reconstruction_error", "threshold", "pipeline"} <= set(body)
    assert body["pipeline"] == "no_mask_raw_volume"
    assert body["threshold"] == pytest.approx(0.02 * config.MIXED_TISSUE_THRESHOLD_MULTIPLIER)
    assert isinstance(body["is_anomaly"], bool)
    assert body["reconstruction_error"] >= 0.0

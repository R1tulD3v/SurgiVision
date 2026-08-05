"""Persistence integration tests — predict writes history, endpoints read it.

Runs entirely on the throwaway SQLite DB from the ``api_client`` fixture; no
Postgres required.
"""
import os
import tempfile

import numpy as np
import pytest

import api
from spleen_3d_model import Spleen3DAutoencoder


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


def test_predict_persists_and_history_lists_it(api_client):
    nib = pytest.importorskip("nibabel")
    api.app.dependency_overrides[api.get_model] = lambda: Spleen3DAutoencoder().eval()

    # History starts empty.
    assert api_client.get("/api/v1/analyses").json() == []

    volume = (np.random.default_rng(0).random((40, 40, 30)).astype(np.float32) * 400 - 100)
    files = {"file": ("scan.nii", _nifti_bytes(nib, volume), "application/octet-stream")}
    r = api_client.post("/api/v1/predict", files=files)
    assert r.status_code == 200, r.text

    hist = api_client.get("/api/v1/analyses")
    assert hist.status_code == 200
    items = hist.json()
    assert len(items) == 1
    rec = items[0]
    assert rec["filename"] == "scan.nii"
    assert rec["pipeline"] == "no_mask_raw_volume"
    assert isinstance(rec["is_anomaly"], bool)
    assert "created_at" in rec

    one = api_client.get(f"/api/v1/analyses/{rec['id']}")
    assert one.status_code == 200
    assert one.json()["id"] == rec["id"]


def test_get_missing_analysis_returns_404(api_client):
    assert api_client.get("/api/v1/analyses/999999").status_code == 404

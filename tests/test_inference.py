"""Tests for the shared inference helpers."""
import numpy as np
import pytest

import inference


def test_normalize_hu_range():
    v = np.linspace(-1000, 1000, 64 * 64 * 64).reshape(64, 64, 64)
    n = inference.normalize_hu(v)
    assert n.min() >= 0.0
    assert n.max() <= 1.0


def test_normalize_hu_endpoints():
    lo, hi = -200, 300
    v = np.array([[[lo, hi]]], dtype=np.float32)
    n = inference.normalize_hu(v, (lo, hi))
    assert np.isclose(n[0, 0, 0], 0.0)
    assert np.isclose(n[0, 0, 1], 1.0)


def test_normalize_hu_rejects_bad_window():
    with pytest.raises(ValueError):
        inference.normalize_hu(np.zeros((2, 2, 2)), (10, 10))


def test_apply_spleen_mask_zeroes_outside(synthetic_volume, synthetic_mask):
    masked = inference.apply_spleen_mask(synthetic_volume, synthetic_mask)
    assert np.all(masked[synthetic_mask <= 0] == 0)
    inside = synthetic_mask > 0
    assert np.allclose(masked[inside], synthetic_volume[inside])
    # original volume must not be mutated
    assert not np.shares_memory(masked, synthetic_volume)


def test_reconstruction_error_map_shape_and_sign(model, synthetic_volume, torch_device):
    mse, emap = inference.reconstruction_error_map(model, synthetic_volume, torch_device)
    assert emap.shape == (64, 64, 64)
    assert mse >= 0.0
    assert float(emap.min()) >= 0.0  # abs error is non-negative (not random noise)


def test_scalar_error_matches_map(model, synthetic_volume, torch_device):
    mse_only = inference.reconstruction_error(model, synthetic_volume, torch_device)
    mse_pair, _ = inference.reconstruction_error_map(model, synthetic_volume, torch_device)
    assert np.isclose(mse_only, mse_pair)


@pytest.mark.parametrize(
    "error,threshold,expected",
    [
        (0.02, 0.015, True),
        (0.01, 0.015, False),
        (0.015, 0.015, False),  # strictly greater than threshold
    ],
)
def test_decide_anomaly(error, threshold, expected):
    d = inference.decide_anomaly(error, threshold)
    assert d["is_anomaly"] is expected
    assert np.isclose(d["confidence"], error / threshold)
    assert d["threshold"] == threshold


def test_decide_anomaly_zero_threshold_is_safe():
    d = inference.decide_anomaly(0.01, 0.0)
    assert d["confidence"] == 0.0


def test_preprocess_raw_volume_shape_and_range():
    vol = (np.random.default_rng(1).random((90, 90, 50)).astype(np.float32) * 500 - 200)
    out = inference.preprocess_raw_volume(vol)
    assert out.shape == (64, 64, 64)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_crop_and_resize_to_spleen():
    vol = (np.random.default_rng(2).random((90, 90, 60)).astype(np.float32) * 500 - 200)
    mask = np.zeros((90, 90, 60), dtype=np.uint8)
    mask[30:60, 30:60, 20:40] = 1  # a cuboid "spleen"
    out_vol, out_mask = inference.crop_and_resize_to_spleen(vol, mask)
    assert out_vol.shape == (64, 64, 64)
    assert out_mask.shape == (64, 64, 64)
    assert out_vol.min() >= 0.0 and out_vol.max() <= 1.0
    assert (out_mask > 0).sum() > 0


def test_crop_and_resize_empty_mask_returns_none():
    vol = np.zeros((32, 32, 32), dtype=np.float32)
    mask = np.zeros((32, 32, 32), dtype=np.uint8)
    v, m = inference.crop_and_resize_to_spleen(vol, mask)
    assert v is None and m is None

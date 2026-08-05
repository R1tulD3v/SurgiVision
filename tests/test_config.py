"""Tests for the central config module."""
import importlib
from pathlib import Path

import config


def test_paths_are_path_objects():
    assert isinstance(config.DATA_ROOT, Path)
    assert isinstance(config.MODELS_DIR, Path)
    assert isinstance(config.AUTOENCODER_PATH, Path)
    assert isinstance(config.HYBRID_MODEL_PATH, Path)


def test_default_checkpoints_live_under_models_dir():
    assert config.AUTOENCODER_PATH.parent == config.MODELS_DIR
    assert config.HYBRID_MODEL_PATH.parent == config.MODELS_DIR


def test_constants_are_sane():
    assert 0 < config.DEFAULT_THRESHOLD < 1
    lo, hi = config.HU_WINDOW
    assert hi > lo
    assert config.TARGET_SIZE == (64, 64, 64)
    assert config.MAX_UPLOAD_MB > 0
    assert ".nii.gz" in config.ALLOWED_UPLOAD_EXTENSIONS


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SURGIVISION_DATA_ROOT", str(tmp_path))
    try:
        importlib.reload(config)
        assert config.DATA_ROOT == tmp_path.resolve()
    finally:
        # Restore defaults so other tests see the unmodified config.
        monkeypatch.delenv("SURGIVISION_DATA_ROOT", raising=False)
        importlib.reload(config)

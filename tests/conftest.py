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
def torch_device() -> torch.device:
    # Tests run on CPU for determinism and portability.
    # (Named torch_device, not device, to avoid clashing with
    # pytest-playwright's own `device` fixture used by the e2e tests.)
    return torch.device("cpu")


@pytest.fixture(scope="session")
def model(torch_device):
    """A freshly-initialized (untrained) autoencoder in eval mode."""
    inference.set_global_seed(0)
    net = Spleen3DAutoencoder().to(torch_device)
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


@pytest.fixture
def api_client(tmp_path):
    """FastAPI TestClient backed by a throwaway SQLite database.

    Overrides the ``get_session`` dependency so API/persistence tests need no
    Postgres. Imports are lazy so the pure-unit tests don't require FastAPI.
    """
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import api
    import db
    import db_models  # noqa: F401  (registers models on Base.metadata)

    url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    eng = create_engine(url, connect_args={"check_same_thread": False}, future=True)
    db.Base.metadata.create_all(eng)
    TestSession = sessionmaker(bind=eng, expire_on_commit=False)

    def override_session():
        session = TestSession()
        try:
            yield session
        finally:
            session.close()

    api.app.dependency_overrides[db.get_session] = override_session
    try:
        yield TestClient(api.app)
    finally:
        api.app.dependency_overrides.clear()

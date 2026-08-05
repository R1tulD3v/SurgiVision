"""FastAPI inference service for SurgiVision.

Exposes the spleen-CT anomaly detector over HTTP so any client (or the demo UI)
can call it independently of Streamlit. This is the service boundary that the
later Phase-2 work (persistence, auth) builds on.

Run locally:
    pip install -r requirements.txt
    uvicorn api:app --reload --port 8000
    # interactive docs at http://127.0.0.1:8000/docs

Note: `/predict` uses the mask-free raw-volume heuristic (no organ
localization). Automatic spleen segmentation (MONAI) is the planned upgrade;
results without a mask are coarser than the mask-based pipeline.
"""
from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import config
import inference

app = FastAPI(
    title="SurgiVision API",
    version="1.0.0",
    description="Unsupervised spleen-CT anomaly detection (3D autoencoder).",
)


class HealthResponse(BaseModel):
    status: str


class ModelInfo(BaseModel):
    model_available: bool
    model_path: str
    target_size: list[int]
    default_threshold: float


class PredictResponse(BaseModel):
    is_anomaly: bool
    confidence: float
    reconstruction_error: float
    threshold: float
    pipeline: str


@lru_cache(maxsize=1)
def _load_model_cached():
    # Cached so the checkpoint is loaded once per process, not per request.
    return inference.load_autoencoder()


def get_model():
    """Dependency: return the autoencoder, or 503 if no checkpoint is present."""
    try:
        return _load_model_cached()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model not available: {exc}. Train a model or set "
                "SURGIVISION_AUTOENCODER_PATH."
            ),
        )


@app.get("/healthz", response_model=HealthResponse, tags=["meta"])
def healthz():
    """Liveness probe — does not require a model."""
    return HealthResponse(status="ok")


@app.get("/api/v1/model", response_model=ModelInfo, tags=["meta"])
def model_info():
    """Report whether a model is available and the active configuration."""
    return ModelInfo(
        model_available=config.AUTOENCODER_PATH.exists(),
        model_path=str(config.AUTOENCODER_PATH),
        target_size=list(config.TARGET_SIZE),
        default_threshold=config.DEFAULT_THRESHOLD,
    )


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(default=None),
    model=Depends(get_model),
):
    """Run anomaly detection on an uploaded NIfTI CT volume."""
    name = (file.filename or "").lower()
    if not name.endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=415, detail="Only NIfTI (.nii/.nii.gz) is supported.")

    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File exceeds the {config.MAX_UPLOAD_MB} MB limit."
        )

    import nibabel as nib  # imported lazily so the module loads without nibabel

    suffix = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            volume_data = nib.load(tmp_path).get_fdata()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read NIfTI volume: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    volume = inference.preprocess_raw_volume(volume_data)
    error, _ = inference.reconstruction_error_map(model, volume)
    base = threshold if threshold is not None else config.DEFAULT_THRESHOLD
    effective = base * config.MIXED_TISSUE_THRESHOLD_MULTIPLIER
    decision = inference.decide_anomaly(error, effective)
    return PredictResponse(
        is_anomaly=decision["is_anomaly"],
        confidence=decision["confidence"],
        reconstruction_error=error,
        threshold=effective,
        pipeline="no_mask_raw_volume",
    )

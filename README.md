# SurgiVision

[![CI](https://github.com/R1tulD3v/SurgiVision/actions/workflows/ci.yml/badge.svg)](https://github.com/R1tulD3v/SurgiVision/actions/workflows/ci.yml)

**Unsupervised anomaly detection in spleen CT scans using a 3D convolutional autoencoder.**

SurgiVision trains a 3D autoencoder on *normal* spleen tissue and flags scans whose
reconstruction error is high — the idea being that the model reconstructs healthy
anatomy well and pathological tissue (cysts, infarcts, lacerations, masses) poorly.
It ships with a preprocessing pipeline, training loop, an evaluation/metrics suite,
and an interactive Streamlit demo that produces a downloadable PDF report.

> ⚠️ **Not a medical device.** This is a research/education project and a portfolio
> demo. Its outputs are **not** validated for clinical use and must never inform real
> diagnosis or treatment. See **Limitations** below — please read them honestly.

---

## Limitations (read this first)

This project is an honest work-in-progress. Known limitations:

- **Inference currently depends on a spleen segmentation mask.** The main detection
  path zeroes out non-spleen voxels using the dataset's ground-truth label. On a
  truly unseen scan you would not have that mask, so an automated segmentation step
  (e.g. MONAI) is required to make this genuinely end-to-end. *(Planned — see the
  roadmap.)*
- **Evaluation uses synthetic anomalies.** Pathologies are simulated by intensity
  manipulation, and the decision threshold is calibrated on normal training volumes.
  Reported metrics are therefore optimistic and should be treated as a sanity check,
  not real-world performance.
- **The 2D-image upload path is a non-clinical demo.** The model is trained on 3D
  volumes; a stacked 2D image is out-of-distribution and not meaningful.
- **A trained model and the dataset are not included** in this repository (they are
  large binaries). You must download the data and train, or supply your own
  checkpoint — see below.

The full assessment and improvement plan lives in
[`Project_upgrade_strategy.md`](Project_upgrade_strategy.md).

---

## Pipeline

```
NIfTI CT volume
   │  spleen_preprocessing.py   (crop to spleen bbox, HU window → [0,1], resize 64³)
   ▼
3D Autoencoder  (spleen_3d_model.py)
   │  training_pipeline.py      (MSE reconstruction loss, train on normal tissue)
   ▼
Reconstruction error  ──►  threshold  ──►  Normal / Anomaly
   │  inference.py             (shared error-map + decision helpers)
   ▼
Streamlit demo + PDF report   (streamlit_universal_demo.py)
```

## Project structure

| File | Purpose |
|------|---------|
| `config.py` | Central paths & constants (override via env vars). |
| `inference.py` | Shared helpers: HU normalize, error map, decision, secure model load. |
| `spleen_3d_model.py` | The 3D convolutional autoencoder. |
| `spleen_preprocessing.py` | NIfTI loading, cropping, normalization, resizing. |
| `training_pipeline.py` | Dataset + training loop for the autoencoder. |
| `spleen_anomaly_detector_fixed.py` | Reconstruction-error anomaly detector. |
| `enhanced_anomaly_creator.py` | Synthetic pathology generator + adaptive thresholds. |
| `hybrid_anomaly_detector_fixed.py` | Hybrid (AE + classifier + spatial + attention) detector. |
| `model_analysis.py` | Metrics & plots (ROC/AUC, PR, confusion matrix, …). |
| `internet_ct_tester.py` | Batch sanity tester. |
| `streamlit_universal_demo.py` | Interactive demo UI + PDF report. |
| `tests/` | Pytest suite (runs without the dataset). |

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

pip install -r requirements.txt
# For a CPU-only PyTorch build:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Data

Download **Task09_Spleen** from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
and place it so the layout is:

```
data/Task09_Spleen/
  imagesTr/  *.nii.gz
  labelsTr/  *.nii.gz
  imagesTs/  *.nii.gz
```

By default the code looks for `./data/Task09_Spleen` and `./models/`. Override with
environment variables (see [Configuration](#configuration)).

## Train

```bash
python training_pipeline.py        # writes models/best_spleen_3d_autoencoder.pth
```

## Run the demo

```bash
streamlit run streamlit_universal_demo.py
```

## Run the API

A FastAPI service ([`api.py`](api.py)) exposes the detector over HTTP:

```bash
uvicorn api:app --reload --port 8000
# interactive docs (Swagger UI): http://127.0.0.1:8000/docs
```

| Method & path | Purpose |
|---------------|---------|
| `GET /healthz` | liveness probe (no model needed) |
| `GET /api/v1/model` | model availability + active config |
| `POST /api/v1/predict` | upload a NIfTI CT volume → JSON anomaly result |
| `GET /api/v1/analyses` | recent analyses (history / audit trail) |
| `GET /api/v1/analyses/{id}` | one analysis by id |

Example:
```bash
curl -F "file=@scan.nii.gz" -F "threshold=0.015" http://127.0.0.1:8000/api/v1/predict
```
Returns `503` until a trained model is present. `/predict` currently uses the
mask-free raw-volume heuristic (see **Limitations**); automatic spleen
segmentation is the planned upgrade.

### Database & persistence

Every prediction is persisted (history + audit trail). The store is configured
by `SURGIVISION_DATABASE_URL` — a local SQLite file by default, PostgreSQL in
production. Schema is managed with **Alembic**:

```bash
alembic upgrade head          # create/upgrade the schema
```

Run the API with Postgres via Docker Compose:
```bash
docker compose up --build     # starts Postgres + the API on :8000
```
(The trained model is mounted from `./models`, not baked into the image.)

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite is self-contained (synthetic fixtures) and does **not** require the
dataset or a trained model.

## Docker

```bash
docker build -t surgivision .
docker run -p 8501:8501 \
  -e SURGIVISION_MODELS_DIR=/models -v "$PWD/models:/models:ro" \
  -e SURGIVISION_DATA_ROOT=/data/Task09_Spleen -v "$PWD/data:/data:ro" \
  surgivision
```

Then open http://localhost:8501.

> Note: the Docker image has not been built/published as part of this commit; the
> `Dockerfile` is provided as the build recipe.

## Configuration

All paths and constants live in `config.py` and can be overridden via environment
variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `SURGIVISION_DATA_ROOT` | `./data/Task09_Spleen` | Dataset root (imagesTr/, labelsTr/). |
| `SURGIVISION_MODELS_DIR` | `./models` | Directory for checkpoints. |
| `SURGIVISION_AUTOENCODER_PATH` | `<MODELS_DIR>/best_spleen_3d_autoencoder.pth` | Autoencoder checkpoint. |
| `SURGIVISION_HYBRID_MODEL_PATH` | `<MODELS_DIR>/best_hybrid_detector.pth` | Hybrid model checkpoint. |

Copy `.env.example` to `.env` as a starting point.

## Security note

Model checkpoints are loaded with `torch.load(..., weights_only=True)` so that
loading a `.pth` file cannot execute arbitrary code. Only load checkpoints from
sources you trust.

## Roadmap

See [`Project_upgrade_strategy.md`](Project_upgrade_strategy.md) for the full,
prioritized plan (API backend, persistence, MONAI segmentation, real explainability,
LLM-drafted reports, CI/CD, observability, and more).

## Acknowledgements

- Dataset: **Medical Segmentation Decathlon — Task09 Spleen** (http://medicaldecathlon.com/).

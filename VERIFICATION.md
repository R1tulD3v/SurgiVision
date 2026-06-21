# Verification Workflow

Run this after **every phase**, in order. Only continue to the next phase once
all layers pass (fix issues first).

```
unit tests  →  lint + compile  →  E2E browser smoke  →  manual check  →  proceed
```

## Layer 1 — Automated unit tests (logic)

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
Covers: config resolution, the autoencoder forward pass / latent shape, the
shared inference helpers (HU normalize, real reconstruction-error map, anomaly
decision), and the reproducible data splits. **No dataset or model required.**

## Layer 2 — Lint & compile (quality / syntax)

```bash
ruff check .
python -m py_compile *.py tests/*.py
```

## Layer 3 — E2E browser smoke test (Playwright)

One-time setup:
```bash
pip install pytest-playwright
python -m playwright install chromium
```
Run (two terminals):
```bash
# terminal 1 — start the app
python -m streamlit run streamlit_universal_demo.py --server.headless true --server.port 8765

# terminal 2 — drive a real browser against it
APP_URL=http://127.0.0.1:8765 pytest tests/e2e
```
The E2E test is **environment-adaptive**: with no model it asserts the app boots
and reports "Model not found"; with a model present it asserts the controls
render. It also fails on any uncaught JS error. (Excluded from the default
`pytest` run because it needs a live server + browser.)

## Layer 4 — Manual verification (MUST, with a real model)

Place a trained checkpoint at `models/best_spleen_3d_autoencoder.pth` and the
dataset at `data/Task09_Spleen/` (or set `SURGIVISION_MODELS_DIR` /
`SURGIVISION_DATA_ROOT`), launch the app, then click through:

- [ ] App loads with the header and **no** "Model not found" error.
- [ ] **Training Volume Test** → pick a volume → *Analyze* → metrics dashboard
      appears; the anomaly heatmap is a **real reconstruction-error map** (not
      random noise); the 3D plot renders; the PDF report downloads.
- [ ] **Upload Medical File** → upload a `.nii.gz` → result + heatmap render.
      Then upload an unsupported/oversized file → a **validation error** is shown
      and processing stops.
- [ ] **Synthetic Pathology Demo** → each pathology type produces a result.
- [ ] The **2D image** path shows the out-of-distribution warning.

## Phase 1 — result (recorded)

| Layer | Result |
|-------|--------|
| Unit tests | ✅ 28 passed |
| Lint (ruff) | ✅ clean |
| Compile | ✅ all modules |
| E2E (Playwright) | ✅ 4 passed against live app (title, header, model-state, no JS errors) |
| App boots | ✅ renders header + correct config-resolved model path |
| Manual (with model) | ⏳ to be done locally by the maintainer |

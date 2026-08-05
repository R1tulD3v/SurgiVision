# SurgiVision — Upgrade & Deploy Checklist

Goal: finish the upgrade in phases and end with a **live, shareable demo** for interviews.
Legend: ✅ done · 🟡 partial · ⬜ not started

---

## 0. State of the world (what exists right now)

- **GitHub repo** `R1tulD3v/SurgiVision`: has the refactored code on two open PRs (not yet merged):
  - **PR #3** — Phase 1 (config, inference, tests, CI, Docker, integrity + security fixes).
  - **PR #4** — Phase 2 batch 1 (FastAPI service). Stacked on #3.
- **Local-only assets** (correctly NOT in GitHub — too large):
  - Trained model `best_spleen_3d_autoencoder.pth` (**387 MB**) + `best_hybrid_detector.pth` (164 MB), under `unetpp_project/`.
  - `Task09_Spleen` dataset.
  - `unetpp_project/surgivision_hf/` — a **Hugging Face Spaces deploy scaffold** (model in Git LFS, `DEPLOY.md`), **never pushed** (no git remote) → **no live URL yet**. Its `app.py` is the **old, pre-refactor** code (still has the random-noise heatmap + unsafe model load).

> ⚠️ `unetpp_project/` is a nested git repo with 387 MB models — it must be **git-ignored** in the main repo, never committed.

---

## 1. Already done ✅

- ✅ Config module, no hardcoded paths, `.env` support
- ✅ Shared `inference.py` + secure `torch.load(weights_only=True)`
- ✅ Integrity fixes: real reconstruction-error heatmap, honest leakage-aware evaluation, upload validation, 2D caveat
- ✅ Bug fix: `encode()` latent dimension
- ✅ 34 automated tests, `ruff`+`mypy` config, GitHub Actions CI
- ✅ Dockerfile, honest README, verification runbook, Playwright E2E + MCP config
- ✅ FastAPI service (`/healthz`, `/api/v1/model`, `/api/v1/predict`)

## 2. Remaining ⬜ — ordered toward a LIVE demo

### Phase A — Verify with the REAL model (now possible; was blocked on the model)
- ⬜ Point config at the real model/data via env vars and load the 387 MB checkpoint with our code
- ⬜ Run the **full test suite** + **Streamlit app** + **API** against the real model
- ⬜ Manual click-through: Training Volume Test → real heatmap; Upload; Synthetic Pathology; PDF export
- ⬜ Fix anything the real model surfaces

### Phase B — Consolidate GitHub
- ⬜ Merge **PR #3**, then **PR #4** into `main`
- ⬜ Add `unetpp_project/`, `Project_Mastery_for_Interview.md` to `.gitignore` (protect the repo)

### Phase C — Prepare the live deploy (Hugging Face Spaces, from the REFACTORED code)
- ⬜ Build a deploy folder from our clean code (not the old `app.py`)
- ⬜ HF `README.md` YAML header (sdk: streamlit), `requirements.txt` with `opencv-python-headless`
- ⬜ Model + a few sample volumes tracked in **Git LFS**
- ⬜ Verify the deploy build locally (headless)

### Phase D — Go LIVE  (needs YOUR Hugging Face auth)
- ⬜ **You:** create a HF **write token**, `hf auth login`
- ⬜ Create the Space + push (`git push space main`) → permanent live URL
- ⬜ Smoke-test the live Space

### Phase E — Backend depth (optional, after the live link; strong talking points)
- ⬜ PostgreSQL persistence + analysis history/audit
- ⬜ JWT auth + role-based access
- ⬜ MONAI auto-segmentation (removes the mask dependency), Grad-CAM, LLM report
- ⬜ Observability (structured logs, metrics)

---

## 3. Recommended path

**Fastest impressive result = deploy the *refactored* Streamlit app to Hugging Face Spaces** (A → C → D).
- HF Spaces is already scaffolded, free, handles the 387 MB model via LFS, and gives a permanent URL.
- We deploy the *clean* code (real heatmap, honest eval), not the old buggy `app.py`.
- The FastAPI service, Postgres, and auth are excellent **talking points** and can follow the live link.

## 4. What only you can do
- Hugging Face **write token** + `hf auth login` (I can't enter credentials).
- Merging PRs on GitHub (or I can guide/point you).

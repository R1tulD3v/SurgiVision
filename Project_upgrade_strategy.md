# SurgiVision — Project Upgrade Strategy

> **Prepared as:** Staff ML Engineer / AI Architect review, wearing the hats of Senior SDE, Product Engineer, Tech Lead, System Architect, Security Engineer, QA Engineer, DevOps/SRE, and Hiring Manager.
> **Date:** 2026-06-21
> **Scope:** Brutally honest assessment of the current repository and a practical, market-relevant upgrade plan for SDE, backend, full-stack, AI, ML, and data-analyst interviews.

---

## One-line "why this project becomes stronger"

**SurgiVision becomes interview-grade the moment it stops being a single-folder Streamlit ML demo and becomes a service-oriented, tested, observable medical-AI *product* — a real inference API with persistence, honest evaluation, in-browser explainability, and an LLM-drafted report layer — because that exact shift is the gap between "I trained a model" and "I shipped an ML system," which is what these interviews actually probe.**

---

## How to read this document

Every claim is tagged:

- **FACT** — directly verified by reading the code/repo in its current state.
- **INFERENCE** — a reasoned conclusion from the code; not 100% certain but well-grounded.
- **RECOMMENDATION** — a proposed change, not a statement about the current state.

Scoring uses 7 dimensions (each 1–5). Six are *benefits* (higher = better): Interview Value, Market Relevance, Demo Impact, Technical Depth, Resume Value, Long-term Usefulness. One is **Effort** (higher = harder). The composite is:

> **Net Priority Score = (sum of 6 benefit dimensions) − Effort** — range 1 to 29.

---

## Reviewer's verdict (the honest snapshot)

**Is this impressive or just decorated?** Right now it is **a genuinely above-average ML prototype wrapped in a deliberately impressive-looking UI, but it is decorated in the places that matter most to an experienced reviewer.**

- The **model work is real and non-trivial** — a 3D convolutional autoencoder for volumetric CT, unsupervised anomaly detection via reconstruction error, a hybrid ensemble head, and a proper metrics suite (ROC/AUC/PR/confusion matrix). This is well above the typical "MNIST classifier" or "CRUD app" student project. **FACT.**
- But the **engineering around the model is near-zero**: no API, no database, no tests, no CI, no containerization, no auth, no logging, no config, no dependency file. **FACT.**
- And there are **three credibility landmines** a strong interviewer will step on within five minutes: (1) the anomaly heatmap shown in the main demo path is random noise, (2) inference secretly depends on the ground-truth segmentation mask, and (3) evaluation is done on synthetic anomalies with a threshold derived from the training set. **FACT / INFERENCE.**

The good news: the hard part (a working volumetric DL model) is already done. The upgrades below are mostly *engineering and honesty* work — high-leverage, very teachable, and exactly the stuff interviews reward.

---

## Compact upgrade table (TL;DR version)

| # | Upgrade | Category | Net Score | Effort | Verdict |
|---|---------|----------|:---------:|:------:|---------|
| 1 | LLM-drafted radiology report (Anthropic API) | AI | 26 | Med | **Do now** |
| 2 | FastAPI inference service (decouple model from UI) | Backend | 25 | Low-Med | **Do now** |
| 3 | PostgreSQL persistence + analysis history/audit | Backend/DB | 24 | Med | **Do now** |
| 4 | MONAI spleen segmentation (remove mask dependency) | ML | 24 | Med-High | **Do now** (correctness) |
| 5 | Grad-CAM real explainability (kill the fake heatmap) | AI/ML | 24 | Med | **Do now** (integrity) |
| 6 | RAG over radiology guidelines for grounded reports | AI | 23 | Med-High | Do next |
| 7 | Security hardening (weights_only, validation, rate-limit, secrets, audit) | Security | 22 | Low-Med | **Do now** (mandatory) |
| 8 | Dockerize + docker-compose (app+db+cache) | DevOps | 22 | Low | **Do now** |
| 9 | Honest evaluation methodology (held-out split, no leakage) | ML | 22 | Low-Med | **Do now** (mandatory) |
| 10 | React/Next.js + NiiVue in-browser medical viewer | Frontend | 22 | High | Do next |
| 11 | Cloud deployment (HF Spaces / Render → AWS) | Cloud | 21 | Med | Do next |
| 12 | CI/CD with GitHub Actions (lint/type/test/build) | DevOps | 21 | Low | **Do now** |
| 13 | Analytics dashboard over results DB | Data | 21 | Low-Med | Do next |
| 14 | pytest test suite + coverage | Testing | 21 | Low | **Do now** (mandatory) |
| 15 | Drift/data monitoring (Evidently) | Data/MLOps | 20 | Med | Do later |
| 16 | JWT auth + role-based access control | Security | 20 | Med | Do next |
| 17 | Observability (structured logs, Prometheus/Grafana, Sentry) | Observability | 20 | Med | Do later |
| 18 | Experiment tracking + model registry (MLflow/W&B) + DVC | MLOps | 20 | Med | Do later |
| 19 | Code-quality overhaul (config, types, ruff, de-dup, requirements) | Code quality | 17 | Low | **Do now** (foundation) |
| 20 | Perf: ONNX/TorchScript export + batch inference | Performance | 16 | Med | Do later |

> **Note on ordering:** the requested default priority is Frontend → Backend → AI → Data → ML → Cloud/DevOps → Security → Testing → … I am **partially overriding** it (as permitted) because this specific project "strongly argues otherwise": its frontend already exists (Streamlit), while its weakest and most credibility-damaging gaps are **ML correctness and basic backend/engineering hygiene**. Security, testing, and code quality are treated as **mandatory cross-cutting layers** regardless of rank — see Section 7.

---

# 1. Current Project DNA Analysis

## 1.1 What the project actually is (verified overview)

**FACT.** SurgiVision is an **unsupervised anomaly-detection pipeline for spleen CT scans** built on the Medical Segmentation Decathlon **Task09_Spleen** dataset. The flow is:

1. **Preprocess** a NIfTI (`.nii.gz`) CT volume — load with `nibabel`, crop to the spleen bounding box using the label mask, window Hounsfield Units to `[-200, 300]`, normalize to `[0,1]`, resize to `64×64×64` with `scipy.ndimage.zoom`. ([spleen_preprocessing.py](spleen_preprocessing.py))
2. **Train** a 3D convolutional **autoencoder** (3× Conv3d/BN/ReLU/MaxPool encoder → Linear bottleneck `32768→512→32768` → 3× ConvTranspose3d decoder) on *spleen-only normal tissue* using MSE reconstruction loss, Adam, `ReduceLROnPlateau`. ([spleen_3d_model.py](spleen_3d_model.py), [training_pipeline.py](training_pipeline.py))
3. **Detect anomalies** by reconstruction error: compute a threshold as `mean + 3·std` over the first 10 training volumes; flag a volume if its error exceeds the threshold. ([spleen_anomaly_detector_fixed.py](spleen_anomaly_detector_fixed.py))
4. **Generate synthetic pathologies** (cyst, infarct, laceration, hyperdense mass, multiple metastases) by directly manipulating voxel intensities, used as positive test cases. ([enhanced_anomaly_creator.py](enhanced_anomaly_creator.py))
5. **Hybrid ensemble** — a frozen autoencoder plus a classification head on encoded features, a 3D-CNN spatial analyzer on the error map, and an attention module; combined with weighted voting. ([hybrid_anomaly_detector_fixed.py](hybrid_anomaly_detector_fixed.py))
6. **Evaluate** with accuracy/precision/recall/F1/specificity/sensitivity/AUC-ROC and matplotlib/seaborn plots. ([model_analysis.py](model_analysis.py))
7. **Serve a demo** via Streamlit — three modes (test on training volume, upload a file, synthetic-pathology demo), Plotly 3D scatter visualization, anomaly heatmap, and a polished PDF report via `reportlab`. ([streamlit_universal_demo.py](streamlit_universal_demo.py))

**FACT.** The repo contains **only source code, a README, and a logo**. There is no `data/`, no `models/` (trained weights), and every path is hardcoded relative to a non-existent `src/` layout (e.g., `../data/Task09_Spleen`, `../models/best_spleen_3d_autoencoder.pth`). **The code therefore cannot run from a fresh clone** without external artifacts and a directory restructure.

**Inferred dependency set (INFERENCE, from imports):** `torch`, `numpy`, `nibabel`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `pandas`, `streamlit`, `plotly`, `opencv-python` (`cv2`), `Pillow`, `reportlab`, `requests`. There is **no `requirements.txt`** pinning any of these. **FACT.**

## 1.2 Layer-by-layer analysis

For each layer: **what exists**, **what's strong**, **what's weak/missing**, **interview questions it already supports**, and **the highest-leverage upgrade**.

### Frontend
- **Exists (FACT):** A single Streamlit app with custom CSS, three demo modes, Plotly 3D scatter rendering, a 2D/3D heatmap, a patient-details sidebar form, a detection-sensitivity slider, and downloadable PDF reports.
- **Strong:** It is genuinely demo-ready and looks professional. The PDF report (`generate_pdf_bytes`) is detailed and polished. Plotly gives interactive 3D out of the box.
- **Weak/Missing:** Streamlit *is* the entire application — UI, business logic, and model inference are fused. No component model, no state management, no routing, no real medical-imaging viewer. The **2D-image path is medically meaningless** (it stacks a 2D image 64× into a fake volume and multiplies the threshold by 10 — [streamlit_universal_demo.py:287](streamlit_universal_demo.py:287), [:297](streamlit_universal_demo.py:297)). The **training-mode heatmap is random noise** (`np.random.random((64,64,64))*0.001` — [streamlit_universal_demo.py:218](streamlit_universal_demo.py:218), [:705](streamlit_universal_demo.py:705)).
- **Interview Qs it supports today:** "Walk me through your UI." "How does Streamlit caching work?" (`st.cache_resource`). "How do you render 3D data in a browser?"
- **Top upgrade:** Split UI from backend; add a real medical viewer (NiiVue/Cornerstone3D) and a results/history dashboard.
- **Maturity: intermediate.**

### Backend
- **Exists (FACT):** **None as a distinct layer.** All logic lives in scripts and inside the Streamlit process.
- **Strong:** N/A.
- **Weak/Missing:** No API service, no request/response contracts, no separation of concerns, no async handling of slow 3D inference, no service that another client could call.
- **Interview Qs it supports today:** Almost none — and this is the single biggest gap for SDE/backend roles.
- **Top upgrade:** A **FastAPI** inference service (`/predict`, `/health`, `/reports/{id}`) with Pydantic schemas and OpenAPI docs.
- **Maturity: beginner (effectively absent).**

### Database
- **Exists (FACT):** None. No persistence of any kind.
- **Weak/Missing:** No record of analyses, no patient/scan store, no audit trail, no way to build analytics over time.
- **Interview Qs it supports today:** None.
- **Top upgrade:** PostgreSQL + SQLAlchemy + Alembic; persist each analysis (pseudonymized patient, scan metadata, error, threshold, decision, model version, timestamp).
- **Maturity: beginner (absent).**

### APIs
- **Exists (FACT):** None (no REST/GraphQL/gRPC). The closest thing is Streamlit's internal callbacks.
- **Top upgrade:** Same FastAPI service; version it (`/api/v1/...`), document with OpenAPI, add a typed client.
- **Maturity: beginner (absent).**

### Authentication / Authorization
- **Exists (FACT):** None. The app is fully open; anyone who can reach it can run inference and download PHI-style reports.
- **Weak/Missing:** No users, no sessions, no RBAC, no tenant isolation — a serious omission for a *medical* app.
- **Interview Qs it supports today:** None.
- **Top upgrade:** JWT auth + role-based access (radiologist / technician / admin / read-only).
- **Maturity: beginner (absent).**

### Cloud / Infrastructure
- **Exists (FACT):** None. Local-only, run-from-folder.
- **Weak/Missing:** No object storage for scans/models, no managed DB, no IaC, no environments.
- **Top upgrade:** Containerize, push model to object storage, deploy to a managed platform (HF Spaces/Render to start; AWS ECS/Fargate + S3 + RDS as the "serious" story).
- **Maturity: beginner (absent).**

### DevOps / CI-CD
- **Exists (FACT):** None. No `Dockerfile`, no GitHub Actions, no `.gitignore` (earlier commits even tracked `.pyc` and a `Not useful/` junk folder — visible in git history), no dependency pinning.
- **Top upgrade:** `Dockerfile` + `docker-compose.yml` + GitHub Actions pipeline (ruff → mypy → pytest → build image).
- **Maturity: beginner (absent).**

### Security
- **Exists (FACT):** Effectively none, and there are concrete risks:
  - `torch.load(model_path, ...)` is called **without `weights_only=True`** in at least four files ([spleen_anomaly_detector_fixed.py:14](spleen_anomaly_detector_fixed.py:14), [hybrid_anomaly_detector_fixed.py:17](hybrid_anomaly_detector_fixed.py:17), [enhanced_anomaly_creator.py:166](enhanced_anomaly_creator.py:166), [streamlit_universal_demo.py](streamlit_universal_demo.py) via the detector). Loading an untrusted `.pth` can execute arbitrary code (pickle deserialization). **FACT.**
  - File upload accepts up to ~200 MB with no content validation beyond extension; the file is written to a temp path and parsed. **FACT.**
  - No secrets management, no rate limiting, no audit logging, no PHI handling (patient name/ID are entered and embedded in PDFs with zero protection). **FACT.**
- **Top upgrade:** See the Security Hardening Bundle (Section 5) — it's mandatory.
- **Maturity: beginner (weak/at-risk).**

### Testing
- **Exists (FACT):** No automated tests. The `test_*` functions (e.g., `test_model_architecture`, `test_corrected_detection`) are **manual print-based smoke scripts**, not assertions, and `internet_ct_tester.py` even calls `input()` ([internet_ct_tester.py:282](internet_ct_tester.py:282)) — it cannot run in CI.
- **Top upgrade:** A `pytest` suite with a tiny synthetic fixture volume so tests run without the full dataset.
- **Maturity: beginner (absent).**

### Observability
- **Exists (FACT):** None. Logging is `print()` with emojis everywhere. No metrics, no tracing, no error tracking.
- **Top upgrade:** Structured logging (loguru/structlog), Prometheus metrics (latency, throughput, error rate), Sentry.
- **Maturity: beginner (absent).**

### AI / ML features
- **Exists (FACT):** This is the project's real asset — a 3D CNN autoencoder, reconstruction-error anomaly detection, a multi-head hybrid ensemble (classification + spatial CNN + attention), synthetic pathology generation, and a metrics suite.
- **Strong:** Volumetric deep learning is legitimately advanced; the unsupervised "train on normal, flag deviations" framing is sound and interesting; the metrics suite shows ML literacy.
- **Weak/Missing (the credibility gaps):**
  - **INFERENCE (high confidence):** **Inference requires the ground-truth spleen mask.** The detector zeroes out non-spleen voxels using `mask > 0` ([spleen_anomaly_detector_fixed.py:98-100](spleen_anomaly_detector_fixed.py:98)) — but on a genuinely new scan you do **not** have that mask. So the headline "analyze any spleen CT" is not actually supported by the main pipeline.
  - **FACT:** The Streamlit "Upload" path tries to *match the uploaded filename against training filenames* and, if matched, re-runs the training sample **with its ground-truth mask** ([streamlit_universal_demo.py:202-215](streamlit_universal_demo.py:202)). That inflates apparent capability on "uploads."
  - **INFERENCE:** Evaluation uses **synthetic anomalies** (intensity blobs) as positives and derives the threshold from the **training set** ([spleen_anomaly_detector_fixed.py:31-64](spleen_anomaly_detector_fixed.py:31)). Reported metrics (e.g., high AUC) are therefore optimistic and don't reflect real-world generalization — this is **data leakage** and **circular evaluation**.
  - No explainability (the "heatmap" is fake in the main path), no uncertainty quantification, no GenAI/LLM layer, no experiment tracking, magic-number thresholds (`0.015`, `0.008756`, ×5, ×10).
- **Interview Qs it supports today:** "Explain autoencoder-based anomaly detection." "Why unsupervised here?" "How did you choose the threshold?" (careful — the honest answer reveals the leakage). "What's reconstruction error?"
- **Top upgrade:** MONAI segmentation to remove the mask dependency + honest evaluation + Grad-CAM.
- **Maturity: strong (model) but with serious credibility gaps.**

### Data analysis
- **Exists (FACT):** A solid one-shot evaluation harness — confusion matrix, ROC, PR curve, error distribution, per-pathology accuracy, metrics summary (sklearn + matplotlib + seaborn). ([model_analysis.py](model_analysis.py))
- **Strong:** Demonstrates real metrics understanding (sensitivity/specificity matter clinically).
- **Weak/Missing:** It's a static, run-once analysis. No cohort/longitudinal analytics, no dashboard over many runs, no data-quality profiling, no drift analysis, nothing backed by a database.
- **Interview Qs it supports today:** "Why is AUC better than accuracy for imbalanced data?" "Explain precision vs recall in a clinical context."
- **Top upgrade:** A live analytics dashboard over a results DB + drift monitoring.
- **Maturity: intermediate.**

### Performance
- **Exists (FACT):** CPU/GPU device handling, `st.cache_resource` to cache the loaded model, `torch.cuda.empty_cache()` calls.
- **Weak/Missing:** Per-request preprocessing with no caching, no batch inference, no ONNX/TorchScript, no mixed precision; the `empty_cache()` inside the training loop ([training_pipeline.py:99](training_pipeline.py:99)) actually *hurts* throughput.
- **Top upgrade:** ONNX Runtime export for fast CPU inference + batch endpoint + input-hash caching.
- **Maturity: intermediate (low).**

### Code quality
- **Exists (FACT):** Reasonable file-level separation and readable names.
- **Weak/Missing:** Hardcoded paths everywhere; magic numbers; **duplicated inference logic** (the "mask → tensor → reconstruct → MSE error" block is copy-pasted across ~6 files); **dead code** (`create_pdf_bytes` is defined but unused — the app uses `generate_pdf_bytes`); **duplicate imports** (reportlab/`BytesIO`/`datetime` imported 2–3× in the Streamlit file — [streamlit_universal_demo.py:18-24](streamlit_universal_demo.py:18), [:442-451](streamlit_universal_demo.py:442)); no type hints; no config layer; a typo in user-facing output ("Anomally detected" — [streamlit_universal_demo.py:585](streamlit_universal_demo.py:585)).
- **Top upgrade:** Package structure + config module + shared `inference` module + ruff/mypy/pre-commit.
- **Maturity: beginner → intermediate.**

### Scalability
- **Exists (FACT):** None by design — single-process Streamlit, synchronous inference, no queue.
- **Top upgrade:** Stateless FastAPI workers behind a load balancer + background job queue (Celery/RQ + Redis) for 3D inference.
- **Maturity: beginner.**

### Maintainability
- **Exists (FACT):** Medium — files are small and readable.
- **Weak/Missing:** No tests, no docs beyond a 3-line README, no config, heavy duplication, won't run from a clean clone.
- **Maturity: beginner → intermediate.**

## 1.3 Maturity scorecard (one line per layer)

| Layer | Maturity | One-line summary |
|-------|----------|------------------|
| Frontend | **intermediate** | Polished Streamlit demo, but UI/logic/model are fused and one path shows fake data. |
| Backend | **beginner (absent)** | No service layer — logic lives in scripts and the UI process. |
| Database | **beginner (absent)** | No persistence, history, or audit trail. |
| APIs | **beginner (absent)** | Nothing another client could call. |
| Auth/Authz | **beginner (absent)** | Fully open medical app with PHI fields and zero access control. |
| Cloud/Infra | **beginner (absent)** | Local-only, run-from-folder. |
| DevOps/CI-CD | **beginner (absent)** | No Docker, CI, `.gitignore`, or pinned deps. |
| Security | **beginner (at-risk)** | `torch.load` RCE risk, unvalidated uploads, no secrets/PHI handling. |
| Testing | **beginner (absent)** | Print-based smoke functions, one with `input()`; nothing CI-runnable. |
| Observability | **beginner (absent)** | `print()` only — no logs/metrics/tracing. |
| AI/ML | **strong (with gaps)** | Real 3D DL, but mask-dependent inference + leaky, synthetic-only evaluation. |
| Data analysis | **intermediate** | Good one-shot metrics; no live/longitudinal analytics. |
| Performance | **intermediate (low)** | Basic device handling; no batching/ONNX; per-request preprocessing. |
| Code quality | **beginner→intermediate** | Readable but hardcoded paths, magic numbers, dup/dead code, no types/config. |
| Scalability | **beginner** | Single-process, synchronous. |
| Maintainability | **beginner→intermediate** | Small files, but no tests/docs/config; won't run from a clean clone. |

**Overall:** *Strong ML prototype, hackathon-grade engineering.* The model carries it; the system around it does not yet exist.

## 1.4 Integrity issues — call these out before an interviewer does (brutal honesty)

These are not "nice to fix later." Each one, if discovered live, can sink an otherwise good interview. Fixing them is *itself* a great talking point ("I found and corrected a validity bug in my own pipeline").

1. **Fake heatmap in the main demo path. FACT.** Training-mode and training-pipeline uploads display `np.random.random((64,64,64))*0.001` as the "anomaly heatmap." It is pure noise. Fix with Grad-CAM or the genuine reconstruction-error map (Upgrade #5).
2. **Hidden mask dependency at inference. INFERENCE (high).** The model can't localize the spleen on an unseen scan; it relies on the dataset's ground-truth mask. Fix with a segmentation step (Upgrade #4).
3. **Leaky / circular evaluation. INFERENCE (high).** Threshold from the training set + synthetic-only positives → optimistic metrics. Fix with a held-out split and (ideally) real abnormal data (Upgrade #9).
4. **"Internet CT tester" doesn't use the internet. FACT.** `download_sample_cts()` just re-reads training indices 10/15/20 and labels them "internet" cases ([internet_ct_tester.py:21-42](internet_ct_tester.py:21)).
5. **2D image support is a gimmick. INFERENCE.** Stacking a 2D image into a pseudo-volume and feeding it to a spleen-CT model is medically meaningless; the ×10 threshold confirms it's a hack.
6. **`torch.load` without `weights_only`. FACT.** Deserialization RCE risk (Upgrade #7).

---

# 2. Missing Pieces (gap analysis)

Mapped to the classic "does this sound senior?" checklist, with a verdict on how much each gap matters **for this project type** (a medical-AI system aimed at SDE/backend/AI/ML/data roles).

| Gap | Present? | Severity for THIS project | Why |
|-----|:--------:|:--------------------------:|-----|
| Real AI/ML component | ✅ Yes (strong) | — | This is the project's strength; protect it by fixing validity. |
| Honest evaluation / no leakage | ❌ No | **Critical** | Leaky metrics are the #1 thing that discredits an ML project. |
| Inference without ground-truth labels | ❌ No | **Critical** | Mask-dependent inference means it can't run on real new scans. |
| Explainability (real, not fake) | ❌ No | **Critical** | The fake heatmap is an integrity problem; XAI is also hot. |
| Backend API / service boundary | ❌ No | **High** | Biggest gap for SDE/backend roles; unlocks everything else. |
| Database / persistence / audit | ❌ No | **High** | No history, no analytics, no audit — unacceptable for medical. |
| Auth + RBAC | ❌ No | **High** | Mandatory framing for a healthcare app; common interview topic. |
| Tests + CI | ❌ No | **High** | "Do you write tests?" is asked in nearly every SDE loop. |
| Containerization / deploy | ❌ No | **High** | "How would you ship this?" — currently no answer. |
| Meaningful caching | ❌ No (only model cache) | **Medium** | Easy win; common system-design probe. |
| GenAI / LLM layer | ❌ No | **High (opportunity)** | The most current, highest-signal addition you can make in 2026. |
| Data pipeline / analytics story | ⚠️ Partial | **Medium-High** | One-shot plots only; no dashboard/cohort/drift story. |
| Observability (logs/metrics/traces) | ❌ No | **Medium** | Differentiates "I deployed it" from "I operated it." |
| Rate limiting / abuse protection | ❌ No | **Medium** | Quick to add; good security talking point. |
| Reproducibility (pinned deps, seeds, config) | ❌ No | **High** | Won't run from clean clone today — fix first. |
| Production thinking (versioning, registry, rollback) | ❌ No | **Medium** | MLOps maturity signal. |
| Scaling story (queue, workers, async) | ❌ No | **Medium** | System-design payoff once the API exists. |

**Which gaps matter MOST for this project type (ranked):**
1. **ML validity** (leakage, mask dependency, fake heatmap) — fix or your strongest asset becomes a liability.
2. **Backend API + persistence** — converts a notebook-grade demo into a system.
3. **Reproducibility + tests + CI + Docker** — table stakes; cheap; asked constantly.
4. **Security/auth** — mandatory for a *medical* app; cheap signal.
5. **GenAI report layer** — the single most "current" differentiator.
6. **Analytics + observability** — turns it into an operable product and a data story.

---

# 3. Best Upgrade Ideas (detailed cards)

Each card: what it does · why it matters · roles it helps · market relevance · interview value (with a sample question + what a strong answer sounds like) · complexity · time · dependencies · risk · demo value · level.

> Time estimates assume a motivated student working part-time; "d" = days, "w" = weeks.

---

### Upgrade #1 — LLM-drafted radiology report (Anthropic API)
- **What:** Feed the structured numeric outputs (reconstruction error, threshold, decision, pathology pattern, spleen-region stats) into Claude to generate a **structured preliminary report** — Findings, Impression, Recommendation — with explicit "AI-generated, requires radiologist review" guardrails. Use the latest Claude model (e.g., `claude-opus-4-8` for quality or `claude-haiku-4-5` for cost/latency). Force **structured output** (JSON schema) and add a rule-based guardrail that the LLM may only describe what the numbers support (no invented measurements).
- **Why it matters:** Clinical-NLP / LLM report drafting is one of the hottest real applications of GenAI in healthcare right now. It turns "a number" into "a usable artifact."
- **Roles:** AI engineer (primary), Full-stack, Backend.
- **Market relevance:** Very high (2026). GenAI + structured output + guardrails is exactly what AI-product teams hire for.
- **Interview value:** **Very high.** *Q: "How do you stop the LLM from hallucinating findings?"* Strong answer: "I constrain it to a JSON schema, pass only verified numeric features, add a system-prompt rule that it may not state measurements not present in the input, run a post-generation validator that rejects reports referencing absent fields, and keep a human-in-the-loop disclaimer. I also built a small eval set of (input → expected impression) to catch regressions."
- **Complexity:** Medium · **Time:** 3–5 d · **Deps:** FastAPI (#2) ideally; Anthropic API key + secrets management (#7).
- **Risk:** Medium (hallucination, PHI leaving your environment — mitigate with pseudonymization and a clear disclaimer).
- **Demo value:** Very high (upload → instant readable report). · **Level:** Medium.

### Upgrade #2 — FastAPI inference service
- **What:** Extract all inference into a FastAPI app: `POST /api/v1/predict` (multipart scan upload → JSON result), `GET /healthz`, `GET /reports/{id}`, `POST /reports/{id}/pdf`. Pydantic request/response models, auto OpenAPI/Swagger docs, dependency-injected model loader. Streamlit (or React) becomes a *client*.
- **Why it matters:** Creates the service boundary every backend/SDE interview expects; unlocks auth, caching, scaling, and a real architecture diagram.
- **Roles:** Backend (primary), SDE, Full-stack.
- **Market relevance:** Very high; FastAPI is the default modern Python API framework.
- **Interview value:** **Very high.** *Q: "How would you serve a slow 3D model without blocking requests?"* Strong answer: "Synchronous for small inputs, but I move heavy 3D inference to a background task/queue (RQ/Celery + Redis), return a job id immediately, and let the client poll `/jobs/{id}` or receive a webhook. The API stays stateless so I can scale workers horizontally."
- **Complexity:** Low-Medium · **Time:** 3–5 d · **Deps:** the shared `inference` module from the code-quality refactor (#19).
- **Risk:** Low · **Demo value:** Medium (Swagger UI is a clean demo) · **Level:** Medium.

### Upgrade #3 — PostgreSQL persistence + analysis history & audit
- **What:** Add Postgres + SQLAlchemy + Alembic migrations. Tables: `users`, `scans` (pseudonymized patient ref, modality, shape, storage URI), `analyses` (scan_id, model_version, error, threshold, decision, confidence, created_by, created_at), `audit_log`. Every prediction is persisted; the UI gets a "History" view.
- **Why it matters:** No serious app is stateless about its results. Enables analytics, audit (mandatory in healthcare), reproducibility, and model-version comparisons.
- **Roles:** Backend, Full-stack, Data engineer.
- **Market relevance:** High (universal).
- **Interview value:** **Very high.** *Q: "Design the schema for storing analyses and supporting an audit trail."* Strong answer covers normalization, an append-only audit log, soft deletes, indices on `created_at`/`model_version`, and storing large blobs (scans) in object storage with only URIs in the DB.
- **Complexity:** Medium · **Time:** 4–6 d · **Deps:** FastAPI (#2).
- **Risk:** Low · **Demo value:** Medium-High (history + audit views) · **Level:** Medium.

### Upgrade #4 — MONAI spleen segmentation (remove the mask dependency)
- **What:** Add a segmentation stage (MONAI pretrained spleen model or a small trained U-Net) that **produces the spleen mask at inference time**, so the pipeline works on a raw, unseen CT with no ground-truth label. This closes the project's biggest correctness gap.
- **Why it matters:** Without it, the system literally cannot run on a new patient. With it, the end-to-end claim becomes true, and you gain MONAI — the standard medical-imaging DL framework — on your résumé.
- **Roles:** ML engineer (primary), AI engineer.
- **Market relevance:** Very high (MONAI/segmentation are core medical-AI skills).
- **Interview value:** **Very high.** *Q: "Your detector needs the spleen region — where does that come from in production?"* Strong answer: "Originally I leaned on dataset masks, which is leakage at inference time. I added a MONAI segmentation front-end so the spleen is localized automatically; I measured the downstream impact of imperfect masks on the anomaly score and added morphological cleanup."
- **Complexity:** Medium-High · **Time:** 1–2 w · **Deps:** MONAI; some compute for fine-tuning.
- **Risk:** Medium (segmentation errors propagate) · **Demo value:** High (auto-overlay of the segmented spleen) · **Level:** Advanced.

### Upgrade #5 — Grad-CAM real explainability (replace the fake heatmap)
- **What:** Use Captum (or a manual hook) to produce a true saliency/error map showing *where* the model sees deviation, and render it as the heatmap and a 3D overlay. Delete the `np.random.random` placeholder.
- **Why it matters:** Fixes an integrity problem **and** adds explainable-AI, which is both clinically essential and a hot interview topic.
- **Roles:** ML engineer, AI engineer.
- **Market relevance:** High (XAI in regulated domains).
- **Interview value:** **High.** *Q: "How do you explain a deep model's decision to a clinician?"* Strong answer discusses reconstruction-error localization vs gradient-based saliency, their failure modes, and overlaying on the original slice for trust.
- **Complexity:** Medium · **Time:** 3–5 d · **Deps:** trained model; Captum.
- **Risk:** Low-Medium · **Demo value:** Very high (visual, intuitive) · **Level:** Medium.

### Upgrade #6 — RAG over radiology guidelines (grounded recommendations)
- **What:** Build a small retrieval layer (pgvector or Chroma) over curated, citable spleen-lesion management references; have the report LLM (#1) cite retrieved snippets so recommendations are grounded, not invented.
- **Why it matters:** RAG is the dominant pattern for trustworthy LLM apps; "grounded + cited" is exactly what reduces hallucination concerns in medicine.
- **Roles:** AI engineer (primary).
- **Market relevance:** Very high.
- **Interview value:** **Very high.** *Q: "How is RAG different from fine-tuning, and why choose it here?"* Strong answer: freshness, citeability, no retraining, cheaper, and auditable sources — critical in a clinical setting.
- **Complexity:** Medium-High · **Time:** 1–1.5 w · **Deps:** #1, a vector store, a curated corpus.
- **Risk:** Medium (corpus quality, licensing of medical text) · **Demo value:** High (answers with citations) · **Level:** Advanced.

### Upgrade #7 — Security hardening (cross-cutting, mandatory)
- **What:** (a) `torch.load(..., weights_only=True)` + checksum/signature verification of model files; (b) upload validation — magic-byte sniffing, size caps, NIfTI/DICOM header sanity checks, reject on parse failure; (c) secrets via env/`.env` + `pydantic-settings` (no keys in code); (d) rate limiting (`slowapi`); (e) audit logging of every inference and report download; (f) PHI handling — pseudonymize patient identifiers, encrypt at rest, restrict report access.
- **Why it matters:** It's a *medical* app; security is non-negotiable and a cheap, high-signal differentiator.
- **Roles:** Security-focused engineer, Backend, SDE.
- **Market relevance:** High (always).
- **Interview value:** **Very high.** *Q: "What's the risk of `pickle`/`torch.load`?"* Strong answer: arbitrary code execution on load; mitigate with `weights_only`, signed artifacts, and loading only from trusted storage.
- **Complexity:** Low-Medium · **Time:** 3–5 d (spread across features) · **Deps:** FastAPI (#2) for most controls.
- **Risk:** Low · **Demo value:** Low-Medium (show a blocked malicious upload) · **Level:** Medium.

### Upgrade #8 — Dockerize + docker-compose
- **What:** Multi-stage `Dockerfile` (slim CPU base), `docker-compose.yml` wiring app + Postgres + Redis. `.dockerignore`, non-root user, pinned base image.
- **Why it matters:** "Runs anywhere," reproducible, the foundation for any deployment. Fixes the "won't run from clean clone" problem.
- **Roles:** DevOps/SRE, Backend, SDE.
- **Market relevance:** Very high (universal).
- **Interview value:** **High.** *Q: "Why multi-stage builds?"* Strong answer: smaller images, no build toolchain in the runtime layer, faster cold starts, smaller attack surface.
- **Complexity:** Low · **Time:** 2–3 d · **Deps:** `requirements.txt` (#19).
- **Risk:** Low · **Demo value:** Medium (`docker compose up` just works) · **Level:** Beginner-Medium.

### Upgrade #9 — Honest evaluation methodology (cross-cutting, mandatory)
- **What:** Proper train/val/**held-out test** split fixed up front; compute the threshold on validation only; report metrics on the untouched test set; add k-fold CV; clearly label synthetic-anomaly results as a *sanity check*, not real performance; if any real abnormal cases are obtainable, evaluate on them and report honestly (even if numbers drop).
- **Why it matters:** Converts the project's biggest liability into its biggest credibility asset.
- **Roles:** ML engineer (primary), Data analyst.
- **Market relevance:** High.
- **Interview value:** **Very high.** *Q: "How do you know your model generalizes?"* A strong, leakage-aware answer is rare in student projects and immediately signals maturity.
- **Complexity:** Low-Medium · **Time:** 3–5 d · **Deps:** none.
- **Risk:** Low (other than your metrics getting more honest) · **Demo value:** Medium · **Level:** Medium.

### Upgrade #10 — React/Next.js frontend + NiiVue medical viewer
- **What:** A Next.js app that consumes the FastAPI API: upload, live result, **NiiVue** (WebGL) in-browser NIfTI volume rendering with the segmented spleen + Grad-CAM overlay, a history table, and the analytics dashboard. (Cornerstone3D/OHIF if you add DICOM.)
- **Why it matters:** Real full-stack story + a *domain-correct* viewer (not a Plotly scatter), which is far more impressive for medical imaging.
- **Roles:** Frontend, Full-stack.
- **Market relevance:** High.
- **Interview value:** **High.** *Q: "How do you render large 3D volumes in the browser?"* Strong answer covers WebGL/texture-based volume rendering, streaming/downsampling, and offloading heavy compute to the backend.
- **Complexity:** High · **Time:** 2–3 w · **Deps:** #2 (API), #4/#5 (things worth viewing).
- **Risk:** Medium (scope) · **Demo value:** Very high · **Level:** Advanced.

### Upgrade #11 — Cloud deployment
- **What:** Start with a free/cheap live demo (Hugging Face Spaces or Render/Railway/Fly.io) with the model pulled from object storage; document an AWS production path (ECS/Fargate + S3 for scans/models + RDS Postgres + CloudFront for the frontend).
- **Why it matters:** A **live URL** is the single highest-conversion demo asset; cloud literacy is expected.
- **Roles:** Cloud/DevOps, Full-stack.
- **Market relevance:** High.
- **Interview value:** **High.** *Q: "Where do model weights live in production?"* Strong answer: object storage with versioned keys, pulled at startup or baked into the image depending on size/latency trade-offs.
- **Complexity:** Medium · **Time:** 3–5 d (managed) · **Deps:** #8.
- **Risk:** Medium (cost, secrets in cloud) · **Demo value:** Very high (shareable link) · **Level:** Medium.

### Upgrade #12 — CI/CD with GitHub Actions
- **What:** Pipeline: `ruff` (lint+format) → `mypy` (types) → `pytest` (with coverage gate) → build & (optionally) push Docker image → deploy on tag. Add pre-commit hooks mirroring CI.
- **Why it matters:** Demonstrates engineering discipline; catches regressions; expected at every level.
- **Roles:** DevOps/SRE, SDE.
- **Market relevance:** Very high.
- **Interview value:** **High.** *Q: "What's in your CI pipeline and why?"* Strong answer ties each stage to a failure it prevents.
- **Complexity:** Low · **Time:** 2–3 d · **Deps:** #14, #19, #8.
- **Risk:** Low · **Demo value:** Medium (green checks + badges) · **Level:** Beginner-Medium.

### Upgrade #13 — Analytics dashboard over the results DB
- **What:** A dashboard (Streamlit page, or React, or Metabase) over `analyses`: scans/day, anomaly rate over time, reconstruction-error distribution, per-pathology detection rates, latency percentiles, model-version comparison, exportable CSV.
- **Why it matters:** This is the **data-analyst/data-engineer story** — turning operational data into insight.
- **Roles:** Data analyst (primary), Data engineer, Full-stack.
- **Market relevance:** High.
- **Interview value:** **High.** *Q: "What would you monitor for this product and why?"* Strong answer connects metrics to clinical/operational decisions and to model health.
- **Complexity:** Low-Medium · **Time:** 4–6 d · **Deps:** #3.
- **Risk:** Low · **Demo value:** High · **Level:** Medium.

### Upgrade #14 — pytest test suite + coverage (cross-cutting, mandatory)
- **What:** Unit tests for preprocessing (output shape `64³`, normalization in `[0,1]`, bbox cropping), model forward pass, threshold logic, anomaly decision boundaries, and PDF generation; an API integration test; a tiny synthetic fixture volume so tests need no dataset; coverage reporting with a gate.
- **Why it matters:** "Do you test?" is asked in nearly every loop; tests also make all later refactors safe.
- **Roles:** QA/SDET, SDE, Backend.
- **Market relevance:** High.
- **Interview value:** **High.** *Q: "How do you test ML code that needs big data/GPUs?"* Strong answer: synthetic fixtures, deterministic seeds, shape/contract tests, separating I/O from logic, mocking the model where appropriate.
- **Complexity:** Low · **Time:** 4–6 d · **Deps:** #19 (refactor makes code testable).
- **Risk:** Low · **Demo value:** Medium · **Level:** Beginner-Medium.

### Upgrade #15 — Data/model drift monitoring
- **What:** Use Evidently (or custom checks) to track input distribution (HU stats, volume sizes), prediction distribution, and error-score drift over time; alert on shift.
- **Why it matters:** MLOps maturity; "models degrade" is a senior insight.
- **Roles:** ML engineer, Data engineer, MLOps.
- **Market relevance:** High.
- **Interview value:** **Medium-High.** *Q: "How would you know your model is getting worse in production?"* Strong answer: proxy metrics (input/output drift) when labels are delayed, plus periodic re-evaluation.
- **Complexity:** Medium · **Time:** 4–6 d · **Deps:** #3, #13.
- **Risk:** Low · **Demo value:** Medium · **Level:** Advanced.

### Upgrade #16 — JWT auth + role-based access control
- **What:** Auth (JWT or a provider like Auth0/Clerk); roles radiologist / technician / admin / read-only; protect endpoints; scope report access to the creating user/org.
- **Why it matters:** Mandatory for a medical app; RBAC is a frequent interview topic.
- **Roles:** Backend, Security, Full-stack.
- **Market relevance:** High.
- **Interview value:** **High.** *Q: "JWT vs sessions — trade-offs?"* Strong answer: statelessness/scaling vs revocation difficulty; refresh-token rotation; where to store tokens safely.
- **Complexity:** Medium · **Time:** 4–6 d · **Deps:** #2, #3.
- **Risk:** Medium (auth bugs are security bugs) · **Demo value:** Medium · **Level:** Medium.

### Upgrade #17 — Observability stack
- **What:** Structured logging (loguru/structlog) with correlation IDs replacing every `print`; Prometheus metrics (inference latency histogram, request count, error rate, model-load time); Grafana dashboard; Sentry for exceptions.
- **Why it matters:** Separates "I deployed it" from "I can operate it."
- **Roles:** DevOps/SRE, Backend.
- **Market relevance:** Medium-High.
- **Interview value:** **Medium-High.** *Q: "What are the four golden signals?"* Strong answer: latency, traffic, errors, saturation — mapped to your metrics.
- **Complexity:** Medium · **Time:** 4–6 d · **Deps:** #2, #8.
- **Risk:** Low · **Demo value:** Medium (a live Grafana board is nice) · **Level:** Medium.

### Upgrade #18 — Experiment tracking + model registry + DVC
- **What:** MLflow or Weights & Biases for runs/metrics/artifacts; a model registry with versions and stage promotion (staging→prod); DVC (or Git LFS) for dataset/model versioning so the repo no longer assumes external files.
- **Why it matters:** Reproducibility and MLOps maturity; also makes the repo runnable.
- **Roles:** ML engineer, MLOps.
- **Market relevance:** High.
- **Interview value:** **Medium-High.** *Q: "How do you track and reproduce experiments?"* Strong answer: logged params/metrics/artifacts, versioned data/code, registry-based promotion with the exact commit.
- **Complexity:** Medium · **Time:** 4–6 d · **Deps:** #19.
- **Risk:** Low · **Demo value:** Medium · **Level:** Medium.

### Upgrade #19 — Code-quality overhaul (foundation, do early)
- **What:** `requirements.txt`/`pyproject.toml` with pinned deps; a `surgivision/` package (`models/`, `data/`, `inference/`, `api/`, `reporting/`, `config.py`); a config layer (`pydantic-settings` + YAML) to kill hardcoded paths and magic numbers; **extract the duplicated inference block into one `inference` function**; remove dead code (`create_pdf_bytes`) and duplicate imports; add type hints + `ruff` + `mypy`; fix the "Anomally" typo; add a `.gitignore`.
- **Why it matters:** Everything else (tests, API, CI) is cheaper and cleaner on this foundation; it also makes the repo actually runnable.
- **Roles:** SDE, Backend, all.
- **Market relevance:** Medium (table stakes).
- **Interview value:** **Medium-High** as a story: "I refactored 6 copies of inference into one tested module and introduced config + typing."
- **Complexity:** Low · **Time:** 4–6 d · **Deps:** none (do first).
- **Risk:** Low · **Demo value:** Low (but enables everything) · **Level:** Beginner-Medium.

### Upgrade #20 — Performance: ONNX/TorchScript + batch inference
- **What:** Export the model to ONNX, serve via ONNX Runtime for faster CPU inference; add a batch endpoint; cache preprocessing by input hash; remove the per-iteration `empty_cache()` from training.
- **Why it matters:** Concrete, measurable latency/throughput wins — great for a "I made it 3× faster" story.
- **Roles:** ML engineer, Backend, Performance.
- **Market relevance:** Medium.
- **Interview value:** **Medium.** *Q: "How did you speed up inference?"* Strong answer: profiled first, found preprocessing dominated, cached it, exported to ONNX, batched — with before/after numbers.
- **Complexity:** Medium · **Time:** 4–6 d · **Deps:** trained model; #2.
- **Risk:** Low-Medium (numerical parity after export) · **Demo value:** Medium (latency chart) · **Level:** Medium.

---

# 4. Priority Ranking Table (scoring model)

Dimensions scored 1–5. **Net = (Interview + Market + Demo + Depth + Resume + Long-term) − Effort.**

| Rank | Upgrade | Cat | Interview | Market | Demo | Depth | Resume | Long-term | Effort | **Net** | Verdict |
|:---:|---------|-----|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|---|
| 1 | LLM-drafted report | AI | 5 | 5 | 5 | 4 | 5 | 5 | 3 | **26** | Do now |
| 2 | FastAPI service | Backend | 5 | 5 | 3 | 4 | 5 | 5 | 2 | **25** | Do now |
| 3 | Postgres + history/audit | Backend/DB | 5 | 4 | 4 | 4 | 5 | 5 | 3 | **24** | Do now |
| 4 | MONAI segmentation | ML | 5 | 5 | 4 | 5 | 5 | 4 | 4 | **24** | Do now* |
| 5 | Grad-CAM XAI | AI/ML | 5 | 5 | 5 | 4 | 4 | 4 | 3 | **24** | Do now* |
| 6 | RAG guidelines | AI | 5 | 5 | 4 | 4 | 5 | 4 | 4 | **23** | Do next |
| 7 | Security hardening | Security | 5 | 4 | 2 | 4 | 4 | 5 | 2 | **22** | Do now (mand.) |
| 8 | Dockerize | DevOps | 4 | 5 | 3 | 3 | 4 | 5 | 2 | **22** | Do now |
| 9 | Honest evaluation | ML | 5 | 4 | 2 | 5 | 4 | 4 | 2 | **22** | Do now (mand.) |
| 10 | React + NiiVue | Frontend | 4 | 4 | 5 | 4 | 5 | 4 | 4 | **22** | Do next |
| 11 | Cloud deploy | Cloud | 4 | 4 | 5 | 3 | 4 | 4 | 3 | **21** | Do next |
| 12 | CI/CD | DevOps | 4 | 5 | 2 | 3 | 4 | 5 | 2 | **21** | Do now |
| 13 | Analytics dashboard | Data | 4 | 4 | 4 | 3 | 4 | 4 | 2 | **21** | Do next |
| 14 | pytest + coverage | Testing | 5 | 4 | 2 | 3 | 4 | 5 | 2 | **21** | Do now (mand.) |
| 15 | Drift monitoring | Data/MLOps | 4 | 4 | 3 | 4 | 4 | 4 | 3 | **20** | Do later |
| 16 | JWT + RBAC | Security | 5 | 4 | 3 | 3 | 4 | 4 | 3 | **20** | Do next |
| 17 | Observability | Observability | 4 | 4 | 3 | 4 | 4 | 4 | 3 | **20** | Do later |
| 18 | Experiment tracking + registry | MLOps | 4 | 4 | 3 | 4 | 4 | 4 | 3 | **20** | Do later |
| 19 | Code-quality overhaul | Code quality | 4 | 3 | 1 | 3 | 3 | 5 | 2 | **17** | Do now (foundation) |
| 20 | ONNX/perf | Performance | 3 | 3 | 3 | 4 | 3 | 3 | 3 | **16** | Do later |

`*` = lower Net than the very top items only because Effort is higher, but flagged **Do now** because they fix correctness/integrity, which outranks raw score.

### Effort-vs-impact quadrants

- **High impact / low effort (do first):** Code-quality overhaul (#19), Dockerize (#8), CI/CD (#12), pytest (#14), Security hardening (#7), Honest evaluation (#9), FastAPI (#2).
- **High impact / higher effort (flagship builds):** LLM report (#1), MONAI segmentation (#4), Postgres (#3), Grad-CAM (#5), RAG (#6), React+NiiVue (#10).
- **Medium impact / medium effort (depth/MLOps):** Analytics (#13), Observability (#17), Drift (#15), Experiment tracking (#18), Cloud (#11), JWT/RBAC (#16).
- **Lower priority:** ONNX/perf (#20) — valuable but optimize only after the system exists.

### Category-grouped view (honoring the requested default order)

| Requested priority | Category | Lead upgrade(s) here |
|:--:|---|---|
| 1 | Frontend | React + NiiVue viewer (#10); fix fake heatmap as part of #5 |
| 2 | Backend | FastAPI (#2), Postgres (#3) |
| 3 | AI | LLM report (#1), RAG (#6), Grad-CAM (#5) |
| 4 | Data analysis | Analytics dashboard (#13), drift (#15) |
| 5 | ML | MONAI seg (#4), honest eval (#9), experiment tracking (#18) |
| 6 | Cloud/DevOps | Docker (#8), CI/CD (#12), deploy (#11) |
| 7 | Security (mandatory) | Hardening (#7), JWT/RBAC (#16) |
| 8 | Testing (mandatory) | pytest + coverage (#14) |
| 9 | Code quality | Overhaul (#19) |
| 10 | Performance | ONNX/batch (#20) |
| 11 | Observability | Logs/metrics/Sentry (#17) |
| 12 | Misc polish | README, badges, architecture diagram, demo video |

---

# 5. Feature Bundles

Each bundle is a coherent story you can present as one "chapter" of the project and one résumé bullet.

### A. Backend Robustness Bundle
- **Goal:** Turn the demo into a real service with state.
- **Features:** FastAPI (#2) + Postgres/history/audit (#3) + caching + JWT/RBAC (#16) + structured logging (#17).
- **Why together:** They form the canonical request → auth → inference → persist → audit path; each depends on the previous.
- **Interview-ready for:** **Backend engineer / SDE / Full-stack.** Yes — this bundle alone makes the project credible for backend loops.

### B. AI / GenAI Bundle
- **Goal:** A trustworthy, current GenAI layer on top of the model.
- **Features:** LLM report drafting (#1) + RAG over guidelines (#6) + a "explain this result" chat.
- **Why together:** Structured output + retrieval grounding + guardrails is the standard trustworthy-LLM stack.
- **Interview-ready for:** **AI engineer.** Yes — this is the most "2026-current" bundle.

### C. ML Credibility Bundle
- **Goal:** Make the science defensible.
- **Features:** MONAI segmentation (#4) + honest evaluation (#9) + Grad-CAM (#5) + uncertainty (MC-dropout) + experiment tracking/registry (#18).
- **Why together:** Removes leakage, removes mask dependency, adds explainability and reproducibility — the four things that make ML work believable.
- **Interview-ready for:** **ML engineer.** Yes — and it neutralizes the integrity landmines.

### D. Data Analytics Bundle
- **Goal:** Insight from operational data.
- **Features:** Postgres (#3) + analytics dashboard (#13) + drift monitoring (#15) + CSV/export + SQL views.
- **Why together:** You can't analyze what you don't store; dashboard + drift are the consumption layer.
- **Interview-ready for:** **Data analyst / Data engineer.** Yes for analyst; partial for data engineer (add a real pipeline/orchestrator to fully satisfy DE).

### E. Cloud & Deployment Bundle
- **Goal:** Ship it and prove you can.
- **Features:** Docker (#8) + CI/CD (#12) + cloud deploy (#11) + secrets + object storage for models.
- **Why together:** The standard build→test→containerize→deploy chain.
- **Interview-ready for:** **DevOps/SRE / Full-stack.** Yes for the pipeline story; add IaC (Terraform) + Kubernetes for senior DevOps depth.

### F. Security Hardening Bundle (cross-cutting)
- **Goal:** Make a medical app you'd actually trust.
- **Features:** `weights_only` + signed models, upload validation, rate limiting, secrets management, audit logging, PHI pseudonymization/encryption, RBAC (#16).
- **Why together:** Together they cover the OWASP-style surface for a file-upload ML service handling sensitive data.
- **Interview-ready for:** **Security-focused engineer / Backend.** Yes as a differentiator on top of any other bundle.

### G. Testing & Quality Bundle (cross-cutting)
- **Goal:** Engineering discipline.
- **Features:** Code-quality overhaul (#19) + pytest/coverage (#14) + ruff/mypy/pre-commit + CI gate (#12).
- **Why together:** Refactor enables tests; tests enable safe CI; CI enforces quality.
- **Interview-ready for:** **SDE / QA-SDET.** Yes as table-stakes proof.

### H. Frontend Showcase Bundle
- **Goal:** A domain-correct, impressive UI.
- **Features:** React/Next.js (#10) + NiiVue/Cornerstone viewer + history + analytics views + Grad-CAM overlay.
- **Why together:** A real medical viewer + dashboards is the visual proof of the whole system.
- **Interview-ready for:** **Frontend / Full-stack.** Yes once the API exists.

---

# 6. Phase-wise Roadmap

> Sequencing principle: **make it run and honest → make it a service → make it impressive → make it production-grade.** Security, testing, and code quality are threaded through every phase, not bolted on at the end.

### Phase 1 — Quick wins & integrity fixes (≈1.5–2.5 weeks)
- **Build:** Code-quality overhaul (#19: requirements, package layout, config, kill dup/dead code, `.gitignore`) · honest evaluation (#9) · Grad-CAM to replace the fake heatmap (#5, at minimum stop showing noise) · `torch.load(weights_only=True)` + basic upload validation (part of #7) · a starter pytest suite (#14) · Dockerfile (#8) · a real README with architecture diagram + setup.
- **Why this phase:** It makes the repo **runnable from a clean clone**, removes the three integrity landmines, and adds the cheapest high-signal engineering hygiene. Nothing later is safe or demoable without this.
- **Time:** ~2 weeks · **Skills learned:** packaging, config management, leakage-aware evaluation, XAI basics, Docker, pytest. · **Interview benefit:** "I found and fixed a validity bug in my own pipeline" + "it's containerized and tested." · **Demo benefit:** `docker compose up` works; the heatmap is now real.

### Phase 2 — Strong résumé builders: become a system (≈3–4 weeks)
- **Build:** FastAPI inference service (#2) · Postgres + history/audit (#3) · JWT auth + RBAC (#16) · CI/CD (#12) · cloud deploy with a **live URL** (#11) · structured logging (start of #17).
- **Why this phase:** This is the jump from "ML script" to "ML product." It unlocks every backend/SDE/full-stack interview question and gives you a shareable link.
- **Time:** ~3–4 weeks · **Skills learned:** API design, schema/migrations, authn/z, pipelines, deployment. · **Interview benefit:** end-to-end architecture you can whiteboard. · **Demo benefit:** live, multi-user, with history and audit.

### Phase 3 — Standout, current differentiators (≈3–5 weeks)
- **Build:** MONAI segmentation to remove the mask dependency (#4) · LLM-drafted report (#1) · RAG over guidelines (#6) · analytics dashboard (#13).
- **Why this phase:** MONAI makes the end-to-end claim *true*; the GenAI layer makes it *current*; the dashboard makes it a *data story*. This is what people remember.
- **Time:** ~4–5 weeks · **Skills learned:** medical segmentation, LLM app design, RAG, guardrails/evals, analytics. · **Interview benefit:** the rare student project that is honest, end-to-end, *and* GenAI-native. · **Demo benefit:** upload a raw CT → auto-segment → detect → explain → AI-drafted, cited report → dashboard updates.

### Phase 4 — Advanced / production-grade (ongoing)
- **Build:** Full observability (Prometheus/Grafana/Sentry, #17) · experiment tracking + model registry + DVC (#18) · drift monitoring (#15) · React/NiiVue frontend (#10) · ONNX/perf (#20) · IaC (Terraform) + autoscaling/queue for 3D inference.
- **Why this phase:** Operability, reproducibility, and scale — the senior/MLOps signals.
- **Time:** ongoing · **Skills learned:** SRE, MLOps, advanced frontend, IaC. · **Interview benefit:** "operated," not just "shipped." · **Demo benefit:** dashboards, metrics, a polished domain-specific UI.

---

# 7. Security, Testing & Code-Quality as Mandatory Layers

For **every** major upgrade: how to test it, what security risk it introduces, how to reduce that risk, what refactor keeps it clean, and what to log/measure.

### Cross-cutting baseline (apply once, benefits all)
- **Testing baseline:** a synthetic fixture volume; deterministic seeds; separate I/O from logic so functions are unit-testable; contract tests on every API endpoint; coverage gate in CI.
- **Security baseline:** `torch.load(weights_only=True)` + model checksums; validate every upload (magic bytes, size, header sanity); secrets only via env/`pydantic-settings`; rate limiting; audit log; pseudonymize PHI; least-privilege DB user; dependency scanning (`pip-audit`/Dependabot).
- **Code-quality baseline:** one shared `inference` module (kill the 6× duplication); config object (no magic numbers/paths); type hints + `mypy`; `ruff` format/lint; pre-commit.
- **Observability baseline:** structured logs with correlation IDs (replace all `print`); a latency histogram + error counter per endpoint; Sentry on exceptions.

### Per-upgrade specifics

| Upgrade | How to test | Security risk it adds | Risk mitigation | Refactor to stay clean | Logs/metrics to add |
|---|---|---|---|---|---|
| LLM report (#1) | Golden-set (input→expected impression) eval; validator that rejects reports citing absent fields; snapshot tests | PHI sent to a third party; prompt injection via filenames/notes | Pseudonymize before sending; strip/escape user text; output schema validation; disclaimer | `reporting/llm.py` with a typed `ReportRequest`/`ReportResponse`; prompt templates in files | LLM latency, token usage/cost, validation-failure rate |
| FastAPI (#2) | Endpoint contract tests (200/4xx/5xx), schema validation tests | Larger attack surface; SSRF/path issues on upload | Pydantic validation; size/type limits; no user-controlled file paths | Thin routers → service layer → `inference` module | request count, latency p50/p95/p99, error rate |
| Postgres (#3) | Repository unit tests on a test DB; migration up/down tests | SQL injection; PHI at rest | ORM/parameterized queries; encryption at rest; least-privilege role | Repository pattern; Alembic migrations | query latency, connection-pool saturation, audit writes |
| MONAI seg (#4) | Dice/IoU on a held-out set; shape/range asserts; "mask present" invariant | Large model artifact provenance | Signed/verified weights; trusted storage | `data/segmentation.py` behind an interface | seg latency, mask-volume distribution (drift proxy) |
| Grad-CAM (#5) | Assert saliency shape/finiteness; regression test vs a saved map | Low | — | `inference/explain.py` | explain latency |
| RAG (#6) | Retrieval precision@k on labeled queries; citation-presence test | Poisoned/incorrect corpus; injection | Curated, versioned corpus; cite sources; sanitize queries | `reporting/retrieval.py`; versioned index | retrieval latency, hit rate, empty-retrieval rate |
| Security hardening (#7) | Negative tests (malicious upload rejected, oversized rejected, RCE payload blocked) | — (reduces risk) | This *is* the mitigation | central `security/` module | blocked-upload count, rate-limit hits, auth failures |
| Docker (#8) | `docker build` in CI; container smoke test | Vulnerable base image; running as root | Pinned slim base; non-root user; image scan (Trivy) | `.dockerignore`; multi-stage | build time, image size, cold-start time |
| Honest eval (#9) | The eval *is* a test; assert no train/test overlap | Low | — | `evaluation/` module; fixed split file | metrics per model version |
| React/NiiVue (#10) | Component tests; Playwright e2e (upload→result) | XSS; large-file DoS in browser | Sanitize/escape; client-side size checks; CSP | API client layer; typed DTOs | page load, viewer FPS, API error surfacing |
| Cloud deploy (#11) | Post-deploy health/smoke checks | Exposed secrets; public buckets | Secret manager; private buckets + signed URLs; HTTPS | env-specific config | uptime, deploy success rate, 5xx after deploy |
| CI/CD (#12) | The pipeline tests everything else; test the pipeline on a branch | Supply-chain (actions/deps) | Pin action SHAs; `pip-audit`; least-privilege tokens | reusable workflow steps | pipeline duration, flaky-test rate, coverage trend |
| Analytics (#13) | Query/aggregation unit tests on seeded data | PHI exposure in aggregates | Aggregate-only views; role-gate access | SQL views / a `reporting` query layer | dashboard query latency |
| pytest (#14) | (is the testing layer) | — | — | fixtures in `conftest.py` | coverage %, test duration |
| Drift (#15) | Tests on synthetic drifted vs stable data | False alerts causing alarm fatigue | Tuned thresholds; severity levels | `monitoring/drift.py` | drift scores, alert count |
| JWT/RBAC (#16) | Authz matrix tests (role × endpoint), token expiry/refresh tests | Broken access control (OWASP #1) | Server-side checks on every route; short-lived tokens + rotation | central auth dependency | login success/fail, 401/403 counts |
| Observability (#17) | Assert log structure; metric-endpoint test | Logging PHI/secrets | Redact sensitive fields; sample logs | logging config module | the golden signals |
| Experiment tracking (#18) | Smoke test logging API; registry promotion test | Artifact tampering | Checksums; access-controlled registry | training emits runs via one helper | run count, promotion events |
| ONNX/perf (#20) | Numerical-parity test (PyTorch vs ONNX within tolerance) | — | — | `inference/onnx_runtime.py` behind same interface | latency before/after, throughput |

---

# 8. Interview Value Summary (role-based)

### Role → which upgrades to lead with

| Role | Lead upgrades | Why these |
|------|---------------|-----------|
| **SDE (generalist)** | #2 FastAPI, #14 tests, #12 CI/CD, #19 refactor, #8 Docker | Proves clean code, services, testing, shipping. |
| **Backend engineer** | #2 API, #3 DB/audit, #16 auth, #7 security, scaling/queue | Service design, data modeling, authz, reliability. |
| **Full-stack engineer** | #10 React/NiiVue, #2 API, #3 DB, #11 deploy | End-to-end ownership with a real UI. |
| **AI engineer** | #1 LLM report, #6 RAG, #5 Grad-CAM, guardrails/evals | GenAI app design + trustworthiness. |
| **ML engineer** | #4 MONAI, #9 honest eval, #5 XAI, #18 tracking/registry | Modeling, validity, reproducibility, MLOps. |
| **Data analyst** | #13 analytics dashboard, #9 metrics, #15 drift | Insight from operational data; metric fluency. |
| **Data engineer** | #3 DB, #13 pipeline/dashboard, #18 DVC, ingestion | Storage, versioning, pipelines (add an orchestrator for full credit). |
| **Cloud/DevOps engineer** | #8 Docker, #12 CI/CD, #11 deploy, #17 observability, IaC | Build→ship→operate. |
| **Security-focused engineer** | #7 hardening, #16 RBAC, #1 PHI handling | File-upload + PHI + model-deserialization threat model. |

### Sample interview exchanges (per flagship upgrade)

- **MONAI segmentation (#4).** *Q:* "At inference, where does the spleen region come from?" *Strong A:* "Originally from dataset masks — which is leakage, since real scans don't ship with labels. I added a MONAI segmentation front-end so the spleen is localized automatically, then measured how mask error affects the anomaly score and added morphological cleanup. It's now genuinely end-to-end on raw CT."
- **LLM report (#1).** *Q:* "How do you prevent hallucinated findings?" *Strong A:* "Only verified numeric features go in; the model is constrained to a JSON schema; a system rule forbids stating measurements not present in the input; a post-validator rejects non-conforming output; and there's a human-review disclaimer. I track validation-failure rate as a quality metric."
- **FastAPI + scaling (#2).** *Q:* "How do you serve slow 3D inference at scale?" *Strong A:* "Stateless API workers; heavy inference goes to a Redis-backed job queue; the client gets a job id and polls or gets a webhook; workers scale horizontally; the model loads once per worker and is cached."
- **Honest evaluation (#9).** *Q:* "How do you know it generalizes?" *Strong A:* "Fixed held-out test set never used for threshold selection; threshold tuned on validation only; synthetic anomalies are a sanity check, not headline metrics; I report on real abnormal cases where available, even when the numbers are humbling."
- **Security (#7).** *Q:* "Risks of loading a `.pth`?" *Strong A:* "Pickle deserialization can execute arbitrary code; I use `weights_only=True`, verify a checksum/signature, and only load from trusted storage."

### Résumé depth each upgrade adds (sample bullets)

- "Built a **FastAPI** microservice serving a **3D CNN** medical-imaging model with async job processing, JWT/RBAC, and OpenAPI docs."
- "Added a **MONAI** segmentation front-end to remove a ground-truth-mask dependency, making anomaly detection **end-to-end on raw CT**."
- "Shipped a **GenAI report layer** (Claude + RAG over clinical guidelines) with schema-constrained, **hallucination-guarded**, source-cited output."
- "Established **leakage-free evaluation** (held-out test, validation-only thresholding) and **Grad-CAM** explainability for clinician trust."
- "Containerized with **Docker**, automated **CI/CD** (ruff/mypy/pytest, coverage gate), deployed to the cloud with object-storage-backed model artifacts."
- "Instrumented **Prometheus/Grafana + Sentry** and an **analytics dashboard** with **drift monitoring** over a Postgres results store."

### Resume one-liner for the whole project
> *SurgiVision — an end-to-end medical-AI platform: MONAI spleen segmentation → 3D-autoencoder anomaly detection → Grad-CAM explainability → LLM/RAG report drafting, served via a tested, secured FastAPI + Postgres backend with CI/CD, cloud deployment, and an analytics/observability stack.*

---

# 9. Final Recommended Next 5 Upgrades

If you do nothing else, do these — in this order. They deliver the most credibility per unit effort and convert the project from "decorated demo" to "defensible system."

1. **Code-quality + integrity foundation (#19 + #9 + the #5 heatmap fix + `weights_only` from #7).**
   *Why first:* Makes the repo run from a clean clone, removes the three integrity landmines, and makes everything after it safe. Cheap, fast, and the difference between "impressive" and "embarrassing under questioning." (~1.5 weeks)
2. **FastAPI inference service (#2) + pytest + CI/CD + Docker (#14, #12, #8).**
   *Why:* Turns scripts into a tested, containerized, continuously-built **service** — the core SDE/backend signal and the platform for everything else. (~1.5–2 weeks)
3. **Postgres persistence + history/audit (#3) + JWT/RBAC (#16).**
   *Why:* State, audit, and access control — mandatory for a medical app and the backbone of analytics. (~1.5 weeks)
4. **MONAI segmentation (#4) + Grad-CAM (#5 full).**
   *Why:* Makes the end-to-end claim *true* (no mask dependency) and the explainability *real*. This is the standout ML chapter. (~2 weeks)
5. **LLM-drafted report (#1) + a live cloud deployment (#11).**
   *Why:* The most "2026-current" differentiator plus a shareable live URL — your highest-conversion demo asset. Add RAG (#6) and the analytics dashboard (#13) next as fast follows. (~1.5–2 weeks)

---

## Appendix: at-a-glance compact summary

- **Today:** A real 3D-autoencoder anomaly detector with a polished Streamlit demo — but no API, DB, tests, CI, auth, or deployment, plus three integrity issues (fake heatmap, hidden mask dependency, leaky evaluation). *Strong model, near-absent system.*
- **Do now:** Foundation + integrity fixes → FastAPI + tests + CI + Docker → Postgres + auth.
- **Do next:** MONAI segmentation + Grad-CAM → LLM/RAG report + cloud deploy + analytics.
- **Do later:** Observability, experiment tracking/registry, drift monitoring, React/NiiVue, ONNX/perf, IaC.
- **Mandatory throughout:** security, testing, code quality.

> **Why this project becomes stronger (one line):** Because the upgrades convert a single-folder ML demo into a tested, secured, observable, end-to-end medical-AI *system* with honest evaluation, real explainability, and a current GenAI layer — which is exactly the evidence SDE, backend, full-stack, AI, ML, and data interviews are looking for.

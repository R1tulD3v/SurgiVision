# syntax=docker/dockerfile:1

# ---- Stage 1: build dependencies into a venv ----
FROM python:3.12-slim AS builder

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# Install the CPU-only PyTorch build first (smaller, no CUDA), then the rest.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: slim runtime image ----
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# System libraries required by opencv-python at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY . .
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Mount the dataset/model at runtime, e.g.:
#   docker run -p 8501:8501 \
#     -e SURGIVISION_MODELS_DIR=/models -v /host/models:/models:ro \
#     -e SURGIVISION_DATA_ROOT=/data/Task09_Spleen -v /host/data:/data:ro \
#     surgivision
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "streamlit_universal_demo.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]

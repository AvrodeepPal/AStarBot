# ---------- Stage 1: builder ----------
# Installs everything into a self-contained venv. CPU-only torch is pulled
# from the PyTorch index first so pip never downloads multi-GB CUDA wheels.
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY rag ./rag
COPY interfaces ./interfaces
COPY scripts ./scripts

RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install .

# Bake the embedding model into the image so cold starts don't download it.
# Set --build-arg PREFETCH_MODEL=false to skip (image shrinks by ~440 MB).
ARG PREFETCH_MODEL=true
ENV HF_HOME=/opt/hf-cache
RUN if [ "$PREFETCH_MODEL" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-base-en-v1.5')"; \
    fi

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=0 \
    TOKENIZERS_PARALLELISM=false

# libgomp1: OpenMP runtime needed by torch on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 astarbot

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=astarbot:astarbot /opt/hf-cache /opt/hf-cache

WORKDIR /app
COPY --chown=astarbot:astarbot . .

USER astarbot
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\",\"8000\")}/health')" || exit 1

CMD ["python", "main.py"]

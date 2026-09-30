# ---------- Stage 1: builder ----------
# Installs everything into a self-contained venv.
#
# Default (INSTALL_LOCAL_EMBED=false): slim image for EMBEDDING_BACKEND=hf.
# No torch, no sentence-transformers, no model weights.
#
# --build-arg INSTALL_LOCAL_EMBED=true: adds the [local-embed] extra (CPU-only
# torch from the PyTorch index, so pip never pulls multi-GB CUDA wheels) and
# bakes BGE into the image, for EMBEDDING_BACKEND=local on a host with the
# RAM for it (~1 GB+).
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

ARG INSTALL_LOCAL_EMBED=false
RUN pip install --upgrade pip \
    && if [ "$INSTALL_LOCAL_EMBED" = "true" ]; then \
         pip install torch --index-url https://download.pytorch.org/whl/cpu \
         && pip install ".[local-embed]"; \
       else \
         pip install .; \
       fi

# Bake the embedding model into local-embed images so cold starts don't
# download it. --build-arg PREFETCH_MODEL=false skips it (~440 MB smaller).
ARG PREFETCH_MODEL=true
ENV HF_HOME=/opt/hf-cache
RUN mkdir -p /opt/hf-cache \
    && if [ "$INSTALL_LOCAL_EMBED" = "true" ] && [ "$PREFETCH_MODEL" = "true" ]; then \
         python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-base-en-v1.5')"; \
       fi

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

ARG INSTALL_LOCAL_EMBED=false

# jemalloc replaces glibc malloc for every process in the container via
# LD_PRELOAD (no code change). It returns freed pages to the OS on a timer
# instead of holding them, which keeps RSS flat in a long-lived Python
# process: background_thread purges in the background, dirty/muzzy decay
# control how fast, and narenas:2 avoids per-core arenas a 1-vCPU box
# can't use. The library path is arch-specific, so it's symlinked to one
# fixed path that works on both amd64 (Railway) and arm64 (Apple silicon).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=0 \
    TOKENIZERS_PARALLELISM=false \
    LD_PRELOAD=/usr/local/lib/libjemalloc.so.2 \
    MALLOC_CONF=background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:0,narenas:2

# libjemalloc2: the allocator (runtime package only; -dev is for compiling).
# libgomp1: OpenMP runtime torch needs on slim images (local-embed only).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libjemalloc2 \
       $( [ "$INSTALL_LOCAL_EMBED" = "true" ] && echo libgomp1 ) \
    && ln -s "/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2" /usr/local/lib/libjemalloc.so.2 \
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

# main.py runs one supervised uvicorn worker, recycled every
# WORKER_MAX_REQUESTS requests (see main.py for why not plain uvicorn).
CMD ["python", "main.py"]

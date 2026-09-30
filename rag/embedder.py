"""Query embedding backends.

Two implementations behind one interface (`embed_query` / `aembed_query`):

    LocalBGEEmbedder   sentence-transformers in-process. Needs the
                       `[local-embed]` extra (torch); ~800 MB resident.
    HFEmbedder         Hugging Face Inference API. Needs HF_TOKEN only;
                       nothing model-sized is ever loaded into RAM.

Both return a unit-length `settings.embedding_dim` vector for the SAME model
(BAAI/bge-base-en-v1.5), so switching backends never touches the Pinecone
index. BGE is asymmetric: the query prefix is applied here, documents are
embedded without it (scripts/embed.py, which always runs locally).

`settings.embedding_backend` picks the backend ("auto" = HF when HF_TOKEN is
set). The torch import is deferred to `load_embedder`, so an "hf" process
never imports it even when it is installed.
"""

import asyncio
import logging
import math
import time

import requests

from rag.config import settings

log = logging.getLogger("astarbot.embedder")


class EmbedderError(RuntimeError):
    """Raised when a query embedding cannot be produced."""


def load_embedder():
    """Load the local SentenceTransformer. Shared by LocalBGEEmbedder and the scripts."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # slim install without the [local-embed] extra
        raise EmbedderError(
            "Local embeddings need the [local-embed] extra (pip install -e '.[local-embed]'), "
            "or set HF_TOKEN / EMBEDDING_BACKEND=hf to use the Hugging Face API."
        ) from exc

    log.info(
        "loading_embedding_model",
        extra={"model": settings.embedding_model, "device": settings.embedding_device},
    )
    model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
    # Renamed upstream; keep the old name as a fallback for older installs.
    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dim = get_dim()
    if dim != settings.embedding_dim:
        raise EmbedderError(
            f"Embedding model produces {dim}-d vectors but EMBEDDING_DIM={settings.embedding_dim}; "
            "the Pinecone index dimension must match the model."
        )
    return model


def _unit(vec: list[float]) -> list[float]:
    """L2-normalise so HF and local vectors are interchangeable for cosine search."""
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


class LocalBGEEmbedder:
    backend = "local"

    def __init__(self) -> None:
        self.model = load_embedder()

    def embed_query(self, query: str) -> list[float]:
        vec = self.model.encode(
            settings.query_prefix + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()

    async def aembed_query(self, query: str) -> list[float]:
        # CPU-bound: run in a worker thread so the event loop stays free.
        return await asyncio.to_thread(self.embed_query, query)


class HFEmbedder:
    """Hugging Face Inference API (hf-inference provider, feature-extraction).

    Free-tier accounts get a small monthly credit and are rate-limited, which
    is ample for a portfolio bot. 503 (model loading) and 429 (rate limit) are
    retried with exponential backoff; anything else fails fast. One pooled
    `requests.Session` keeps the TLS connection warm between queries.
    """

    backend = "hf"

    def __init__(self) -> None:
        if not settings.hf_token:
            raise EmbedderError("EMBEDDING_BACKEND=hf requires HF_TOKEN to be set")
        self.url = settings.hf_embedding_endpoint
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {settings.hf_token}"})

    def _parse(self, data) -> list[float]:
        # A single string input returns a flat vector; tolerate a batch of one.
        if isinstance(data, list) and data and isinstance(data[0], list):
            data = data[0]
        if not isinstance(data, list) or not data or not isinstance(data[0], int | float):
            raise EmbedderError(f"Unexpected HF response shape: {str(data)[:120]}")
        if len(data) != settings.embedding_dim:
            raise EmbedderError(f"HF returned {len(data)}-d, expected {settings.embedding_dim}")
        return _unit([float(x) for x in data])

    def embed_query(self, query: str) -> list[float]:
        payload = {"inputs": settings.query_prefix + query}
        last_err = "no attempt made"
        for attempt in range(settings.hf_max_retries):
            try:
                r = self.session.post(self.url, json=payload, timeout=settings.hf_timeout_seconds)
            except requests.RequestException as exc:
                last_err = str(exc)[:200]
                log.warning("hf_embed_fail", extra={"attempt": attempt, "err": last_err})
                time.sleep(min(2**attempt, 4))
                continue

            if r.status_code in (429, 503):
                # 503: the model is being loaded on HF's side; 429: rate limited.
                wait = min(2**attempt * (2 if r.status_code == 429 else 1), 8)
                last_err = f"HTTP {r.status_code}"
                log.warning(
                    "hf_rate_limited" if r.status_code == 429 else "hf_model_loading",
                    extra={"attempt": attempt, "wait_s": wait},
                )
                time.sleep(wait)
                continue
            if r.status_code >= 400:
                # 401/403/404 will not fix themselves; don't burn retries on them.
                raise EmbedderError(f"HF embedding HTTP {r.status_code}: {r.text[:200]}")
            return self._parse(r.json())

        raise EmbedderError(f"HF embedding failed after {settings.hf_max_retries} attempts: {last_err}")

    async def aembed_query(self, query: str) -> list[float]:
        # Blocking HTTP (with its retry sleeps) in a worker thread; callers
        # can await it alongside the guard LLM without blocking the loop.
        return await asyncio.to_thread(self.embed_query, query)


def build_embedder() -> LocalBGEEmbedder | HFEmbedder:
    """Return the embedder for the configured (resolved) backend."""
    backend = settings.resolved_embedding_backend
    log.info("embedder_backend", extra={"backend": backend, "model": settings.embedding_model})
    return HFEmbedder() if backend == "hf" else LocalBGEEmbedder()

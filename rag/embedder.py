"""Query embedding backends.

Two implementations behind one interface (`embed_query` / `aembed_query`):

    LocalBGEEmbedder     sentence-transformers in-process. Needs the
                         `[local-embed]` extra (torch); ~800 MB resident.
    CloudflareEmbedder   Cloudflare Workers AI. Needs CF_API_TOKEN and
                         CF_ACCOUNT_ID; nothing model-sized is ever loaded
                         into RAM; free tier is 10,000 requests/day.

Both return a unit-length `settings.embedding_dim` vector for the SAME model
(BAAI/bge-base-en-v1.5), so switching backends never touches the Pinecone
index. BGE is asymmetric: the query prefix is applied here, documents are
embedded without it (scripts/embed.py, which always runs locally).

`settings.embedding_backend` picks the backend ("auto" = CF when both CF_*
are set). The torch import is deferred to `load_embedder`, so a "cf" process
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
            "or set CF_API_TOKEN + CF_ACCOUNT_ID / EMBEDDING_BACKEND=cf to use Cloudflare Workers AI."
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
    """L2-normalise so API and local vectors are interchangeable for cosine search."""
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


class CloudflareEmbedder:
    """Cloudflare Workers AI (`@cf/baai/bge-base-en-v1.5`).

    Same model as the local backend. Two details keep its vectors in the
    index's space: the request asks for CLS pooling (Workers AI defaults to
    mean, sentence-transformers' BGE uses CLS), and the output is
    L2-normalised client-side. 429 and 5xx are retried with exponential
    backoff; other 4xx (bad token, wrong account ID) fail fast.

    Response: {"result": {"shape": [1, 768], "data": [[...]], "pooling": "cls"},
               "success": true, "errors": [], "messages": []}
    """

    backend = "cf"

    def __init__(self) -> None:
        if not settings.cf_api_token or not settings.cf_account_id:
            raise EmbedderError("EMBEDDING_BACKEND=cf requires CF_API_TOKEN and CF_ACCOUNT_ID")
        self.url = settings.cf_embedding_endpoint
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {settings.cf_api_token}"})

    def _parse(self, body) -> list[float]:
        if not isinstance(body, dict) or not body.get("success", False):
            errors = body.get("errors") if isinstance(body, dict) else body
            raise EmbedderError(f"Cloudflare returned success=false: {str(errors)[:200]}")
        result = body.get("result")
        # Documented shape is {"data": [[...]], "shape": [...]}; tolerate a bare list.
        rows = result.get("data") if isinstance(result, dict) else result
        if isinstance(rows, list) and rows and isinstance(rows[0], list):
            rows = rows[0]
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], int | float):
            raise EmbedderError(f"Unexpected Cloudflare response shape: {str(result)[:120]}")
        if len(rows) != settings.embedding_dim:
            raise EmbedderError(f"Cloudflare returned {len(rows)}-d, expected {settings.embedding_dim}")
        pooling = result.get("pooling") if isinstance(result, dict) else None
        if pooling not in (None, settings.cf_pooling):
            raise EmbedderError(f"Cloudflare used {pooling} pooling, expected {settings.cf_pooling}")
        return _unit([float(x) for x in rows])

    def embed_query(self, query: str) -> list[float]:
        payload = {"text": [settings.query_prefix + query], "pooling": settings.cf_pooling}
        last_err = "no attempt made"
        for attempt in range(settings.cf_max_retries):
            try:
                r = self.session.post(self.url, json=payload, timeout=settings.cf_timeout_seconds)
            except requests.RequestException as exc:
                last_err = str(exc)[:200]
                log.warning("cf_embed_fail", extra={"attempt": attempt, "err": last_err})
                time.sleep(min(2**attempt, 4))
                continue

            if r.status_code == 429 or r.status_code >= 500:
                # 429: rate limited (daily free-tier cap or burst); 5xx: transient.
                wait = min(2**attempt * (2 if r.status_code == 429 else 1), 8)
                last_err = f"HTTP {r.status_code}"
                log.warning(
                    "cf_rate_limited" if r.status_code == 429 else "cf_server_error",
                    extra={"attempt": attempt, "status": r.status_code, "wait_s": wait},
                )
                time.sleep(wait)
                continue
            if r.status_code >= 400:
                # 401/403/404 will not fix themselves; don't burn retries on them.
                raise EmbedderError(f"Cloudflare embedding HTTP {r.status_code}: {r.text[:200]}")
            return self._parse(r.json())

        raise EmbedderError(
            f"Cloudflare embedding failed after {settings.cf_max_retries} attempts: {last_err}"
        )

    async def aembed_query(self, query: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, query)


def build_embedder() -> LocalBGEEmbedder | CloudflareEmbedder:
    """Return the embedder for the configured (resolved) backend."""
    backend = settings.resolved_embedding_backend
    log.info("embedder_backend", extra={"backend": backend, "model": settings.embedding_model})
    return CloudflareEmbedder() if backend == "cf" else LocalBGEEmbedder()

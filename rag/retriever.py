"""Query embedding + Pinecone top-k retrieval, with an in-process LRU cache.

The query encoder is BAAI/bge-base-en-v1.5 (768-d), run either in-process or
via the Hugging Face Inference API (`rag.embedder`); the retriever does not
care which. BGE is asymmetric: queries carry `settings.query_prefix`,
documents do not, and the prefix lives in exactly one place (`rag.embedder`).

Two entry points share one pipeline:
    aretrieve   async; used by the engine. Raises RetrievalError when the
                embedder or Pinecone fails, so the engine can answer
                "technical snag" instead of "off-topic".
    retrieve    sync convenience wrapper; errors are logged and yield [].
"""

import asyncio
import logging
import threading
from collections import OrderedDict

from pinecone import Pinecone

from rag.config import settings
from rag.embedder import EmbedderError, build_embedder
from rag.knowledge import DEFAULT_PRIORITY

log = logging.getLogger("astarbot.retriever")


class RetrievalError(RuntimeError):
    """The embedder or Pinecone failed; distinct from "nothing relevant found"."""


def _field(obj, key, default=None):
    """Pinecone match objects behave like both attrs and dicts across SDK versions."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def cache_key(query: str) -> str:
    """Case- and whitespace-insensitive key.

    BGE-base-en-v1.5 uses an uncased tokenizer that also collapses whitespace,
    so these variants embed identically; folding them raises the hit rate at
    no cost to accuracy.
    """
    return " ".join(query.lower().split())


class RetrievalCache:
    """Small thread-safe LRU of finished retrievals.

    Values are stored as tuples of dicts and copied on the way out, so a
    caller mutating its results can never poison a later hit. Only non-empty
    results are stored: an empty list may be a transient failure.
    """

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, tuple[dict, ...]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> list[dict] | None:
        with self._lock:
            value = self._data.get(key)
            if value is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
        return [dict(r) for r in value]

    def put(self, key: str, results: list[dict]) -> None:
        if not results:
            return
        with self._lock:
            self._data[key] = tuple(dict(r) for r in results)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


class PineconeRetriever:
    def __init__(self) -> None:
        self.embedder = build_embedder()
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.pc.Index(settings.pinecone_index_name)
        size = settings.retrieval_cache_size
        self.cache: RetrievalCache | None = RetrievalCache(size) if size > 0 else None

    def _query_kwargs(self, vec: list[float]) -> dict:
        return {
            "vector": vec,
            "top_k": max(settings.fetch_k, settings.top_k),
            "include_metadata": True,
            # Vectors are only needed for MMR; skip the 10 x 768 floats otherwise.
            "include_values": settings.mmr_lambda < 1.0,
            "namespace": settings.pinecone_namespace,
        }

    def _cached(self, query: str, request_id: str) -> list[dict] | None:
        if self.cache is None:
            return None
        results = self.cache.get(cache_key(query))
        log.info(
            "retrieval_cache",
            extra={
                "request_id": request_id,
                "hit": results is not None,
                "size": len(self.cache),
                "maxsize": self.cache.maxsize,
                "hits": self.cache.hits,
                "misses": self.cache.misses,
            },
        )
        return results

    def _store(self, query: str, results: list[dict]) -> None:
        if self.cache is not None:
            self.cache.put(cache_key(query), results)

    async def aretrieve(self, query: str, request_id: str = "-") -> list[dict]:
        """Return up to top_k matches, best first. See `_postprocess` for the stages.

        Each result carries {id, title, text, links, tags, source, priority,
        score, final_score}. Raises RetrievalError on embedder/Pinecone
        failure; [] means Pinecone answered but nothing survived filtering.
        """
        cached = self._cached(query, request_id)
        if cached is not None:
            return cached
        try:
            vec = await self.embedder.aembed_query(query)
        except EmbedderError as exc:
            log.error("embed_fail", extra={"request_id": request_id, "err": str(exc)[:200]})
            raise RetrievalError(str(exc)) from exc
        try:
            # The Pinecone client is synchronous; keep it off the event loop.
            res = await asyncio.to_thread(self.index.query, **self._query_kwargs(vec))
        except Exception as exc:  # noqa: BLE001
            log.error("retrieval_fail", extra={"request_id": request_id, "err": str(exc)[:200]})
            raise RetrievalError(str(exc)) from exc
        results = self._postprocess(res, request_id)
        self._store(query, results)
        return results

    def retrieve(self, query: str, request_id: str = "-") -> list[dict]:
        """Synchronous variant for scripts and tests. Failures are logged and yield []."""
        cached = self._cached(query, request_id)
        if cached is not None:
            return cached
        try:
            vec = self.embedder.embed_query(query)
            res = self.index.query(**self._query_kwargs(vec))
        except Exception as exc:  # noqa: BLE001
            log.error("retrieval_fail", extra={"request_id": request_id, "err": str(exc)[:200]})
            return []
        results = self._postprocess(res, request_id)
        self._store(query, results)
        return results

    def _postprocess(self, res, request_id: str) -> list[dict]:
        """Stages 2-4 of retrieval; Pinecone can only do stage 1:
          1. vector search for `fetch_k` candidates (over-fetch)
          2. drop anything below `min_retrieval_score` on the RAW cosine score,
             which is the quantity the threshold was calibrated against
          3. re-rank by `cosine + priority_weight * priority`, so an
             editorially important entry wins a near-tie
          4. MMR selection down to `top_k` (`mmr_lambda` < 1.0), so the FAQ
             copy of a fact does not crowd out a different, useful entry
        """
        use_mmr = settings.mmr_lambda < 1.0
        matches = _field(res, "matches", []) or []
        candidates: list[dict] = []
        for m in matches:
            score = float(_field(m, "score", 0.0) or 0.0)
            if score < settings.min_retrieval_score:
                continue
            meta = _field(m, "metadata", {}) or {}
            priority = int(meta.get("priority", DEFAULT_PRIORITY) or DEFAULT_PRIORITY)
            candidates.append(
                {
                    "id": _field(m, "id"),
                    "title": meta.get("title", ""),
                    "text": meta.get("text", ""),
                    "links": list(meta.get("links", []) or []),
                    "tags": list(meta.get("tags", []) or []),
                    "source": meta.get("source", ""),
                    "priority": priority,
                    "score": score,
                    "final_score": score + settings.priority_weight * priority,
                    "_values": _field(m, "values", None) if use_mmr else None,
                }
            )

        candidates.sort(key=lambda r: r["final_score"], reverse=True)
        if use_mmr:
            results = _mmr_select(candidates, settings.top_k, settings.mmr_lambda)
        else:
            results = candidates[: settings.top_k]
        for c in candidates:
            c.pop("_values", None)

        log.info(
            "retrieval",
            extra={
                "request_id": request_id,
                "fetch_k": settings.fetch_k,
                "top_k": settings.top_k,
                "n_candidates": len(candidates),
                "n_results": len(results),
                "n_dropped_below_threshold": len(matches) - len(candidates),
                "top_score": results[0]["score"] if results else None,
            },
        )
        return results


def _mmr_select(candidates: list[dict], k: int, lam: float) -> list[dict]:
    """Greedy maximal marginal relevance over already-sorted candidates.

    Relevance is `final_score` (cosine + priority); redundancy is the cosine
    between candidate vectors, which Pinecone returned alongside the match.
    If any candidate lacks a vector (older SDK, `include_values` ignored) the
    function degrades to plain top-k so retrieval never fails on this step.
    """
    if len(candidates) <= k:
        return candidates
    vectors = [c.get("_values") for c in candidates]
    if any(v is None or len(v) == 0 for v in vectors):
        return candidates[:k]

    import numpy as np  # only paid for when MMR is actually on

    mat = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms == 0, 1.0, norms)
    sim = mat @ mat.T
    rel = np.asarray([c["final_score"] for c in candidates], dtype="float32")

    picked: list[int] = [0]  # the best candidate is always in
    remaining = list(range(1, len(candidates)))
    while remaining and len(picked) < k:
        redundancy = sim[np.ix_(remaining, picked)].max(axis=1)
        mmr = lam * rel[remaining] - (1.0 - lam) * redundancy
        best = remaining[int(np.argmax(mmr))]
        picked.append(best)
        remaining.remove(best)
    return [candidates[i] for i in picked]

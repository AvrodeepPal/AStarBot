"""Local BGE embeddings + Pinecone top-k retrieval.

The embedding model (BAAI/bge-base-en-v1.5, 768-d) runs in-process via
sentence-transformers: no per-query API cost, no network hop, one ~440 MB
download cached under HF_HOME.

BGE is asymmetric. Queries are embedded WITH `settings.query_prefix`;
documents (scripts/embed.py) are embedded WITHOUT it. Mixing these up
noticeably degrades recall, so the prefix lives in exactly one place here.
"""

import logging

import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from rag.config import settings
from rag.knowledge import DEFAULT_PRIORITY

log = logging.getLogger("astarbot.retriever")


def _field(obj, key, default=None):
    """Pinecone match objects behave like both attrs and dicts across SDK versions."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def load_embedder() -> SentenceTransformer:
    """Shared by the retriever and the embed script so both use one model."""
    log.info(
        "loading_embedding_model",
        extra={"model": settings.embedding_model, "device": settings.embedding_device},
    )
    model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
    # Renamed upstream; keep the old name as a fallback for older installs.
    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dim = get_dim()
    if dim != settings.embedding_dim:
        raise RuntimeError(
            f"Embedding model produces {dim}-d vectors but EMBEDDING_DIM={settings.embedding_dim}; "
            "the Pinecone index dimension must match the model."
        )
    return model


class PineconeRetriever:
    def __init__(self) -> None:
        self.embedder = load_embedder()
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.pc.Index(settings.pinecone_index_name)

    def _embed_query(self, query: str) -> list[float]:
        vec = self.embedder.encode(
            settings.query_prefix + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()

    def retrieve(self, query: str, request_id: str = "-") -> list[dict]:
        """Return up to top_k matches, best first.

        Four stages, because Pinecone can only do the first one:
          1. vector search for `fetch_k` candidates (over-fetch)
          2. drop anything below `min_retrieval_score` on the RAW cosine score,
             which is the quantity the threshold was calibrated against
          3. re-rank by `cosine + priority_weight * priority`, so an
             editorially important entry wins a near-tie
          4. MMR selection down to `top_k` (`mmr_lambda` < 1.0), so the FAQ
             copy of a fact does not crowd out a different, useful entry

        Each result carries {id, title, text, links, tags, source, priority,
        score, final_score}. Pinecone errors are logged and yield an empty
        list; the engine maps that to a refusal rather than raising.
        """
        use_mmr = settings.mmr_lambda < 1.0
        try:
            vec = self._embed_query(query)
            res = self.index.query(
                vector=vec,
                top_k=max(settings.fetch_k, settings.top_k),
                include_metadata=True,
                include_values=use_mmr,
                namespace=settings.pinecone_namespace,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("retrieval_fail", extra={"request_id": request_id, "err": str(exc)[:200]})
            return []

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

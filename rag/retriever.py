"""Local BGE embeddings + Pinecone top-k retrieval.

The embedding model (BAAI/bge-base-en-v1.5, 768-d) runs in-process via
sentence-transformers: no per-query API cost, no network hop, one ~440 MB
download cached under HF_HOME.

BGE is asymmetric. Queries are embedded WITH `settings.query_prefix`;
documents (scripts/embed.py) are embedded WITHOUT it. Mixing these up
noticeably degrades recall, so the prefix lives in exactly one place here.
"""

import logging

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
    dim = model.get_sentence_embedding_dimension()
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

        Three stages, because Pinecone can only do the first one:
          1. vector search for `fetch_k` candidates (over-fetch)
          2. drop anything below `min_retrieval_score` on the RAW cosine score,
             which is the quantity the threshold was calibrated against
          3. re-rank by `cosine + priority_weight * priority` and keep `top_k`,
             so an editorially important entry wins a near-tie

        Each result carries {id, title, text, links, tags, source, priority,
        score, final_score}. Pinecone errors are logged and yield an empty
        list; the engine maps that to a refusal rather than raising.
        """
        try:
            vec = self._embed_query(query)
            res = self.index.query(
                vector=vec,
                top_k=max(settings.fetch_k, settings.top_k),
                include_metadata=True,
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
                }
            )

        candidates.sort(key=lambda r: r["final_score"], reverse=True)
        results = candidates[: settings.top_k]

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

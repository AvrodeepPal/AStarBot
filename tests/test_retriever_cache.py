import asyncio
from unittest.mock import MagicMock

import pytest

from rag import retriever as retriever_mod
from rag.config import settings
from rag.retriever import RetrievalCache, RetrievalError, cache_key

MATCHES = {"matches": [{"id": "a", "score": 0.8, "metadata": {"text": "A", "title": "T"}}]}


class StubEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_query(self, q):
        self.calls += 1
        return [0.0] * settings.embedding_dim

    async def aembed_query(self, q):
        return self.embed_query(q)


def _retriever(monkeypatch, size=64, result=MATCHES):
    monkeypatch.setattr(settings, "retrieval_cache_size", size)
    embedder = StubEmbedder()
    index = MagicMock()
    index.query.return_value = result
    pc = MagicMock()
    pc.Index.return_value = index
    monkeypatch.setattr(retriever_mod, "build_embedder", lambda: embedder)
    monkeypatch.setattr(retriever_mod, "Pinecone", lambda api_key: pc)
    return retriever_mod.PineconeRetriever(), embedder, index


def test_repeat_query_hits_cache(monkeypatch):
    r, embedder, index = _retriever(monkeypatch)
    first = r.retrieve("What are his hobbies?")
    second = r.retrieve("What are his hobbies?")
    assert first == second
    assert index.query.call_count == 1 and embedder.calls == 1
    assert r.cache.hits == 1 and r.cache.misses == 1


def test_key_folds_case_and_whitespace(monkeypatch):
    assert cache_key("  What are   his HOBBIES? ") == "what are his hobbies?"
    r, _, index = _retriever(monkeypatch)
    r.retrieve("What are his hobbies?")
    r.retrieve("what are  his hobbies?")
    assert index.query.call_count == 1


def test_different_queries_miss(monkeypatch):
    r, _, index = _retriever(monkeypatch)
    r.retrieve("hobbies")
    r.retrieve("projects")
    assert index.query.call_count == 2


def test_size_zero_disables_cache(monkeypatch):
    r, _, index = _retriever(monkeypatch, size=0)
    assert r.cache is None
    r.retrieve("q")
    r.retrieve("q")
    assert index.query.call_count == 2


def test_empty_results_are_not_cached(monkeypatch):
    r, _, index = _retriever(monkeypatch, result={"matches": []})
    assert r.retrieve("q") == [] and r.retrieve("q") == []
    assert index.query.call_count == 2


def test_hits_are_defensive_copies(monkeypatch):
    r, _, _ = _retriever(monkeypatch)
    r.retrieve("q")[0]["text"] = "poisoned"
    assert r.retrieve("q")[0]["text"] == "A"


def test_lru_evicts_oldest():
    c = RetrievalCache(2)
    c.put("a", [{"id": "a"}])
    c.put("b", [{"id": "b"}])
    c.get("a")  # a is now most recent
    c.put("c", [{"id": "c"}])
    assert c.get("b") is None and c.get("a") is not None and len(c) == 2


def test_async_path_shares_the_cache(monkeypatch):
    r, _, index = _retriever(monkeypatch)
    asyncio.run(r.aretrieve("q"))
    r.retrieve("q")
    assert index.query.call_count == 1


def test_async_path_raises_on_infrastructure_failure(monkeypatch):
    r, _, index = _retriever(monkeypatch)
    index.query.side_effect = RuntimeError("pinecone down")
    with pytest.raises(RetrievalError):
        asyncio.run(r.aretrieve("q"))
    assert r.retrieve("q") == []  # the sync wrapper keeps its [] contract

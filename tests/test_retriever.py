from unittest.mock import MagicMock

import numpy as np

from rag import retriever as retriever_mod
from rag.config import settings


def _match(mid, score, values=None, **meta):
    m = {"id": mid, "score": score, "metadata": meta}
    if values is not None:
        m["values"] = values
    return m


def _make_retriever(matches, monkeypatch):
    embedder = MagicMock()
    embedder.encode.return_value = np.zeros(settings.embedding_dim, dtype="float32")
    index = MagicMock()
    index.query.return_value = {"matches": matches}
    pc = MagicMock()
    pc.Index.return_value = index
    monkeypatch.setattr(retriever_mod, "load_embedder", lambda: embedder)
    monkeypatch.setattr(retriever_mod, "Pinecone", lambda api_key: pc)
    return retriever_mod.PineconeRetriever(), embedder, index


def test_query_prefix_applied_and_over_fetched(monkeypatch):
    r, embedder, index = _make_retriever([], monkeypatch)
    r.retrieve("what projects?")
    assert embedder.encode.call_args.args[0] == settings.query_prefix + "what projects?"
    assert embedder.encode.call_args.kwargs["normalize_embeddings"] is True
    # Pinecone is asked for fetch_k candidates, not top_k, so we can re-rank.
    assert index.query.call_args.kwargs["top_k"] == max(settings.fetch_k, settings.top_k)
    assert index.query.call_args.kwargs["namespace"] == settings.pinecone_namespace


def test_result_shape_and_fields(monkeypatch):
    matches = [_match("a", 0.9, text="A", title="Title A", links=["u"], tags=["x"], priority=4, source="faq")]
    r, _, _ = _make_retriever(matches, monkeypatch)
    out = r.retrieve("q")[0]
    assert out["id"] == "a" and out["title"] == "Title A" and out["source"] == "faq"
    assert out["links"] == ["u"] and out["tags"] == ["x"] and out["priority"] == 4
    assert out["score"] == 0.9
    assert out["final_score"] == 0.9 + settings.priority_weight * 4


def test_missing_metadata_fields_get_defaults(monkeypatch):
    r, _, _ = _make_retriever([_match("bare", 0.6, text="T")], monkeypatch)
    out = r.retrieve("q")[0]
    assert out["title"] == "" and out["links"] == [] and out["tags"] == [] and out["source"] == ""
    assert out["priority"] == 3  # knowledge.DEFAULT_PRIORITY


def test_priority_breaks_a_near_tie(monkeypatch):
    # 0.01 apart in cosine; priority 5 vs 2 is worth 0.06 at the default weight.
    matches = [
        _match("low_prio", 0.71, text="L", priority=2),
        _match("high_prio", 0.70, text="H", priority=5),
    ]
    r, _, _ = _make_retriever(matches, monkeypatch)
    assert [m["id"] for m in r.retrieve("q")] == ["high_prio", "low_prio"]


def test_priority_cannot_overturn_a_clear_winner(monkeypatch):
    matches = [_match("far_better", 0.80, text="B", priority=2), _match("weak", 0.60, text="W", priority=5)]
    r, _, _ = _make_retriever(matches, monkeypatch)
    assert [m["id"] for m in r.retrieve("q")] == ["far_better", "weak"]


def test_priority_weight_zero_is_pure_cosine(monkeypatch):
    monkeypatch.setattr(settings, "priority_weight", 0.0)
    matches = [_match("hi_cos", 0.71, text="L", priority=2), _match("hi_prio", 0.70, text="H", priority=5)]
    r, _, _ = _make_retriever(matches, monkeypatch)
    assert [m["id"] for m in r.retrieve("q")] == ["hi_cos", "hi_prio"]


def test_results_capped_at_top_k(monkeypatch):
    matches = [_match(f"e{i}", 0.9 - i / 100, text="T") for i in range(settings.fetch_k)]
    r, _, _ = _make_retriever(matches, monkeypatch)
    assert len(r.retrieve("q")) == settings.top_k


def test_min_score_filters_on_raw_cosine(monkeypatch):
    monkeypatch.setattr(settings, "min_retrieval_score", 0.5)
    # 0.45 raw would clear 0.5 after a priority-5 boost; the filter must run first.
    matches = [_match("lo", 0.45, text="L", priority=5), _match("hi", 0.9, text="H", priority=2)]
    r, _, _ = _make_retriever(matches, monkeypatch)
    assert [m["id"] for m in r.retrieve("q")] == ["hi"]


def test_attribute_style_matches(monkeypatch):
    m = MagicMock()
    m.id, m.score, m.metadata = "x", 0.6, {"text": "T", "tags": ["t"], "priority": 3}
    r, _, _ = _make_retriever([m], monkeypatch)
    out = r.retrieve("q")
    assert len(out) == 1 and out[0]["id"] == "x" and out[0]["text"] == "T"


def test_empty_and_error_paths(monkeypatch):
    r, _, index = _make_retriever([], monkeypatch)
    assert r.retrieve("q") == []
    index.query.side_effect = RuntimeError("pinecone down")
    assert r.retrieve("q") == []


def test_mmr_skips_a_near_duplicate(monkeypatch):
    """faq-gate and self-gate say the same thing; only one should take a slot."""
    monkeypatch.setattr(settings, "top_k", 2)
    monkeypatch.setattr(settings, "mmr_lambda", 0.7)
    gate = [1.0, 0.0, 0.0]
    gate_copy = [0.99, 0.14, 0.0]  # cosine ~0.99 with `gate`
    other = [0.0, 1.0, 0.0]
    matches = [
        _match("self-gate", 0.80, values=gate, text="G", priority=3),
        _match("faq-gate", 0.79, values=gate_copy, text="G2", priority=3),
        _match("self-now", 0.70, values=other, text="N", priority=3),
    ]
    r, _, index = _make_retriever(matches, monkeypatch)
    assert [m["id"] for m in r.retrieve("q")] == ["self-gate", "self-now"]
    assert index.query.call_args.kwargs["include_values"] is True
    # the raw vector never leaks into the result dicts
    assert all("_values" not in m for m in r.retrieve("q"))


def test_mmr_disabled_at_lambda_one(monkeypatch):
    monkeypatch.setattr(settings, "top_k", 2)
    monkeypatch.setattr(settings, "mmr_lambda", 1.0)
    matches = [
        _match("a", 0.80, values=[1, 0], text="A"),
        _match("a2", 0.79, values=[1, 0], text="A2"),
        _match("b", 0.70, values=[0, 1], text="B"),
    ]
    r, _, index = _make_retriever(matches, monkeypatch)
    assert [m["id"] for m in r.retrieve("q")] == ["a", "a2"]
    assert index.query.call_args.kwargs["include_values"] is False


def test_mmr_degrades_to_top_k_without_vectors(monkeypatch):
    monkeypatch.setattr(settings, "top_k", 2)
    monkeypatch.setattr(settings, "mmr_lambda", 0.7)
    matches = [_match("a", 0.80, text="A"), _match("a2", 0.79, text="A2"), _match("b", 0.70, text="B")]
    r, _, _ = _make_retriever(matches, monkeypatch)
    assert [m["id"] for m in r.retrieve("q")] == ["a", "a2"]

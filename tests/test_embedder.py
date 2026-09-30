import math
from unittest.mock import MagicMock

import pytest
import requests

from rag import embedder as embedder_mod
from rag.config import settings
from rag.embedder import EmbedderError, HFEmbedder, LocalBGEEmbedder, build_embedder

DIM = settings.embedding_dim


def _resp(status, payload=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.text = str(payload)
    return r


@pytest.fixture
def hf(monkeypatch):
    monkeypatch.setattr(settings, "hf_token", "hf_test")
    monkeypatch.setattr(settings, "hf_max_retries", 3)
    sleeps: list[float] = []
    monkeypatch.setattr(embedder_mod.time, "sleep", sleeps.append)
    e = HFEmbedder()
    e.session = MagicMock()
    e.sleeps = sleeps
    return e


def test_build_embedder_picks_backend(monkeypatch):
    monkeypatch.setattr(settings, "hf_token", "hf_test")
    monkeypatch.setattr(settings, "embedding_backend", "hf")
    assert isinstance(build_embedder(), HFEmbedder)

    monkeypatch.setattr(embedder_mod, "load_embedder", lambda: MagicMock())
    monkeypatch.setattr(settings, "embedding_backend", "local")
    assert isinstance(build_embedder(), LocalBGEEmbedder)

    monkeypatch.setattr(settings, "embedding_backend", "auto")
    assert isinstance(build_embedder(), HFEmbedder)  # token present -> hf


def test_hf_requires_token(monkeypatch):
    monkeypatch.setattr(settings, "hf_token", "")
    with pytest.raises(EmbedderError, match="HF_TOKEN"):
        HFEmbedder()


def test_hf_applies_query_prefix_and_normalises(hf):
    hf.session.post.return_value = _resp(200, [2.0] * DIM)
    vec = hf.embed_query("what projects?")
    assert hf.session.post.call_args.kwargs["json"] == {"inputs": settings.query_prefix + "what projects?"}
    assert math.isclose(sum(x * x for x in vec), 1.0, rel_tol=1e-6)


def test_hf_accepts_batch_of_one(hf):
    hf.session.post.return_value = _resp(200, [[1.0] + [0.0] * (DIM - 1)])
    assert hf.embed_query("q")[0] == 1.0


def test_hf_retries_503_and_429_then_succeeds(hf):
    hf.session.post.side_effect = [_resp(503), _resp(429), _resp(200, [1.0] * DIM)]
    assert len(hf.embed_query("q")) == DIM
    assert hf.session.post.call_count == 3
    assert len(hf.sleeps) == 2


def test_hf_retries_network_errors(hf):
    hf.session.post.side_effect = [requests.ConnectionError("reset"), _resp(200, [1.0] * DIM)]
    assert len(hf.embed_query("q")) == DIM


def test_hf_gives_up_after_max_retries(hf):
    hf.session.post.return_value = _resp(503)
    with pytest.raises(EmbedderError, match="after 3 attempts"):
        hf.embed_query("q")
    assert hf.session.post.call_count == 3


def test_hf_auth_error_fails_fast(hf):
    hf.session.post.return_value = _resp(401, {"error": "Invalid credentials"})
    with pytest.raises(EmbedderError, match="401"):
        hf.embed_query("q")
    assert hf.session.post.call_count == 1


def test_hf_wrong_dimension_rejected(hf):
    hf.session.post.return_value = _resp(200, [1.0] * 384)
    with pytest.raises(EmbedderError, match="384-d"):
        hf.embed_query("q")


def test_local_embedder_missing_extra_is_actionable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "sentence_transformers":
            raise ImportError("no torch here")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(EmbedderError, match="local-embed"):
        embedder_mod.load_embedder()

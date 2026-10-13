import math
from unittest.mock import MagicMock

import pytest
import requests

from rag import embedder as embedder_mod
from rag.config import settings
from rag.embedder import CloudflareEmbedder, EmbedderError, LocalBGEEmbedder, build_embedder

DIM = settings.embedding_dim


def _resp(status, payload=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.text = str(payload)
    return r


@pytest.fixture
def cf(monkeypatch):
    monkeypatch.setattr(settings, "cf_api_token", "cf_test")
    monkeypatch.setattr(settings, "cf_account_id", "acct")
    monkeypatch.setattr(settings, "cf_max_retries", 3)
    sleeps: list[float] = []
    monkeypatch.setattr(embedder_mod.time, "sleep", sleeps.append)
    e = CloudflareEmbedder()
    e.session = MagicMock()
    e.sleeps = sleeps
    return e


def _cf_ok(vec, pooling="cls"):
    result = {"shape": [1, len(vec)], "data": [vec], "pooling": pooling}
    return _resp(200, {"result": result, "success": True})


def test_build_embedder_picks_backend(monkeypatch):
    monkeypatch.setattr(settings, "cf_api_token", "cf_test")
    monkeypatch.setattr(settings, "cf_account_id", "acct")
    monkeypatch.setattr(settings, "embedding_backend", "cf")
    assert isinstance(build_embedder(), CloudflareEmbedder)

    monkeypatch.setattr(embedder_mod, "load_embedder", lambda: MagicMock())
    monkeypatch.setattr(settings, "embedding_backend", "local")
    assert isinstance(build_embedder(), LocalBGEEmbedder)

    monkeypatch.setattr(settings, "embedding_backend", "auto")
    assert isinstance(build_embedder(), CloudflareEmbedder)  # creds present -> cf

    monkeypatch.setattr(settings, "cf_account_id", "")
    assert isinstance(build_embedder(), LocalBGEEmbedder)  # incomplete creds -> local


@pytest.mark.parametrize("missing", ["cf_api_token", "cf_account_id"])
def test_cf_requires_token_and_account(monkeypatch, missing):
    monkeypatch.setattr(settings, "cf_api_token", "cf_test")
    monkeypatch.setattr(settings, "cf_account_id", "acct")
    monkeypatch.setattr(settings, missing, "")
    with pytest.raises(EmbedderError, match="CF_API_TOKEN and CF_ACCOUNT_ID"):
        CloudflareEmbedder()


def test_cf_request_shape_prefix_pooling_and_normalises(cf):
    cf.session.post.return_value = _cf_ok([2.0] * DIM)
    vec = cf.embed_query("what projects?")
    call = cf.session.post.call_args
    assert call.args[0] == "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/baai/bge-base-en-v1.5"
    assert call.kwargs["json"] == {"text": [settings.query_prefix + "what projects?"], "pooling": "cls"}
    assert math.isclose(sum(x * x for x in vec), 1.0, rel_tol=1e-6)


def test_cf_normalises_known_vector(cf, monkeypatch):
    monkeypatch.setattr(settings, "embedding_dim", 3)
    cf.session.post.return_value = _cf_ok([3.0, 4.0, 0.0])
    assert cf.embed_query("q") == pytest.approx([0.6, 0.8, 0.0])


def test_cf_tolerates_bare_list_result(cf):
    cf.session.post.return_value = _resp(200, {"result": [[1.0] + [0.0] * (DIM - 1)], "success": True})
    assert cf.embed_query("q")[0] == 1.0


def test_cf_success_false_rejected(cf):
    cf.session.post.return_value = _resp(200, {"result": None, "success": False, "errors": [{"code": 5007}]})
    with pytest.raises(EmbedderError, match="success=false"):
        cf.embed_query("q")


def test_cf_wrong_pooling_rejected(cf):
    # Mean-pooled vectors are not in the index's (CLS) space; refuse them.
    cf.session.post.return_value = _cf_ok([1.0] * DIM, pooling="mean")
    with pytest.raises(EmbedderError, match="mean pooling"):
        cf.embed_query("q")


def test_cf_wrong_dimension_rejected(cf):
    cf.session.post.return_value = _cf_ok([1.0] * 384)
    with pytest.raises(EmbedderError, match="384-d"):
        cf.embed_query("q")


def test_cf_retries_429_5xx_and_network_then_succeeds(cf, monkeypatch):
    monkeypatch.setattr(settings, "cf_max_retries", 4)
    cf.session.post.side_effect = [
        _resp(429), _resp(502), requests.ConnectionError("reset"), _cf_ok([1.0] * DIM),
    ]
    assert len(cf.embed_query("q")) == DIM
    assert cf.session.post.call_count == 4
    assert len(cf.sleeps) == 3


def test_cf_gives_up_after_max_retries(cf):
    cf.session.post.return_value = _resp(503)
    with pytest.raises(EmbedderError, match="after 3 attempts"):
        cf.embed_query("q")
    assert cf.session.post.call_count == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_cf_client_errors_fail_fast(cf, status):
    cf.session.post.return_value = _resp(status, {"success": False, "errors": [{"message": "nope"}]})
    with pytest.raises(EmbedderError, match=str(status)):
        cf.embed_query("q")
    assert cf.session.post.call_count == 1
    assert cf.sleeps == []


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

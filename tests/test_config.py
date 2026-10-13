import pytest
from pydantic import ValidationError

from rag.config import Settings


def test_backend_is_normalised():
    assert Settings(embedding_backend=" CF ").embedding_backend == "cf"
    assert Settings(embedding_backend="Local").embedding_backend == "local"


@pytest.mark.parametrize("bad", ["hf", "huggingface", "cloudflare"])
def test_unknown_backend_fails_fast(bad):
    with pytest.raises(ValidationError):
        Settings(embedding_backend=bad)


def test_auto_backend_resolves_on_cf_credentials():
    cf = {"cf_api_token": "t", "cf_account_id": "a"}
    assert Settings(embedding_backend="auto", **cf).resolved_embedding_backend == "cf"
    # both CF values are needed; one alone falls through to local
    for half in ({"cf_api_token": "t", "cf_account_id": ""}, {"cf_api_token": "", "cf_account_id": "a"}):
        assert Settings(embedding_backend="auto", **half).resolved_embedding_backend == "local"
    # an explicit choice always wins
    assert Settings(embedding_backend="local", **cf).resolved_embedding_backend == "local"


def test_cf_endpoint_and_pooling():
    s = Settings(cf_account_id="abc123", cf_embedding_model="@cf/baai/bge-base-en-v1.5")
    assert s.cf_embedding_endpoint == (
        "https://api.cloudflare.com/client/v4/accounts/abc123/ai/run/@cf/baai/bge-base-en-v1.5"
    )
    assert Settings(cf_pooling=" CLS ").cf_pooling == "cls"
    with pytest.raises(ValidationError):
        Settings(cf_pooling="max")


def test_v240_latency_defaults(monkeypatch):
    for var in ("MMR_LAMBDA", "RECENT_TURNS_IN_PROMPT", "REASONING_EFFORT", "RETRIEVAL_CACHE_SIZE"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.mmr_lambda == 1.0
    assert s.recent_turns_in_prompt == 2
    assert s.reasoning_effort == "low"  # the lowest gpt-oss accepts; "" is NOT "off"
    assert s.retrieval_cache_size == 64
    assert s.enable_streaming_endpoint is True


def test_negative_cache_size_rejected():
    with pytest.raises(ValidationError):
        Settings(retrieval_cache_size=-1)

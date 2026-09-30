import pytest
from pydantic import ValidationError

from rag.config import Settings


def test_backend_is_normalised():
    assert Settings(embedding_backend=" HF ").embedding_backend == "hf"


def test_unknown_backend_fails_fast():
    with pytest.raises(ValidationError):
        Settings(embedding_backend="huggingface")


def test_auto_backend_resolves_on_hf_token():
    assert Settings(embedding_backend="auto", hf_token="").resolved_embedding_backend == "local"
    assert Settings(embedding_backend="auto", hf_token="hf_x").resolved_embedding_backend == "hf"
    # an explicit choice always wins
    assert Settings(embedding_backend="local", hf_token="hf_x").resolved_embedding_backend == "local"


def test_hf_endpoint_derived_from_model_unless_overridden():
    s = Settings(embedding_model="BAAI/bge-base-en-v1.5", hf_embedding_url="")
    assert s.hf_embedding_endpoint == (
        "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5/pipeline/feature-extraction"
    )
    assert Settings(hf_embedding_url="https://example.test/e").hf_embedding_endpoint == "https://example.test/e"


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

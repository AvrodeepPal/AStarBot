from rag.config import settings
from rag.llm import LLMChain


def test_guard_probability_parsing(monkeypatch):
    monkeypatch.setattr(settings, "guard_threshold", 0.5)
    assert LLMChain.parse_guard_output("0.0012") is True
    assert LLMChain.parse_guard_output("0.97") is False
    assert LLMChain.parse_guard_output("1") is False


def test_guard_label_parsing():
    assert LLMChain.parse_guard_output("safe") is True
    assert LLMChain.parse_guard_output("UNSAFE") is False
    assert LLMChain.parse_guard_output("jailbreak detected") is False
    assert LLMChain.parse_guard_output("") is True  # fail-open on garbage

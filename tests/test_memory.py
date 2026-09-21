from rag.config import settings
from rag.memory import summarize_conversation
from tests.conftest import FakeLLM


def test_returns_previous_when_no_messages():
    assert summarize_conversation("prev", [], FakeLLM()) == "prev"


def test_returns_previous_on_llm_failure():
    llm = FakeLLM(summary=None)
    assert summarize_conversation("prev", [{"role": "user", "content": "hi"}], llm) == "prev"


def test_clips_to_max_lines_and_strips_controls():
    llm = FakeLLM(summary="\n".join(f"line\x00 {i}" for i in range(10)))
    out = summarize_conversation(None, [{"role": "user", "content": "hi"}], llm)
    assert out.count("\n") == settings.summary_max_lines - 1
    assert "\x00" not in out


def test_transcript_includes_previous_summary_and_turns():
    llm = FakeLLM()
    summarize_conversation("earlier stuff", [{"role": "user", "content": "hello"}], llm)
    user_turn = llm.summary_calls[0][1][1]
    assert "earlier stuff" in user_turn and "USER: hello" in user_turn

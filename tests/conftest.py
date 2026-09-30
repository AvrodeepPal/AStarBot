"""Shared fixtures. No network: Pinecone, Groq and the embedding model are stubbed."""

import os

# Must run before any `rag.*` import: Settings() is built at import time.
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("ENABLE_PROMPT_GUARD_LLM", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")
# Tests never reach a real embedder; pin the backend so a developer's HF_TOKEN
# in .env can't change which class build_embedder() returns.
os.environ.setdefault("EMBEDDING_BACKEND", "local")
os.environ.setdefault("HF_TOKEN", "")

import pytest  # noqa: E402


class FakeRetriever:
    """Returns canned matches; records the last query for assertions."""

    def __init__(self, results=None):
        self.results = results if results is not None else [
            {
                "id": "self-intro",
                "source": "self",
                "title": "Introduction",
                "text": "I'm Avrodeep Pal, an AI Systems Engineer based in Barrackpore.",
                "links": ["https://github.com/AvrodeepPal"],
                "tags": ["intro"],
                "priority": 5,
                "score": 0.82,
                "final_score": 0.92,
            },
            {
                "id": "prog-high-level",
                "source": "studies",
                "title": "High-level languages",
                "text": "Python, Java and SQL day to day.",
                "links": [],
                "tags": ["skills"],
                "priority": 4,
                "score": 0.71,
                "final_score": 0.79,
            },
        ]
        self.last_query = None

    def retrieve(self, query, request_id="-"):
        self.last_query = query
        return list(self.results)

    async def aretrieve(self, query, request_id="-"):
        # Looked up at call time so tests can monkeypatch `retrieve`.
        return self.retrieve(query, request_id)


class FakeLLM:
    """Scripted LLM chain. `answers` is consumed in order by invoke_chat."""

    def __init__(self, answers=None, safe=True, summary="Visitor asked about education.\nTone: professional"):
        self.answers = list(answers or ["Avrodeep is an AI Systems Engineer at GenAIus."])
        self.safe = safe
        self.summary = summary
        self.chat_calls: list = []
        self.summary_calls: list = []

    def check_safe(self, text, request_id="-"):
        return self.safe

    def invoke_chat(self, messages, request_id="-"):
        self.chat_calls.append(messages)
        if not self.answers:
            return None, -1
        return self.answers.pop(0), 0

    def invoke_summarize(self, messages, request_id="-"):
        self.summary_calls.append(messages)
        return self.summary

    # Async surface used by the engine; delegates to the scripted sync methods.

    async def acheck_safe(self, text, request_id="-"):
        return self.check_safe(text, request_id)

    async def ainvoke_chat(self, messages, request_id="-"):
        return self.invoke_chat(messages, request_id)

    async def ainvoke_summarize(self, messages, request_id="-"):
        return self.invoke_summarize(messages, request_id)

    async def astream_chat(self, messages, request_id="-"):
        """Stream the next scripted answer a few characters at a time."""
        answer, tier = self.invoke_chat(messages, request_id)
        if answer is None:
            return
        for i in range(0, len(answer), 7):
            yield tier, answer[i : i + 7]


@pytest.fixture
def fake_engine(monkeypatch):
    """A real RAGEngine with retriever + LLM swapped for fakes."""
    from rag import engine as engine_mod

    retriever = FakeRetriever()
    llm = FakeLLM()
    monkeypatch.setattr(engine_mod, "PineconeRetriever", lambda: retriever)
    monkeypatch.setattr(engine_mod, "LLMChain", lambda: llm)
    eng = engine_mod.RAGEngine()
    eng._fake_retriever = retriever
    eng._fake_llm = llm
    return eng

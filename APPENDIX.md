# AStarBot: Technical Appendix

This is the reference companion to the [README](README.md). It covers the platforms AStarBot runs on, the repository layout, every Make target and environment variable, the API contract, notable code paths, the test suite, and deployment.

App version `2.4.1` · Prompt version `v2.4.0` · Python ≥ 3.11

---

## Contents

1. [Platforms and services](#1-platforms-and-services)
2. [Directory structure](#2-directory-structure)
3. [Makefile reference](#3-makefile-reference)
4. [Configuration](#4-configuration)
5. [API reference](#5-api-reference)
6. [Pipeline internals](#6-pipeline-internals)
7. [Test suite](#7-test-suite)
8. [Knowledge base and indexing](#8-knowledge-base-and-indexing)
9. [Deployment](#9-deployment)
10. [Logging and observability](#10-logging-and-observability)
11. [Design notes](#11-design-notes)
12. [Troubleshooting and rollback levers](#12-troubleshooting-and-rollback-levers)

---

## 1. Platforms and services

| Service | Role | Notes |
|---|---|---|
| **Pinecone** (serverless, AWS `us-east-1`) | Vector store | Index `astarbot`, 768-d, cosine, namespace `astarbot` |
| **Groq** | LLM inference | Guard, primary, fallback and summarizer tiers, all env-configurable |
| **Cloudflare Workers AI** | Query embeddings (`cf` backend) | `@cf/baai/bge-base-en-v1.5`, free tier of 10,000 requests/day |
| **sentence-transformers** | Query embeddings (`local` backend), document indexing, calibration | `BAAI/bge-base-en-v1.5`, about 440 MB of weights, about 1 GB RSS with torch |
| **Render** | Hosts the API | Docker web service, `PORT` injected, free instances spin down after 15 min idle |
| **Streamlit Community Cloud** | Hosts the beta UI | Thin HTTP client, 1 GB tier |
| **GitHub Codespaces / devcontainer** | Development environment | Python 3.12 Bookworm image |

| Tier | Default model | Temperature | Purpose |
|---|---|---|---|
| guard | `meta-llama/llama-prompt-guard-2-86m` | n/a | Jailbreak probability on the raw question. Fails open. |
| primary | `openai/gpt-oss-20b` | 0.4 | Grounded answer, `reasoning_effort=low` |
| fallback | `openai/gpt-oss-120b` | 0.4 | Used on any primary error or empty completion |
| summarizer | `openai/gpt-oss-20b` | 0.0 | Memory summary of 3 to 5 lines |

### Python dependencies (`pyproject.toml`)

| Group | Packages |
|---|---|
| core | `python-dotenv`, `pydantic`, `pydantic-settings`, `fastapi`, `uvicorn[standard]>=0.30`, `pinecone`, `langchain-core`, `langchain-groq`, `requests`, `numpy`, `tqdm` |
| `[local-embed]` | `sentence-transformers`, `torch` (CPU wheels via the PyTorch index) |
| `[ui]` | `streamlit` |
| `[dev]` | `pytest`, `httpx`, `ruff` |

The core install is about 114 MB with no torch. The full development environment is about 1.4 GB.

---

## 2. Directory structure

```
AStarBot/
├── main.py                     Uvicorn launcher: one worker under a Multiprocess supervisor
├── rag/                        Core library (no web or UI imports)
│   ├── __init__.py             __version__
│   ├── config.py               Settings: every env var, defaults, validators, derived properties
│   ├── log.py                  JSON-lines logging formatter
│   ├── guardrails.py           Input sanitisation, injection regex, output and stream sanitisers
│   ├── prompt.py               8-block versioned prompt, refusal constants, message builders
│   ├── knowledge.py            Entry schema → embedding text / Pinecone metadata / context block
│   ├── followup.py             Follow-up detection → standalone retrieval query
│   ├── embedder.py             LocalBGEEmbedder, CloudflareEmbedder, build_embedder()
│   ├── retriever.py            Pinecone query, threshold, priority re-rank, MMR, LRU cache
│   ├── llm.py                  LLMChain: guard / primary / fallback / summarizer, streaming
│   ├── memory.py               Stateless conversation summarisation (sync and async)
│   └── engine.py               RAGEngine: the 8-step orchestrator
├── interfaces/
│   ├── api.py                  FastAPI app: /chat, /chat/stream, /summarize, /health, /version
│   ├── streamlit_app.py        Beta UI: HTTP client with streaming, warm-up, /summarize
│   ├── cli.py                  Terminal client running the engine in-process
│   └── session.py              trim_window(): the client memory policy, shared by all clients
├── scripts/
│   ├── embed.py                Rebuild the Pinecone namespace from data/*.json (--dry-run)
│   ├── calibrate.py            On-topic vs off-topic score distributions (offline)
│   └── check_embedder.py       Cloudflare vs local BGE cosine parity check
├── data/                       Knowledge base: self, academics, experience, projects,
│                               studies, personality, faq (*.json, about 99 entries)
├── tests/                      14 test modules plus conftest.py (all offline)
├── .devcontainer/              Codespaces / VS Code container definition
├── Dockerfile                  Multi-stage build: slim (default) or local-embed
├── .dockerignore               Keeps tests, docs, caches and .env out of the image
├── Makefile                    Every common command (see §3)
├── pyproject.toml              Package metadata, extras, ruff and pytest config
├── requirements.txt            `-e .[ui]` shim for Streamlit Cloud
├── .env.example                Annotated template for every setting
└── APPENDIX.md                 This file
```

Dependency direction runs one way: `interfaces/` → `rag/engine.py` → the other `rag/` modules → `rag/config.py`. `scripts/` reuse `rag.knowledge` and `rag.embedder` so documents are indexed with the same rules that queries are embedded with.

---

## 3. Makefile reference

`PYTHON` can be overridden (`make test PYTHON=.venv/bin/python`). `TORCH_CPU` adds the PyTorch CPU wheel index so pip never pulls CUDA builds.

| Target | Command | Use it for |
|---|---|---|
| `install` | `pip install -e ".[ui]"` | API and UI with the Cloudflare backend, no torch |
| `install-local` | `pip install $(TORCH_CPU) -e ".[ui,local-embed]"` | Adds local BGE for `EMBEDDING_BACKEND=local` |
| `dev` | `pip install $(TORCH_CPU) -e ".[dev,ui,local-embed]"` | Everything: tests, lint, embed, calibrate |
| `api` | `python main.py` | Serve the API on `HOST:PORT` (default `0.0.0.0:8000`) |
| `ui` | `streamlit run interfaces/streamlit_app.py` | Streamlit client (port 8501) |
| `cli` | `python -m interfaces.cli` | Terminal chat with `/debug`, `/summary`, `/clear` |
| `embed` | `python -m scripts.embed` | Rebuild the Pinecone index |
| `embed-dry` | `python -m scripts.embed --dry-run` | Parse and embed locally without touching Pinecone |
| `calibrate` | `python -m scripts.calibrate` | Check whether `MIN_RETRIEVAL_SCORE` can separate topics |
| `check-embed` | `python -m scripts.check_embedder` | Cloudflare vs local vectors (cosine ≥ 0.999) |
| `test` | `pytest -v` | Full offline suite |
| `lint` | `ruff check .` | Lint (`E`, `F`, `I`, `B`, `UP`, line length 110) |
| `docker-build` | `docker build -t astarbot:2.4.1 .` | Slim image |
| `docker-build-local` | `docker build --build-arg INSTALL_LOCAL_EMBED=true -t astarbot:2.4.1-local .` | Image with torch and BGE baked in |
| `docker-run` | `docker run --rm -p 8000:8000 --env-file .env astarbot:2.4.1` | Run the slim image locally |

A typical first session:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
cp .env.example .env              # fill in the keys
make dev
make test && make lint
make embed-dry                    # sanity-check the knowledge base
make embed                        # write to Pinecone
make calibrate                    # optional: inspect score separation
make check-embed                  # only if using the Cloudflare backend
make api                          # in one terminal
make ui                           # in another; set ASTARBOT_API_URL=http://localhost:8000
```

---

## 4. Configuration

All settings live in `rag/config.py` as one `pydantic-settings` class. Values come from the process environment first, then from `.env`. Field names map to upper-cased variables (`top_k` ← `TOP_K`). `.env.example` documents every variable.

| Group | Variables (defaults) |
|---|---|
| Pinecone | `PINECONE_API_KEY`, `PINECONE_INDEX_NAME=astarbot`, `PINECONE_NAMESPACE=astarbot` |
| Embedding | `EMBEDDING_BACKEND=auto`, `EMBEDDING_MODEL=BAAI/bge-base-en-v1.5`, `EMBEDDING_DIM=768`, `EMBEDDING_DEVICE=cpu`, `QUERY_PREFIX` |
| Cloudflare | `CF_API_TOKEN`, `CF_ACCOUNT_ID`, `CF_EMBEDDING_MODEL=@cf/baai/bge-base-en-v1.5`, `CF_POOLING=cls`, `CF_TIMEOUT_SECONDS=15`, `CF_MAX_RETRIES=3` (1..6) |
| Retrieval | `TOP_K=5`, `FETCH_K=10`, `PRIORITY_WEIGHT=0.02`, `MIN_RETRIEVAL_SCORE=0.0`, `MMR_LAMBDA=1.0`, `RETRIEVAL_CACHE_SIZE=64` |
| Groq | `GROQ_API_KEY`, `GUARD_MODEL`, `PRIMARY_LLM_MODEL`, `FALLBACK_LLM_MODEL`, `SUMMARIZER_LLM_MODEL`, `TEMPERATURE=0.4`, `SUMMARIZER_TEMPERATURE=0.0`, `MAX_ANSWER_TOKENS=400`, `REASONING_EFFORT=low`, `GUARD_THRESHOLD=0.5`, `LLM_TIMEOUT_SECONDS=30` |
| Memory | `ENABLE_SUMMARY=true`, `SUMMARY_TRIGGER_AFTER=10`, `RECENT_TURNS_IN_PROMPT=2`, `SUMMARY_MAX_LINES=5`, `MAX_RECENT_MESSAGES=12`, `MESSAGES_AFTER_SUMMARY=4` |
| Guardrails | `MAX_QUESTION_CHARS=500`, `MAX_MESSAGE_CHARS=2000`, `MAX_SUMMARY_CHARS=2000`, `MAX_MESSAGES_SENT=20`, `MAX_ANSWER_CHARS=2000`, `ENABLE_PROMPT_GUARD_LLM=true` |
| API | `ENABLE_STREAMING_ENDPOINT=true`, `WORKER_MAX_REQUESTS=500`, `FRONTEND_ORIGIN` (comma-separated, or `*`) |
| Runtime | `HOST=0.0.0.0`, `PORT=8000`, `LOG_LEVEL=INFO`, `DEBUG=false` |
| Streamlit only | `ASTARBOT_API_URL` (read by `streamlit_app.py`, not by `Settings`) |

Validators fail at startup instead of degrading quietly:

```python
@field_validator("embedding_backend")
@classmethod
def _check_backend(cls, v: str) -> str:
    # A typo silently falling back to "local" would load torch on a 512 MB
    # container, the exact thing "cf" exists to avoid.
    v = v.strip().lower()
    if v not in {"auto", "cf", "local"}:
        raise ValueError("EMBEDDING_BACKEND must be 'auto', 'cf' or 'local'")
    return v

@property
def resolved_embedding_backend(self) -> str:
    if self.embedding_backend == "auto":
        return "cf" if self.cf_api_token and self.cf_account_id else "local"
    return self.embedding_backend
```

> **Gotcha:** keep `CF_API_TOKEN` and `CF_ACCOUNT_ID` empty unless you mean to use Cloudflare. With `auto`, any non-empty pair, even placeholder values, switches the backend.

> **Gotcha:** `REASONING_EFFORT=` (empty) does **not** turn reasoning off. gpt-oss accepts only `low`, `medium` or `high`. An empty value omits the parameter, so Groq uses its default (`medium`), which is slower and can use up `MAX_ANSWER_TOKENS` before any visible text is produced.

---

## 5. API reference

### Schemas (`interfaces/api.py`)

```python
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=settings.max_message_chars)

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=settings.max_question_chars)
    recent_messages: list[Message] = Field(default_factory=list, max_length=settings.max_messages_sent)
    summary: str | None = Field(default=None, max_length=settings.max_summary_chars)

class ChatResponse(BaseModel):
    answer: str
    updated_summary: str | None = None
```

| Method | Path | Request | Response |
|---|---|---|---|
| `POST` | `/chat` | `ChatRequest` | `ChatResponse` (JSON) |
| `POST` | `/chat/stream` | `ChatRequest` | `text/plain; charset=utf-8` raw text, `Cache-Control: no-cache`, `X-Accel-Buffering: no`. Returns 404 when disabled. |
| `POST` | `/summarize` | `{recent_messages, summary?}` | `{updated_summary}`. Makes no model call below `SUMMARY_TRIGGER_AFTER`. |
| `GET` | `/health` | n/a | `{"status": "ok"}` |
| `GET` | `/version` | n/a | Versions, models, `embedding_backend`, `top_k`, `retrieval_cache_size`, `streaming_enabled`, `initial_message`, `client_window{…}` |

Schema violations return **422** before the request reaches the engine. Every response carries `X-Request-Id`: the incoming header if the client sent one, otherwise an 8-character hex id. The same id appears on the engine's log lines.

### Examples

```bash
# Blocking
curl -s localhost:8000/chat -H 'Content-Type: application/json' -d '{
  "question": "How did he do it?",
  "recent_messages": [
    {"role": "user", "content": "Tell me about the credit risk project"},
    {"role": "assistant", "content": "…"}
  ],
  "summary": null
}' | jq

# Streaming (-N disables curl buffering)
curl -N localhost:8000/chat/stream -H 'Content-Type: application/json' \
  -d '{"question": "What are his hobbies?"}'

# Client policy and backend info
curl -s localhost:8000/version | jq '.client_window, .embedding_backend'
```

### Stateless memory contract

The server stores nothing, so every client must apply the same window policy. The reference implementation is `interfaces/session.py`:

```python
def trim_window(messages, previous_summary, new_summary):
    turns = [m for m in messages if m["role"] in {"user", "assistant"}]
    if new_summary and new_summary != previous_summary:
        return turns[-settings.messages_after_summary:]   # 4
    return turns[-settings.max_recent_messages:]          # 12
```

1. Start with `recent_messages = [initial assistant message]` and `summary = null`.
2. After each answer, append the user turn and the assistant turn.
3. Trim the window with the rule above.
4. Send `updated_summary` back as `summary` on the next request.

**Streaming clients** receive no `updated_summary`. Once the window, including the new exchange, reaches `client_window.summary_trigger_after` messages, they call `POST /summarize` *after* rendering the answer and then apply the same trim rule. `interfaces/streamlit_app.py` follows this pattern and falls back to `/chat` when `/version` reports `streaming_enabled: false` or `/chat/stream` returns 404.

---

## 6. Pipeline internals

### 6.1 Orchestration and concurrency (`rag/engine.py`)

Steps 1 to 5 are shared by `achat` and `astream_chat` in `_prepare()`. The guard and retrieval run concurrently because neither reads the other's output:

```python
retrieval = self.retriever.aretrieve(retrieval_q, request_id)
if settings.enable_prompt_guard_llm:
    safe, contexts = await asyncio.gather(
        self.llm.acheck_safe(q, request_id), retrieval, return_exceptions=True
    )
    if safe is False:                       # unsafe wins, even if retrieval failed
        return _Plan(summ, early_answer=REFUSAL_UNSAFE, outcome="injection_guard")
    if isinstance(contexts, BaseException):
        if not isinstance(contexts, RetrievalError):
            raise contexts
        return _Plan(summ, early_answer=STATIC_FALLBACK, outcome="retrieval_failed")
```

`RAGEngine.chat()` is a synchronous wrapper for the CLI. It reuses one persistent `asyncio.Runner`, because calling `asyncio.run()` on every turn would strand the clients' connection pools on closed loops. `achat` never raises: any unhandled error is logged and answered with `STATIC_FALLBACK`.

**Outcomes** recorded in `chat_done`: `answered`, `input_rejected`, `injection_regex`, `injection_guard`, `retrieval_failed`, `no_context`, `llm_failed`, `empty_answer`.

### 6.2 Input guardrails (`rag/guardrails.py`)

- Strips control characters and ANSI escapes but keeps tabs and newlines. Truncates at a word boundary.
- Bounds and whitelists history roles. Validates the summary.
- A regex screen catches common injection phrasings before any model call:

```python
INJECTION_PATTERNS = (
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|preceding)\s+(instructions?|prompts?|rules?)",
    r"\byou\s+are\s+now\b",
    r"\bpretend\s+(to\s+be|you\s+are)\b",
    r"\bsystem\s*prompt\b",
    r"\bdeveloper\s+mode\b",
    # … 14 patterns in total, compiled case-insensitively into one regex
)
```

### 6.3 Follow-up rewrite (`rag/followup.py`)

Pronouns such as "it" or "that one" and continuation openers such as "and…" or "tell me more" mark a question as a follow-up. For **retrieval only**, the previous user question is prepended so the vector search keeps the topic. The model still answers the question as typed.

```python
_REFERENT_RE = re.compile(
    r"\b(it|its|that|this|those|these|them|the same|that one|this one|the project|the paper)\b", re.I)

def standalone_query(question, recent_messages):
    if not is_follow_up(question):
        return question
    previous = last_user_turn(recent_messages)
    if not previous or previous.strip().lower() == question.strip().lower():
        return question
    return f"{previous} {question}"
```

"he" and "his" are left out on purpose, since nearly every question about the subject contains them.

### 6.4 Embeddings (`rag/embedder.py`)

Both backends return a unit-length 768-d vector for the same model, so switching between them never requires reindexing.

```python
# CloudflareEmbedder.embed_query (abridged)
payload = {"text": [settings.query_prefix + query], "pooling": settings.cf_pooling}
r = self._session.post(settings.cf_embedding_endpoint, json=payload, timeout=settings.cf_timeout_seconds)
if r.status_code == 429 or r.status_code >= 500:
    wait = min(2**attempt * (2 if r.status_code == 429 else 1), 8)   # 429 backs off twice as hard
    ...
if r.status_code >= 400:
    raise EmbedderError(...)                                         # other 4xx fail fast
return _unit(self._parse(r.json()))                                  # L2-normalise client-side
```

- `pooling: "cls"` is required. Workers AI defaults to mean pooling, but the index was built with sentence-transformers' BGE, which uses CLS pooling. Mean-pooled vectors land in a different space and retrieval degrades without raising any error.
- The response is rejected on `success: false`, a wrong dimension, or a pooling mode that doesn't match the request.
- `LocalBGEEmbedder` imports `sentence_transformers` lazily, so a `cf` process never loads torch even when torch is installed.
- BGE is asymmetric. `QUERY_PREFIX` is added to queries only, and documents are embedded without it.

### 6.5 Retrieval (`rag/retriever.py`)

| Stage | What happens |
|---|---|
| 0. Cache | LRU keyed on `" ".join(query.lower().split())`. Stores non-empty results only and returns a defensive copy on each hit. |
| 1. Search | Pinecone returns `FETCH_K=10` candidates by cosine similarity |
| 2. Threshold | Drop candidates whose **raw** cosine is below `MIN_RETRIEVAL_SCORE` (off by default; see §8) |
| 3. Re-rank | `final = cosine + PRIORITY_WEIGHT × priority` (0.02 × 2..5) breaks near-ties without overturning a clear winner |
| 4. MMR | Optional (`MMR_LAMBDA < 1.0`): `λ·relevance − (1−λ)·max_sim_to_picked`, down to `TOP_K=5` |

```python
def _mmr_select(candidates, k, lam):
    ...
    picked, remaining = [0], list(range(1, len(candidates)))
    while remaining and len(picked) < k:
        redundancy = sim[np.ix_(remaining, picked)].max(axis=1)
        mmr = lam * rel[remaining] - (1.0 - lam) * redundancy
        best = remaining[int(np.argmax(mmr))]
        picked.append(best); remaining.remove(best)
    return [candidates[i] for i in picked]
```

MMR falls back to plain top-k if Pinecone returns no vectors. numpy is imported only when MMR is on. Infrastructure failures raise `RetrievalError`, which produces the "technical snag" answer rather than the off-topic refusal.

### 6.6 Prompt (`rag/prompt.py`)

Eight blocks always appear in the same order:

1. identity
2. scope
3. grounding rules
4. domain discipline
5. injection defence
6. refusal guidance
7. style and length
8. summary, recent exchange, context and question

Blocks 1 to 7 form the **system** turn. Block 8, which holds everything untrusted, goes in the **user** turn, and the recent exchange appears there as `ROLE: text` lines. A forged assistant message therefore never gains the authority of a real one. Refusal strings are module constants (`REFUSAL_OFFTOPIC`, `REFUSAL_UNSAFE`, `REFUSAL_JAILBREAK`, `STATIC_FALLBACK`, plus three private-question variants that end in the same contact pointer), so guardrails and the model produce identical text. Bump `PROMPT_VERSION` after any change to the wording.

### 6.7 LLM chain (`rag/llm.py`)

```python
def _make_llm(model, temperature, max_tokens, reasoning) -> ChatGroq:
    kwargs = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
              "api_key": settings.groq_api_key, "timeout": settings.llm_timeout_seconds,
              "max_retries": 0}  # the fallback tier is the retry policy
    if reasoning and settings.reasoning_effort and "reasoning_effort" in ChatGroq.model_fields:
        kwargs["reasoning_effort"] = settings.reasoning_effort
    return ChatGroq(**kwargs)
```

- **Guard parsing.** The guard returns either a probability string or a label. Anything that can't be parsed counts as safe (fail-open), and the failure is logged at `ERROR` as `guard_fail_open`.
- **Blocking.** The chain tries primary, then fallback. An empty completion is treated as a failure, because it usually means reasoning used up the token budget.
- **Streaming.** A tier that fails *before its first token* hands over to the next tier. A tier that breaks *after* sending tokens stops the stream (`llm_stream_broken`) so a second answer is never spliced onto the first. Chunks are kept unstripped so the spaces between words survive.

### 6.8 Output guardrails while streaming

`StreamSanitizer` applies the same rules as `sanitize_answer`, incrementally:

- It holds back the last *N* characters, where *N* is the length of the longest leak marker (about 44). This lets it remove a marker split across chunks, such as `"###"` + `" SCO"` + `"PE"`.
- An unclosed `<think>` holds back everything after it and is dropped at the end of the stream.
- It enforces `MAX_ANSWER_CHARS` with a word-boundary cut and `…`.
- If nothing survives, the engine yields `STATIC_FALLBACK`.

---

## 7. Test suite

The suite has **163 tests in 14 modules**, takes under a second, and needs no network. `tests/conftest.py` sets the environment *before* any `rag.*` import, because `Settings()` is built at import time:

```python
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("ENABLE_PROMPT_GUARD_LLM", "false")
os.environ.setdefault("EMBEDDING_BACKEND", "local")   # a developer's CF_* in .env can't leak in
os.environ.setdefault("CF_API_TOKEN", "")
os.environ.setdefault("CF_ACCOUNT_ID", "")
```

It also provides `FakeRetriever` and `FakeLLM` doubles with scripted sync and async methods.

| Module | Tests | Covers |
|---|---:|---|
| `test_api.py` | 16 | All routes, `X-Request-Id` echo, 422 limits, stream content type, streaming toggle, `/summarize` trigger |
| `test_config.py` | 8 | Backend normalisation and validation, `auto` resolution, CF endpoint and pooling, defaults |
| `test_embed.py` | 12 | Data loading, duplicate ids, `--dry-run`, abort-before-delete, dimension guard |
| `test_embedder.py` | 16 | Backend selection, request shape, normalisation, 429/5xx retries, fast-fail on 4xx, pooling and dimension checks |
| `test_engine.py` | 13 | Happy path, every refusal path, summary trigger, follow-up rewrite, history as data |
| `test_followup.py` | 5 | Referent and continuation detection, no double prepend |
| `test_guardrails.py` | 25 | Sanitisation, truncation, role whitelist, injection patterns, leak and `<think>` stripping |
| `test_knowledge.py` | 9 | Embedding text vs context block, metadata, validation of the real data files |
| `test_llm.py` | 2 | Guard probability and label parsing |
| `test_memory.py` | 4 | Summary fallbacks, line clipping, transcript format |
| `test_prompt.py` | 12 | Block order, refusal templates, untrusted data kept out of the system turn |
| `test_retriever.py` | 13 | Query prefix, priority re-rank bounds, threshold, MMR and its degradation |
| `test_retriever_cache.py` | 9 | Hit/miss, key folding, LRU eviction, defensive copies, async/sync sharing |
| `test_stream.py` | 19 | Split-marker sanitising, stream fallbacks, no splice, guard ∥ retrieval timing |

```bash
make test                                   # whole suite, verbose
pytest tests/test_stream.py -q              # one module
pytest -k "mmr or cache" -q                 # by keyword
pytest tests/test_engine.py::test_happy_path
make lint                                   # ruff: E, F, I, B, UP
```

The concurrency test pins the gather behaviour: two 0.2 s fake steps must finish in under 0.35 s.

---

## 8. Knowledge base and indexing

### Entry schema (`rag/knowledge.py`)

```jsonc
{
  "meta": { "source": "faq", "last_updated": "YYYY-MM-DD" },
  "entries": [
    {
      "id": "faq-example",            // required, unique across all files
      "title": "Short topic title",
      "question": "Literal FAQ question (optional)",
      "text": "The fact itself.",      // required
      "links": ["https://…"],
      "tags": ["topic"],
      "aliases": ["phrasings visitors actually type"],
      "priority": 3,                    // 2..5, default 3
      "last_verified": "YYYY-MM-DD"
    }
  ]
}
```

Two derived strings are deliberately different:

| Function | Content | Used for |
|---|---|---|
| `embedding_text()` | title + question + text + aliases | The vector, so aliases help retrieval |
| `context_block()` | title + text + `Links:` | What the LLM sees: no ids, tags or aliases |

Pinecone metadata holds `source, text, title, question, links, tags, priority, last_verified`.

### Withheld facts

A fact that is deliberately kept private but gets asked often is written as an entry that says so, rather than left out. `self-age-birthday` is the example: it gives his age (24, as of October 2026) and states that the exact birthday is concealed, then points to the Contact Me section. Without such an entry, the question falls through to the generic private refusal. With it, the bot answers what is shared and declines the rest in one reply. Time-sensitive facts like age carry an "as of" date in the text and a `last_verified` date. Re-check them when you bump `meta.last_updated`.

### `make embed` order of operations

1. Glob every `data/*.json` file, so adding a file needs no code change.
2. Normalise every entry. Malformed entries and duplicate ids are skipped with a warning.
3. Embed **everything** locally, without the query prefix, normalised.
4. Refuse to continue if the index dimension is not 768.
5. Only then clear namespace `astarbot` and upsert.

A parse or model failure therefore aborts before anything is deleted, and entries removed from JSON leave no stale vectors behind.

### Calibration

`make calibrate` embeds the knowledge base in memory and scores a fixed set of probe questions: 37 on-topic, 8 off-topic, and a few canaries for entries that must *not* surface. On the current data the worst on-topic score is about 0.50 and the best off-topic score is about 0.61. The two distributions overlap, so `MIN_RETRIEVAL_SCORE` stays at `0.0` and the prompt's scope block handles refusals. Re-run calibration after editing `data/`.

### Embedder parity

`make check-embed` embeds five probe questions with both Cloudflare and local BGE. It exits non-zero if the worst cosine is below 0.999. Run it before switching the API to `EMBEDDING_BACKEND=cf`.

---

## 9. Deployment

### Docker image

| Stage | Contents |
|---|---|
| builder | `python:3.12-slim`, venv at `/opt/venv`, `pip install .` (or CPU torch + `.[local-embed]` and a pre-fetched BGE when `INSTALL_LOCAL_EMBED=true`) |
| runtime | `python:3.12-slim`, `libjemalloc2` (plus `libgomp1` for torch), non-root user `astarbot` (uid 1000), `HEALTHCHECK` on `/health`, `CMD ["python", "main.py"]` |

| Build arg | Default | Effect |
|---|---|---|
| `INSTALL_LOCAL_EMBED` | `false` | `true` adds torch and sentence-transformers (needs about 1 GB RAM at runtime) |
| `PREFETCH_MODEL` | `true` | Bakes BGE into local-embed images. `false` makes the image about 440 MB smaller. |

jemalloc is preloaded for every process and returns freed pages to the OS quickly:

```dockerfile
ENV LD_PRELOAD=/usr/local/lib/libjemalloc.so.2 \
    MALLOC_CONF=background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:0,narenas:2
RUN ln -s "/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2" /usr/local/lib/libjemalloc.so.2
```

The symlink makes the same path work on amd64 (Render) and arm64 (Apple silicon).

### Worker supervision (`main.py`)

`uvicorn --workers 1 --limit-max-requests N` only *exits* after N requests. With a single worker, uvicorn starts no supervisor to respawn it. `main.py` therefore starts the supervisor explicitly:

```python
config = uvicorn.Config("interfaces.api:app", host=settings.host, port=settings.port,
                        workers=1, limit_max_requests=settings.worker_max_requests or None,
                        timeout_graceful_shutdown=60)
sock = config.bind_socket()
Multiprocess(config, sockets=[sock]).run()
```

The parent process holds the socket and never imports the app. When the worker reaches its limit, it drains in-flight requests, including open streams, and the supervisor respawns it in about 2 to 3 s. Connections that arrive in the meantime wait in the socket backlog rather than being refused. `WORKER_MAX_REQUESTS=0` runs a plain server with no recycling.

### Render (API)

1. Create a **Web Service** from the repo with the **Docker** runtime. **Leave the Docker Command empty** so the image's `CMD ["python", "main.py"]` runs.
2. Set environment variables from `.env.example`: `PINECONE_API_KEY`, `GROQ_API_KEY`, `CF_API_TOKEN`, `CF_ACCOUNT_ID`, `EMBEDDING_BACKEND=cf`, `REASONING_EFFORT=low`, and `FRONTEND_ORIGIN`. Render injects `PORT`, and `Settings` binds `0.0.0.0:$PORT`.
3. Set the **Health Check Path** to `/health`.
4. Verify with `GET /version`, which should report `"embedding_backend": "cf"`. The boot logs should contain `engine_ready` and `warmup_ok`.

Measured idle memory is about 150 MB in total with `cf` and about 915 MB with `local`, so only `cf` fits the 512 MB free instance.

Free instances spin down after 15 minutes without traffic, and the first request after that waits for a cold start (up to about a minute). Both clients hide most of it. The Streamlit UI pings `/health`, and the portfolio frontend calls `/version` once the page is idle, so the container is usually awake by the time a visitor sends a question. The portfolio's request timeout is 90 s, and the UI's read timeout is 120 s.

### Streamlit Community Cloud (UI)

- Main file: `interfaces/streamlit_app.py`. Dependencies come from `requirements.txt` (`-e .[ui]`, no torch).
- Secrets: `ASTARBOT_API_URL = "https://<your-service>.onrender.com"`. No Cloudflare, Pinecone or Groq keys are needed, because the UI never embeds or retrieves anything itself.
- On page load, a background thread sends one `/health` ping per session to wake a sleeping container while the visitor reads.

### Devcontainer / Codespaces

The devcontainer uses `mcr.microsoft.com/devcontainers/python:1-3.12-bookworm` and installs `.[dev,ui,local-embed]` with CPU torch. It auto-starts Streamlit and forwards ports 8501 (UI) and 8000 (API). The Python, Pylance and Ruff extensions are preinstalled.

---

## 10. Logging and observability

Every log record is one JSON object on stdout (`rag/log.py`):

```json
{"ts": "2026-10-13T12:00:00.123Z", "level": "INFO", "logger": "astarbot.engine",
 "msg": "chat_done", "request_id": "a1b2c3d4", "prompt_version": "v2.4.0",
 "embedding_backend": "cf", "top_k": 5, "retrieval_top_score": 0.71,
 "llm_tier_used": 0, "outcome": "answered", "latency_ms": 1480, "stream": true, "ttft_ms": 1210}
```

| Event | Meaning |
|---|---|
| `chat_request` / `chat_stream_request` | API received a request (message count, whether a summary was sent) |
| `injection_blocked` | `layer`: `regex` or `prompt_guard` |
| `followup_rewritten` | The retrieval query was expanded |
| `retrieval` / `retrieval_cache` | Candidate counts and top score / cache `hit`, `size`, `hits`, `misses` |
| `cf_rate_limited` / `cf_server_error` | Embedder retry with `attempt` and `wait_s` |
| `llm_ok` / `llm_fail` / `llm_all_failed` | Blocking chain progress (`tier` 0 = primary, 1 = fallback) |
| `llm_stream_ok` / `llm_stream_broken` | Stream finished cleanly / broke after its first token |
| `guard_fail_open` (ERROR) | Request served without the guard layer |
| `chat_done` | Final outcome, latency, and `ttft_ms` for streams |

Raw questions are logged only at `DEBUG`. Third-party loggers (`httpx`, `urllib3`, `pinecone`, `sentence_transformers`) are capped at `WARNING`.

---

## 11. Design notes

These decisions came from measurement, not assumption.

- **Latency was an environment problem.** Locally, the code answered in 1.3 to 2.9 s. On the original 512 MB Railway container (the API has since moved to Render), the API took around 30 s because torch and BGE pushed the worker to about 860 MB RSS, which caused memory stalls and restarts. Moving query embeddings off-process fixed it. Streaming, concurrency and caching are secondary improvements.
- **Hosted embeddings moved from Hugging Face to Cloudflare** because the Hugging Face free tier is credit-gated and starts returning 401/402 once the credit runs out. Cloudflare serves the same model with a daily request quota. The vector space is unchanged as long as CLS pooling is requested.
- **The guard and retrieval run in parallel.** The results match sequential execution exactly, and only wall-clock time changes. In the rare unsafe case, one retrieval is wasted.
- **Retrieval failures produce the "technical snag" message, not an off-topic refusal**, so a visitor isn't told a valid question is out of scope during an outage.
- **The LRU cache stores retrievals, never answers.** Answers depend on the summary, the recent exchange and a temperature of 0.4.
- **Recent exchange is 2 messages**, the previous question and its answer. That is enough for "how did he do *it*" to resolve, and it roughly halves the history tokens in each prompt.
- **MMR is off by default.** Calibration showed no duplicate crowding, and MMR requires Pinecone to return every candidate's raw vector.
- **Groq free tier.** Bursts of about 12 prompts of about 3k tokens in one minute can hit the tokens-per-minute limit on both tiers. Watch for `llm_fail … 429`.

---

## 12. Troubleshooting and rollback levers

Every lever below is an environment change. None needs a code change.

| Symptom | Check / lever |
|---|---|
| Every answer is "technical snag" | Look for `embed_fail` / `cf_*` or Pinecone errors. Verify the `CF_*` values, or set `EMBEDDING_BACKEND=local` on a host with about 1 GB RAM. |
| Retrieval quality dropped after switching to Cloudflare | Run `make check-embed` and confirm `CF_POOLING=cls` |
| Slow first token / empty primary answers | Make sure `REASONING_EFFORT=low` is set and not empty |
| Streaming renders badly behind a proxy | Set `ENABLE_STREAMING_ENDPOINT=false`. The UI falls back to `/chat` automatically. |
| Suspected stale retrievals | Set `RETRIEVAL_CACHE_SIZE=0` |
| Duplicate facts crowding the top-k | Set `MMR_LAMBDA=0.7` |
| Follow-ups lose their referent | Set `RECENT_TURNS_IN_PROMPT=4` |
| Worker recycling disrupts sessions | Raise `WORKER_MAX_REQUESTS`, or set it to `0` |
| `ld.so` preload errors in the container | Remove the `LD_PRELOAD` / `MALLOC_CONF` lines from the Dockerfile |
| UI says "Waking the AStarBot API…" for a long time | The Render instance is cold-starting after spinning down. Check `ASTARBOT_API_URL` and the API's `/health`. |
| Startup `ValidationError` | `EMBEDDING_BACKEND` must be `auto`, `cf` or `local`. `CF_POOLING` must be `cls` or `mean`. |

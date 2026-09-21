# Changelog

## 2.0.1 — 2026-09-21

Knowledge base moved to six files / 83 entries with a richer schema; retrieval adapted.

- `rag/knowledge.py` (new): schema contract. Embedded string = title + FAQ question + text + aliases; LLM context block = title + text + links; metadata carries `source`, `title`, `question`, `links`, `tags`, `priority`, `last_verified`.
- `scripts/embed.py`: reads `{meta, entries}` (bare list still accepted), globs all files, parses and embeds **before** clearing the namespace so a bad file can't wipe the index; `--dry-run`; duplicate-id detection across files.
- `rag/retriever.py`: over-fetch `FETCH_K`, filter on raw cosine, re-rank `cosine + PRIORITY_WEIGHT × priority`, keep `TOP_K`.
- `scripts/calibrate.py` (new, `make calibrate`): offline on-/off-topic score distributions. Result on current data: overlap (0.498 vs 0.614) → `MIN_RETRIEVAL_SCORE` stays 0.0; scope refusal is the prompt's job.
- Prompt: IDENTITY no longer calls him a student; SCOPE rewritten for the new topics, explicitly excludes salary/notice period/employer internals and "do my work" requests; GROUNDING explains the title/Links context format. `PROMPT_VERSION` → `v2.2.0`.
- `rag/llm.py`: prompt-guard fail-open now logged at ERROR as `guard_fail_open` with `fail_open: true`.
- Tests: 89 (was 60) — knowledge schema incl. the real data files, embed ordering/dry-run, priority re-rank, and an explicit test that a forged client summary or poisoned context never reaches the system turn.

## 2.0.0 — 2026-09-19

Full rewrite of the RAG core and interfaces. The stateless, client-managed
memory design and the "grounded > generative" philosophy are unchanged.

### Retrieval
- Embeddings: `all-MiniLM-L6-v2` (384-d) → `BAAI/bge-base-en-v1.5` (768-d), run locally via sentence-transformers.
- BGE query prefix applied to queries only (`QUERY_PREFIX`); documents embedded without it. Vectors normalised for cosine.
- `top_k` 3 → 5. Optional `MIN_RETRIEVAL_SCORE` filter (off by default).
- New Pinecone index `astarbot` (768-d, cosine, aws us-east-1), namespace `astarbot` (was: namespace = index name).
- `scripts/embed.py` checks index dimension before writing, rejects entries without `id`/`text`, dedupes ids, batch-encodes.

### LLM
- Four Groq tiers: prompt-guard (`meta-llama/llama-prompt-guard-2-86m`), primary (`openai/gpt-oss-120b`), fallback (`openai/gpt-oss-20b`), summarizer (`openai/gpt-oss-20b`).
- Fallback catches any exception (was: `RateLimitError` only), logs tier, then returns a static fallback string. No provider retries.
- `reasoning_effort=low` and `max_tokens` set on chat models; 30 s timeout.

### Guardrails (defense in depth)
- Layer 1 `rag/guardrails.py`: control/ANSI stripping, length caps at word boundary, role whitelist, message-window cap, regex injection screen, output leak-marker stripping and length cap.
- Layer 2: prompt-guard LLM classification (fail-open, toggle `ENABLE_PROMPT_GUARD_LLM`).
- Layer 3: `INJECTION_DEFENSE` prompt block.
- Explicit refusal taxonomy with fixed strings: private / off-topic / unsafe / jailbreak + static fallback.

### Prompt
- Seven ordered blocks (identity, scope, grounding, injection defense, refusals, style, memory+context+question), `PROMPT_VERSION = "v2.1.0"` logged per request and served at `/version`.
- Instructions go in the system turn, untrusted material (summary, context, question) in the user turn.
- Scope now explicitly includes what the knowledge base publishes (career goals, compensation expectations, contact, resume, hobbies) and excludes everything else.

### API
- Moved to `interfaces/api.py`. Added `GET /version`, `X-Request-Id` middleware, CORS from `FRONTEND_ORIGIN`.
- Strict Pydantic limits: question ≤ 500 chars, ≤ 20 messages, message ≤ 2000 chars, roles `user|assistant`.
- Engine constructed in FastAPI lifespan (testable, no import-time side effects).

### Interfaces
- `interfaces/cli.py`: `/clear`, `/debug` (sources + scores + tier + latency), `/summary`, `/help`.
- `interfaces/streamlit_app.py`: sidebar diagnostics, clear button, debug toggle, `st.secrets` → env bridge for Streamlit Cloud.
- `interfaces/session.py`: shared window-trimming policy (keep 12; keep 4 after a fresh summary) for CLI, Streamlit and the React frontend to mirror.

### Config / ops
- `rag/config.py`: typed `Settings` (pydantic-settings); fails fast on missing `PINECONE_API_KEY` / `GROQ_API_KEY`. `.env.example` covers every field.
- `rag/log.py`: JSON logs with `request_id`, `prompt_version`, `top_k`, `retrieval_top_score`, `llm_tier_used`, `latency_ms`. Raw questions at DEBUG only.
- `pyproject.toml` replaces `requirements.txt` (kept as a thin `-e .` shim with the CPU torch index for Streamlit Cloud / devcontainer).
- Dockerfile: multi-stage, venv copy, CPU-only torch, embedding model baked in (`--build-arg PREFETCH_MODEL=false` to skip), non-root user, healthcheck.
- Added `.dockerignore`, `Makefile`, `tests/` (60 tests, offline), devcontainer bumped to Python 3.12.

### Removed
- Root `api.py`, `app.py`, `cli.py`; v1 `rag/prompt.py` blocks (`SYSTEM_PROMPT`, `RAG_RULES`, `REFUSAL_RULES`, `FALLBACK_MESSAGE`); `TOP_K_RETRIEVAL`, `MAX_RECENT_MESSAGES`-driven summarisation.

## 1.x
- Initial stateless RAG backend: MiniLM embeddings, Pinecone k=3, Groq primary/fallback, summarised memory, FastAPI + Streamlit + CLI.

# AStarBot v2.4 — Grounded, Bounded, Stateless Portfolio Assistant

AStarBot is a Retrieval-Augmented Generation (RAG) backend that answers questions about **Avrodeep Pal** — education, projects, skills, interests — from a curated knowledge base. It is deliberately *not* a general chatbot: every fact comes from retrieved context, everything else is politely refused, and the server keeps no session state.

```
Client (React / Streamlit / CLI)          holds: recent_messages[], summary
        │  POST /chat or /chat/stream {question, recent_messages, summary}
        ▼
FastAPI  interfaces/api.py  (async)       request_id, CORS, strict schemas
        ▼
RAGEngine  rag/engine.py
   1. input guardrails      sanitise, cap lengths, whitelist roles
   2. regex injection screen
   3. follow-up rewrite     retrieval query only
   4. prompt-guard LLM   ┐  llama-prompt-guard-2-86m (fail-open)
   5. retrieval          ┘  concurrent · LRU cache → BGE-base 768-d (HF API or local) → Pinecone top-5
   6. prompt + LLM chain    gpt-oss-20b → gpt-oss-120b → static fallback  (streamed on /chat/stream)
   7. output guardrails     strip leaks, cap length  (incremental when streaming)
   8. summarise (if window full)   gpt-oss-20b, temp 0  (/summarize for streaming clients)
        │  {answer, updated_summary}  or  text/plain token stream
        ▼
Client trims its window and sends the summary back next turn
```

Design principles: **grounded** (context only), **bounded** (refuse out of scope), **stateless** (no Redis/DB), **deterministic** (no agents/tools), **auditable** (layered versioned prompt, JSON logs), **explicit** (plain try/except fallbacks).

---

## Quick start

```bash
cp .env.example .env            # fill PINECONE_API_KEY, GROQ_API_KEY (+ HF_TOKEN for HF embeddings)
make dev                        # pip install -e ".[dev,ui,local-embed]" with CPU torch
make install                    # …or: API + UI only, no torch (needs HF_TOKEN)
make embed                      # (re)build the Pinecone index from data/*.json
make calibrate                  # check on-/off-topic score separation (offline)
make cli                        # python -m interfaces.cli
make ui                         # streamlit run interfaces/streamlit_app.py
make api                        # python main.py  → http://localhost:8000
make test                       # pytest (offline; Pinecone/Groq/HF are mocked)
make check-embed                # HF API vs local BGE vectors should match (cosine ≥ 0.999)
```

Requires Python ≥ 3.11 (3.12 recommended). Query embeddings run either through the Hugging Face Inference API (`EMBEDDING_BACKEND=hf`, no torch) or in-process (`local`, needs the `[local-embed]` extra; the first run downloads `BAAI/bge-base-en-v1.5`, ~440 MB, into `HF_HOME`). `auto` (default) picks `hf` whenever `HF_TOKEN` is set. Document embedding (`make embed`) and `make calibrate` always run locally. See [CHANGELOG.md](CHANGELOG.md) for what changed in v2.4.0 and why.

### API

| Method | Path       | Body / Response |
|--------|------------|-----------------|
| POST   | `/chat`    | `{question, recent_messages?: [{role, content}], summary?}` → `{answer, updated_summary}` |
| POST   | `/chat/stream` | same body → `text/plain; charset=utf-8` token stream (no summary; `ENABLE_STREAMING_ENDPOINT`) |
| POST   | `/summarize` | `{recent_messages, summary?}` → `{updated_summary}` (no-op below `SUMMARY_TRIGGER_AFTER`) |
| GET    | `/health`  | `{"status": "ok"}` |
| GET    | `/version` | app/prompt version, models, embedding backend, streaming flag, cache size, initial message, client window policy |

Limits (422 if exceeded): question ≤ 500 chars, ≤ 20 messages of ≤ 2000 chars, roles `user`/`assistant`, summary ≤ 2000 chars. Every response carries `X-Request-Id`.

---

## Stateless memory contract (clients must follow)

1. Start with `recent_messages = [initial assistant message]`, `summary = null`.
2. After each `/chat`, append the user turn and the answer.
3. If `updated_summary` **differs** from the summary you sent, keep only the last **4** turns (`MESSAGES_AFTER_SUMMARY`); otherwise keep the last **12** (`MAX_RECENT_MESSAGES`).
4. Send `updated_summary` back as `summary` on the next request.

The server summarises when it receives ≥ `SUMMARY_TRIGGER_AFTER` (10) turns. `interfaces/session.py` is the reference implementation; mirror it in the React hook.

**Streaming clients** (`/chat/stream`) get text only, no `updated_summary`. After rendering the answer, if the window (including the new exchange) has ≥ `client_window.summary_trigger_after` messages, call `POST /summarize` and treat its `updated_summary` exactly as in step 3. `interfaces/streamlit_app.py` does this.

Below that threshold the model still sees the last `RECENT_TURNS_IN_PROMPT` (2) messages, i.e. the previous question and its answer, rendered as `ROLE: text` lines inside the user turn (never as real assistant messages, so a forged history carries no authority). Follow-ups that lean on a referent ("how did he do **it**", "and the results?") also get the previous user question prepended to the *retrieval* query only (`rag/followup.py`), so the vector search carries the topic while the model answers the question as typed.

---

## Knowledge base

`data/*.json` — six files, one vector per entry, no chunking. Every file is globbed, so adding one needs no code change.

```json
{ "meta": { "source": "faq", "last_updated": "2026-09-21" },
  "entries": [ { "id": "faq-gate", "title": "Is he preparing for GATE?", "question": "…", "text": "…",
                 "links": ["https://…"], "tags": ["gate"], "aliases": ["gate plans"], "priority": 5 } ] }
```

`rag/knowledge.py` is the schema contract. Two derived strings are deliberately different:

- **embedded** (`embedding_text`): `title + question + text + aliases` — the title makes a chunk like "Not right now…" unambiguous, and aliases only help retrieval if they're inside the vector.
- **shown to the LLM** (`context_block`): `title + text + Links:` — clean text, no ids/tags/aliases.

Pinecone metadata: `text, title, question, links, tags, priority, last_verified, source`.

`make embed` parses and embeds **everything first**, then clears namespace `astarbot` and upserts — a malformed file aborts before anything is deleted, and entries removed from JSON leave no stale vectors. `make embed-dry` does the same without touching Pinecone. The script refuses to write to an index whose dimension ≠ 768.

Pinecone index: `astarbot` · 768-d · cosine · aws us-east-1 · namespace `astarbot`.

---

## Retrieval

Four stages, because Pinecone only does the first:

0. LRU cache (`RETRIEVAL_CACHE_SIZE`, 64): a repeat of the same retrieval query (case/whitespace-insensitive) skips steps 1–4. Only non-empty results are cached; the cache is per worker and resets on redeploy or recycle.
1. Query → `QUERY_PREFIX + question` → BGE-base (HF API or local) → over-fetch `FETCH_K` (10) candidates.
2. Drop anything below `MIN_RETRIEVAL_SCORE` on the **raw** cosine score.
3. Re-rank by `cosine + PRIORITY_WEIGHT × priority` (0.02 × 2..5 = up to 0.06 — reorders near-ties, never overturns a clear winner).
4. Optional MMR down to `TOP_K` (5) with `MMR_LAMBDA` < 1.0: each next pick is `λ·relevance − (1−λ)·max-cosine-to-already-picked`, so `faq-gate` doesn't take a slot when `self-gate` is already in. **Off by default since v2.4.0** (`MMR_LAMBDA=1.0`) because calibration shows no duplicate crowding and MMR needs every candidate's raw vector returned by Pinecone; set `0.7` to re-enable.

An embedder or Pinecone failure (e.g. an HF rate limit) is answered with `STATIC_FALLBACK` ("technical snag"), not the off-topic refusal.

**`MIN_RETRIEVAL_SCORE` defaults to 0.0, and that's measured, not lazy.** `make calibrate` runs 32 real questions and 8 off-topic ones against the current data with no network: worst on-topic 0.498, best off-topic 0.617 ("recipe for biryani" is genuinely close to `pers-food`). The distributions overlap, so no threshold separates them and refusal belongs to the prompt's SCOPE block. It also checks a few canary questions for entries that must *not* surface (the growth-area entry once leaked into "what is he working on now" because its title collided). Re-run after editing the knowledge base; raise the threshold only if the groups actually separate.

## Prompt (`rag/prompt.py`, `PROMPT_VERSION`)

Eight fixed-order blocks: identity → scope → grounding → domain discipline → injection defense → refusal taxonomy → style/length → summary + recent exchange + context + question. Blocks 1–7 ride in the system turn; block 8 (all untrusted material) in the user turn. Bump `PROMPT_VERSION` on any wording change; it is logged per request and exposed at `/version`.

Domain discipline keeps the four threads (job, research, GATE, personal) apart — a question about his work doesn't get his GATE plans appended, and the weakness entry is used only when weaknesses are asked about. Style matches depth to the verb ("what is X" → a line; "explain X" → the numbers the entry actually holds, up to 120 words) and forbids grading his skills ("highly proficient") — the model describes what he built and lets the visitor judge.

Refusal strings are constants (`REFUSAL_OFFTOPIC`, `REFUSAL_UNSAFE`, `REFUSAL_JAILBREAK`, `STATIC_FALLBACK`) so responses are consistent whether the guardrail or the model produces them. Private/compensation refusals come in three variants (`REFUSAL_PRIVATE_VARIANTS`), each ending in the same `CONTACT_POINTER` to the portfolio's Contact Me section; the model picks the opener that fits the tone and avoids the one it used last.

## LLM tiers (Groq)

| Tier | Model | Job |
|------|-------|-----|
| guard | `meta-llama/llama-prompt-guard-2-86m` | jailbreak probability on the raw question; ≥ `GUARD_THRESHOLD` → refuse. Fail-open. |
| primary | `openai/gpt-oss-20b` | grounded answer (`REASONING_EFFORT=low`) |
| fallback | `openai/gpt-oss-120b` | any primary exception, or a primary stream that fails before its first token |
| summarizer | `openai/gpt-oss-20b` @ temp 0 | 3–5 line memory summary |

All models are env-configurable. The guard runs concurrently with retrieval; neither reads the other's output, so results are identical to running them in turn. gpt-oss cannot disable reasoning: `low` is the fastest setting, and an empty `REASONING_EFFORT` falls back to the provider default (medium).

## Logging

One JSON object per line (`rag/log.py`). `chat_done` records carry `request_id` (the API's `X-Request-Id`), `prompt_version`, `embedding_model`, `embedding_backend`, `top_k`, `retrieval_top_score`, `llm_tier_used` (0 primary, 1 fallback, −1 all failed), `outcome`, `latency_ms`, `stream`, and for streams `ttft_ms` (time to first token). `retrieval_cache` records carry `hit`, `size`, running `hits`/`misses`. Raw questions appear only at `DEBUG`.

---

## Layout

```
.
├── main.py                    uvicorn launcher: one supervised, recycled worker
├── rag/
│   ├── config.py              typed Settings from env
│   ├── log.py                 JSON logging
│   ├── guardrails.py          input/output sanitisation, injection regex
│   ├── prompt.py              8-block versioned prompt, refusal constants
│   ├── knowledge.py           entry schema → embedding text / metadata / context block
│   ├── followup.py            follow-up → standalone retrieval query
│   ├── embedder.py            query embeddings: HF Inference API or local BGE
│   ├── retriever.py           Pinecone + re-rank + MMR + LRU cache
│   ├── llm.py                 guard / primary / fallback / summarizer chain
│   ├── memory.py              summarisation
│   └── engine.py              orchestrator
├── interfaces/
│   ├── api.py                 FastAPI
│   ├── streamlit_app.py       beta UI
│   ├── cli.py                 terminal client
│   └── session.py             shared client window policy
├── scripts/embed.py           index rebuild (--dry-run)
├── scripts/calibrate.py       score-distribution check for MIN_RETRIEVAL_SCORE
├── scripts/check_embedder.py  HF API vs local BGE parity check
├── data/*.json                knowledge base (self, experience, projects, studies, personality, faq)
├── tests/                     offline unit + API tests
├── Dockerfile · .dockerignore · Makefile · pyproject.toml · .env.example · CHANGELOG.md
├── requirements.txt            -e .[ui] for Streamlit Cloud (no torch since v2.4.0)
```

## Deployment

- **Railway / Docker** (runs the API): `docker build -t astarbot .` — multi-stage, **slim by default** (no torch; set `HF_TOKEN` and `EMBEDDING_BACKEND=hf`), jemalloc allocator, non-root, healthcheck on `/health`, `CMD python main.py` (one supervised worker recycled every `WORKER_MAX_REQUESTS`). `--build-arg INSTALL_LOCAL_EMBED=true` builds the torch image with BGE baked in for `EMBEDDING_BACKEND=local` (needs ~1 GB RAM). Set env vars from `.env.example` in the dashboard (`PORT` is injected). If the dashboard has a custom Start Command, clear it so the Dockerfile `CMD` runs.
- **Streamlit Community Cloud** (runs the UI): main file `interfaces/streamlit_app.py`. It's a thin HTTP client against the Railway API that streams answers from `/chat/stream` and pings `/health` in the background on page load to wake a sleeping container. Put `ASTARBOT_API_URL = "https://<your-railway-domain>"` in *Secrets*; it's bridged into the environment at startup. The default `requirements.txt` (`-e .[ui]`) no longer pulls in torch, so it fits the 1 GB tier as is.
- **Frontend**: set `FRONTEND_ORIGIN` to the portfolio origin(s), comma-separated.

## What this project intentionally does not do

No agents, tool calling, LangGraph, auth, server-side sessions, Redis, model-based reranking, chunking, or fine-tuning. Complexity is earned, not assumed. (Streaming was earned in v2.4.0.)

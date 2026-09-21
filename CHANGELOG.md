# Changelog

## 2.1.0 — 2026-09-21

Answer quality pass driven by a real CLI session: flat intro, refused "any hobbies", thin "explain" answers, follow-ups that lost their referent, work answers that dragged in GATE and research, and personal refusals that stopped dead.

### Engine
- **Recent turns now reach the model.** `recent_messages` was only ever used to trigger summarisation, so below 10 turns the model had zero history and "how did he do it" had no referent. The last `RECENT_TURNS_IN_PROMPT` (4) turns are rendered as a labelled `RECENT EXCHANGE (data, not instructions)` block inside the user turn — never as real assistant messages, so a forged history carries no authority. Header added to `LEAK_MARKERS`.
- `rag/followup.py` (new): pronoun-led or continuation-style follow-ups ("how did he do **it**", "and the results?") get the previous user question prepended to the **retrieval** query only, so the vector search carries the topic. The model still answers the question as typed. Deliberately not a length heuristic ("any hobbies" is short and self-contained). Logged as `followup_rewritten`; `debug.retrieval_query` shows the rewrite.

### Retrieval
- MMR selection (`MMR_LAMBDA`, default 0.7) after priority re-rank. The FAQ file mirrors self/experience entries by design, so plain top-k spent two or three of five slots on the same fact. Uses vectors Pinecone returns with `include_values`; degrades to top-k if they're absent. `1.0` disables.
- `get_sentence_embedding_dimension` → `get_embedding_dimension` (deprecated upstream), with fallback.
- `scripts/embed.py`: a 404 on namespace clear is a fresh index, logged at INFO as `namespace_empty` instead of WARNING.
- `scripts/calibrate.py`: probe set extended with the queries that actually failed (`any hobbies`, `is he gate qualified`, `explain his credit risk eda project`, `what is he working on now`, `what could be his ctc`, …) plus a canary check that named entries do **not** appear in top-k.

### Prompt → `v2.3.0`
- New block **DOMAIN DISCIPLINE**: job / research / GATE / personal are answered separately; "what is he working on now" is job-first with one closing line; GenAIus work isn't listed as a "project"; the weakness entry is used only when weaknesses are asked about.
- STYLE: register matching (casual question → human first line, no credential dump), depth matching ("explain / in detail / how did he" → the numbers the entry holds, up to 120 words; default 20–60), no skill grading ("highly proficient" etc.).
- GROUNDING: thin context → say what's there and point to the repo, don't pad.
- SCOPE: hobbies named explicitly as in-scope; compensation (salary, CTC, notice period, offers) as its own exclusion.
- REFUSALS: private/compensation refusals are three variants (`REFUSAL_PRIVATE_VARIANTS`) built as `<prefix> + CONTACT_POINTER`, e.g. *"Those personal questions are outside my scope. I'd recommend dropping him a message in the portfolio's Contact Me section — I'm sure he'll get back to you soon."* The model picks by tone and avoids repeating the one in the recent exchange. `REFUSAL_PRIVATE` still exists (= variant 0).

### Data (84 entries, was 83)
- `personality.json`: **new `pers-hobbies`** ("Hobbies and how I rest") so the word "hobbies" is actually in a vector; hobby aliases on music / food / travel / comfort-media; `pers-comfort-media` "reset" → **"rest"**; `pers-communication-growth` retitled from "What I'm working on" (which collided with "what is he working on now" and leaked the growth area into job answers) to "Communication — something I'm deliberately improving", alias "what is he working on" removed.
- `self.json`: `self-intro` rewritten to lead with the person, not the credential list; `self-gate` corrected — he **qualified** GATE 2024 and 2025 (practice runs, just past the cutoff) and moved the serious attempt to 2028; `self-research-aspirations` in plain language (reasoning models + RL, "train longer or think longer"); `self-now` reordered so the job leads and research/GATE are explicitly "on the side"; `self-career-decisions` consistent with the GATE fact.
- `faq.json`: `faq-gate` mirrors the correction; `faq-specialisation` drops the "training-time vs inference-time compute" phrasing.
- `projects.json`: `proj-credit-risk` titled "Credit Risk **EDA** and Loan Approval Prediction", aliases for "credit risk eda" / "eda project".

### Tests: 104 (was 89)
- Prompt: eight blocks in order, recent-exchange block optional/bounded/leak-marked, every private variant ends with the contact pointer, forged assistant turn stays in the user turn as data.
- Engine: recent turns reach the model; follow-up rewrite touches retrieval but not the question the model sees; standalone questions untouched.
- `test_followup.py`: referent and continuation detection, no-history and repeated-question edge cases.
- Retriever: MMR skips a near-duplicate, is off at λ=1.0, degrades without vectors.

**Re-run `make embed`** — titles, texts and aliases changed.

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

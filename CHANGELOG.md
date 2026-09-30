# Changelog

All notable changes to AStarBot. Versions follow the app version in `rag/__init__.py`; the prompt has its own `PROMPT_VERSION` (logged per request, exposed at `GET /version`).

---

## [2.4.0] — 2026-09-30 · Latency & memory release

**Prompt:** `v2.3.0 → v2.4.0` · **App:** `2.1.0 → 2.4.0` · **Answer quality:** unchanged by design (no prompt coverage removed, same models, same retrieval set when MMR is off)

### TL;DR

| Problem | Fix in 2.4.0 | Where |
|---|---|---|
| ~30 s per reply on Railway | Offload query embeddings to the Hugging Face Inference API, so torch never loads. The API process drops from ~860 MB to ~150 MB, below the 512 MB free-tier cap it was overrunning | `rag/embedder.py`, `EMBEDDING_BACKEND` |
| Visitor stares at a spinner for the whole answer | Token streaming (`POST /chat/stream`) rendered with `st.write_stream` | `rag/llm.py`, `rag/engine.py`, `interfaces/api.py`, `interfaces/streamlit_app.py` |
| First question after idle pays the cold start | Streamlit pings `/health` in the background on page load, and the API warms the embedder/Pinecone/cache at boot | `interfaces/streamlit_app.py`, `RAGEngine.warmup` |
| Guard LLM and retrieval ran one after the other | Run concurrently with `asyncio.gather`; all routes are `async` | `rag/engine.py` |
| Repeat questions redo embedding + Pinecone | Bounded in-process LRU cache of retrievals | `rag/retriever.py` |
| Prompt tokens grew every turn | Recent exchange 4 → 2 messages, MMR off (no vectors fetched), prompt wording compressed | `rag/config.py`, `rag/prompt.py` |
| Slow RSS creep in a long-lived worker | jemalloc allocator + worker recycling under a supervisor | `Dockerfile`, `main.py` |
| 1.4 GB install, torch on Streamlit Cloud | torch/sentence-transformers → `[local-embed]` extra, streamlit → `[ui]` extra | `pyproject.toml`, `requirements.txt` |

---

### Why it was slow: diagnosis

The discussion listed four code changes as suspects (model swap, prompt growth, 4-turn recent exchange, MMR with `include_values`). I benchmarked the **previous code (HEAD `bc9c17c`) locally** against the same Groq/Pinecone account and knowledge base:

| Question | Old code, `/chat` (local) |
|---|---|
| tell me about him | 2.91 s |
| what projects has he built | 2.45 s |
| explain his credit risk eda project | 2.25 s |
| how did he do it, what data and algos (follow-up) | 1.94 s |
| what are his hobbies | 1.33 s |

**The code by itself answers in 1.3–2.9 s.** The prompt growth, recent exchange and MMR together cost well under a second. The ~30 s on Railway comes from the environment:

1. **Memory.** With local BGE, the API worker measured **863 MB RSS** (torch + sentence-transformers + model weights). Railway's free tier gives 512 MB, so the container runs over its limit, stalls on memory pressure, and risks OOM restarts. Each restart re-pays the ~20 s model load.
2. **Cold start.** With App Sleeping on, the first request after idle waits for the container to boot and load torch plus the 440 MB model before any work starts.
3. **`REASONING_EFFORT` empty does not mean "off."** gpt-oss on Groq only accepts `low` / `medium` / `high` and cannot disable reasoning. An empty value just omits the parameter, so Groq applies the model default (**medium**), which is *more* reasoning than the previous `low`. Reasoning tokens are generated before the first visible token and come out of the `MAX_ANSWER_TOKENS` (400) budget. If reasoning uses up the budget, the primary returns an empty completion, which triggers a fallback call (a second full LLM round trip). **Set `REASONING_EFFORT=low` on Railway.**

So the real fix is (1) and (2): get torch out of the 512 MB container. Streaming, concurrency, caching and prompt trimming are the second tier: they cut hundreds of milliseconds and make the wait *feel* shorter, but on their own they would not have fixed 30 s.

---

### Deploy checklist

**Railway (API) — Variables**

| Variable | Set to | Why |
|---|---|---|
| `HF_TOKEN` | your HF token (read scope) | enables the HF embedding backend |
| `EMBEDDING_BACKEND` | `hf` | explicit is safer than `auto`; the slim image has no torch |
| `REASONING_EFFORT` | `low` | empty = provider default = medium (see diagnosis) |
| `MMR_LAMBDA` | delete, or `1.0` | new default is 1.0 |
| `RECENT_TURNS_IN_PROMPT` | delete, or `2` | new default is 2 |
| `WORKER_MAX_REQUESTS` | optional, default `500` | `0` disables recycling |

**Railway (API) — Settings**
- **Start Command:** clear any custom command so the Dockerfile's `CMD ["python", "main.py"]` runs. Plain `uvicorn … --workers 1 --limit-max-requests N` stops serving after N requests (see Phase 6).
- The default Docker build is now the **slim** image (no torch). Only pass `INSTALL_LOCAL_EMBED=true` as a build arg if you want the local backend and have ~1 GB of RAM.
- App Sleeping can stay on; the Streamlit warm-up absorbs most of the wake time, and a slim container boots in seconds instead of ~20 s.

**Streamlit Community Cloud (UI)**
- Python dependencies: `requirements.txt` (default). It's now `-e .[ui]` with no torch. `requirements-streamlit.txt` is gone.
- Secrets unchanged (`ASTARBOT_API_URL`). `HF_TOKEN` is **not** needed here.

**Before switching the backend**, optionally run `make check-embed` locally with `HF_TOKEN` set. It embeds four probe questions through both the HF API and local BGE and expects cosine ≥ 0.999. This confirms the HF-side vectors match the ones the Pinecone index was built with.

---

### Changes, in implementation order

Each phase can be deployed and rolled back on its own. The order below is the order they were built in: every phase only depends on phases above it.

#### Phase 1 — Config and env surface

New settings in `rag/config.py` (all have defaults; nothing new is *required* unless you choose the HF backend):

```python
embedding_backend: str = "auto"      # "auto" | "hf" | "local"
hf_token: str = ""
hf_embedding_url: str = ""           # empty = derived from EMBEDDING_MODEL
hf_timeout_seconds: float = 15.0
hf_max_retries: int = 3              # 1..6
retrieval_cache_size: int = 64       # 0 disables
enable_streaming_endpoint: bool = True
worker_max_requests: int = 500       # 0 disables recycling
```

Changed defaults, per the decisions in the discussion:

| Setting | Before | After | Effect |
|---|---|---|---|
| `mmr_lambda` | 0.7 | **1.0** (off) | Pinecone no longer returns 10 × 768 raw floats per query; no numpy similarity matrix per request |
| `recent_turns_in_prompt` | 4 | **2** | Counted in *messages*, so 2 = previous question + its answer. Enough for "how did he do **it**" to resolve (verified live, below), and roughly halves the history tokens in every prompt |
| `reasoning_effort` | `low` | `low` (kept) | Comment and `.env.example` now say explicitly that empty ≠ off |

Validation added:
- `EMBEDDING_BACKEND` must be `auto`, `hf` or `local` (case/whitespace-insensitive). A typo fails at startup instead of silently loading torch on a 512 MB box.
- `RETRIEVAL_CACHE_SIZE ≥ 0`, `WORKER_MAX_REQUESTS ≥ 0`, `HF_MAX_RETRIES` in 1..6.

Derived properties:
- `settings.resolved_embedding_backend`: `auto` → `hf` if `HF_TOKEN` is set, else `local`.
- `settings.hf_embedding_endpoint`: `https://router.huggingface.co/hf-inference/models/<EMBEDDING_MODEL>/pipeline/feature-extraction` unless `HF_EMBEDDING_URL` overrides it.

`GET /version` now also returns `embedding_backend`, `streaming_enabled`, `retrieval_cache_size` and `client_window.summary_trigger_after` (the Streamlit client reads the last one).

`.env.example` was rewritten with the new sections: embedding backend + `HF_TOKEN`, cache, `REASONING_EFFORT=low` with the empty-value warning, `ENABLE_STREAMING_ENDPOINT`, `WORKER_MAX_REQUESTS`. `HF_TOKEN` is deliberately left **empty** there. With `auto`, any non-empty value, even a placeholder, switches the backend to HF.

#### Phase 2 — Embedding backend abstraction (hybrid HF / local)

New module **`rag/embedder.py`**. Both backends share one interface, `embed_query(str) -> list[float]` / `aembed_query`, and return a unit-length 768-d vector for the **same** model. The Pinecone index is untouched when you switch.

```python
def build_embedder() -> LocalBGEEmbedder | HFEmbedder:
    backend = settings.resolved_embedding_backend
    return HFEmbedder() if backend == "hf" else LocalBGEEmbedder()
```

- **`HFEmbedder`**: `POST {inputs: QUERY_PREFIX + query}` to the HF router with `Authorization: Bearer $HF_TOKEN`.
  - One pooled `requests.Session`, so the TLS connection stays warm between queries.
  - **503** (model loading) and **429** (rate limit) retry with exponential backoff (429 backs off twice as hard, capped at 8 s). Network errors retry. **401/403/404 fail immediately** instead of burning retries on errors that won't fix themselves.
  - Accepts a flat vector or a batch of one, rejects the wrong dimension, and **L2-normalises client-side** so HF and local vectors can be swapped for cosine search.
- **`LocalBGEEmbedder`**: the previous in-process sentence-transformers path. `load_embedder()` moved here from `rag/retriever.py` and **imports `sentence_transformers` lazily**, so an `hf` process never imports torch even if it's installed. A missing `[local-embed]` extra gives an actionable error.
- `scripts/embed.py` and `scripts/calibrate.py` import `load_embedder` from `rag.embedder` and **always run locally**. Documents are embedded once, offline, with no query prefix.
- New **`scripts/check_embedder.py`** (`make check-embed`): the HF vs local parity check described in the deploy checklist.

**Deviation from the plan:** the spec's URL (`api-inference.huggingface.co/pipeline/feature-extraction/…`) **no longer resolves**. HF moved serverless inference behind `router.huggingface.co/hf-inference/…`. Checked on 2026-09-30: the old host fails DNS, and the router returns 401 without a token (the endpoint exists).

**Is it free?** HF free accounts get a small monthly inference credit and are rate-limited. A single-sentence embedding is a tiny request, so a portfolio bot's traffic fits comfortably, but it isn't unlimited. If you see sustained `hf_rate_limited` log lines, set `EMBEDDING_BACKEND=local` on a host with ~1 GB RAM, or move to a paid embedding API.

**Measured memory** (macOS arm64, no jemalloc, idle after warm-up):

| Backend | Worker | Supervisor + tracker | Total |
|---|---|---|---|
| `local` (torch + BGE) | **863 MB** | 37 + 16 MB | ~915 MB |
| `hf` (slim install) | **100 MB** | 37 + 16 MB | ~153 MB |

#### Phase 3 — Prompt compression (`PROMPT_VERSION v2.4.0`)

Wording only. **Every rule and every refusal string is still present.**

- **`SCOPE_RULES`**: the "must NOT discuss" list merged from 7 bullets to 6 by folding "politics, religion…" and "general knowledge, current events, trivia" into one line. "Writing, reviewing or debugging their code" is kept in full (the spec's draft had dropped "reviewing").
- **`REFUSAL_GUIDANCE`**: the three private-refusal variants used to be spelled out in full, repeating `CONTACT_POINTER` three times. Now the three **openers** are listed once, followed by "followed by a space and then exactly: `<CONTACT_POINTER>`". The model still emits exactly one of `REFUSAL_PRIVATE_VARIANTS`.
- **`STYLE_AND_LENGTH`**: the "match depth to the verb" bullet was tightened with `/` and `->`.
- `DOMAIN_DISCIPLINE`, `RAG_CONSTRAINTS`, `INJECTION_DEFENSE`, `SYSTEM_IDENTITY` were not touched.

System prompt: **7,295 → 7,006 characters** (~70–80 tokens). It's a small saving, but it applies to every request. The larger per-request token cut comes from `RECENT_TURNS_IN_PROMPT` 4 → 2.

#### Phase 4 — LRU retrieval cache

`RetrievalCache` in `rag/retriever.py`: a small thread-safe `OrderedDict` LRU, one per worker. The async and sync paths share it.

- **Key:** the *retrieval* query (after follow-up rewriting), lower-cased with whitespace collapsed. BGE-base-en-v1.5's tokenizer is uncased and whitespace-insensitive, so "What are his hobbies?" and "what are  his hobbies?" embed identically and share one entry.
- **Value:** the final top-k (after threshold, priority re-rank, MMR), stored as tuples of dicts and **copied on every hit**, so a caller mutating results can't poison the cache.
- **Only non-empty results are cached.** An empty result may be a transient failure; caching it would keep refusing a valid question until eviction.
- **Not cached:** the LLM answer. It depends on the summary and recent exchange, and runs at temperature 0.4.
- **Memory:** 64 entries × ~5 matches × ~1 KB ≈ 320 KB. It resets on redeploy or worker recycle, which is fine.
- **Log:** a `retrieval_cache` line per request with `hit`, `size`, `maxsize` and running `hits`/`misses`, to show hit rates in Railway's logs.

**Deviation:** the spec used `functools.lru_cache` inside `__init__`. That doesn't work for the async path, can't skip caching empty results, and needs a hits-before/after trick to log hits. The explicit LRU is about 30 lines and does all three.

Also in this phase: retrieval failures are now **distinguishable from "nothing relevant."** `aretrieve` raises `RetrievalError` when the embedder or Pinecone fails, and the engine answers with `STATIC_FALLBACK` ("I hit a technical snag…", outcome `retrieval_failed`). Before, those failures got the **off-topic** refusal, which would mislead a visitor during an HF rate limit. The sync `retrieve()` keeps its old "log and return `[]`" contract for scripts and tests.

#### Phase 5 — Streamlit warm-up

```python
def warm_backend() -> None:
    if st.session_state.get("backend_warmed"):
        return
    st.session_state.backend_warmed = True
    threading.Thread(target=ping, daemon=True).start()   # GET /health, 120 s timeout
```

**Deviation from the plan:**
- **Per session, not `@st.cache_resource`.** `cache_resource` caches *process-wide* on Streamlit's server. After the first visitor ever, it would never ping again, even when Railway had gone back to sleep.
- **In a background thread, not blocking.** A blocking 120 s call would hold the page blank while Railway wakes up. Now the page renders instantly and the container wakes while the visitor reads and types.
- **`/version` failures are no longer cached.** Before, a sleeping API at first load was cached as "unreachable" for 5 minutes. Now `_fetch_version` raises, `st.cache_data` doesn't cache exceptions, and the next rerun retries. The sidebar shows "Waking the AStarBot API…" instead of an error.

Server side, the API also warms itself at boot with `RAGEngine.warmup()`, a background task in the lifespan. It runs one retrieval for "Who is Avrodeep?", which wakes the HF model, opens the Pinecone connection and seeds the cache with the most common opener. `/health` does not wait for it, and a failure is only logged.

#### Phase 6 — Docker and worker hardening

**jemalloc** (`Dockerfile`, runtime stage):
```dockerfile
ENV LD_PRELOAD=/usr/local/lib/libjemalloc.so.2 \
    MALLOC_CONF=background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:0,narenas:2
RUN apt-get install -y --no-install-recommends libjemalloc2 \
 && ln -s "/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2" /usr/local/lib/libjemalloc.so.2
```
jemalloc is a drop-in `malloc` replacement. glibc's allocator tends to keep freed memory reserved for the process. jemalloc groups allocations into size classes (less fragmentation) and, with `background_thread` plus the decay settings, returns idle pages to the OS within about a second. It typically cuts RSS by 10–20% in long-running Python services, with no code change. `narenas:2` avoids per-core arenas a 1-vCPU container can't use. The library is **symlinked to a fixed path** because its directory is arch-specific (`x86_64-linux-gnu` on Railway, `aarch64-linux-gnu` on Apple silicon). A hard-coded x86 path would make every process on arm64 print a preload error. Rollback: remove the two `ENV` lines.

**Worker recycling** (`main.py`). **Deviation from the plan, and a correction:** the spec's `uvicorn … --workers 1 --limit-max-requests 500` described a "rolling restart with overlap." That's not how uvicorn behaves with one worker. I read uvicorn's source: only `--workers > 1` starts the `Multiprocess` supervisor. With a single worker, reaching the limit simply **exits the process**, and the container stops serving until Railway restarts it. So `main.py` starts the supervisor explicitly, with one worker:

```python
config = uvicorn.Config("interfaces.api:app", host=..., port=..., workers=1,
                        limit_max_requests=settings.worker_max_requests or None,
                        timeout_graceful_shutdown=60)
sock = config.bind_socket()
Multiprocess(config, sockets=[sock]).run()
```

- The parent holds the listening socket for the container's lifetime and never imports the app (~37 MB).
- At the limit, the worker finishes in-flight requests, **including open streams**, then exits. `timeout_graceful_shutdown=60` stops a stalled client from holding it forever.
- The supervisor respawns the worker. Connections that arrive meanwhile **wait in the socket backlog** instead of being refused.
- Verified locally with a limit of 3: the worker exited, the supervisor respawned it, and the request that arrived during the swap got a 200 after 0.62 s.
- **Side effects:** the LRU cache resets on each recycle. With the slim image a respawn takes ~2–3 s; with the local backend it would re-load BGE (~20 s). At 500 requests that's a couple of times per busy beta day. `WORKER_MAX_REQUESTS=0` turns recycling off.
- The Dockerfile `CMD` stays `python main.py`, so no Railway start command is needed (clear any custom one).

#### Phase 7 — Async parallel pipeline

- **`rag/llm.py`**: `acheck_safe`, `ainvoke_chat`, `ainvoke_summarize` (on `ChatGroq.ainvoke`), plus `astream_chat` (Phase 8). The sync `check_safe`/`invoke_chat` were replaced; `invoke_summarize` stays for the sync `summarize_conversation`.
- **`rag/memory.py`**: `asummarize_conversation`, sharing its message building and clipping with the sync version.
- **`rag/retriever.py`**: `aretrieve`. The embedder call and Pinecone's sync client run in `asyncio.to_thread`, so they don't block the event loop.
- **`rag/engine.py`**: steps 1–5 are factored into `_prepare()`, shared by `achat` and `astream_chat`. The guard and retrieval are **gathered**:

```python
retrieval = self.retriever.aretrieve(retrieval_q, request_id)
if settings.enable_prompt_guard_llm:
    safe, contexts = await asyncio.gather(
        self.llm.acheck_safe(q, request_id), retrieval, return_exceptions=True
    )
    if safe is False:                          # unsafe wins, even if retrieval failed
        return _Plan(summ, early_answer=REFUSAL_UNSAFE, outcome="injection_guard")
    if isinstance(contexts, BaseException):
        ...                                    # RetrievalError -> STATIC_FALLBACK
```

**Why quality is unaffected:** the guard reads only the raw question, and retrieval reads only the (rewritten) question. Neither reads the other's output, so the results are byte-for-byte what sequential execution would give. Only wall-clock time changes: the guard's Groq round trip overlaps with embedding + Pinecone. The final LLM call still waits for retrieval, which it must. Cost: in the rare unsafe case, one retrieval is thrown away. A test pins the concurrency (two 0.2 s steps finish in < 0.35 s).

- **`interfaces/api.py`**: every route is `async def`. `/chat` awaits `engine.achat`, so one worker serves overlapping visitors on the event loop instead of tying up a thread-pool slot per request.
- **Request-id correlation:** the API's `X-Request-Id` is now passed into the engine. Before, the engine generated its own id, so `chat_request` and `chat_done` log lines couldn't be joined.
- **CLI compatibility:** `RAGEngine.chat()` is a sync wrapper over **one persistent `asyncio.Runner`**. A fresh `asyncio.run()` per turn would strand the Groq/HF async connection pools on closed loops. It must not be called inside a running loop; everything under the API awaits `achat` directly.

**Deviation:** the spec suggested `PineconeAsyncio`. The sync client in `to_thread` is equivalent for one query per request, needs no extra `pinecone[asyncio]` dependency, and keeps one code path for sync and async.

#### Phase 8 — Streaming endpoint and frontend

**`POST /chat/stream`**: same request schema and limits as `/chat`. It returns `text/plain; charset=utf-8` raw text (not SSE frames) with `Cache-Control: no-cache` and `X-Accel-Buffering: no`, so proxies don't buffer. Returns 404 when `ENABLE_STREAMING_ENDPOINT=false`.

**`LLMChain.astream_chat`** yields `(tier, text)` chunks from `ChatGroq.astream`:
- A tier that fails or streams **nothing before its first token** hands over to the fallback.
- A tier that breaks **after** tokens were sent stops the stream (logged as `llm_stream_broken`). The visitor already has half an answer, and splicing a second model's answer onto it would be worse. (The spec's version would have restarted on the fallback mid-answer.)
- Chunk text is taken **unstripped**. The old `_text()` strips whitespace, which would have glued streamed words together. A new `_raw_text()` is used for streams.

**Output guardrails while streaming.** **Deviation:** the spec accepted that leak markers could reach the visitor mid-stream and only sanitised the *logged* copy. Instead, a new **`StreamSanitizer`** (`rag/guardrails.py`) applies the same rules as `sanitize_answer` incrementally:
- It holds back the last 44 characters (the longest leak marker) until more text arrives, so a marker split across chunks (`"###"`, `" SCO"`, `"PE"`) is still removed before release.
- An unclosed `<think>` holds everything after it, and it is dropped at end of stream if never closed.
- The `MAX_ANSWER_CHARS` cap is enforced with a word-boundary cut and `…`.
- The hold-back delays the visible stream by a few tokens.
- If nothing survives (all tiers failed, or the answer was empty after sanitising), the stream ends with `STATIC_FALLBACK`.

Refusals (regex/guard/no-context/retrieval-failed/invalid input) arrive as one chunk and never call the LLM. `chat_done` for streams logs `stream: true` and **`ttft_ms`** (time to first token).

**Summaries with streaming.** **Deviation:** the plan's "Option A" let the summary go stale during streaming sessions. That would silently drop long-term memory past 12 messages. Instead there's a new **`POST /summarize`** (`{recent_messages, summary}` → `{updated_summary}`). It applies the *same* trigger rule server-side (a no-op with no model call below `SUMMARY_TRIGGER_AFTER` or with summaries disabled). The Streamlit client calls it **after the answer is already on screen**, only once its window reaches `client_window.summary_trigger_after`, so it never delays the answer. `/chat` still summarises inline as before.

**Streamlit** (`interfaces/streamlit_app.py`):
- `answer_streaming()`: `st.spinner("Thinking…")` covers only the wait for the **first** chunk, then `st.write_stream` renders the rest as it arrives. `r.encoding = "utf-8"` is set so a multi-byte character split across chunks is decoded correctly.
- Falls back to `/chat` (`answer_blocking()`) when `/version` reports `streaming_enabled: false`, when the API is pre-2.4 (no such key), or on a 404 from `/chat/stream`. **Turning streaming off is a pure server-side switch**; the UI adapts without a redeploy.
- The sidebar shows the embedding backend and streaming state.

**Live check** (real Groq + Pinecone, local BGE, requests paced 14 s apart to stay under Groq's free-tier TPM):

| Question | TTFT | Full answer |
|---|---|---|
| tell me about him | 2.41 s | 2.54 s |
| what projects has he built | 2.36 s | 2.67 s |
| explain his credit risk eda project | 1.96 s | 2.29 s |
| how did he do it, what data and algos (follow-up) | 2.05 s | 2.31 s |
| what are his hobbies | 2.10 s | 2.22 s |

Streaming works end to end, and the follow-up resolved to the credit-risk project with the 2-message recent exchange. Note how close TTFT is to the full time: Groq generates the visible answer in ~0.3 s, and most of the wait is **reasoning + prefill before the first visible token**. Streaming improves the feel, but the real latency levers are `REASONING_EFFORT=low`, fewer prompt tokens, and a container that isn't over its memory limit.

The same local setup with non-streaming `/chat` on the new code: 1.58 / 1.57 / 1.28 / 1.38 s vs. 2.91 / 2.45 / 2.25 / 1.94 s on the old code. These are single runs over the public internet, so treat them as indicative, not a benchmark.

#### Phase 9 — Dependency cleanup

`pyproject.toml`:
```toml
dependencies = [ ..., "uvicorn[standard]>=0.30.0", "pinecone>=5.0.0", "langchain-core>=0.2.0",
                 "langchain-groq>=0.1.5", "requests>=2.31.0", "numpy>=1.26.0", "tqdm>=4.66.0" ]

[project.optional-dependencies]
local-embed = ["sentence-transformers>=2.6.1", "torch>=2.2.0"]
ui          = ["streamlit>=1.33.0"]
dev         = ["pytest>=8.0.0", "httpx>=0.27.0", "ruff>=0.4.0"]
```
- **Removed from core:** `torch`, `sentence-transformers` (→ `local-embed`) and `streamlit` (→ `ui`; the API never imports it, and it pulls in pandas/pyarrow).
- **Added explicitly:** `requests` (HF client; was only transitive) and `numpy` (MMR, calibrate; was only transitive). numpy is now imported lazily inside `_mmr_select`, so it costs nothing while MMR is off.
- `uvicorn` floor raised to 0.30, where the `Multiprocess` supervisor respawns dead workers.
- A fresh `pip install .` is **114 MB** vs **1.4 GB** for the old full environment.
- **`requirements.txt`** → `-e .[ui]`, light enough for Streamlit Cloud. That's why `requirements-streamlit.txt` (already deleted in the working tree) isn't needed.
- **`Dockerfile`**: `INSTALL_LOCAL_EMBED=false` by default, so the slim image is the default. `true` installs CPU torch + `[local-embed]` and bakes BGE in (`PREFETCH_MODEL` now only applies there). `libgomp1` is installed only for the torch image.
- **`Makefile`**: `install` (slim + UI), `install-local` (+ torch), `dev` (everything), `check-embed`, `docker-build` (slim), `docker-build-local`. Docker tags bumped to 2.4.0.
- **`.devcontainer`** installs `.[dev,ui,local-embed]`.

---

### API changes (backwards compatible)

| Endpoint | Change |
|---|---|
| `POST /chat` | Unchanged contract. Now `async`; engine logs carry the API's `X-Request-Id`. Retrieval infrastructure failures now return `STATIC_FALLBACK` instead of the off-topic refusal. |
| `POST /chat/stream` | **New.** Same body; `text/plain` token stream; 404 when disabled. |
| `POST /summarize` | **New.** `{recent_messages, summary}` → `{updated_summary}`; no-op below the trigger. |
| `GET /version` | **Added:** `embedding_backend`, `streaming_enabled`, `retrieval_cache_size`, `client_window.summary_trigger_after`. |

### Tests

`make test`: **155 passed** (104 existing + 51 new), `ruff check .` clean. Everything is offline; Pinecone, Groq and HF are mocked.

- **New files:**
  - `tests/test_config.py`: backend validation/normalisation, `auto` resolution, endpoint derivation, new defaults.
  - `tests/test_embedder.py`: backend selection, token requirement, prefix + normalisation, 503/429/network retries, give-up after `HF_MAX_RETRIES`, 401 fails fast, wrong dimension, missing `[local-embed]` message.
  - `tests/test_retriever_cache.py`: hit/miss, case/whitespace folding, size 0, empty results not cached, defensive copies, LRU eviction, async/sync share the cache, `RetrievalError`.
  - `tests/test_stream.py`: `StreamSanitizer` (split markers, split/unclosed `<think>`, cap), streaming happy path/refusal/failure/unhandled error, `achat` == `chat`, **guard ∥ retrieval timing**, unsafe-wins-over-retrieval-failure, retrieval failure → snag not off-topic, `/summarize` trigger rule, stream tier semantics (fallback before first token, no splice after, whitespace preserved, empty primary).
- **`tests/test_api.py`:** `/chat/stream` content type, request id, refusal, 422, disabled → 404; `/summarize` both sides of the trigger; `/version` fields.
- **Existing tests, adjusted plumbing only:**
  - `conftest.py` fakes gained async methods that delegate to the scripted sync ones, and pin `EMBEDDING_BACKEND=local` / empty `HF_TOKEN` so a developer's `.env` can't change test behaviour.
  - `test_retriever.py` now patches `rag.embedder.load_embedder`, so the query-prefix assertion runs through the real `LocalBGEEmbedder`.
  - `test_prompt.py::test_refusal_templates_embedded_in_prompt` asserts the openers + pointer instead of three full strings, as Phase 3 requires. The contact-pointer and variant tests are unchanged and still pass.

### Rollback levers (env only, no code change)

| Symptom | Lever |
|---|---|
| HF rate limits / errors (`hf_rate_limited`, `embed_fail`) | `EMBEDDING_BACKEND=local` (needs the torch image and ~1 GB RAM) |
| Streaming renders badly | `ENABLE_STREAMING_ENDPOINT=false`; Streamlit falls back to `/chat` automatically |
| Suspected stale retrievals | `RETRIEVAL_CACHE_SIZE=0` |
| Duplicate facts crowding top-k | `MMR_LAMBDA=0.7` |
| Follow-ups losing their referent | `RECENT_TURNS_IN_PROMPT=4` |
| Recycling disrupts sessions | raise `WORKER_MAX_REQUESTS`, or `0` to disable |
| jemalloc problems | remove the `LD_PRELOAD` / `MALLOC_CONF` lines from the Dockerfile |

### Known limitations and follow-ups

- **Groq free-tier rate limits.** During benchmarking, ~12 prompts of ~3k tokens within one minute hit the **tokens-per-minute** limit on `gpt-oss-120b`. The 20b fallback absorbed the first one, then both tiers were limited and requests got `STATIC_FALLBACK`. A burst of beta testers can hit the same wall. Watch for `llm_fail … 429` in the logs.
- The Docker image was **not built in this environment** (no Docker available). The slim install was verified with a fresh venv (`pip install .` → API boots on the `hf` backend, ~100 MB worker), and the supervisor/recycling was verified with uvicorn directly. Build once on Railway and check the boot logs for `embedder_backend` `"hf"` and no `ld.so` preload errors.
- HF vector parity with the index was not verified here (no `HF_TOKEN` in this environment). Run `make check-embed` once before switching Railway to `hf`.
- Deferred by decision: Redis/semantic caching, paid tiers, `PineconeAsyncio` / `httpx.AsyncClient` rewrites.

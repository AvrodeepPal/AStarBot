# Software Requirements Specification (SRS)

## Project: **AStarBot v2 — Backend (RAG-based Portfolio Assistant)**

Version 2.0.0 · 2026-09-19 · supersedes the v1 SRS

---

## 1. Introduction

### 1.1 Purpose

AStarBot is a **backend-only, Retrieval-Augmented Generation (RAG) chatbot** that answers queries about Avrodeep Pal's **education, projects, skills, and interests** from a controlled, verified knowledge base.

The backend is deterministic, stateless, grounded (no hallucination by construction), guarded in depth, and deployable as a containerised API.

### 1.2 Scope

Covers: knowledge ingestion and retrieval, guardrails, prompt design, LLM fallback, stateless memory, API / Streamlit / CLI interfaces, Docker deployment (Railway or similar).

Excludes: the React frontend (a consumer of the API), authentication, user accounts, long-term personalisation, agents, tools, MCP, LangGraph, streaming.

---

## 2. System Overview

```
Question
   ↓
Input guardrails (sanitise, bound, regex injection screen)
   ↓
Prompt-guard LLM (llama-prompt-guard-2-86m, fail-open)
   ↓
Retrieval (BGE-base 768-d local embedding → Pinecone, k=5)
   ↓
Prompt assembly (7 versioned blocks; system turn = rules, user turn = data)
   ↓
LLM chain (gpt-oss-120b → gpt-oss-20b → static fallback)
   ↓
Output guardrails (leak stripping, length cap)
   ↓
Conditional summary update (gpt-oss-20b, temp 0)
   ↓
{answer, updated_summary}
```

Key philosophy: **explicit control > autonomous behaviour**.

---

## 3. Functional Requirements

### 3.1 Knowledge Ingestion
- The system SHALL read `data/*.json`; each entry SHALL contain `id`, `text`, `tags`.
- `scripts/embed.py` SHALL embed each entry as one vector (no chunking) with `BAAI/bge-base-en-v1.5`, normalised, **without** the query prefix.
- It SHALL clear the namespace before upserting (idempotent) and SHALL refuse to run if the index dimension ≠ 768.

### 3.2 Query Handling
- The system SHALL embed the question with the BGE query prefix and retrieve **top k = 5** entries.
- The system SHALL answer **only from retrieved context**.
- If no context is retrieved, the system SHALL return `REFUSAL_OFFTOPIC`.

### 3.3 Guardrails
- Input SHALL be stripped of control/ANSI characters and truncated at word boundaries (question ≤ 500, message ≤ 2000, summary ≤ 2000 chars, ≤ 20 messages, roles ∈ {user, assistant}).
- Questions matching known injection patterns SHALL be answered with `REFUSAL_JAILBREAK` without calling any LLM.
- When enabled, the prompt-guard model SHALL classify the question; unsafe → `REFUSAL_UNSAFE`. Guard failures SHALL fail open.
- The prompt SHALL instruct the model to treat summary, context and question as data, never instructions.
- Model output SHALL be stripped of prompt markers and capped at 1200 chars.
- Refusals SHALL use fixed strings: private, off-topic, unsafe, jailbreak, static fallback.

### 3.4 Context Memory
- The system SHALL NOT store memory server-side.
- The client SHALL send `recent_messages` and `summary`; the server SHALL return `updated_summary`.
- The server SHALL regenerate the summary when it receives ≥ 10 turns; the summary SHALL be ≤ 5 lines.
- Clients SHALL keep ≤ 12 turns, and only the last 4 once a fresh summary arrives.

### 3.5 LLM Fallback
- The system SHALL try the primary model, then the fallback model on **any** exception, then return `STATIC_FALLBACK`.
- Fallback logic SHALL be explicit try/except; no framework retries.
- The tier that served each request SHALL be logged.

### 3.6 Interfaces
- FastAPI: `POST /chat`, `GET /health`, `GET /version`; `X-Request-Id` on every response; CORS from `FRONTEND_ORIGIN`.
- Streamlit: chat UI with beta disclaimer, sidebar diagnostics, clear button, debug toggle.
- CLI: `/clear`, `/debug`, `/summary`, `/help`, `/exit`.

---

## 4. Non-Functional Requirements

| Area | Requirement |
|------|-------------|
| Reliability | No single LLM dependency; every failure path returns a well-formed response |
| Performance | Embedding local (no network); one chat LLM call per request plus optional guard and summary calls |
| Security | No tools, no web access, no code execution; defense-in-depth against prompt injection; strict payload limits; non-root container |
| Observability | JSON logs with `request_id`, `prompt_version`, `latency_ms`, `llm_tier_used`, `retrieval_top_score`; no PII at INFO |
| Maintainability | Typed config, versioned prompt, isolated modules, offline test suite |
| Portability | Multi-stage Docker image, CPU-only torch, model baked in |

---

## 5. Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| API | FastAPI + Uvicorn |
| Config | pydantic-settings |
| Embeddings | sentence-transformers · `BAAI/bge-base-en-v1.5` (768-d, local) |
| Vector DB | Pinecone (index `astarbot`, cosine, aws us-east-1) |
| LLMs | Groq via langchain-groq: `llama-prompt-guard-2-86m`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b` |
| UI (beta) | Streamlit |
| Container | Docker (multi-stage, non-root) |
| Deployment | Railway or similar (API); Streamlit Community Cloud (beta UI) |
| Tests | pytest, httpx |

---

## 6. Folder Structure

```
astarbot/
├── main.py                 uvicorn entry point
├── rag/                    config, log, guardrails, prompt, retriever, llm, memory, engine
├── interfaces/             api.py, streamlit_app.py, cli.py, session.py
├── scripts/embed.py        index rebuild
├── data/                   knowledge base JSON
├── tests/                  offline tests
├── Dockerfile, .dockerignore, Makefile, pyproject.toml, requirements.txt (shim), .env.example
└── README.md, SRS.md, CHANGELOG.md
```

---

## 7. Data Design

### 7.1 Knowledge entry
```json
{ "id": "education-001", "text": "I'm currently pursuing my MCA at Jadavpur University…", "tags": ["education", "background"] }
```

### 7.2 Pinecone metadata
`text`, `tags`, `source` (file stem). Vector id = entry `id`.

---

## 8. API Design

### 8.1 Request
```json
{ "question": "string (1–500)", "recent_messages": [{ "role": "user|assistant", "content": "string (1–2000)" }], "summary": "string | null (≤2000)" }
```

### 8.2 Response
```json
{ "answer": "string", "updated_summary": "string | null" }
```

### 8.3 Version
```json
{ "version": "2.0.0", "prompt_version": "v2.1.0", "embedding_model": "BAAI/bge-base-en-v1.5", "embedding_dim": 768, "top_k": 5, "primary_model": "...", "fallback_model": "...", "initial_message": "...", "client_window": { "max_recent_messages": 12, "messages_after_summary": 4 } }
```

---

## 9. Environment Configuration

See `.env.example`; every variable maps to a field in `rag/config.py`. Required: `PINECONE_API_KEY`, `GROQ_API_KEY`. Everything else has a default.

---

## 10. Deployment Architecture

```
React frontend ──CORS──▶ FastAPI container (Railway) ──▶ Pinecone
                                   │                 └──▶ Groq
                                   └── local BGE embeddings (in-process)
Streamlit Cloud (beta) ── same RAGEngine in-process
```

Single stateless container, port from `PORT`, horizontally scalable.

---

## 11. Design Rationale

> AStarBot is retrieval-centric, not action-centric. Agents, tools, or orchestration frameworks would add complexity without improving answer quality for a fixed, curated knowledge base. The system therefore uses explicit RAG orchestration, layered guardrails, an explicit LLM fallback chain, and stateless summarised memory.

---

## 12. Project Status

v2 implemented: core, interfaces, tests, container. Pending: populate the new 768-d index (`make embed`), deploy API, wire the React frontend to `/chat` and `/version`.

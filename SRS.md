# Software Requirements Specification (SRS)

## Project: **AStarBot - Backend (RAG-based Portfolio Assistant)**

---

## 1. Introduction

### 1.1 Purpose

AStarBot is a **backend-only, Retrieval-Augmented Generation (RAG) chatbot system** designed to answer queries about a student’s **education, projects, skills, and interests** using a controlled, verified knowledge base.

The backend is:

* deterministic
* stateless
* secure against hallucination
* deployable as a containerized API service

This document defines the **architecture, technologies, data flow, and constraints** of the system.

---

### 1.2 Scope

This SRS covers:

* Backend logic only
* Knowledge ingestion & retrieval
* Memory summarization (no persistent storage)
* API and CLI interfaces
* Deployment via Docker (Koyeb)

This SRS **excludes**:

* Frontend/UI
* Authentication
* User accounts
* Long-term personalization
* Agents, tools, MCP, LangGraph

---

### 1.3 Definitions & Acronyms

| Term      | Meaning                              |
| --------- | ------------------------------------ |
| RAG       | Retrieval-Augmented Generation       |
| LLM       | Large Language Model                 |
| k         | Number of retrieved context chunks   |
| MiniLM    | SentenceTransformer all-MiniLM-L6-v2 |
| Stateless | No server-side session persistence   |

---

## 2. System Overview

AStarBot follows a **classic RAG pipeline**:

```
User Query
   ↓
Context Retrieval (Pinecone, k=3)
   ↓
Prompt Assembly (System + Rules + Context + Memory)
   ↓
LLM Invocation (Groq → fallback)
   ↓
Response
   ↓
Conversation Summary Update
```

Key design philosophy:

> **Explicit control > autonomous behavior**

---

## 3. Functional Requirements

### 3.1 Knowledge Ingestion

* The system SHALL read structured JSON files from `/data`
* Each JSON entry SHALL contain:

  * `id`
  * `text`
  * `tags`
* A manual script SHALL embed all JSON files into Pinecone
* Embeddings SHALL use `sentence-transformers/all-MiniLM-L6-v2`

---

### 3.2 Query Handling

* The system SHALL accept a user question
* The system SHALL retrieve **top k = 3** relevant chunks
* The system SHALL answer **only from retrieved context**
* If context is insufficient, the system SHALL respond with a fallback message

---

### 3.3 Context Memory

* The system SHALL NOT store memory server-side
* The system SHALL support **conversation summarization**
* The system SHALL:

  * accept a summary string from the client
  * update the summary when message count exceeds a threshold
* The summary SHALL be concise (3–5 lines)

---

### 3.4 Safety & Refusal Handling

* The system SHALL refuse:

  * personal/private questions
  * unsupported topics
  * speculative queries
* The refusal SHALL:

  * be polite
  * be professional
  * redirect to allowed topics

---

### 3.5 Fallback Handling

* The system SHALL support a fallback LLM
* If the primary model fails (rate limit, error):

  * fallback model SHALL be invoked
* Fallback logic SHALL be explicit (try/except)

---

## 4. Non-Functional Requirements

### 4.1 Reliability

* No single LLM dependency
* Graceful degradation on failure

### 4.2 Performance

* Retrieval time < 1 second (Pinecone)
* Single LLM call per request (except summarization)

### 4.3 Security

* No external tools
* No web access
* No execution capabilities
* Context-only answering rule enforced

### 4.4 Maintainability

* Modular folder structure
* Explicit orchestration logic
* No hidden framework magic

---

## 5. Technology Stack

### 5.1 Core Technologies

| Layer         | Technology           |
| ------------- | -------------------- |
| Language      | Python 3.12          |
| API Framework | FastAPI              |
| Container     | Docker               |
| Deployment    | Koyeb                |
| LLM Provider  | Groq                 |
| Embeddings    | SentenceTransformers |
| Vector DB     | Pinecone             |

---

### 5.2 AI / NLP Components

* Primary LLM (configurable via `.env`)
* Fallback LLM (configurable via `.env`)
* Embedding model: `all-MiniLM-L6-v2`
* Non-streaming LLM calls (`invoke()`)

---

## 6. Folder Structure

```
astarbot/
├── data/                    # Knowledge base (JSON)
│   ├── self_data.json
│   ├── education.json
│   └── projects.json
│
├── rag/                     # Core RAG logic
│   ├── engine.py            # Retrieval + prompt + LLM orchestration
│   ├── retriever.py         # Pinecone connection & k=3 search
│   ├── prompt.py            # System, refusal, fallback prompts
│   └── memory.py            # Conversation summarization logic
│
├── scripts/
│   └── embed.py             # Manual embedding pipeline
│
├── api.py                   # FastAPI routes + exception handling
├── cli.py                   # Local testing interface
├── main.py                  # Uvicorn entry point
│
├── Dockerfile
├── requirements.txt
└── .env
```

---

## 7. Data Design

### 7.1 JSON Knowledge Format

```json
{
  "id": "education-001",
  "text": "I'm currently pursuing my MCA at Jadavpur University...",
  "tags": ["education", "background"]
}
```

### 7.2 Pinecone Metadata

Each vector stores:

* `id`
* `text`
* `tags`
* `source_file`

---

## 8. API Design

### 8.1 Request Schema (Conceptual)

```json
{
  "session_id": "string",
  "question": "string",
  "recent_messages": [],
  "summary": "string | null"
}
```

### 8.2 Response Schema

```json
{
  "answer": "string",
  "updated_summary": "string | null"
}
```

---

## 9. Environment Configuration

### 9.1 `.env` File

```env
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
PINECONE_ENV=

GROQ_API_KEY=

PRIMARY_LLM_MODEL=
FALLBACK_LLM_MODEL=
TEMPERATURE=0.7

EMBEDDING_MODEL=
TOP_K_RETRIEVAL=

MAX_RECENT_MESSAGES=
ENABLE_SUMMARY=

DEBUG=
LOG_LEVEL=
```

---

## 10. Deployment Architecture

```
Client (Frontend)
      ↓
FastAPI (Docker Container)
      ↓
RAG Engine
  ↙        ↘
Pinecone   Groq LLM
```

* Single container
* Exposes port 8000
* Stateless by design

---


## 11. Design Rationale (Interview-Ready)

> *“AStarBot is retrieval-centric, not action-centric. Introducing agents, tools, or orchestration frameworks would increase complexity without improving answer quality. Therefore, the system uses explicit RAG orchestration with stateless summarized memory.”*

---

## 12. Project Status

Requirements finalized
Architecture locked
Technology stack finalized
Implementation phase pending

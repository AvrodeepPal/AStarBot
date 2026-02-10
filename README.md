# AStarBot – Backend for a Personal AI Portfolio Assistant

AStarBot is a **backend-only Retrieval-Augmented Generation (RAG) system** designed to power a personal AI assistant for Avrodeep Pal.
The system answers questions about education, projects, skills, and interests using a **curated knowledge base** and modern large language models, while maintaining strict control over accuracy, tone, and scope.

This project prioritizes **engineering clarity, determinism, and maintainability** over autonomous or agent-based behavior.

---

## 1. Project Motivation

Personal portfolio chatbots often suffer from one or more of the following issues:

* hallucinated or exaggerated information
* poor grounding in verified data
* excessive framework abstraction (agents, tools, orchestration layers)
* fragile session memory implementations
* unclear separation between data, prompts, and model logic

AStarBot was built to address these issues by following a few core principles:

* **All factual answers must come from retrieved data**
* **Memory should be explicit, bounded, and inspectable**
* **The backend should remain stateless**
* **Complexity should be earned, not assumed**

---

## 2. System Overview

AStarBot is a classic RAG pipeline with a deliberately minimal surface area.

At a high level, the system performs the following steps for every query:

1. Embed the user query
2. Retrieve the top-k most relevant knowledge entries
3. Assemble a constrained prompt using:

   * system identity
   * hard safety rules
   * retrieved context
   * summarized conversation memory
4. Invoke an LLM to generate a response
5. Optionally update the conversation summary
6. Return the answer and updated summary to the client

The backend itself **does not store sessions or user state**.

---

## 3. Architectural Design

### 3.1 High-Level Data Flow

```
Client (CLI / Frontend)
        |
        |  question, recent_messages, summary
        v
FastAPI (/chat)
        |
        v
RAG Engine
 ├── Retriever (Pinecone, k=3)
 ├── Prompt Builder
 ├── Primary LLM
 ├── Fallback LLM
 └── Summarization LLM (optional)
        |
        v
Response + Updated Summary
```

---

### 3.2 Stateless Memory via Summarization

Rather than using Redis or server-side session storage, AStarBot uses **stateless summarized memory**:

* The client sends:

  * recent conversation turns (bounded window)
  * a short summary of earlier conversation
* When the window grows beyond a threshold, the backend:

  * generates a new summary using a lightweight LLM
  * returns the updated summary to the client
* The client stores and sends this summary on the next request

This approach has several advantages:

* no external memory infrastructure
* no session coupling to a specific server instance
* easy horizontal scaling
* explicit, inspectable memory state

---

## 4. Knowledge Base Design

### 4.1 Data Format

All knowledge is stored in JSON files under `/data`.

Each entry contains:

* a **semantic ID** (stable, human-readable)
* text content
* descriptive tags

Example:

```json
{
  "id": "project-002",
  "text": "Built a stock price prediction model using LSTM networks...",
  "tags": ["project", "machine-learning", "lstm"]
}
```

Design choices:

* **Semantic IDs** are preferred over UUIDs for debuggability
* No chunking is used; each JSON entry maps to one vector
* The dataset is assumed to be curated and relatively small

---

### 4.2 Embedding & Indexing

* Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2`
* Vector dimensionality: 384
* Similarity metric: cosine
* Storage: Pinecone

The ingestion script (`scripts/embed.py`) performs a **full rebuild**:

* deletes all existing vectors in the namespace
* re-embeds all JSON entries
* upserts them with metadata

This makes the process idempotent and avoids stale or duplicated data.

---

## 5. Retrieval Strategy

* Top-k retrieval (`k = 3`)
* No metadata filtering at query time
* Metadata is preserved for future extensions (e.g., scoring, tone control)

The system intentionally avoids:

* hybrid search
* reranking models
* agent-driven retrieval

The goal is **predictable, explainable retrieval**, not maximum recall at all costs.

---

## 6. Prompt Engineering Strategy

Prompting is handled manually (string-based), not via high-level abstractions.

The prompt is composed of several conceptual layers:

1. **System identity**

   * defines who AStarBot is
   * constrains scope to portfolio-related topics

2. **Hard RAG rules**

   * answer only from provided context
   * no guessing or external knowledge

3. **Safety and refusal rules**

   * polite refusal for unsupported or personal questions
   * redirection to allowed topics

4. **Style and tone guidelines**

   * professional but approachable
   * concise, non-robotic responses
   * length constraints

5. **Conversation summary (if present)**

6. **Retrieved context**

7. **User question**

This structure makes it easy to reason about **why** the model answered in a certain way.

---

## 7. Model Strategy

### 7.1 Separation of Model Responsibilities

The system deliberately uses **different models for different tasks**:

* **Primary LLM**

  * used for answering user questions
  * chosen for reasoning quality and availability

* **Fallback LLM**

  * used when the primary model hits rate limits or errors
  * prioritizes speed and reliability

* **Summarization LLM**

  * used only for memory summarization
  * deterministic, low temperature, low cost

This separation reduces coupling and improves robustness.

---

### 7.2 Invocation Style

* All model calls use `.invoke()`
* No streaming
* No tool calls
* No agent loops

This keeps execution linear and debuggable.

---

## 8. API Design

### 8.1 Endpoint Overview

* `POST /chat` – main interaction endpoint
* `GET /health` – health check

The API is intentionally minimal.

---

### 8.2 Request Contract

```json
{
  "question": "string",
  "recent_messages": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "summary": "string | null"
}
```

---

### 8.3 Response Contract

```json
{
  "answer": "string",
  "updated_summary": "string | null"
}
```

This contract makes frontend integration straightforward and explicit.

---

## 9. CLI Tooling

AStarBot includes a CLI (`cli.py`) for local testing.

The CLI:

* simulates a client-managed memory window
* updates and stores summaries
* prints answers directly to stdout

It is useful for:

* prompt tuning
* retrieval debugging
* sanity-checking summaries
* testing fallback behavior

---

## 10. Deployment Model

### 10.1 Containerization

* Python 3.12
* Slim base image
* Single container
* No sidecar services

### 10.2 Runtime Characteristics

* Stateless
* Single worker by default
* Horizontal scaling supported via platform (e.g. Koyeb)

---

## 11. Explicit Non-Goals

AStarBot intentionally does **not** include:

* autonomous agents
* tool calling
* web browsing
* long-term personalization
* user authentication
* fine-tuned models

These were excluded to maintain clarity, correctness, and maintainability.

---

## 12. What This Project Demonstrates

From an engineering perspective, AStarBot demonstrates:

* practical RAG system design
* disciplined prompt engineering
* stateless memory via summarization
* controlled fallback strategies
* production-ready API design
* clear separation of concerns

It is designed to be **understandable, extensible, and defensible**.

---

## 13. License

MIT License.

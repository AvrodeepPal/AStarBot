"""AStarBot v2 — grounded, bounded, stateless RAG core.

Modules:
    config      typed settings loaded once from the environment
    log         structured JSON logging
    guardrails  input sanitisation, injection detection, output sanitisation
    prompt      versioned, layered prompt blocks and refusal templates
    followup    follow-up question -> standalone retrieval query
    embedder    query embeddings: local BGE or Hugging Face Inference API
    retriever   Pinecone top-k search + priority re-rank + MMR + LRU cache
    llm         multi-tier Groq chain (guard / primary / fallback / summarizer)
    memory      stateless conversation summarisation
    engine      orchestrator that wires the pipeline together
"""

__version__ = "2.4.0"

"""AStarBot v2 — grounded, bounded, stateless RAG core.

Modules:
    config      typed settings loaded once from the environment
    log         structured JSON logging
    guardrails  input sanitisation, injection detection, output sanitisation
    prompt      versioned, layered prompt blocks and refusal templates
    followup    follow-up question -> standalone retrieval query
    retriever   local BGE embeddings + Pinecone top-k search + MMR
    llm         multi-tier Groq chain (guard / primary / fallback / summarizer)
    memory      stateless conversation summarisation
    engine      orchestrator that wires the pipeline together
"""

__version__ = "2.1.0"

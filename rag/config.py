"""Typed runtime configuration.

Every environment variable the application reads is declared here, once.
`Settings()` is instantiated at import time. `PINECONE_API_KEY` and
`GROQ_API_KEY` default to "" so this module (and anything that only needs
client-side constants, like `interfaces.session`) can be imported without
those secrets — e.g. the Streamlit UI, which talks to the API over HTTP and
never touches Pinecone/Groq directly. Code that actually calls those
services (`rag.retriever`, `rag.llm`) still fails fast if the key is blank.

Values are read from the process environment first, then from a `.env`
file in the working directory. Field names map to upper-cased env vars
(`top_k` <- `TOP_K`).
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Pinecone --------------------------------------------------------
    pinecone_api_key: str = ""
    pinecone_index_name: str = "astarbot"
    pinecone_namespace: str = "astarbot"

    # ---- Embedding (local BGE or Hugging Face Inference API) -------------
    # Where the QUERY encoder runs. Same model and dimension either way, so the
    # Pinecone index never changes; only the process that encodes does.
    #   "hf"    Hugging Face Inference API; needs HF_TOKEN; no torch in RAM
    #   "local" sentence-transformers in-process; needs the [local-embed] extra
    #   "auto"  "hf" when HF_TOKEN is set, otherwise "local"
    embedding_backend: str = "auto"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    embedding_device: str = "cpu"  # "cuda" / "mps" if available
    # BGE is asymmetric: this prefix goes on QUERIES only, never on documents.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    hf_token: str = ""
    # Empty = derived from EMBEDDING_MODEL (see `hf_embedding_endpoint`).
    hf_embedding_url: str = ""
    hf_timeout_seconds: float = 15.0
    hf_max_retries: int = Field(default=3, ge=1, le=6)

    # ---- Retrieval -------------------------------------------------------
    top_k: int = 5
    # Pinecone ranks by cosine only, so we over-fetch and re-rank locally to let
    # `priority` break ties between near-equally similar entries.
    fetch_k: int = 10
    # final = cosine + priority_weight * priority (priority runs 2..5).
    # At 0.02 a priority-5 entry gains 0.10 over a priority-0 one: enough to
    # reorder neighbours a few hundredths apart, never enough to promote a
    # genuinely worse match. Set to 0.0 to rank by cosine alone.
    priority_weight: float = 0.02
    # Matches whose RAW cosine score is below this are dropped before re-ranking.
    # Default 0.0 (off), and that is a measured decision, not laziness:
    # `make calibrate` on the current knowledge base shows the on-topic and
    # off-topic score distributions OVERLAP (worst on-topic 0.498, best
    # off-topic 0.614 — "recipe for biryani" is genuinely close to pers-food).
    # No threshold separates them, so topic refusal belongs to the prompt's
    # SCOPE block, not to a number here. Re-run `make calibrate` after editing
    # data/*.json; only raise this if the two groups actually separate.
    min_retrieval_score: float = 0.0
    # Maximal marginal relevance over the over-fetched candidates. The FAQ
    # file deliberately mirrors self/experience entries (faq-gate vs self-gate,
    # faq-location vs self-location-contact), so plain top-k spends two or
    # three of its five slots on the same fact. MMR picks each next result by
    #   mmr_lambda * relevance - (1 - mmr_lambda) * max_similarity_to_picked
    # so a near-duplicate is skipped in favour of something new. 1.0 disables
    # it (pure relevance order); 0.7 is a conventional starting point.
    # Default 1.0 since v2.4.0: `make calibrate` shows no duplicate crowding in
    # top-k, and MMR costs a 10 x 768-float `include_values` payload plus a
    # numpy similarity matrix on every request.
    mmr_lambda: float = 1.0
    # In-process LRU over finished retrievals, keyed on the normalised
    # retrieval query. 64 entries is well under 1 MB. 0 disables it.
    retrieval_cache_size: int = Field(default=64, ge=0)

    # ---- Groq LLM tiers --------------------------------------------------
    groq_api_key: str = ""
    guard_model: str = "meta-llama/llama-prompt-guard-2-86m"
    primary_llm_model: str = "openai/gpt-oss-20b"
    fallback_llm_model: str = "openai/gpt-oss-120b"
    summarizer_llm_model: str = "openai/gpt-oss-20b"
    temperature: float = 0.4
    summarizer_temperature: float = 0.0
    max_answer_tokens: int = 400
    # Reasoning effort for models that support it. gpt-oss on Groq accepts only
    # low / medium / high and cannot switch reasoning off; EMPTY does not mean
    # "none", it means "don't send the parameter", which leaves the provider
    # default (medium for gpt-oss). "low" is therefore the fastest setting.
    # Reasoning tokens are generated before the first visible token and are
    # drawn from the same max_tokens budget as the answer.
    reasoning_effort: str = "low"
    # Prompt-guard emits a jailbreak probability; >= threshold is treated as unsafe.
    guard_threshold: float = 0.5
    llm_timeout_seconds: float = 30.0

    # ---- Memory ----------------------------------------------------------
    enable_summary: bool = True
    # How many of the most recent turns ride in the user prompt as a labelled
    # RECENT EXCHANGE block. This is short-term memory so "how did he do it"
    # has a referent; the summary remains the long-term memory. 0 disables.
    # Counted in messages: 2 = the last user question + the answer to it,
    # which is all a referent like "it" ever points back to.
    recent_turns_in_prompt: int = 2
    summary_trigger_after: int = 10   # summarise once the client window reaches this many turns
    summary_max_lines: int = 5
    max_recent_messages: int = 12     # window size clients keep between requests
    messages_after_summary: int = 4   # turns clients keep once a fresh summary arrives

    # ---- Guardrails ------------------------------------------------------
    max_question_chars: int = 500
    max_message_chars: int = 2000
    max_summary_chars: int = 2000
    max_messages_sent: int = 20
    max_answer_chars: int = 2000
    enable_prompt_guard_llm: bool = True

    # ---- API -------------------------------------------------------------
    # POST /chat/stream (token streaming). /chat is always available.
    enable_streaming_endpoint: bool = True
    # Recycle the uvicorn worker after this many requests to cap slow RSS
    # growth; main.py keeps a supervisor alive to respawn it. 0 disables.
    worker_max_requests: int = Field(default=500, ge=0)

    # ---- CORS ------------------------------------------------------------
    # Comma-separated list of allowed origins, or "*" for any.
    frontend_origin: str = "*"

    # ---- Runtime ---------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    debug: bool = False

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        return v.upper()

    @field_validator("embedding_backend")
    @classmethod
    def _check_backend(cls, v: str) -> str:
        # Fail at startup: a typo silently falling back to "local" would load
        # torch on a 512 MB container, the exact thing "hf" exists to avoid.
        v = v.strip().lower()
        if v not in {"auto", "hf", "local"}:
            raise ValueError("EMBEDDING_BACKEND must be 'auto', 'hf' or 'local'")
        return v

    @field_validator("reasoning_effort")
    @classmethod
    def _normalise_reasoning(cls, v: str) -> str:
        return v.strip().lower()

    @property
    def resolved_embedding_backend(self) -> str:
        """The backend actually used: "auto" resolved against HF_TOKEN."""
        if self.embedding_backend == "auto":
            return "hf" if self.hf_token else "local"
        return self.embedding_backend

    @property
    def hf_embedding_endpoint(self) -> str:
        return self.hf_embedding_url or (
            f"https://router.huggingface.co/hf-inference/models/{self.embedding_model}"
            "/pipeline/feature-extraction"
        )

    @property
    def cors_origins(self) -> list[str]:
        """Parse FRONTEND_ORIGIN into the list CORSMiddleware expects."""
        return [o.strip() for o in self.frontend_origin.split(",") if o.strip()] or ["*"]


settings = Settings()

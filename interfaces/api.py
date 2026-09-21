"""FastAPI backend — the deployment target (Railway or similar).

Stateless by design: the client sends `recent_messages` + `summary` and
receives `updated_summary` back. No sessions, no auth, no streaming.

Endpoints:
    POST /chat      main chat
    GET  /health    liveness
    GET  /version   app + prompt + embedding info
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import __version__
from rag.config import settings
from rag.engine import RAGEngine
from rag.log import setup_logging
from rag.prompt import INITIAL_MESSAGE, PROMPT_VERSION

setup_logging(settings.log_level)
log = logging.getLogger("astarbot.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loads the embedding model and connects to Pinecone once per process.
    app.state.engine = RAGEngine()
    yield


app = FastAPI(
    title="AStarBot API",
    description="Grounded, bounded, stateless RAG backend for AStarBot",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a short request id to every request/response for log correlation."""
    request.state.request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:8]
    response = await call_next(request)
    response.headers["X-Request-Id"] = request.state.request_id
    return response


# ---- schemas -------------------------------------------------------------
# Limits here mirror rag.guardrails: FastAPI rejects absurd payloads with 422
# before they ever reach the engine.


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=settings.max_message_chars)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=settings.max_question_chars)
    recent_messages: list[Message] = Field(default_factory=list, max_length=settings.max_messages_sent)
    summary: str | None = Field(default=None, max_length=settings.max_summary_chars)


class ChatResponse(BaseModel):
    answer: str
    updated_summary: str | None = None


# ---- routes --------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest, request: Request) -> ChatResponse:
    log.info(
        "chat_request",
        extra={
            "request_id": request.state.request_id,
            "n_messages": len(payload.recent_messages),
            "has_summary": payload.summary is not None,
        },
    )
    result = request.app.state.engine.chat(
        question=payload.question,
        recent_messages=[m.model_dump() for m in payload.recent_messages],
        summary=payload.summary,
    )
    return ChatResponse(answer=result["answer"], updated_summary=result["updated_summary"])


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict:
    return {
        "version": __version__,
        "prompt_version": PROMPT_VERSION,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "top_k": settings.top_k,
        "primary_model": settings.primary_llm_model,
        "fallback_model": settings.fallback_llm_model,
        "initial_message": INITIAL_MESSAGE,
        "client_window": {
            "max_recent_messages": settings.max_recent_messages,
            "messages_after_summary": settings.messages_after_summary,
        },
    }

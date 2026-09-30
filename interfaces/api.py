"""FastAPI backend — the deployment target (Railway or similar).

Stateless by design: the client sends `recent_messages` + `summary` and
receives `updated_summary` back. No sessions, no auth.

Endpoints:
    POST /chat          main chat, one JSON response
    POST /chat/stream   same pipeline, answer streamed as raw UTF-8 text
    POST /summarize     fold a full client window into a fresh summary
                        (streaming clients; /chat does this inline)
    GET  /health        liveness
    GET  /version       app + prompt + embedding info + client window policy

All routes are `async def`: the engine awaits Groq, the embedder and
Pinecone (the last two in worker threads), so one worker serves several
visitors concurrently instead of blocking a thread-pool slot per request.
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    # Builds the embedder (local model or HF client) and connects to Pinecone
    # once per process. The warm-up runs in the background so /health answers
    # as soon as the engine exists; it wakes the HF model, opens the Pinecone
    # connection and seeds the retrieval cache.
    app.state.engine = RAGEngine()
    warmup = asyncio.create_task(app.state.engine.warmup())
    yield
    warmup.cancel()


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


class SummarizeRequest(BaseModel):
    recent_messages: list[Message] = Field(default_factory=list, max_length=settings.max_messages_sent)
    summary: str | None = Field(default=None, max_length=settings.max_summary_chars)


class SummarizeResponse(BaseModel):
    updated_summary: str | None = None


# ---- routes --------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, request: Request) -> ChatResponse:
    log.info(
        "chat_request",
        extra={
            "request_id": request.state.request_id,
            "n_messages": len(payload.recent_messages),
            "has_summary": payload.summary is not None,
        },
    )
    result = await request.app.state.engine.achat(
        question=payload.question,
        recent_messages=[m.model_dump() for m in payload.recent_messages],
        summary=payload.summary,
        request_id=request.state.request_id,
    )
    return ChatResponse(answer=result["answer"], updated_summary=result["updated_summary"])


@app.post("/chat/stream")
async def chat_stream_endpoint(payload: ChatRequest, request: Request) -> StreamingResponse:
    """Token-streamed /chat. The body is plain UTF-8 text, not SSE frames.

    Output guardrails run incrementally on the server (StreamSanitizer), so
    the client can render chunks as they arrive. `updated_summary` is not
    part of a stream; streaming clients call POST /summarize once their
    window reaches `summary_trigger_after` (see GET /version).
    """
    if not settings.enable_streaming_endpoint:
        raise HTTPException(status_code=404, detail="Streaming is disabled")
    log.info(
        "chat_stream_request",
        extra={
            "request_id": request.state.request_id,
            "n_messages": len(payload.recent_messages),
            "has_summary": payload.summary is not None,
        },
    )
    stream = request.app.state.engine.astream_chat(
        question=payload.question,
        recent_messages=[m.model_dump() for m in payload.recent_messages],
        summary=payload.summary,
        request_id=request.state.request_id,
    )
    return StreamingResponse(
        stream,
        media_type="text/plain; charset=utf-8",
        # Ask proxies (Railway's edge included) not to buffer the body.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_endpoint(payload: SummarizeRequest, request: Request) -> SummarizeResponse:
    updated = await request.app.state.engine.asummarize(
        recent_messages=[m.model_dump() for m in payload.recent_messages],
        summary=payload.summary,
        request_id=request.state.request_id,
    )
    return SummarizeResponse(updated_summary=updated)


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.get("/version")
async def version() -> dict:
    return {
        "version": __version__,
        "prompt_version": PROMPT_VERSION,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "embedding_backend": settings.resolved_embedding_backend,
        "top_k": settings.top_k,
        "retrieval_cache_size": settings.retrieval_cache_size,
        "streaming_enabled": settings.enable_streaming_endpoint,
        "primary_model": settings.primary_llm_model,
        "fallback_model": settings.fallback_llm_model,
        "initial_message": INITIAL_MESSAGE,
        "client_window": {
            "max_recent_messages": settings.max_recent_messages,
            "messages_after_summary": settings.messages_after_summary,
            "summary_trigger_after": settings.summary_trigger_after,
        },
    }

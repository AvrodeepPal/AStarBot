# api.py

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

from rag.engine import RAGEngine


# App setup
app = FastAPI(
    title="AStarBot API",
    description="Backend RAG service for AStarBot",
    version="2.0.0",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("astarbot-api")

engine = RAGEngine()


# Request / Response models
class Message(BaseModel):
    role: str = Field(..., examples=["user", "assistant"])
    content: str


class ChatRequest(BaseModel):
    question: str
    recent_messages: Optional[List[Message]] = []
    summary: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    updated_summary: Optional[str] = None


# Routes
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    """
    Main chat endpoint.

    The backend is stateless:
    - recent_messages and summary are provided by the client
    - updated_summary is returned to the client
    """

    logger.info("Received chat request")

    result = engine.chat(
        question=payload.question,
        recent_messages=[m.model_dump() for m in payload.recent_messages],
        summary=payload.summary,
    )

    return ChatResponse(
        answer=result["answer"],
        updated_summary=result["updated_summary"],
    )


# Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}

# main.py
"""
FastAPI server for AStarBot RAG chatbot
Provides REST API endpoints for the chatbot functionality
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.rag_chain import generate_rag_response
import uvicorn

# FastAPI app instance
app = FastAPI(
    title="AStarBot RAG API",
    description="Backend API for Avrodeep Pal's portfolio chatbot",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    question: str
    
class ChatResponse(BaseModel):
    answer: str
    status: str = "success"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "🤖 AStarBot RAG backend is running!",
        "status": "healthy",
        "version": "1.0.0",
        "about": "AI assistant for Avrodeep Pal's portfolio"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "AStarBot RAG API",
        "endpoints": ["/", "/health", "/chat", "/chat-simple"]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for RAG queries
    
    Args:
        request: ChatRequest with user's question
        
    Returns:
        ChatResponse with AI-generated answer
    """
    try:
        # Validate input
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=400, 
                detail="Question cannot be empty"
            )
        
        # Generate RAG response
        answer = generate_rag_response(request.question.strip())
        
        return ChatResponse(answer=answer)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again."
        )

@app.post("/chat-simple")
async def chat_simple(request: dict):
    """
    Simplified chat endpoint for easy frontend integration
    Accepts: {"question": "your question here"}
    Returns: {"answer": "response here"}
    """
    try:
        question = request.get("question", "").strip()
        
        if not question:
            return {"error": "No question provided"}
        
        answer = generate_rag_response(question)
        return {"answer": answer}
        
    except Exception as e:
        print(f"❌ Error in simple chat endpoint: {e}")
        return {"error": "Something went wrong. Please try again."}

# Development server
if __name__ == "__main__":
    print("🚀 Starting AStarBot RAG API server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )

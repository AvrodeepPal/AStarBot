"""Uvicorn launcher for the FastAPI backend (Docker / Railway entry point)."""

import uvicorn

from rag.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "interfaces.api:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )

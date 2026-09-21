# main.py

import os
import uvicorn

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )

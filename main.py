"""Uvicorn launcher for the FastAPI backend (Docker / Railway entry point).

One worker process, recycled every WORKER_MAX_REQUESTS requests.

Why a supervisor for a single worker: `uvicorn --workers 1
--limit-max-requests N` does NOT restart anything — with one worker there is
no supervisor, so after N requests the process simply exits and the
container stops serving until the platform restarts it. uvicorn's
Multiprocess supervisor is only used for --workers > 1, so it is started
here explicitly with one worker instead:

    - the parent holds the listening socket for the life of the container
    - the worker drains in-flight requests (including open streams), exits
    - the parent respawns it; connections arriving meanwhile wait in the
      socket backlog for the ~2-3 s re-import instead of being refused

The parent never imports the app, so it costs a few tens of MB, and a single
worker means one embedder, one retrieval cache and one Pinecone pool.
WORKER_MAX_REQUESTS=0 runs a plain single-process server with no recycling.
"""

import uvicorn
from uvicorn.supervisors import Multiprocess

from rag.config import settings


def main() -> None:
    config = uvicorn.Config(
        "interfaces.api:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        workers=1,
        log_level=settings.log_level.lower(),
        limit_max_requests=settings.worker_max_requests or None,
        # Headroom for a stream to finish during a recycle; a slow client
        # can't hold the old worker open forever.
        timeout_graceful_shutdown=60,
    )
    if not settings.worker_max_requests:
        uvicorn.Server(config).run()
        return
    sock = config.bind_socket()
    Multiprocess(config, sockets=[sock]).run()


if __name__ == "__main__":
    main()

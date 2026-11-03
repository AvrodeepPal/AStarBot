"""Per-IP sliding-window rate limit for the paid endpoints.

In-memory, because main.py runs a single worker. A recycled worker starts
with an empty table, which is fine: the limit protects the Groq, Cloudflare
and Pinecone quotas from a script, not from a patient human.

Applied as a route dependency to /chat, /chat/stream and /summarize only;
/health and /version stay open because the frontends call them to wake the
server.
"""

import logging
import time
from collections import deque

from fastapi import HTTPException, Request

log = logging.getLogger("astarbot.api")

RATE_LIMIT_DETAIL = "Whoa, slow down! Please wait a minute before asking more questions."

# Drop idle IPs once the table reaches this size, so rotating addresses
# can't grow it without bound.
_SWEEP_AT = 10_000


def client_ip(request: Request) -> str:
    """Best guess at the visitor's IP behind Render's proxies.

    Render appends to a client-supplied X-Forwarded-For rather than
    replacing it, so the first entry can be spoofed. Prefer True-Client-IP,
    which Cloudflare (in front of Render) sets itself; fall back to the
    first X-Forwarded-For entry, then the socket peer (local dev, tests).
    """
    true_ip = request.headers.get("true-client-ip", "").strip()
    if true_ip:
        return true_ip
    forwarded = request.headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    if first:
        return first
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """At most `limit` requests per `window` seconds per IP (sliding window)."""

    def __init__(self, limit: int, window: float, clock=time.monotonic):
        self.limit = limit
        self.window = window
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> float | None:
        """Record a hit. Return None if allowed, else seconds until a slot frees."""
        if self.limit <= 0:
            return None
        now = self._clock()
        cutoff = now - self.window
        if len(self._hits) >= _SWEEP_AT:
            self._hits = {k: q for k, q in self._hits.items() if q and q[-1] > cutoff}
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= self.limit:
            return hits[0] + self.window - now
        hits.append(now)
        return None


async def rate_limit(request: Request) -> None:
    """FastAPI dependency: 429 once the caller's IP is over its limit."""
    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    ip = client_ip(request)
    retry_after = limiter.check(ip)
    if retry_after is not None:
        log.warning(
            "rate_limited",
            extra={"request_id": getattr(request.state, "request_id", None), "path": request.url.path},
        )
        raise HTTPException(
            status_code=429,
            detail=RATE_LIMIT_DETAIL,
            headers={"Retry-After": str(max(1, int(retry_after) + 1))},
        )

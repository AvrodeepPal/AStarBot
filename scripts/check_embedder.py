"""Check that Cloudflare Workers AI and local BGE produce the same query vectors.

    python -m scripts.check_embedder    # needs CF_API_TOKEN + CF_ACCOUNT_ID + the [local-embed] extra

The Pinecone index was built with local BGE (scripts/embed.py). Switching the
QUERY encoder to Workers AI is only safe if both produce the same vector for
the same text; this embeds a few probe questions both ways and prints the
cosine between them. Expect >= 0.999 (float noise only). Anything lower means
the API is pooling or prefixing differently (check CF_POOLING=cls): keep
EMBEDDING_BACKEND=local until that is resolved. Makes a handful of CF calls.
"""

import sys
import time

from rag.config import settings
from rag.embedder import CloudflareEmbedder, LocalBGEEmbedder

PROBES = [
    "Who is Avrodeep?",
    "what are his hobbies",
    "explain his credit risk eda project",
    "Is he preparing for GATE?",
    "how did he do it",
]
MIN_COSINE = 0.999


def main() -> int:
    cf, local = CloudflareEmbedder(), LocalBGEEmbedder()
    print(f"CF endpoint : {settings.cf_embedding_endpoint} (pooling={settings.cf_pooling})")
    worst = 1.0
    for q in PROBES:
        t0 = time.perf_counter()
        a = cf.embed_query(q)
        cf_ms = (time.perf_counter() - t0) * 1000
        b = local.embed_query(q)
        cos = sum(x * y for x, y in zip(a, b, strict=True))  # both unit-length
        worst = min(worst, cos)
        print(f"  {cos:.6f}  {cf_ms:6.0f} ms  {q}")
    ok = worst >= MIN_COSINE
    print(f"\nworst cosine {worst:.6f} -> {'OK, backends are interchangeable' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

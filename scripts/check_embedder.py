"""Check that the Hugging Face API and local BGE produce the same query vectors.

    python -m scripts.check_embedder        # needs HF_TOKEN + the [local-embed] extra

The Pinecone index was built with local BGE (scripts/embed.py). Switching the
QUERY encoder to the HF Inference API is only safe if both produce the same
vector for the same text; this embeds a few probe questions both ways and
prints the cosine between them. Expect >= 0.999 (float noise only). Anything
lower means the API is pooling or prefixing differently: keep
EMBEDDING_BACKEND=local until that is resolved. Makes a handful of HF calls.
"""

import sys
import time

from rag.config import settings
from rag.embedder import HFEmbedder, LocalBGEEmbedder

PROBES = [
    "Who is Avrodeep?",
    "what are his hobbies",
    "explain his credit risk eda project",
    "Is he preparing for GATE?",
]
MIN_COSINE = 0.999


def main() -> int:
    hf, local = HFEmbedder(), LocalBGEEmbedder()
    print(f"HF endpoint : {settings.hf_embedding_endpoint}")
    worst = 1.0
    for q in PROBES:
        t0 = time.perf_counter()
        a = hf.embed_query(q)
        hf_ms = (time.perf_counter() - t0) * 1000
        b = local.embed_query(q)
        cos = sum(x * y for x, y in zip(a, b, strict=True))  # both unit-length
        worst = min(worst, cos)
        print(f"  {cos:.6f}  {hf_ms:6.0f} ms  {q}")
    ok = worst >= MIN_COSINE
    print(f"\nworst cosine {worst:.6f} -> {'OK, backends are interchangeable' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

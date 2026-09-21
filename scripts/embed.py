"""Rebuild the Pinecone index from data/*.json.

    python -m scripts.embed            # full rebuild
    python -m scripts.embed --dry-run  # parse + embed locally, touch nothing

Every file in data/ is read, so adding a file needs no code change. Each entry
becomes exactly one vector (no chunking), embedded with the same local BGE
model the retriever uses and WITHOUT the query prefix (BGE is asymmetric:
prefix on queries only), normalised for cosine similarity.

What gets vectorised is `knowledge.embedding_text`: title + FAQ question +
text + aliases. Metadata keeps the clean `text` separate, so the prompt never
sees the alias soup that only exists to help retrieval.

Order of operations is deliberate: parse and embed EVERYTHING first, and only
then clear the namespace and upsert. A malformed file or a model failure
therefore aborts before anything is deleted. The clear-then-upsert also means
entries removed from the JSON (e.g. the old self_data.json) leave no stale
vectors behind.
"""

import argparse
import json
import logging
from pathlib import Path

from pinecone import Pinecone
from tqdm import tqdm

from rag.config import settings
from rag.knowledge import build_metadata, embedding_text, normalize_entry
from rag.log import setup_logging
from rag.retriever import load_embedder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BATCH_SIZE = 100

log = logging.getLogger("astarbot.embed")


def load_records() -> tuple[list[dict], dict[str, int]]:
    """Read every data/*.json. Skips malformed entries and duplicate ids with a warning.

    Accepts both the current ``{"meta": ..., "entries": [...]}`` shape and a
    bare top-level list.
    """
    files = sorted(DATA_DIR.glob("*.json"))
    if not files:
        raise RuntimeError(f"No JSON files found in {DATA_DIR}")

    records: list[dict] = []
    per_file: dict[str, int] = {}
    seen: dict[str, str] = {}  # id -> file it first appeared in

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            entries = payload.get("entries", [])
            source = (payload.get("meta") or {}).get("source") or path.stem
        elif isinstance(payload, list):
            entries, source = payload, path.stem
        else:
            raise RuntimeError(f"{path.name}: expected an object or a list at the top level")

        count = 0
        for raw in entries:
            try:
                entry = normalize_entry(raw, source)
            except (ValueError, TypeError) as exc:
                log.warning("skip_entry", extra={"file": path.name, "err": str(exc)})
                continue
            if entry["id"] in seen:
                log.warning(
                    "duplicate_id",
                    extra={"file": path.name, "id": entry["id"], "first_seen_in": seen[entry["id"]]},
                )
                continue
            seen[entry["id"]] = path.name
            records.append(entry)
            count += 1
        per_file[path.name] = count

    if not records:
        raise RuntimeError("No usable entries found in data/*.json")
    return records, per_file


def run_embedding(dry_run: bool = False) -> int:
    setup_logging(settings.log_level)

    print(f"Model     : {settings.embedding_model} ({settings.embedding_dim}-d, {settings.embedding_device})")
    print(f"Index     : {settings.pinecone_index_name}  namespace={settings.pinecone_namespace}")
    if dry_run:
        print("Mode      : DRY RUN (nothing is written to Pinecone)")

    # 1. Parse first: a bad file must abort before anything is deleted.
    records, per_file = load_records()
    print(f"Parsed    : {len(records)} entries from {len(per_file)} file(s)")

    # 2. Embed. Documents get NO query prefix.
    embedder = load_embedder()
    vectors = embedder.encode(
        [embedding_text(r) for r in records],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=True,
    )

    payload = [
        {"id": r["id"], "values": vec.tolist(), "metadata": build_metadata(r)}
        for r, vec in zip(records, vectors, strict=True)
    ]

    if dry_run:
        print("\nDry run — sample embedding input:")
        print("-" * 60)
        print(embedding_text(records[0]))
        print("-" * 60)
        _print_summary(per_file, len(payload), written=False)
        return len(payload)

    # 3. Only now touch Pinecone.
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    stats = index.describe_index_stats()
    index_dim = stats.get("dimension") if isinstance(stats, dict) else getattr(stats, "dimension", None)
    if index_dim and int(index_dim) != settings.embedding_dim:
        raise RuntimeError(
            f"Pinecone index is {index_dim}-d but the model is {settings.embedding_dim}-d. "
            "Create a new index with the matching dimension."
        )

    print(f"Clearing namespace '{settings.pinecone_namespace}' (removes vectors for deleted entries)…")
    try:
        index.delete(delete_all=True, namespace=settings.pinecone_namespace)
    except Exception as exc:  # noqa: BLE001 - an empty namespace raises on some SDKs
        log.warning("namespace_clear_failed", extra={"err": str(exc)[:200]})

    print("Upserting…")
    for i in tqdm(range(0, len(payload), BATCH_SIZE), unit="batch"):
        index.upsert(vectors=payload[i : i + BATCH_SIZE], namespace=settings.pinecone_namespace)

    _print_summary(per_file, len(payload), written=True)
    return len(payload)


def _print_summary(per_file: dict[str, int], total: int, written: bool) -> None:
    print("\nSummary")
    print(f"  {'file':<28}{'entries':>8}")
    for name, n in sorted(per_file.items()):
        print(f"  {name:<28}{n:>8}")
    target = f"-> namespace '{settings.pinecone_namespace}'" if written else "(not written)"
    print(f"  {'total':<28}{total:>8}  {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild the AStarBot Pinecone index.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and embed locally without writing to Pinecone.",
    )
    args = parser.parse_args()
    run_embedding(dry_run=args.dry_run)

# scripts/embed.py
"""
Rebuilds the Pinecone index from scratch.

Steps:
1. Deletes all vectors in the target namespace
2. Reads all JSON files from /data
3. Embeds each entry (no chunking)
4. Upserts vectors with semantic IDs

Safe to run multiple times.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

import pinecone
from sentence_transformers import SentenceTransformer


# Environment
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

NAMESPACE = "astarbot"
BATCH_SIZE = 100


# Sanity checks
if not all([PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME]):
    raise RuntimeError("Missing Pinecone configuration in .env")

print("🔹 Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)


# Pinecone init
print("🔹 Connecting to Pinecone...")
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV,
)

index = pinecone.Index(PINECONE_INDEX_NAME)


# HARD RESET: delete all vectors in namespace
print(f"Deleting ALL vectors in namespace '{NAMESPACE}'...")
index.delete(delete_all=True, namespace=NAMESPACE)
print("Namespace cleared.")


# Load data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
json_files = list(DATA_DIR.glob("*.json"))

if not json_files:
    raise RuntimeError("No JSON files found in /data directory.")

print(f"🔹 Found {len(json_files)} JSON files.")

vectors = []

for json_file in json_files:
    source_name = json_file.stem

    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    for record in records:
        vector_id = record["id"] # semantic ID
        text = record["text"]
        tags = record.get("tags", [])

        embedding = embedder.encode(text).tolist()

        vectors.append(
            (
                vector_id,
                embedding,
                {
                    "text": text,
                    "tags": tags,
                    "source": source_name,
                },
            )
        )


# Batch upsert
print(f"Uploading {len(vectors)} vectors to Pinecone...")

for i in tqdm(range(0, len(vectors), BATCH_SIZE)):
    batch = vectors[i : i + BATCH_SIZE]
    index.upsert(
        vectors=batch,
        namespace=NAMESPACE,
    )

print("Re-embedding complete. Index is fully rebuilt.")

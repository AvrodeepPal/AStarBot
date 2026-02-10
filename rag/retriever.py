# rag/retriever.py

import os
from dotenv import load_dotenv
import pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", 3))

NAMESPACE = "astarbot"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


class PineconeRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV,
        )
        self.index = pinecone.Index(PINECONE_INDEX_NAME)

    def retrieve(self, query: str) -> list[dict]:
        query_vector = self.embedder.encode(query).tolist()

        result = self.index.query(
            vector=query_vector,
            top_k=TOP_K,
            include_metadata=True,
            namespace=NAMESPACE,
        )

        contexts = []
        for match in result.get("matches", []):
            metadata = match.get("metadata", {})
            contexts.append(
                {
                    "id": match.get("id"),
                    "text": metadata.get("text", ""),
                    "tags": metadata.get("tags", []),
                    "score": match.get("score"),
                }
            )

        return contexts

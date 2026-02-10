# rag/retriever.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient
from sentence_transformers import SentenceTransformer

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
TOP_K = int(os.getenv("TOP_K_RETRIEVAL"))

NAMESPACE = os.getenv("PINECONE_INDEX_NAME")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


class PineconeRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        self.pc = PineconeClient(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

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

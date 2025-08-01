# rag/config.py
"""
Configuration module for RAG chatbot
Handles API setup for OpenRouter (Mistral 24B) and Pinecone
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Load environment variables
load_dotenv()

def setup_pinecone():
    """
    Initialize Pinecone client and return index
    Returns: Pinecone Index object
    """
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "starbot")
        return pc.Index(index_name)
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        raise

def get_openrouter_client():
    """
    Initialize OpenRouter client for Mistral 24B
    Returns: OpenAI client configured for OpenRouter
    """
    try:
        return OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
    except Exception as e:
        print(f"❌ Error setting up OpenRouter client: {e}")
        raise

def get_config():
    """
    Get all configuration values
    Returns: Dictionary with config values
    """
    return {
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_env": os.getenv("PINECONE_ENV", "us-east-1"),
        "pinecone_index_name": os.getenv("PINECONE_INDEX_NAME", "starbot"),
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "llm_model": os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3-0324:free"),
        "embedding_dimension": 384,
        "top_k_retrieval": int(os.getenv("TOP_K_RETRIEVAL", "3")),
        "llm_temperature": float(os.getenv("TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("MAX_TOKENS", "150"))
    }

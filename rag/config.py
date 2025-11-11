# rag/config.py
"""
Configuration module for RAG chatbot with Redis support
Handles API setup for OpenRouter (Mistral), Pinecone, and Redis
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
    Initialize OpenRouter client for Mistral/DeepSeek
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

def validate_redis_connection():
    """
    Test Redis connection
    Returns: bool indicating if Redis is available
    """
    try:
        import redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        r = redis.from_url(redis_url)
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}")
        print("💡 Chat memory will be disabled. Install Redis or check REDIS_URL in .env")
        return False

def get_config():
    """
    Get all configuration values including Redis settings
    Returns: Dictionary with config values
    """
    return {
        # Pinecone settings
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_env": os.getenv("PINECONE_ENV", "us-east-1"),
        "pinecone_index_name": os.getenv("PINECONE_INDEX_NAME", "starbot"),
        
        # OpenRouter/LLM settings
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
        "llm_model": os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3-0324:free"),
        "llm_temperature": float(os.getenv("TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("MAX_TOKENS", "150")),
        
        # Embedding settings
        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "embedding_dimension": 384,
        "top_k_retrieval": int(os.getenv("TOP_K_RETRIEVAL", "3")),
        
        # Redis settings
        "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
        "redis_chat_prefix": os.getenv("REDIS_CHAT_PREFIX", "chat_"),
        "redis_enabled": validate_redis_connection(),
        
        # Memory settings
        "memory_window_size": int(os.getenv("MEMORY_WINDOW_SIZE", "3")),
        "enable_memory": os.getenv("ENABLE_MEMORY", "true").lower() == "true"
    }

def print_config_status():
    """Print configuration status for debugging"""
    config = get_config()
    print("🔧 AStarBot Configuration Status:")
    print(f"  📊 Pinecone: {'✅' if config['pinecone_api_key'] else '❌'}")
    print(f"  🤖 OpenRouter: {'✅' if config['openrouter_api_key'] else '❌'}")
    print(f"  💾 Redis: {'✅' if config['redis_enabled'] else '❌'}")
    print(f"  🧠 Memory: {'✅' if config['enable_memory'] and config['redis_enabled'] else '❌'}")
    print(f"  🔍 Retrieval: Top {config['top_k_retrieval']} chunks")
    print(f"  💭 Memory Window: {config['memory_window_size']} exchanges")

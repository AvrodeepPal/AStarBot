# utils/memory.py
"""
Redis Cloud memory wrapper for AStarBot
Handles conversational memory with limited window size using Redis Cloud
"""

import os
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

class LimitedMemory(ConversationBufferMemory):
    """
    Custom memory class that stores only the last N exchanges in Redis Cloud
    """
    def __init__(self, *args, window_size=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size

    def load_memory_variables(self, inputs):
        """
        Return only the last N messages as context
        window_size * 2 because each exchange has user + AI message
        """
        messages = self.chat_memory.messages[-(self.window_size * 2):]
        return {self.memory_key: messages}

def get_redis_memory(session_id: str):
    """
    Create Redis Cloud-backed memory for chat history
    
    Args:
        session_id (str): Unique identifier for the chat session
        
    Returns:
        LimitedMemory: Memory object with Redis Cloud backend
    """
    # Redis Cloud URL from environment variable
    redis_url = os.getenv("REDIS_URL")
    prefix = os.getenv("REDIS_CHAT_PREFIX", "chat_")
    
    if not redis_url:
        raise ValueError("REDIS_URL environment variable is required")

    chat_history = RedisChatMessageHistory(
        url=redis_url,
        session_id=prefix + session_id,
        ttl=3600  # Optional: Set TTL to 1 hour (3600 seconds)
    )

    memory = LimitedMemory(
        chat_memory=chat_history,
        return_messages=True,
        memory_key="chat_history",
        window_size=3  # Keep last 3 exchanges
    )

    return memory

def clear_session_memory(session_id: str):
    """Clear memory for a specific session"""
    try:
        redis_url = os.getenv("REDIS_URL")
        prefix = os.getenv("REDIS_CHAT_PREFIX", "chat_")
        
        if not redis_url:
            print("❌ REDIS_URL not configured")
            return
            
        history = RedisChatMessageHistory(
            url=redis_url,
            session_id=prefix + session_id
        )
        history.clear()
        print(f"✅ Memory cleared for session: {session_id}")
    except Exception as e:
        print(f"❌ Error clearing memory: {e}")

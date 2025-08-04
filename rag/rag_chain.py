# rag/rag_chain.py (Updated sections)

from utils.memory import get_redis_memory, clear_session_memory

class RAGChatbot:
    def __init__(self, session_id="default"):
        """Initialize with Redis Cloud memory support"""
        self.session_id = session_id
        self.config = get_config()
        
        # Initialize Redis Cloud memory
        self.memory = None
        if self.config.get("enable_memory", True):
            try:
                print("🧠 Connecting to Redis Cloud memory...")
                self.memory = get_redis_memory(session_id)
                print(f"✅ Redis Cloud memory ready for session: {session_id}")
            except Exception as e:
                print(f"⚠️ Redis Cloud memory failed: {e}")
                print("💡 Continuing without memory support")
                self.memory = None
        
        # ... rest of your initialization

    def get_chat_history_string(self):
        """Get formatted chat history from Redis Cloud"""
        if not self.memory:
            return ""
        
        try:
            memory_vars = self.memory.load_memory_variables({})
            messages = memory_vars.get("chat_history", [])
            
            if not messages:
                return ""
            
            # Format messages for the prompt
            formatted_history = []
            for msg in messages:
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    role = "Human" if msg.type == "human" else "Assistant"
                    formatted_history.append(f"{role}: {msg.content}")
            
            return "\n".join(formatted_history[-6:])  # Last 3 exchanges
            
        except Exception as e:
            print(f"⚠️ Error getting chat history from Redis Cloud: {e}")
            return ""

    def save_to_memory(self, question: str, answer: str):
        """Save conversation to Redis Cloud"""
        if not self.memory:
            return
        
        try:
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)
            print("💾 Conversation saved to Redis Cloud")
        except Exception as e:
            print(f"⚠️ Error saving to Redis Cloud: {e}")

# Updated helper function
def get_rag_bot(session_id="default"):
    """Get or create RAG bot with Redis Cloud memory"""
    global rag_bots
    if session_id not in rag_bots:
        rag_bots[session_id] = RAGChatbot(session_id)
    return rag_bots[session_id]

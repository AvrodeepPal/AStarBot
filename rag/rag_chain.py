# rag/rag_chain.py
"""
Main RAG pipeline for AStarBot
Handles query embedding, context retrieval, and response generation
"""

from sentence_transformers import SentenceTransformer
from .config import setup_pinecone, get_openrouter_client, get_config
from .prompt import get_rag_prompt, get_fallback_response

class RAGChatbot:
    def __init__(self):
        """Initialize the RAG chatbot with all required components"""
        self.config = get_config()
        print("🚀 Initializing AStarBot components...")
        
        # Initialize embedding model
        print("📥 Loading sentence transformer...")
        self.embedding_model = SentenceTransformer(self.config["embedding_model"])
        print("✅ Embedding model loaded")
        
        # Initialize Pinecone
        print("🔌 Connecting to Pinecone...")
        self.pinecone_index = setup_pinecone()
        print("✅ Pinecone connected")
        
        # Initialize OpenRouter client
        print("🤖 Setting up OpenRouter/Mistral...")
        self.openrouter_client = get_openrouter_client()
        print("✅ LLM client ready")
        
        self.prompt_template = get_rag_prompt()
        print("✅ AStarBot initialized successfully!")

    def embed_query(self, query: str):
        """
        Convert user query to embedding vector
        Args: query (str): User's question
        Returns: list: Embedding vector
        """
        try:
            return self.embedding_model.encode(query).tolist()
        except Exception as e:
            print(f"❌ Error embedding query: {e}")
            raise

    def retrieve_context(self, query_vector: list, top_k: int = 3):
        """
        Retrieve relevant context from Pinecone
        Args: 
            query_vector (list): Embedded query
            top_k (int): Number of results to retrieve
        Returns: list: Retrieved text contexts
        """
        try:
            results = self.pinecone_index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            contexts = []
            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                text = metadata.get("text", "")
                if text:
                    contexts.append(text)
            
            print(f"✅ Retrieved {len(contexts)} context chunks")
            
            # Also print context for debugging (like your Colab code)
            for i, c in enumerate(contexts, 1):
                print(f"{i}. {c[:120]}{'…' if len(c) > 120 else ''}")
            
            return contexts
            
        except Exception as e:
            print(f"❌ Error retrieving context: {e}")
            return []

    def generate_response(self, contexts: list, question: str):
        """
        Generate response using OpenRouter/Mistral 24B
        Args:
            contexts (list): Retrieved context texts
            question (str): User's question
        Returns: str: Generated response
        """
        try:
            # If no context found, return fallback
            if not contexts:
                return get_fallback_response()
            
            # Build context block (similar to your Colab approach)
            context_block = "\n".join([f"{i+1}. {ctx}" for i, ctx in enumerate(contexts)])
            
            # Format prompt using the template
            formatted_prompt = self.prompt_template.format(
                context=context_block,
                question=question
            )
            
            # Generate response via OpenRouter (upgraded from your HF approach)
            response = self.openrouter_client.chat.completions.create(
                model=self.config["llm_model"],
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=self.config["max_tokens"],
                temperature=self.config["llm_temperature"]
            )
            
            answer = response.choices[0].message.content.strip()
            print("✅ Answer generated")
            return answer
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your question. Please try again!"

    def chat(self, question: str):
        """
        Complete RAG pipeline: embed -> retrieve -> generate
        Args: question (str): User's question
        Returns: str: Final response
        """
        try:
            print(f"🤖 Processing: {question}")
            
            # Step 1: Embed the query
            query_vector = self.embed_query(question)
            print(f"✅ Query embedded (dim = {len(query_vector)})")
            
            # Step 2: Retrieve relevant context
            contexts = self.retrieve_context(query_vector, self.config["top_k_retrieval"])
            
            # Step 3: Generate response
            response = self.generate_response(contexts, question)
            
            return response
            
        except Exception as e:
            print(f"❌ Error in RAG pipeline: {e}")
            return "I'm sorry, something went wrong. Please try asking your question again."

# Global instance for FastAPI (singleton pattern)
rag_bot = None

def get_rag_bot():
    """Get or create RAG bot instance"""
    global rag_bot
    if rag_bot is None:
        rag_bot = RAGChatbot()
    return rag_bot

def generate_rag_response(question: str):
    """
    Simple function to generate RAG response
    Args: question (str): User's question
    Returns: str: Generated response
    """
    bot = get_rag_bot()
    return bot.chat(question)

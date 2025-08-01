# app.py
"""
Terminal-based chat application for testing AStarBot locally
Just like your Colab interactive experience!
"""

import os
from rag.rag_chain import generate_rag_response
from dotenv import load_dotenv

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🤖 AStarBot - Terminal Chat Interface")
    print("=" * 60)
    print("💬 Ask me anything about Avrodeep Pal!")
    print("📝 Type 'quit', 'exit', or 'bye' to end the conversation")
    print("🔄 Type 'clear' to clear the screen")
    print("-" * 60)

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """Main chat loop - similar to your Colab experience"""
    
    # Load environment variables
    load_dotenv()
    
    # Check environment setup
    required_vars = ["PINECONE_API_KEY", "OPENROUTER_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   • {var}")
        print("\n💡 Please check your .env file and try again.")
        return
    
    # Print banner
    clear_screen()
    print_banner()
    
    # Initialize chatbot
    try:
        print("🔄 Initializing AStarBot...")
        # Test the RAG chain by importing it
        from rag.rag_chain import get_rag_bot
        bot = get_rag_bot()
        print("✅ AStarBot is ready!\n")
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        print("💡 Make sure you've run the embedding script first:")
        print("   python embeddings/embed_data.py")
        return
    
    # Chat loop (like your Colab interactive function)
    conversation_count = 0
    
    while True:
        try:
            # Get user input (like your get_user_question function)
            user_input = input("👤 You: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Thanks for chatting with AStarBot! Say hi to Avrodeep!")
                break
            
            elif user_input.lower() == 'clear':
                clear_screen()
                print_banner()
                continue
            
            elif not user_input:
                print("💭 Please ask me something about Avrodeep!")
                continue
            
            # Generate response (like your RAG pipeline)
            print("🤖 AStarBot: ", end="", flush=True)
            
            try:
                response = generate_rag_response(user_input)
                print(response)
                conversation_count += 1
                
            except Exception as e:
                print(f"Sorry, I encountered an error: {e}")
                print("💡 Please try asking your question differently.")
            
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("💡 Please try again or restart the application.")
    
    # Final message
    if conversation_count > 0:
        print(f"\n📊 We had {conversation_count} exchanges about Avrodeep!")
    print("🎯 To use AStarBot in your web app, run: python main.py")

if __name__ == "__main__":
    main()

# app.py
"""
Command-line interface for AStarBot with Redis memory support
Enhanced CLI for testing the RAG chatbot with session management
"""

import os
import sys
from rag.rag_chain import get_rag_bot, clear_bot_memory
from rag.config import print_config_status
from utils.memory import get_memory_info

def print_banner():
    """Print AStarBot banner"""
    print("=" * 60)
    print("🤖 AStarBot CLI - Avrodeep Pal's AI Assistant")
    print("🧠 Enhanced with Redis Memory & Session Support")
    print("=" * 60)

def print_help():
    """Print available commands"""
    print("\n📋 Available Commands:")
    print("  !help     - Show this help message")
    print("  !session  - Show current session info")
    print("  !clear    - Clear current session memory")
    print("  !new      - Start new session")
    print("  !memory   - Show memory info")
    print("  !config   - Show configuration status")
    print("  !quit     - Exit AStarBot")
    print("  Or just ask any question about Avrodeep!")

def main():
    """Main CLI loop"""
    print_banner()
    print_config_status()
    
    # Get session ID from user
    session_input = input("\n📄 Enter session ID (press Enter for 'guest'): ").strip()
    session_id = session_input if session_input else "guest"
    
    print(f"\n🎯 Starting session: {session_id}")
    print("💡 Type !help for commands or ask questions about Avrodeep")
    print("🔥 Try: 'What are your main projects?' or 'Tell me about your education'")
    
    # Get bot instance for this session
    bot = get_rag_bot(session_id)
    
    print("\n" + "=" * 60)
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n[{session_id}] 🧑 You: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() == "!quit":
                print("\n👋 Thanks for chatting with AStarBot! Goodbye!")
                break
                
            elif user_input.lower() == "!help":
                print_help()
                continue
                
            elif user_input.lower() == "!config":
                print("\n🔧 Configuration Status:")
                print_config_status()
                continue
                
            elif user_input.lower() == "!session":
                print(f"\n📊 Current Session: {session_id}")
                memory_info = get_memory_info(session_id)
                if "error" in memory_info:
                    print(f"❌ Error: {memory_info['error']}")
                else:
                    print(f"💭 Total messages: {memory_info.get('total_messages', 0)}")
                    print(f"🔄 Recent exchanges: {len(memory_info.get('last_3_exchanges', []))}")
                continue
                
            elif user_input.lower() == "!clear":
                print(f"\n🧹 Clearing memory for session: {session_id}")
                bot.clear_memory()
                print("✅ Memory cleared!")
                continue
                
            elif user_input.lower() == "!new":
                import uuid
                new_session = str(uuid.uuid4())[:8]
                print(f"\n🆕 Starting new session: {new_session}")
                session_id = new_session
                bot = get_rag_bot(session_id)
                print("✅ New session ready!")
                continue
                
            elif user_input.lower() == "!memory":
                memory_info = get_memory_info(session_id)
                print(f"\n🧠 Memory Info for {session_id}:")
                if "error" in memory_info:
                    print(f"❌ Error: {memory_info['error']}")
                else:
                    print(f"📊 Total messages: {memory_info.get('total_messages', 0)}")
                    recent = memory_info.get('last_3_exchanges', [])
                    if recent:
                        print("💭 Recent conversation:")
                        for i, msg in enumerate(recent[-4:], 1):  # Show last 4 messages
                            role = "Human" if hasattr(msg, 'type') and msg.type == "human" else "Bot"
                            content = msg.content if hasattr(msg, 'content') else str(msg)
                            print(f"  {i}. {role}: {content[:80]}...")
                    else:
                        print("💭 No recent conversation")
                continue
            
            # Process regular questions
            print(f"\n[{session_id}] 🤖 AStarBot: ", end="", flush=True)
            response = bot.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("🔄 Please try again or type !quit to exit")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        print("🔧 Please check your configuration and try again")
        sys.exit(1)

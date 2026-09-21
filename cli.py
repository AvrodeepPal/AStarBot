# cli.py

from rag.engine import RAGEngine
from rag.prompt import INITIAL_MESSAGE

def run_cli():
    print("AStarBot CLI")
    print("Type 'exit' to quit.\n")

    engine = RAGEngine()

    # Conversation state (client-side simulation)
    recent_messages = []
    summary = None

    # Initial assistant message
    print(f"AStarBot: {INITIAL_MESSAGE}\n")
    recent_messages.append(
        {"role": "assistant", "content": INITIAL_MESSAGE}
    )

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = engine.chat(
            question=user_input,
            recent_messages=recent_messages,
            summary=summary,
        )

        answer = result["answer"]
        summary = result["updated_summary"]

        print(f"AStarBot: {answer}\n")

        recent_messages.append({"role": "user", "content": user_input})
        recent_messages.append({"role": "assistant", "content": answer})

        # Keep window bounded
        if len(recent_messages) > 12:
            recent_messages = recent_messages[-12:]


if __name__ == "__main__":
    run_cli()

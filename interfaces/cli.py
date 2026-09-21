"""Local terminal client.

    python -m interfaces.cli

Commands:
    /exit, /quit   leave
    /clear         reset messages and summary
    /debug         toggle retrieval/latency diagnostics
    /summary       print the current server-provided summary
    /help          list commands
"""

import logging

from interfaces.session import trim_window
from rag.config import settings
from rag.engine import RAGEngine
from rag.log import setup_logging
from rag.prompt import INITIAL_MESSAGE, PROMPT_VERSION

HELP = "Commands: /exit /quit /clear /debug /summary /help"


def _print_debug(dbg: dict) -> None:
    print(f"  [debug] request={dbg.get('request_id')} outcome={dbg.get('outcome')} "
          f"tier={dbg.get('llm_tier')} latency={dbg.get('latency_ms')}ms")
    for s in dbg.get("sources", []):
        print(f"  [debug]   {s['id']:<24} score={s['score']:.4f}")
    if dbg.get("summarized"):
        print("  [debug] summary refreshed")


def run_cli() -> None:
    # Keep JSON logs out of the chat transcript unless the user opts in.
    setup_logging("DEBUG" if settings.debug else "WARNING")
    logging.getLogger("astarbot").setLevel(logging.DEBUG if settings.debug else logging.WARNING)

    print(f"AStarBot CLI  ·  prompt {PROMPT_VERSION}  ·  {settings.primary_llm_model}  ·  k={settings.top_k}")
    print(HELP + "\n")

    engine = RAGEngine()
    recent_messages: list[dict] = [{"role": "assistant", "content": INITIAL_MESSAGE}]
    summary: str | None = None
    debug = settings.debug

    print(f"AStarBot: {INITIAL_MESSAGE}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in {"/exit", "/quit", "exit", "quit"}:
            print("Goodbye!")
            return
        if cmd == "/help":
            print(HELP + "\n")
            continue
        if cmd == "/clear":
            recent_messages = [{"role": "assistant", "content": INITIAL_MESSAGE}]
            summary = None
            print("Conversation cleared.\n")
            continue
        if cmd == "/debug":
            debug = not debug
            print(f"Debug {'on' if debug else 'off'}.\n")
            continue
        if cmd == "/summary":
            print(f"Summary: {summary or '(none yet)'}\n")
            continue

        result = engine.chat(
            question=user_input,
            recent_messages=recent_messages,
            summary=summary,
            include_debug=debug,
        )
        answer = result["answer"]
        print(f"AStarBot: {answer}")
        if debug and "debug" in result:
            _print_debug(result["debug"])
        print()

        recent_messages.append({"role": "user", "content": user_input})
        recent_messages.append({"role": "assistant", "content": answer})
        recent_messages = trim_window(recent_messages, summary, result["updated_summary"])
        summary = result["updated_summary"]


if __name__ == "__main__":
    run_cli()

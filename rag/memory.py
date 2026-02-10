# rag/memory.py

import os
from langchain_groq import ChatGroq

SUMMARY_PROMPT = """
Summarize the conversation so far in 3-5 concise lines.
Focus only on topics discussed and user intent.
Do not add new information or assumptions.
Keep the summary factual and neutral.
"""


def summarize_conversation(
    previous_summary: str | None,
    recent_messages: list[dict],
) -> str:
    """
    Generate a concise, factual summary of the conversation.

    Args:
        previous_summary (str | None): Existing summary (if any)
        recent_messages (list[dict]): Recent user/assistant turns

    Returns:
        str: Updated conversation summary
    """

    summary_input = ""

    if previous_summary:
        summary_input += f"Previous summary:\n{previous_summary}\n\n"

    summary_input += "Recent messages:\n"
    for msg in recent_messages:
        role = msg["role"].upper()
        content = msg["content"]
        summary_input += f"{role}: {content}\n"

    summarizer_llm = ChatGroq(
        model=os.getenv("SUMMARIZER_LLM_MODEL"),
        temperature=0.0,  # summaries must be deterministic
    )

    response = summarizer_llm.invoke(
        SUMMARY_PROMPT + "\n\n" + summary_input
    )

    return response.content.strip()

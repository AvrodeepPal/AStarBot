"""Client-side conversation window shared by the CLI and Streamlit app.

The backend is stateless, so every client must apply the same policy:
    - keep at most `max_recent_messages` turns
    - when the server returns a NEW summary, keep only the last
      `messages_after_summary` turns (the summary now covers the rest)

The React frontend should mirror this logic exactly.
"""

from rag.config import settings


def trim_window(messages: list[dict], previous_summary: str | None, new_summary: str | None) -> list[dict]:
    """Return the message window to keep after a response."""
    turns = [m for m in messages if m["role"] in {"user", "assistant"}]
    if new_summary and new_summary != previous_summary:
        return turns[-settings.messages_after_summary:]
    return turns[-settings.max_recent_messages:]
